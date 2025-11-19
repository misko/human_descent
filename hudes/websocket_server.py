#!/usr/bin/env python
# noqa: E501

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import pathlib
import secrets
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from queue import Empty
from threading import Thread

import torch
import torch.multiprocessing as mp

# import multiprocessing
import websockets
from websockets.asyncio.server import serve

from hudes import hudes_pb2
from hudes.high_scores import (
    get_top_scores,
    init_db,
    insert_high_score,
    delete_high_score,
    get_all_scores,
    get_rank,
)
from hudes.model_data_and_subspace import ModelDataAndSubspace
from hudes.models_and_datasets.mnist import (
    MNISTCNN,
    MNISTCNN3,
    MNISTFFNN,
    MNISTCNNFlipped,
    mnist_model_data_and_subpace,
)

client_idx = 0
active_clients = {}
# Track scheduled timeout tasks per client to finalize Speed Runs
active_speedrun_tasks: dict[int, asyncio.Task] = {}


def _cancel_speedrun_timeout(client_id: int):
    task = active_speedrun_tasks.pop(client_id, None)
    if task and not task.done():
        try:
            task.cancel()
            logging.debug("Cancelled speed run timeout task for client %s", client_id)
        except Exception:
            pass


# Map resume tokens to client ids for quick lookup
resume_token_to_client_id: dict[str, int] = {}
# Default Speed Run duration in seconds
# (overridable via HUDES_SPEED_RUN_SECONDS)
SPEED_RUN_SECONDS = int(os.environ.get("HUDES_SPEED_RUN_SECONDS", "120"))
RESUME_SESSION_TTL = int(os.environ.get("HUDES_RESUME_TTL", "120"))


def _pack_messages_len_prefixed(msgs: list[bytes]) -> bytes:
    out = bytearray()
    for m in msgs:
        if not isinstance(m, (bytes, bytearray)):
            continue
        out += len(m).to_bytes(4, "big")
        out += m
    return bytes(out)


def _countdown_kwargs(client) -> dict:
    """Build kwargs for speed_run_seconds_remaining, or empty if N/A.

    Returns a dict that can be splatted into a Control constructor.
    """
    data = {"speed_run_active": bool(getattr(client, "speed_run_active", False))}
    if not getattr(client, "speed_run_end_time", 0):
        return data
    remaining = max(0, int(client.speed_run_end_time - time.time()))
    srs = remaining if data["speed_run_active"] else 0
    data["speed_run_seconds_remaining"] = srs
    return data


def _generate_resume_token() -> str:
    return secrets.token_urlsafe(16)


def _generate_session_token() -> str:
    return secrets.token_urlsafe(12)


def _register_resume_token(client, rotate: bool = False):
    if client.resume_token and not rotate:
        resume_token_to_client_id[client.resume_token] = client.client_id
        return client.resume_token
    if client.resume_token and client.resume_token in resume_token_to_client_id:
        del resume_token_to_client_id[client.resume_token]
    client.resume_token = _generate_resume_token()
    resume_token_to_client_id[client.resume_token] = client.client_id
    return client.resume_token


def _attach_resume_token(control_msg: hudes_pb2.Control, client):
    if getattr(client, "resume_token", None):
        control_msg.resume_token = client.resume_token


async def _send_control_message(client, control_msg: hudes_pb2.Control):
    websocket = client.websocket
    if websocket is None or getattr(websocket, "closed", False):
        return
    _attach_resume_token(control_msg, client)
    try:
        await websocket.send(control_msg.SerializeToString())
    except (
        websockets.exceptions.ConnectionClosedError,
        websockets.exceptions.ConnectionClosedOK,
    ):
        logging.debug(
            "send_control_message: websocket closed for client %s", client.client_id
        )
    except Exception as exc:
        logging.error(
            "send_control_message: failed for client %s: %s",
            client.client_id,
            exc,
        )


async def _handle_resume_request(msg, websocket, client_runner_q):
    resume = msg.resume
    token = resume.token if resume is not None else ""
    if not token:
        await websocket.send(
            hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_RESUME,
                resume=hudes_pb2.Control.Resume(
                    status=hudes_pb2.Control.Resume.RESUME_REJECTED,
                ),
            ).SerializeToString()
        )
        return None
    client_id = resume_token_to_client_id.get(token)
    client = active_clients.get(client_id) if client_id is not None else None
    if client is None:
        await websocket.send(
            hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_RESUME,
                resume=hudes_pb2.Control.Resume(
                    status=hudes_pb2.Control.Resume.RESUME_NOT_FOUND,
                    token=token,
                ),
            ).SerializeToString()
        )
        return None
    if not client.resume_enabled:
        await websocket.send(
            hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_RESUME,
                resume=hudes_pb2.Control.Resume(
                    status=hudes_pb2.Control.Resume.RESUME_REJECTED,
                    token=token,
                ),
            ).SerializeToString()
        )
        return None

    provided_session = resume.client_session_token or ""
    expected_session = client.session_token or ""
    if expected_session and provided_session != expected_session:
        await websocket.send(
            hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_RESUME,
                resume=hudes_pb2.Control.Resume(
                    status=hudes_pb2.Control.Resume.RESUME_REJECTED,
                    token=token,
                ),
            ).SerializeToString()
        )
        return None

    new_session_token = (
        resume.new_client_session_token
        if resume and resume.new_client_session_token
        else _generate_session_token()
    )
    client.session_token = new_session_token
    _register_resume_token(client, rotate=True)

    prev_ws = client.websocket
    if prev_ws is not None and prev_ws is not websocket:
        try:
            await prev_ws.close()
        except Exception:
            pass
    client.websocket = websocket
    client.disconnected_at = None
    client.last_seen = time.time()
    client.resume_last_request_idx = resume.last_request_idx or 0
    client.force_update = True
    client.request_full_val = True
    client.sent_batch = -1
    client_runner_q.put(True)
    await _send_control_message(
        client,
        hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_RESUME,
            resume=hudes_pb2.Control.Resume(
                status=hudes_pb2.Control.Resume.RESUME_OK,
                token=client.resume_token,
                last_request_idx=client.request_idx,
                client_session_token=client.session_token,
            ),
        ),
    )
    return client


# TODO
# move to protobuf https://protobuf.dev/getting-started/pythontutorial/
# TODO need to do brew install brew install protobuf


# TODO cache?
def prepare_batch_example_message(
    batch_size: int,
    batch_idx: int,
    dtype: torch.dtype,
    mad: ModelDataAndSubspace,
    n: int = 4,
):
    batch = mad.get_batch(
        batch_size=batch_size,
        batch_idx=batch_idx,
        dtype=dtype,
        train_or_val="train",
    )

    train_data = batch[0][:n]
    train_labels = batch[1][:n]

    # Flatten the data for easy serialization
    train_data_flattened = train_data.flatten().tolist()
    train_labels_flattened = train_labels.flatten().tolist()

    # Get the shapes
    train_data_shape = list(train_data.size())
    train_labels_shape = list(train_labels.size())

    # Create protobuf message
    return hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_BATCH_EXAMPLES,
        batch_examples=hudes_pb2.BatchExamples(
            type=hudes_pb2.BatchExamples.Type.IMG_BW,
            n=n,
            train_data=train_data_flattened,
            train_data_shape=train_data_shape,
            train_labels=train_labels_flattened,
            train_labels_shape=train_labels_shape,
            batch_idx=batch_idx,
        ),
    )


def listen_and_run(
    in_q: mp.Queue,
    out_q: mp.Queue,
    mad: ModelDataAndSubspace,
):
    mad.move_to_device()
    mad.fuse()
    mad.init_param_model()

    client_weights = {}
    # TODO memory leak, but prevents continuously copying models

    logging.info("listen_and_run: started")
    while True:
        try:
            v = in_q.get()  # blocking
        except EOFError:
            logging.info("listen_and_run: returning")
            return

        if v is None:
            return
        else:
            # prepare weights
            mode, client = v
            # client_id = v["client_id"]

            res = {}
            logging.debug(f"listen_and_run: got {mode}")
            if mode in ("train", "mesh", "loss_line"):
                # reset per-client weights if requested (speed run start)
                if client.force_reset_weights:
                    if client.client_id in client_weights:
                        logging.info(
                            "listen_and_run: reset weights for client %s",
                            client.client_id,
                        )
                        del client_weights[client.client_id]
                    client.force_reset_weights = False
                if client.client_id not in client_weights:
                    client_weights[client.client_id] = mad.saved_weights[
                        torch.float32
                    ].clone()
                client_weights[client.client_id] += mad.delta_from_dims(
                    client.current_step, dtype=torch.float32
                )
                logging.debug(
                    "listen_and_run: client checksum %s",
                    client_weights[client.client_id].abs().mean(),
                )

                res["train"] = mad.train_model_inference_with_delta_weights(
                    client_weights[client.client_id].to(client.dtype),
                    batch_size=min(client.batch_size, mad.max_batch_size),
                    batch_idx=client.batch_idx,
                    dtype=client.dtype,
                )
            if mode == "val":
                res["val"] = mad.val_model_inference_with_delta_weights(
                    client_weights[client.client_id], dtype=client.dtype
                )
            if mode == "mesh":
                res["mesh"] = mad.get_loss_grid(
                    base_weights=client_weights[client.client_id].to(client.dtype),
                    grid_size=min(client.mesh_grid_size, mad.max_grid_size),
                    step_size=client.mesh_step_size / 2,
                    grids=min(client.mesh_grids, mad.max_grids),
                    dims_offset=client.dims_offset,
                    batch_idx=client.batch_idx,
                    batch_size=min(client.batch_size, mad.max_batch_size),
                    dtype=client.dtype,
                )
            if mode == "loss_line":
                res["loss_line"] = mad.get_loss_lines(
                    base_weights=client_weights[client.client_id].to(client.dtype),
                    grid_size=min(client.mesh_grid_size, mad.max_grid_size),
                    step_size=client.mesh_step_size / 2,
                    lines=min(
                        client.loss_lines,
                        client.dims_at_a_time or client.loss_lines,
                    ),
                    dims_offset=client.dims_offset,
                    batch_idx=client.batch_idx,
                    batch_size=min(client.batch_size, mad.max_batch_size),
                    dtype=client.dtype,
                )
            if mode == "sgd":
                res["train"], model_weights = mad.sgd_step(
                    client_weights[client.client_id].to(client.dtype),
                    batch_size=min(client.batch_size, mad.max_batch_size),
                    batch_idx=client.batch_idx,
                    dtype=client.dtype,
                    n_steps=client.sgd,
                )
                if client.dtype != torch.float32:
                    client_weights[client.client_id].data.copy_(
                        model_weights.to(torch.float32)
                    )
            if mode not in ("train", "mesh", "loss_line", "val", "sgd"):
                raise ValueError

            logging.debug(f"listen_and_run: return {mode}")
            out_q.put((client.client_id, mode, res))


@dataclass
class Client:
    client_id: int
    last_seen: float = 0.0
    next_step: dict = field(default_factory=lambda: dict())
    current_step: dict = None
    batch_idx: int = 0
    websocket: None = None
    request_idx: int = 0
    sent_batch: int = -1
    request_full_val: bool = False
    active_inference: int = 0
    active_request_idx: int = -1
    mesh_grid_size: int = -1
    mesh_grids: int = 0
    mesh_step_size: float = 0.0
    mesh_enabled: bool = True
    loss_lines: int = 0
    force_update: bool = False
    dims_offset: int = 0
    dims_at_a_time: int = 0
    seed: int = 0
    dtype: torch.dtype = torch.float32
    batch_size: int = 32
    sgd: int = 0
    total_sgd_steps: int = 0
    total_eval_steps: int = 0
    # Speed run state
    speed_run_active: bool = False
    speed_run_end_time: float = 0.0
    speed_run_seq: int = 0  # increments each time a Speed Run starts
    speed_run_log: list = field(default_factory=list)
    best_val_loss_during_run: float | None = None
    high_score_logged: bool = False
    # Signal to inference worker to drop/reset weights for this
    # client on next op
    force_reset_weights: bool = False
    resume_token: str | None = None
    resume_enabled: bool = False
    resume_last_request_idx: int = 0
    disconnected_at: float | None = None
    session_token: str = ""

    def __getstate__(self):
        state = self.__dict__.copy()
        if "websocket" in state:
            # cant pickle this? who doesnt like pickles?
            del state["websocket"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


async def inference_runner_clients(mad, client_runner_q, inference_q, stop):
    logging.info("inference_runner_clients: started")
    while True:
        try:
            if client_runner_q.empty():
                await asyncio.to_thread(client_runner_q.get, timeout=0.01)
        except Empty:
            pass

        if stop is not None and stop.done():
            logging.info("inference_result_sender: returning")
            return

        while not client_runner_q.empty():
            client_runner_q.get()

        # make requests
        for client_id, client in list(active_clients.items()):
            if client is None:
                continue
            websocket = client.websocket
            if websocket is None or getattr(websocket, "closed", False):
                continue

            # client still waiting for response just skip
            if client.active_inference > 0:
                logging.debug(
                    "inference_runner_clients: skip client %s active_inference=%s",
                    client.client_id,
                    client.active_inference,
                )
                continue

            client.active_request_idx = client.request_idx
            # TODO should be some kind of select not poll()
            if client.sent_batch != client.batch_idx:
                try:
                    logging.debug("inference_runner_clients: send batch msg")
                    msg = prepare_batch_example_message(
                        batch_size=client.batch_size,
                        batch_idx=client.batch_idx,
                        dtype=client.dtype,
                        mad=mad,
                    )
                    _attach_resume_token(msg, client)
                    await websocket.send(msg.SerializeToString())
                except (
                    websockets.exceptions.ConnectionClosedOK,
                    websockets.exceptions.ConnectionClosedError,
                ):
                    pass
                except Exception as e:
                    logging.error("Error in prepare batch", e)
                client.sent_batch = client.batch_idx
                client.force_update = True

            if client.active_inference == 0 and client.sgd > 0:
                client.active_inference += 1
                inference_q.put(("sgd", copy.copy(client)))
                # reset flag after sending to worker to avoid repeated resets
                if client.force_reset_weights:
                    logging.debug(
                        "inference_runner_clients: clear reset flag "
                        "after scheduling sgd"
                    )
                    client.force_reset_weights = False
                client.sgd = 0

                # if we have grids, step size changes mesh
                if (
                    client.mesh_enabled and client.mesh_grids > 0
                ) or client.loss_lines > 0:
                    client.force_update = True

            if client.force_update or (
                len(client.next_step) > 0 and client.active_inference == 0
            ):
                client.current_step = client.next_step
                client.next_step = {}

                next_mode = "train"
                if client.mesh_enabled and client.mesh_grids > 0:
                    next_mode = "mesh"
                elif client.loss_lines > 0:
                    next_mode = "loss_line"

                logging.debug(
                    "inference_runner_clients: req %s %s",
                    next_mode,
                    client.force_update,
                )
                client.active_inference += 1
                inference_q.put((next_mode, copy.copy(client)))
                if client.force_reset_weights:
                    logging.debug(
                        "inference_runner_clients: clear reset flag "
                        "after scheduling %s",
                        next_mode,
                    )
                    client.force_reset_weights = False
                client.force_update = False

            if client.request_full_val:
                # send weight vector for inference
                logging.debug(
                    "inference_runner_clients: req inference for client %s",
                    client.client_id,
                )
                client.active_inference += 1
                inference_q.put(("val", copy.copy(client)))
                if client.force_reset_weights:
                    logging.debug(
                        "inference_runner_clients: clear reset flag "
                        "after scheduling val"
                    )
                    client.force_reset_weights = False
                client.request_full_val = False
                logging.debug(
                    "inference_runner_clients: client %s request_full_val cleared",
                    client.client_id,
                )

            # Note: Do not schedule periodic mesh updates here. Mesh requests
            # are driven by user interactions (force_update/next_step) and
            # initial config changes. The Speed Run countdown is handled on the
            # client side to avoid unnecessary server-side work.


async def inference_result_sender(results_q, stop):
    logging.info("inference_result_sender: started")
    while True:
        if not results_q.empty():
            msg = results_q.get()
        else:
            msg = await asyncio.to_thread(results_q.get)

        if stop is not None and stop.done():
            logging.info("inference_result_sender: returning")
            return

        client_id, train_or_val, res = msg
        client = active_clients[client_id]

        client.active_inference -= 1  # allow next thing to run
        # asyncio.sleep(0.00001)
        websocket = client.websocket
        if websocket is None or getattr(websocket, "closed", False):
            logging.debug(
                "inference_result_sender: websocket missing for client %s", client_id
            )
            continue

        if train_or_val in ("train", "mesh", "loss_line", "sgd"):
            if train_or_val in ("train", "mesh", "loss_line"):
                client.total_eval_steps += 1
            logging.debug("inference_result_sender: sent train to client")
            control_msg = hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
                train_loss_and_preds=hudes_pb2.TrainLossAndPreds(
                    train_loss=res["train"]["train_loss"],
                    preds=res["train"]["train_preds"].cpu().float().flatten().tolist(),
                    preds_shape=list(res["train"]["train_preds"].shape),
                    confusion_matrix=res["train"]["confusion_matrix"]
                    .cpu()
                    .float()
                    .flatten()
                    .tolist(),
                    confusion_matrix_shape=list(res["train"]["confusion_matrix"].shape),
                ),
                request_idx=client.active_request_idx,
                total_sgd_steps=client.total_sgd_steps,
                total_eval_steps=client.total_eval_steps,
                **_countdown_kwargs(client),
            )
            await _send_control_message(client, control_msg)
            logging.debug("inference_result_sender: sent train to client : done")
        if train_or_val == "val":
            logging.debug("inference_result_sender: sent val to client")
            speed_finished = False
            if getattr(client, "speed_run_active", False):
                remaining = max(0, int(client.speed_run_end_time - time.time()))
                if remaining <= 0:
                    client.speed_run_active = False
                    speed_finished = True
            rank = None
            total_scores = None
            if speed_finished:
                logging.info(
                    "Speed run finished for client %s; sending VAL loss %.6f",
                    client.client_id,
                    res["val"]["val_loss"],
                )
                try:
                    init_db()
                    r, t = get_rank(res["val"]["val_loss"])
                    rank = r
                    total_scores = t + 1  # +1 because we are about to add this score
                except Exception as e:
                    logging.error("Failed to calculate rank: %s", e)

            await _send_control_message(
                client,
                hudes_pb2.Control(
                    type=hudes_pb2.Control.CONTROL_VAL_LOSS,
                    val_loss=hudes_pb2.ValLoss(
                        val_loss=res["val"]["val_loss"],
                        rank=rank,
                        total_scores=total_scores,
                    ),
                    request_idx=client.active_request_idx,
                    speed_run_finished=speed_finished,
                    total_eval_steps=client.total_eval_steps,
                    **_countdown_kwargs(client),
                ),
            )
            if client.best_val_loss_during_run is None:
                client.best_val_loss_during_run = res["val"]["val_loss"]
            else:
                client.best_val_loss_during_run = min(
                    client.best_val_loss_during_run, res["val"]["val_loss"]
                )
            logging.debug("inference_result_sender: sent val to client : done")
        if train_or_val == "mesh":
            logging.debug("inference_result_sender: sent mesh to client")
            mesh_tensor = res["mesh"].cpu().float()
            mesh_grid_results_list = mesh_tensor.numpy().flatten().tolist()
            mesh_grid_shape = list(mesh_tensor.shape)
            await _send_control_message(
                client,
                hudes_pb2.Control(
                    type=hudes_pb2.Control.CONTROL_MESHGRID_RESULTS,
                    mesh_grid_results=mesh_grid_results_list,
                    mesh_grid_shape=mesh_grid_shape,
                    total_eval_steps=client.total_eval_steps,
                    **_countdown_kwargs(client),
                ),
            )
            logging.debug("inference_result_sender: sent mesh to client : done")
        if train_or_val == "loss_line":
            logging.debug("inference_result_sender: sent loss line to client")
            loss_tensor = res["loss_line"]
            await _send_control_message(
                client,
                hudes_pb2.Control(
                    type=hudes_pb2.Control.CONTROL_LOSS_LINE_RESULTS,
                    loss_line_results=loss_tensor.flatten().tolist(),
                    loss_line_shape=list(loss_tensor.shape),
                    total_eval_steps=client.total_eval_steps,
                    **_countdown_kwargs(client),
                ),
            )
            logging.debug("inference_result_sender: sent loss line to client : done")
        if train_or_val not in ("train", "val", "mesh", "loss_line", "sgd"):
            raise ValueError

        assert client.active_inference >= 0


async def wait_for_stop(inference_q, results_q, stop, client_runner_q):
    await stop
    results_q.put(None)
    inference_q.put(None)
    client_runner_q.put(True)


def _schedule_speedrun_timeout(client_id: int, duration_sec: int, client_runner_q):
    logging.debug(
        "Scheduling speed run timeout for client %s in %s seconds",
        client_id,
        duration_sec,
    )

    async def _timeout_then_eval():
        try:
            await asyncio.sleep(max(0, duration_sec))
            logging.debug("Speed run timeout fired for client %s", client_id)
            # Verify client still exists and the run sequence matches
            client = active_clients.get(client_id)
            if client is None:
                logging.debug("Speed run timeout: client %s missing", client_id)
                return
            # If a newer speed run started, ignore this timeout
            if time.time() < client.speed_run_end_time:
                logging.debug(
                    "Speed run timeout: client %s has newer run (end_time %.3f > now)",
                    client_id,
                    client.speed_run_end_time,
                )
                return
            # Request a normal validation at run end; client will use it
            # as the final score
            client.request_full_val = True
            logging.debug(
                "Speed run timeout: client=%s set request_full_val=True (active=%s, end_time=%.3f)",
                client_id,
                getattr(client, "speed_run_active", None),
                getattr(client, "speed_run_end_time", 0.0),
            )
            client_runner_q.put(True)
            logging.debug(
                "Speed run timeout enqueued final validation for client %s",
                client_id,
            )
        except Exception as e:
            logging.error("Speed Run timeout scheduling error: %s", e)

    # Cancel previous task if any
    _cancel_speedrun_timeout(client_id)
    task = asyncio.create_task(_timeout_then_eval())
    active_speedrun_tasks[client_id] = task


async def inference_runner(
    mad: ModelDataAndSubspace,
    client_runner_q: mp.Queue,
    stop,
    run_in: str = "process",
):
    if run_in == "process":
        ctx = mp.get_context("spawn")
        inference_q = ctx.Queue()
        results_q = ctx.Queue()
        process_or_thread = ctx.Process(
            target=listen_and_run, args=(inference_q, results_q, mad)
        )
    elif run_in == "thread":  # useful for debuggin
        inference_q = mp.Queue()
        results_q = mp.Queue()
        process_or_thread = Thread(
            target=listen_and_run, args=(inference_q, results_q, mad)
        )
    process_or_thread.daemon = True
    process_or_thread.start()

    logging.info("Inference runner running...")
    await asyncio.gather(
        inference_runner_clients(mad, client_runner_q, inference_q, stop),
        inference_result_sender(results_q, stop),
        wait_for_stop(inference_q, results_q, stop, client_runner_q),
    )
    logging.info("Inference runner stopping...")
    process_or_thread.join()


async def process_client(websocket, client_runner_q):
    global client_idx
    client = None

    try:
        async for message in websocket:
            msg = hudes_pb2.Control()
            msg.ParseFromString(message)

            if client is None:
                if msg.type == hudes_pb2.Control.CONTROL_RESUME:
                    client = await _handle_resume_request(
                        msg, websocket, client_runner_q
                    )
                    if client is None:
                        continue
                    # Resume handshake handled entirely; wait for next message
                    continue
                current_client = client_idx
                client_idx += 1
                client = Client(
                    client_id=current_client,
                    last_seen=time.time(),
                    next_step={},
                    batch_idx=0,
                    websocket=websocket,
                    request_idx=0,
                    active_inference=0,
                    sent_batch=-1,
                )
                active_clients[current_client] = client
                logging.debug("process_client: start for client %s", client.client_id)

            client.last_seen = time.time()

            if client.speed_run_active:
                now = time.time()
                if now < client.speed_run_end_time:
                    client.speed_run_log.append(message)
                else:
                    client.speed_run_active = False

            if msg.type == hudes_pb2.Control.CONTROL_DIMS:
                logging.debug("process_client: %s : control dims", client.client_id)
                for dim_and_step in msg.dims_and_steps:
                    dim = dim_and_step.dim + client.dims_offset
                    client.next_step[dim] = (
                        client.next_step.get(dim, 0) + dim_and_step.step
                    )
                client.request_idx = msg.request_idx
            elif msg.type == hudes_pb2.Control.CONTROL_NEXT_BATCH:
                logging.debug("process_client: %s : next batch", client.client_id)
                client.batch_idx += 1
                client.request_full_val = True
                client.request_idx = msg.request_idx
            elif msg.type == hudes_pb2.Control.CONTROL_NEXT_DIMS:
                logging.debug("process_client: %s : next dims", client.client_id)
                client.dims_offset += client.dims_at_a_time
                client.force_update = True
            elif msg.type == hudes_pb2.Control.CONTROL_CONFIG:
                logging.debug("process_client: %s : control config", client.client_id)
                old_batch_size = client.batch_size
                old_dtype = client.dtype
                old_mesh_enabled = client.mesh_enabled
                old_mesh_grids = client.mesh_grids
                old_loss_lines = client.loss_lines

                client.dims_at_a_time = msg.config.dims_at_a_time
                client.seed = msg.config.seed
                client.mesh_grid_size = msg.config.mesh_grid_size
                client.mesh_grids = msg.config.mesh_grids
                client.mesh_step_size = msg.config.mesh_step_size
                client.batch_size = msg.config.batch_size
                client.dtype = getattr(torch, msg.config.dtype)
                client.mesh_enabled = msg.config.mesh_enabled
                client.loss_lines = msg.config.loss_lines
                client.resume_enabled = getattr(msg.config, "resume_supported", True)
                if getattr(msg.config, "client_session_token", None):
                    client.session_token = msg.config.client_session_token
                elif not client.session_token:
                    client.session_token = _generate_session_token()
                if client.resume_enabled:
                    _register_resume_token(client)

                if (
                    client.mesh_enabled
                    and client.mesh_grids > 0
                    or client.loss_lines > 0
                    or old_batch_size != client.batch_size
                    or old_dtype != client.dtype
                    or old_mesh_enabled != client.mesh_enabled
                    or old_mesh_grids != client.mesh_grids
                    or old_loss_lines != client.loss_lines
                ):
                    client.force_update = True

            elif msg.type == hudes_pb2.Control.CONTROL_QUIT:
                logging.debug("process_client: %s : quit", client.client_id)
                client_runner_q.put(True)
                break
            elif msg.type == hudes_pb2.Control.CONTROL_SGD_STEP:
                if client.speed_run_active:
                    logging.debug("process_client: ignoring SGD during speed run")
                else:
                    client.sgd += msg.sgd_steps
                    client.total_sgd_steps += msg.sgd_steps
                    client.request_idx = msg.request_idx
            elif msg.type == hudes_pb2.Control.CONTROL_SPEED_RUN_START:
                logging.debug("process_client: %s : speed run start", client.client_id)
                client.next_step = {}
                client.current_step = None
                client.batch_idx = 0
                client.request_idx = 0
                client.sent_batch = -1
                client.request_full_val = True
                client.dims_offset = 0
                client.total_sgd_steps = 0
                client.sgd = 0
                client.best_val_loss_during_run = None
                client.speed_run_log = []
                client.high_score_logged = False
                duration = int(
                    os.environ.get(
                        "HUDES_SPEED_RUN_SECONDS",
                        str(SPEED_RUN_SECONDS),
                    )
                )
                client.speed_run_active = True
                client.speed_run_seq += 1
                client.speed_run_end_time = time.time() + max(1, duration)
                logging.info(
                    "Client %d Speed Run started for %d seconds",
                    client.client_id,
                    duration,
                )
                _schedule_speedrun_timeout(client.client_id, duration, client_runner_q)
                client.force_update = True
                client.force_reset_weights = True
            elif msg.type == hudes_pb2.Control.CONTROL_SPEED_RUN_CANCEL:
                logging.debug("process_client: %s : speed run cancel", client.client_id)
                if client.speed_run_active:
                    client.speed_run_active = False
                    client.speed_run_end_time = 0.0
                    client.speed_run_log = []
                    client.best_val_loss_during_run = None
                    _cancel_speedrun_timeout(client.client_id)
                    client.force_update = True
                    client_runner_q.put(True)
                    logging.info(
                        "Client %d Speed Run cancelled and returning to normal play",
                        client.client_id,
                    )
                else:
                    logging.debug(
                        "process_client: %s : cancel ignored (no active run)",
                        client.client_id,
                    )
            elif msg.type == hudes_pb2.Control.CONTROL_HIGH_SCORE_LOG:
                logging.debug("process_client: %s : high score log", client.client_id)
                if client.high_score_logged:
                    logging.debug("process_client: high score already logged; ignoring")
                else:
                    name = (
                        msg.high_score.name.strip().upper()
                        if msg.high_score
                        else "????"
                    )
                    if len(name) != 4 or not name.isalnum():
                        logging.info(
                            "Invalid high score name submitted; ignoring run persistence"
                        )
                        client_runner_q.put(True)
                        continue
                    duration = int(
                        os.environ.get(
                            "HUDES_SPEED_RUN_SECONDS",
                            str(SPEED_RUN_SECONDS),
                        )
                    )
                    score = (
                        client.best_val_loss_during_run
                        if client.best_val_loss_during_run is not None
                        else float("inf")
                    )
                    try:
                        init_db()
                        insert_high_score(
                            name=name,
                            score=score,
                            best_val_loss=score,
                            duration=duration,
                            request_idx=client.request_idx,
                            log_bytes=_pack_messages_len_prefixed(client.speed_run_log),
                        )
                        client.high_score_logged = True
                    except Exception as e:
                        logging.error(f"Failed to persist high score: {e}")
            elif msg.type == hudes_pb2.Control.CONTROL_LEADERBOARD_REQUEST:
                logging.debug(
                    "process_client: %s : leaderboard request", client.client_id
                )
                try:
                    init_db()
                    rows = get_top_scores(limit=10)
                    names = [str(r[0]) for r in rows]
                    scores = [float(r[1]) for r in rows]
                    await _send_control_message(
                        client,
                        hudes_pb2.Control(
                            type=hudes_pb2.Control.CONTROL_LEADERBOARD_RESPONSE,
                            leaderboard_names=names,
                            leaderboard_scores=scores,
                        ),
                    )
                except Exception as e:
                    logging.error("Failed to fetch leaderboard: %s", e)
            elif msg.type == hudes_pb2.Control.CONTROL_RESUME:
                logging.debug(
                    "process_client: %s : duplicate resume ignored",
                    client.client_id,
                )
            else:
                logging.warning("received invalid type from client")

            client_runner_q.put(True)
    except (
        websockets.exceptions.ConnectionClosedOK,
        websockets.exceptions.ConnectionClosedError,
    ):
        pass
    finally:
        if client is not None and client.websocket is websocket:
            client.websocket = None
            client.disconnected_at = time.time()
            if client.resume_enabled:
                _register_resume_token(client)


async def run_server(stop, client_runner_q, server_port, ssl_pem):
    # Standalone HTTP health server on a separate port to avoid interfering
    # with the WebSocket handshake. Defaults to server_port + 1.
    health_server = None
    health_port = int(os.environ.get("HUDES_HEALTH_PORT", server_port + 1))

    async def handle_health(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        try:
            await reader.read(1024)
            # Very small, permissive parser; always return 200
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain; charset=utf-8\r\n"
                b"Content-Length: 2\r\n"
                b"Connection: close\r\n"
                b"Access-Control-Allow-Origin: *\r\n\r\nOK"
            )
            writer.write(response)
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def handle_api(reader, writer):
        req = await reader.read(4096)
        first = (req or b"GET / HTTP/1.1\r\n").split(b"\r\n", 1)[0]
        parts = first.split()
        path = parts[1].decode("utf-8") if len(parts) >= 2 else "/health"

        def send_json(payload: str, status: str = "200 OK"):
            body = payload.encode("utf-8")
            headers = (
                f"HTTP/1.1 {status}\r\n"
                "Content-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, OPTIONS, DELETE\r\n"
                "Access-Control-Allow-Headers: *\r\n\r\n"
            ).encode("utf-8")
            writer.write(headers + body)

        if first.startswith(b"OPTIONS"):
            send_json("", status="204 No Content")
            return

        if first.startswith(b"DELETE") and path.startswith("/api/highscores"):
            try:
                import urllib.parse as _url

                qs = ""
                if "?" in path:
                    _, qs = path.split("?", 1)
                params = _url.parse_qs(qs)
                row_id = int(params.get("id", [0])[0])
                if row_id > 0:
                    init_db()
                    success = delete_high_score(row_id)
                    if success:
                        send_json('{"status": "deleted"}')
                    else:
                        send_json('{"error": "not found"}', status="404 Not Found")
                else:
                    send_json('{"error": "invalid id"}', status="400 Bad Request")
            except Exception:
                send_json(
                    '{"error": "failed"}',
                    status="500 Internal Server Error",
                )
            return

        if path == "/health":
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain; charset=utf-8\r\n"
                b"Content-Length: 2\r\n"
                b"Connection: close\r\n"
                b"Access-Control-Allow-Origin: *\r\n\r\nOK"
            )
            writer.write(response)
        elif path == "/api/top10":
            try:
                init_db()
                rows = get_top_scores()

                def _sf(x):
                    try:
                        fx = float(x)
                        return fx if math.isfinite(fx) else None
                    except Exception:
                        return None

                data = []
                for n, s, ts in rows:
                    data.append({"name": n, "score": _sf(s), "ts": ts})
                payload = json.dumps(data)
                send_json(payload)
            except Exception:
                send_json(
                    '{"error": "failed"}',
                    status="500 Internal Server Error",
                )
        elif path.startswith("/api/highscores"):
            try:
                import urllib.parse as _url

                qs = ""
                if "?" in path:
                    _, qs = path.split("?", 1)
                params = _url.parse_qs(qs)
                offset = int(params.get("offset", [0])[0])
                limit = int(params.get("limit", [10000])[0])
                limit = max(1, min(limit, 10000))
                init_db()
                rows = get_all_scores(offset=offset, limit=limit)

                def _sf(x):
                    try:
                        fx = float(x)
                        return fx if math.isfinite(fx) else None
                    except Exception:
                        return None

                data = []
                for r in rows:
                    data.append(
                        {
                            "id": r[0],
                            "ts": r[1],
                            "name": r[2],
                            "score": _sf(r[3]),
                            "bestValLoss": _sf(r[4]),
                            "duration": r[5],
                            "requestIdx": r[6],
                        }
                    )
                payload = json.dumps(data)
                send_json(payload)
            except Exception:
                send_json(
                    '{"error": "failed"}',
                    status="500 Internal Server Error",
                )
        elif path.startswith("/api/rank"):
            try:
                import urllib.parse as _url

                qs = path.split("?", 1)[1] if "?" in path else ""
                params = _url.parse_qs(qs)
                score = float(params.get("score", ["nan"])[0])
                init_db()
                rank, total = get_rank(score)
                payload = json.dumps({"rank": rank, "total": total})
                send_json(payload)
            except Exception:
                send_json(
                    '{"error": "failed"}',
                    status="500 Internal Server Error",
                )
        else:
            writer.write(
                b"HTTP/1.1 404 Not Found\r\n"
                b"Content-Length: 0\r\n"
                b"Connection: close\r\n"
                b"Access-Control-Allow-Origin: *\r\n\r\n"
            )

    ssl_context = None
    if ssl_pem is not None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        localhost_pem = pathlib.Path(__file__).with_name(ssl_pem)
        ssl_context.load_cert_chain(localhost_pem)
    # Start health HTTP server
    try:
        # Listen on all IPv4 addresses; for IPv6, consider also binding to '::'
        health_server = await asyncio.start_server(
            handle_api, host="0.0.0.0", port=health_port
        )
        logging.info(
            "API server listening on 0.0.0.0:%d (health at /health)",
            health_port,
        )
    except OSError as e:
        logging.warning(
            "Failed to start health server on %d: %s",
            health_port,
            e,
        )

    async with serve(
        partial(process_client, client_runner_q=client_runner_q),
        None,
        server_port,
        ssl=ssl_context,
        # Do not intercept HTTP requests here; health server handles them
    ):
        try:
            await stop
        finally:
            if health_server is not None:
                health_server.close()
                await health_server.wait_closed()


async def run_wrapper(args, stop_future=None):
    if args.device == "mps" and (
        not getattr(torch.backends, "mps", False)
        or not torch.backends.mps.is_available()
    ):
        logging.warning("MPS device not found, using CPU")
        args.device = "cpu"
    if args.device == "cuda" and (
        not getattr(torch, "cuda", False) or not torch.cuda.is_available()
    ):
        logging.warning("CUDA device not found, using CPU")
        args.device = "cpu"
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    client_runner_q = mp.Queue()
    executor = ThreadPoolExecutor(max_workers=12)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)

    param_models = None
    if args.model == "cnn":
        model = MNISTCNN()
    elif args.model == "cnn3":
        model = MNISTCNN3()
    elif args.model == "cnn2":
        model = MNISTCNN()

        param_models = {
            torch.float16: MNISTCNNFlipped(),
            torch.float32: MNISTCNNFlipped(),
        }
    elif args.model == "ffnn":
        model = MNISTFFNN()
    else:
        raise ValueError
    n_params = sum([p.numel() for p in model.parameters()])
    logging.info(f"Initialized model with {n_params} parameters")

    mad = mnist_model_data_and_subpace(
        model=model,
        device=args.device,
        param_models=param_models,
        max_batch_size=args.max_batch_size,
        max_grids=args.max_grids,
        max_grid_size=args.max_grid_size,
    )
    if args.download_dataset_and_exit:
        return
    # Log configured Speed Run duration for this server
    try:
        configured_duration = int(
            os.environ.get("HUDES_SPEED_RUN_SECONDS", str(SPEED_RUN_SECONDS))
        )
    except Exception:
        configured_duration = SPEED_RUN_SECONDS
    logging.info(
        "Speed Run configured duration (seconds): %d",
        configured_duration,
    )
    stop = stop_future or asyncio.get_running_loop().create_future()
    await asyncio.gather(
        run_server(stop, client_runner_q, args.port, args.ssl_pem),
        inference_runner(
            mad, run_in=args.run_in, client_runner_q=client_runner_q, stop=stop
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Hudes: Server")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="ffnn",
        choices=["cnn", "ffnn", "cnn2", "cnn3"],
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-grids",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )
    parser.add_argument(
        "--max-grid-size",
        type=int,
        default=41,
    )
    parser.add_argument(
        "--ssl-pem",
        type=str,
        default=None,
    )
    parser.add_argument("--run-in", type=str, default="process")
    parser.add_argument(
        "--download-dataset-and-exit",
        action="store_true",
    )

    args = parser.parse_args()
    asyncio.run(run_wrapper(args))
