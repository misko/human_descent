#!/usr/bin/env python

import argparse
import asyncio
import copy
import logging
import os
import pathlib
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
from hudes.high_scores import init_db, insert_high_score
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
SPEED_RUN_SECONDS = int(os.environ.get("HUDES_SPEED_RUN_SECONDS", "60"))


def _pack_messages_len_prefixed(msgs: list[bytes]) -> bytes:
    out = bytearray()
    for m in msgs:
        if not isinstance(m, (bytes, bytearray)):
            continue
        out += len(m).to_bytes(4, "big")
        out += m
    return bytes(out)


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
            if mode in ("train", "mesh"):
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
            if mode not in ("train", "mesh", "val", "sgd"):
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
    force_update: bool = False
    dims_offset: int = 0
    dims_at_a_time: int = 0
    seed: int = 0
    dtype: torch.dtype = torch.float32
    batch_size: int = 32
    sgd: int = 0
    total_sgd_steps: int = 0
    # Speed run state
    speed_run_active: bool = False
    speed_run_end_time: float = 0.0
    speed_run_log: list = field(default_factory=list)
    best_val_loss_during_run: float | None = None
    high_score_logged: bool = False
    # Signal to inference worker to drop/reset weights for this
    # client on next op
    force_reset_weights: bool = False

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
        for client_id in range(len(active_clients)):
            client = active_clients[client_id]

            # client still waiting for response just skip
            if client.active_inference > 0:
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
                    ).SerializeToString()
                    await client.websocket.send(msg)
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
                if client.mesh_grids > 0:
                    client.force_update = True

            if client.force_update or (
                len(client.next_step) > 0 and client.active_inference == 0
            ):
                client.current_step = client.next_step
                client.next_step = {}

                if client.mesh_grids > 0:
                    logging.debug(
                        "inference_runner_clients: req mesh %s",
                        client.force_update,
                    )
                    client.active_inference += 1
                    inference_q.put(("mesh", copy.copy(client)))
                    if client.force_reset_weights:
                        logging.debug(
                            "inference_runner_clients: clear reset flag "
                            "after scheduling mesh"
                        )
                        client.force_reset_weights = False
                else:
                    logging.debug(
                        "inference_runner_clients: req train %s",
                        client.force_update,
                    )
                    client.active_inference += 1
                    inference_q.put(("train", copy.copy(client)))
                    if client.force_reset_weights:
                        logging.debug(
                            "inference_runner_clients: clear reset flag "
                            "after scheduling train"
                        )
                        client.force_reset_weights = False
                client.force_update = False

            if client.request_full_val:
                # send weight vector for inference
                logging.debug("inference_runner_clients: req inference")
                client.active_inference += 1
                inference_q.put(("val", copy.copy(client)))
                if client.force_reset_weights:
                    logging.debug(
                        "inference_runner_clients: clear reset flag "
                        "after scheduling val"
                    )
                    client.force_reset_weights = False
                client.request_full_val = False

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
        try:
            if train_or_val in ("train", "mesh", "sgd"):
                # TODO need to be ok with getting errors here
                logging.debug("inference_result_sender: sent train to client")
                # compute remaining seconds to possibly flip state
                if client.speed_run_active:
                    remaining = max(0, int(client.speed_run_end_time - time.time()))
                    if remaining <= 0:
                        client.speed_run_active = False
                msg = hudes_pb2.Control(
                    type=hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
                    train_loss_and_preds=hudes_pb2.TrainLossAndPreds(
                        train_loss=res["train"]["train_loss"],
                        preds=res["train"]["train_preds"]
                        .cpu()
                        .float()
                        .flatten()
                        .tolist(),
                        preds_shape=list(res["train"]["train_preds"].shape),
                        confusion_matrix=res["train"]["confusion_matrix"]
                        .cpu()
                        .float()
                        .flatten()
                        .tolist(),
                        confusion_matrix_shape=list(
                            res["train"]["confusion_matrix"].shape
                        ),
                    ),
                    request_idx=client.active_request_idx,
                    total_sgd_steps=client.total_sgd_steps,
                ).SerializeToString()
                await client.websocket.send(msg)
                logging.debug("inference_result_sender: sent train to client : done")
            if train_or_val == "val":
                logging.debug("inference_result_sender: sent val to client")
                # track best validation loss during speed run
                if client.speed_run_active:
                    remaining = max(0, int(client.speed_run_end_time - time.time()))
                    if remaining <= 0:
                        client.speed_run_active = False
                await client.websocket.send(
                    hudes_pb2.Control(
                        type=hudes_pb2.Control.CONTROL_VAL_LOSS,
                        val_loss=hudes_pb2.ValLoss(
                            val_loss=res["val"]["val_loss"],
                        ),
                        request_idx=client.active_request_idx,
                    ).SerializeToString()
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

                # Convert the tensor to a list of floats and capture the shape
                mesh_tensor = res["mesh"].cpu().float()
                mesh_grid_results_list = mesh_tensor.numpy().flatten().tolist()
                mesh_grid_shape = list(mesh_tensor.shape)

                # Send the message using repeated float and include shape
                # compute remaining seconds and possibly flip active flag
                srs = None
                if client.speed_run_end_time:
                    remaining = max(0, int(client.speed_run_end_time - time.time()))
                    if getattr(client, "speed_run_active", False) and remaining <= 0:
                        client.speed_run_active = False
                    srs = remaining if client.speed_run_active else 0
                await client.websocket.send(
                    hudes_pb2.Control(
                        type=hudes_pb2.Control.CONTROL_MESHGRID_RESULTS,
                        mesh_grid_results=mesh_grid_results_list,
                        mesh_grid_shape=mesh_grid_shape,
                        speed_run_seconds_remaining=srs,
                    ).SerializeToString()
                )
                logging.debug("inference_result_sender: sent mesh to client : done")
            if train_or_val not in ("train", "val", "mesh", "sgd"):
                raise ValueError
        except (
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
        ):
            pass
        assert client.active_inference >= 0


async def wait_for_stop(inference_q, results_q, stop, client_runner_q):
    await stop
    results_q.put(None)
    inference_q.put(None)
    client_runner_q.put(True)


async def inference_runner(
    mad: ModelDataAndSubspace,
    client_runner_q: mp.Queue,
    stop,
    run_in: str = "process",
):
    inference_q = mp.Queue()
    results_q = mp.Queue()

    if run_in == "process":
        process_or_thread = mp.Process(
            target=listen_and_run, args=(inference_q, results_q, mad)
        )
    elif run_in == "thread":  # useful for debuggin
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

    logging.debug(f"process_client: start for client {client_idx}")
    async for message in websocket:
        msg = hudes_pb2.Control()
        msg.ParseFromString(message)
        # Record incoming messages for speed run log if active
        if client.speed_run_active:
            client.speed_run_log.append(message)
            # cutoff logging if time elapsed
            if time.time() >= client.speed_run_end_time:
                client.speed_run_active = False
        if msg.type == hudes_pb2.Control.CONTROL_DIMS:
            logging.debug(f"process_client: {client_idx} : control dims")
            for dim_and_step in msg.dims_and_steps:
                dim = dim_and_step.dim + client.dims_offset
                if dim in client.next_step:
                    client.next_step[dim] += dim_and_step.step
                else:
                    client.next_step[dim] = dim_and_step.step
            client.request_idx = msg.request_idx
        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_BATCH:
            logging.debug(f"process_client: {client_idx} : next batch")
            client.batch_idx += 1
            client.request_full_val = True
            client.request_idx = msg.request_idx

        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_DIMS:
            logging.debug(f"process_client: {client_idx} : next dims")
            client.dims_offset += client.dims_at_a_time
            # ensure we do not reuse dims
            client.force_update = True
        elif msg.type == hudes_pb2.Control.CONTROL_CONFIG:
            logging.debug(f"process_client: {client_idx} : control config")
            old_batch_size = client.batch_size
            old_dtype = client.dtype

            client.dims_at_a_time = msg.config.dims_at_a_time
            client.seed = msg.config.seed
            client.mesh_grid_size = msg.config.mesh_grid_size
            client.mesh_grids = msg.config.mesh_grids
            client.mesh_step_size = msg.config.mesh_step_size
            client.batch_size = msg.config.batch_size
            client.dtype = getattr(torch, msg.config.dtype)

            if (
                client.mesh_grids > 0
                or old_batch_size != client.batch_size
                or old_dtype != client.dtype
            ):  # if we have grids, step size changes mesh
                client.force_update = True

        elif msg.type == hudes_pb2.Control.CONTROL_QUIT:
            logging.debug(f"process_client: {client_idx} : quit")
            client_runner_q.put(True)
            break

        elif msg.type == hudes_pb2.Control.CONTROL_SGD_STEP:
            # Ignore SGD during active speed run
            if client.speed_run_active:
                logging.debug("process_client: ignoring SGD during speed run")
            else:
                client.sgd += msg.sgd_steps
                client.total_sgd_steps += msg.sgd_steps
                client.request_idx = msg.request_idx

        elif msg.type == hudes_pb2.Control.CONTROL_SPEED_RUN_START:
            logging.debug(f"process_client: {client_idx} : speed run start")
            # reset per-client state
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
            # Start timer (allow env override at runtime)
            duration = int(
                os.environ.get("HUDES_SPEED_RUN_SECONDS", str(SPEED_RUN_SECONDS))
            )
            client.speed_run_active = True
            client.speed_run_end_time = time.time() + max(1, duration)
            logging.info(
                "Client %d Speed Run started for %d seconds",
                client.client_id,
                duration,
            )
            # Reset weights by signaling runner with force_update and
            # empty step; inference worker re-initializes when missing.
            # inference worker re-initializes weights when not present.
            client.force_update = True
            client.force_reset_weights = True

        elif msg.type == hudes_pb2.Control.CONTROL_HIGH_SCORE_LOG:
            logging.debug(f"process_client: {client_idx} : high score log")
            if client.high_score_logged:
                logging.debug("process_client: high score already logged; ignoring")
            else:
                # finalize run if still active
                client.speed_run_active = False
                client.high_score_logged = True
                name = msg.high_score.name.strip().upper() if msg.high_score else "????"
                if len(name) != 4 or not name.isalnum():
                    name = "????"
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
                # persist
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
                except Exception as e:
                    logging.error(f"Failed to persist high score: {e}")

        else:
            logging.warning("received invalid type from client")

        client_runner_q.put(True)


async def run_server(stop, client_runner_q, server_port, ssl_pem):
    # Standalone HTTP health server on a separate port to avoid interfering
    # with the WebSocket handshake. Defaults to server_port + 1.
    health_server = None
    health_port = int(os.environ.get("HUDES_HEALTH_PORT", server_port + 1))

    async def handle_health(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
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

    ssl_context = None
    if ssl_pem is not None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        localhost_pem = pathlib.Path(__file__).with_name(ssl_pem)
        ssl_context.load_cert_chain(localhost_pem)
    # Start health HTTP server
    try:
        health_server = await asyncio.start_server(
            handle_health, host="127.0.0.1", port=health_port
        )
        logging.info(
            "Health server listening on http://127.0.0.1:%d/health",
            health_port,
        )
    except OSError as e:
        logging.warning("Failed to start health server on %d: %s", health_port, e)

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
    logging.info("Speed Run configured duration (seconds): %d", configured_duration)
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
