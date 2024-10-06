#!/usr/bin/env python

import argparse
import asyncio
import logging
import multiprocessing
import pickle
import time
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Queue
from threading import Thread

import aioprocessing
import torch
import websockets
from websockets.asyncio.server import serve

from hudes import hudes_pb2
from hudes.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import ModelDataAndSubspace, indexed_loss

client_idx = 0
active_clients = {}

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# TODO
# move to protobuf https://protobuf.dev/getting-started/pythontutorial/
# TODO need to do brew install brew install protobuf


# TODO cache?
def prepare_batch_example_message(
    batch_idx: int, mad: ModelDataAndSubspace, n: int = 4
):
    batch = mad.get_batch(batch_idx)
    return hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_BATCH_EXAMPLES,
        batch_examples=hudes_pb2.BatchExamples(
            type=hudes_pb2.BatchExamples.Type.IMG_BW,
            n=n,
            train_data=pickle.dumps(batch["train"][0][:n].tolist()),
            val_data=pickle.dumps(batch["val"][0][:n].tolist()),
            train_labels=pickle.dumps(batch["train"][1][:n].tolist()),
            val_labels=pickle.dumps(batch["val"][1][:n].tolist()),
            batch_idx=batch_idx,
        ),
    )


def listen_and_run(
    in_q: aioprocessing.AioQueue,
    out_q: aioprocessing.AioQueue,
    mad: ModelDataAndSubspace,
):
    mad.move_to_device()
    mad.fuse()
    mad.init_param_model()
    # breakpoint()

    client_weights = {}  # TODO memory leak, but prevents continously copying models

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
            client_id = v["client_id"]

            logging.info(f"listen_and_run: got {v['mode']}")
            if v["mode"] in ("train", "mesh"):
                if client_id not in client_weights:
                    client_weights[client_id] = mad.saved_weights.clone()
                client_weights[client_id] += mad.delta_from_dims(v["current_step"])

            if v["mode"] == "train":
                res = mad.train_model_inference_with_delta_weights(
                    client_weights[client_id], v["batch_idx"]
                )
            elif v["mode"] == "val":
                res = mad.val_model_inference_with_delta_weights(
                    client_weights[client_id]
                )
            elif v["mode"] == "mesh":
                res = mad.get_loss_grid(
                    base_weights=client_weights[client_id],
                    grid_size=v["grid_size"],
                    step_size=v["step_size"],
                    grids=v["grids"],
                    dims_offset=v["dims_offset"],
                    batch_idx=v["batch_idx"],
                )
            else:
                raise ValueError

            logging.info(f"listen_and_run: return {v['mode']}")
            out_q.put((client_id, v["mode"], res))


@dataclass
class Client:
    last_seen: float = 0.0
    next_step: dict = field(default_factory=lambda: dict())
    batch_idx: int = 0
    websocket: None = None
    request_idx: int = 0
    sent_batch: int = -1
    request_full_val: bool = False
    active_inference: bool = False
    active_request_idx: int = -1
    mesh_grid_size: int = -1
    mesh_grids: int = 0
    mesh_step_size: float = 0.0
    force_update: bool = False
    dims_offset: int = 0
    dims_at_a_time: int = 0
    seed: int = 0
    dtype: str = "float32"


async def inference_runner_clients(mad, event, inference_q, stop):
    logging.info(f"inference_runner_clients: started")
    while True:
        await event.coro_wait()
        if stop is not None and stop.done():
            logging.info(f"inference_result_sender: returning")
            return

        event.clear()

        # make requests
        for client_id in range(len(active_clients)):
            client = active_clients[client_id]

            # client still waiting for response just skip
            if client.active_inference:
                continue

            # TODO should be some kind of select not poll()
            if client.sent_batch != client.batch_idx:
                try:
                    logging.info(f"inference_runner_clients: send batch msg")
                    await client.websocket.send(
                        prepare_batch_example_message(
                            client.batch_idx, mad
                        ).SerializeToString()
                    )
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                client.sent_batch = client.batch_idx
                client.force_update = True

            if client.force_update or len(client.next_step) > 0:
                current_step = client.next_step
                client.next_step = {}

                client.active_inference = True
                client.active_request_idx = client.request_idx

                if client.mesh_grids > 0:
                    logging.info(f"inference_runner_clients: req mesh")
                    await inference_q.coro_put(
                        {
                            "mode": "mesh",
                            "current_step": current_step,
                            "grid_size": client.mesh_grid_size,
                            "step_size": client.mesh_step_size,
                            "grids": client.mesh_grids,
                            "dims_offset": client.dims_offset,
                            "batch_idx": client.batch_idx,
                            "client_id": client_id,
                        }
                    )
                else:
                    logging.info(f"inference_runner_clients: req train")
                    await inference_q.coro_put(
                        {
                            "mode": "train",
                            "current_step": current_step,
                            "client_id": client_id,
                            "batch_idx": client.batch_idx,
                        }
                    )
                client.force_update = False

            if client.request_full_val:
                # send weight vector for inference
                logging.info(f"inference_runner_clients: req inference")
                await inference_q.coro_put(
                    {
                        "mode": "val",
                        "client_id": client_id,
                        "batch_idx": client.batch_idx,
                    }
                )


async def inference_result_sender(results_q, stop):

    logging.info(f"inference_result_sender: started")
    while True:
        msg = await results_q.coro_get()

        if stop is not None and stop.done():
            logging.info(f"inference_result_sender: returning")
            return

        client_id, train_or_val, res = msg
        client = active_clients[client_id]

        try:
            if train_or_val == "train":
                # TODO need to be ok with getting errors here
                logging.info(f"inference_result_sender: sent train to client")
                await client.websocket.send(
                    hudes_pb2.Control(
                        type=hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
                        train_loss_and_preds=hudes_pb2.TrainLossAndPreds(
                            train_loss=res["train_loss"],
                            preds=pickle.dumps(res["train_preds"].cpu().float()),
                            confusion_matrix=pickle.dumps(
                                res["confusion_matrix"].cpu().float()
                            ),
                        ),
                        request_idx=client.active_request_idx,
                    ).SerializeToString()
                )
                logging.info(f"inference_result_sender: sent train to client : done")
            elif train_or_val == "val":
                logging.info(f"inference_result_sender: sent val to client")
                await client.websocket.send(
                    hudes_pb2.Control(
                        type=hudes_pb2.Control.CONTROL_VAL_LOSS,
                        val_loss=hudes_pb2.ValLoss(
                            val_loss=res["val_loss"],
                        ),
                        request_idx=client.active_request_idx,
                    ).SerializeToString()
                )
                logging.info(f"inference_result_sender: sent val to client : done")
                client.request_full_val = False
            elif train_or_val == "mesh":
                logging.info(f"inference_result_sender: sent mesh to client")
                await client.websocket.send(
                    hudes_pb2.Control(
                        type=hudes_pb2.Control.CONTROL_MESHGRID_RESULTS,
                        mesh_grid_results=pickle.dumps(res.cpu().float()),
                    ).SerializeToString()
                )
                logging.info(f"inference_result_sender: sent mesh to client : done")
            else:
                raise ValueError
        except (
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
        ) as e:
            pass
        client.active_inference = False


async def wait_for_stop(inference_q, results_q, stop, event):
    await stop
    results_q.put(None)
    inference_q.put(None)
    event.set()


async def inference_runner(
    mad: ModelDataAndSubspace,
    event: aioprocessing.AioEvent,
    stop,
    run_in: str = "process",
):
    global active_clients
    inference_q = aioprocessing.AioQueue()
    results_q = aioprocessing.AioQueue()

    if run_in == "process":
        process_or_thread = aioprocessing.AioProcess(
            target=listen_and_run, args=(inference_q, results_q, mad)
        )
    elif run_in == "thread":  # useful for debuggin
        process_or_thread = Thread(
            target=listen_and_run, args=(inference_q, results_q, mad)
        )
    process_or_thread.daemon = True
    process_or_thread.start()

    print("Inference runner running...")
    await asyncio.gather(
        inference_runner_clients(mad, event, inference_q, stop),
        inference_result_sender(results_q, stop),
        wait_for_stop(inference_q, results_q, stop, event),
    )
    print("Inference runner stopping...")
    process_or_thread.join()


async def process_client(websocket, event):
    global client_idx, active_clients
    current_client = client_idx
    client_idx += 1
    client = Client(
        last_seen=time.time(),
        next_step={},
        batch_idx=0,
        websocket=websocket,
        request_idx=0,
        active_inference=False,
        sent_batch=-1,
    )
    active_clients[current_client] = client

    logging.info(f"process_client: start for client {client_idx}")
    async for message in websocket:
        msg = hudes_pb2.Control()
        msg.ParseFromString(message)
        if msg.type == hudes_pb2.Control.CONTROL_DIMS:
            logging.info(f"process_client: {client_idx} : control dims")
            for dim_and_step in msg.dims_and_steps:
                dim = dim_and_step.dim + client.dims_offset
                if dim in client.next_step:
                    client.next_step[dim] += dim_and_step.step
                else:
                    client.next_step[dim] = dim_and_step.step
            client.request_idx += 1
        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_BATCH:
            logging.info(f"process_client: {client_idx} : next batch")
            client.batch_idx += 1
            client.request_full_val = True
            # send back batch examples

        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_DIMS:
            logging.info(f"process_client: {client_idx} : next dims")
            client.dims_offset += client.dims_at_a_time  # make sure we dont re-use dims
            # client.mesh_dimA += dims_offset
            # client.mesh_dimB += dims_offset
            print("NEXT DIMS")
            client.force_update = True
        elif msg.type == hudes_pb2.Control.CONTROL_CONFIG:
            logging.info(f"process_client: {client_idx} : control config")
            client.dims_at_a_time = msg.config.dims_at_a_time
            client.seed = msg.config.seed
            client.mesh_grid_size = msg.config.mesh_grid_size
            client.mesh_grids = msg.config.mesh_grids
            client.mesh_step_size = msg.config.mesh_step_size
            client.force_update = True

            # send back batch examples
        elif msg.type == hudes_pb2.Control.CONTROL_QUIT:
            logging.info(f"process_client: {client_idx} : quit")
            break
        # elif msg.type == hudes_pb2.Control.CONTROL_MESHGRID_CONFIG:
        #     logging.info(
        #         f"process_client: {client_idx} : meshgrid, {client.dims_offset}, {config['dims_at_a_time']}"
        #     )

        else:
            logging.warning("received invalid type from client")

        event.set()
        #


async def run_server(stop, event):
    async with serve(partial(process_client, event=event), "localhost", 8765):
        await stop


async def run_wrapper(args):
    event = aioprocessing.AioEvent()
    mad = mnist_model_data_and_subpace(
        model=MNISTFFNN(),
        loss_fn=indexed_loss,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    stop = asyncio.get_running_loop().create_future()
    await asyncio.gather(
        run_server(stop, event),
        inference_runner(mad, run_in="thread", event=event, stop=stop),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hudes: Server")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float")

    args = parser.parse_args()
    asyncio.run(run_wrapper(args))
