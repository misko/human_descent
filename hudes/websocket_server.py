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
import websockets
from websockets.asyncio.server import serve

from hudes import hudes_pb2
from hudes.mnist import (
    MNISTFFNN,
    ModelDataAndSubspaceInference,
    mnist_model_data_and_subpace,
)
from hudes.model_data_and_subspace import indexed_loss

client_idx = 0
active_clients = {}


# TODO
# move to protobuf https://protobuf.dev/getting-started/pythontutorial/
# TODO need to do brew install brew install protobuf


# TODO cache?
def prepare_batch_example_message(
    batch_idx: int, mad: ModelDataAndSubspaceInference, n: int = 4
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
    mad: ModelDataAndSubspaceInference,
):
    mad.move_to_device()
    mad.fuse()

    client_weights = {}  # TODO memory leak, but prevents continously copying models

    while True:
        try:
            v = in_q.get()  # blocking
        except EOFError:
            return
        if v is None:
            return
        else:
            # assume its a tensor
            train_or_val, current_step, client_id, batch_idx = v

            # prepare weights
            if client_id not in client_weights:
                client_weights[client_id] = mad.saved_weights.clone()
            client_weights[client_id] += mad.delta_from_dims(current_step)

            if train_or_val == "train":
                res = mad.train_model_inference_with_delta_weights(
                    client_weights[client_id], batch_idx
                )
            elif train_or_val == "val":
                res = mad.val_model_inference_with_delta_weights(
                    client_weights[client_id]
                )
            out_q.put((client_id, train_or_val, res))


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


async def inference_runner_clients(mad, event, inference_q, stop):
    while True:
        await event.coro_wait()
        if stop is not None and stop.done():
            return

        event.clear()

        # make requests
        for client_id in range(len(active_clients)):
            client = active_clients[client_id]

            # client still waiting for response just skip
            if client.active_inference:
                continue

            force_update = False

            # TODO should be some kind of select not poll()
            if client.sent_batch != client.batch_idx:
                try:
                    await client.websocket.send(
                        prepare_batch_example_message(
                            client.batch_idx, mad
                        ).SerializeToString()
                    )
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                client.sent_batch = client.batch_idx
                force_update = True

            if force_update or len(client.next_step) > 0:
                current_step = client.next_step
                client.next_step = {}

                client.active_inference = True
                client.active_request_idx = client.request_idx
                await inference_q.coro_put(
                    (
                        "train",
                        current_step,
                        client_id,
                        client.batch_idx,
                    )
                )

            if client.request_full_val:
                # send weight vector for inference
                await inference_q.coro_put(
                    (
                        "val",
                        {},
                        client_id,
                        client.batch_idx,
                    )
                )


async def inference_result_sender(results_q, stop):
    while True:
        msg = await results_q.coro_get()

        if stop is not None and stop.done:
            return

        client_id, train_or_val, res = msg
        client = active_clients[client_id]

        try:
            if train_or_val == "train":
                # TODO need to be ok with getting errors here
                await client.websocket.send(
                    hudes_pb2.Control(
                        type=hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
                        train_loss_and_preds=hudes_pb2.TrainLossAndPreds(
                            train_loss=res["train_loss"],
                            preds=pickle.dumps(res["train_preds"]),
                            confusion_matrix=pickle.dumps(res["confusion_matrix"]),
                        ),
                        request_idx=client.active_request_idx,
                    ).SerializeToString()
                )
            elif train_or_val == "val":
                await client.websocket.send(
                    hudes_pb2.Control(
                        type=hudes_pb2.Control.CONTROL_VAL_LOSS,
                        val_loss=hudes_pb2.ValLoss(
                            val_loss=res["val_loss"],
                        ),
                        request_idx=client.active_request_idx,
                    ).SerializeToString()
                )
                client.request_full_val = False
        except websockets.exceptions.ConnectionClosedOK:
            pass
        client.active_inference = False


async def wait_for_stop(inference_q, results_q, stop, event):
    await stop
    results_q.put(None)
    inference_q.put(None)
    event.set()


async def inference_runner(
    mad: ModelDataAndSubspaceInference,
    event: aioprocessing.AioEvent,
    stop=None,
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

    await asyncio.gather(
        inference_runner_clients(mad, event, inference_q, stop),
        inference_result_sender(results_q, stop),
        wait_for_stop(inference_q, results_q, stop, event),
    )
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
    dims_offset = 0

    config = {}
    async for message in websocket:
        msg = hudes_pb2.Control()
        msg.ParseFromString(message)
        if msg.type == hudes_pb2.Control.CONTROL_DIMS:
            for dim_and_step in msg.dims_and_steps:
                dim = dim_and_step.dim + dims_offset
                if dim in client.next_step:
                    client.next_step[dim] += dim_and_step.step
                else:
                    client.next_step[dim] = dim_and_step.step
            client.request_idx += 1
        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_BATCH:
            client.batch_idx += 1
            client.request_full_val = True
            # send back batch examples

        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_DIMS:
            dims_offset += config["dims_at_a_time"]  # make sure we dont re-use dims
        elif msg.type == hudes_pb2.Control.CONTROL_CONFIG:
            config["dims_at_a_time"] = msg.config.dims_at_a_time
            config["seed"] = msg.config.seed
            # send back batch examples
        elif msg.type == hudes_pb2.Control.CONTROL_QUIT:
            break
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
        model=MNISTFFNN(), loss_fn=indexed_loss, device=args.device
    )
    stop = asyncio.get_running_loop().create_future()
    await asyncio.gather(
        run_server(stop, event), inference_runner(mad, run_in="process", event=event)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hudes: Server")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    asyncio.run(run_wrapper(args))
