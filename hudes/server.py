#!/usr/bin/env python

import asyncio
import logging
import multiprocessing
import pickle
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from threading import Thread

import websockets
from websockets.asyncio.server import serve

from hudes import hudes_pb2
from hudes.mnist import MNISTFFNN, ModelDataAndSubspace, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import indexed_loss

client_idx = 0
active_clients = {}

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


def listen_and_run(in_q: Queue, out_q: Queue, mad: ModelDataAndSubspace):
    mad.fuse()

    while True:
        v = in_q.get()  # blocking
        if isinstance(v, str):
            if v == "quit":
                return
        else:
            # assume its a tensor
            train_or_val, weight_tensor, batch_idx = v
            if train_or_val == "train":
                res = mad.train_model_inference_with_delta_weights(
                    weight_tensor, batch_idx
                )
            elif train_or_val == "val":
                res = mad.val_model_inference_with_delta_weights(weight_tensor)
            out_q.put(res)


@dataclass
class Client:
    last_seen: float = 0.0
    next_step: dict = field(default_factory=lambda: dict())
    batch_idx: int = 0
    websocket: None = None
    request_idx: int = 0
    sent_batch: int = -1
    weight_vec: None = None
    request_full_val: bool = False


async def inference_runner(
    mad: ModelDataAndSubspace, stop=None, run_in: str = "process"
):
    global active_clients
    inference_q = Queue()
    results_q = Queue()

    if run_in == "process":
        p = multiprocessing.Process(
            target=listen_and_run, args=(inference_q, results_q, mad)
        )
        p.daemon = True
        p.start()
        # p.join()
    elif run_in == "thread":  # useful for debuggin
        thread = Thread(target=listen_and_run, args=(inference_q, results_q, mad))
        thread.daemon = True
        thread.start()

    mad.fuse()

    while True:
        if stop is not None and stop.done():
            inference_q.put("quit")
            return
        for client_id in range(len(active_clients)):
            client = active_clients[client_id]

            # breakpoint()
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
                current_request_idx = client.request_idx
                client.next_step = {}

                if client.weight_vec is None:
                    client.weight_vec = mad.blank_weight_vec()
                # update weight vector for client
                if len(current_step) > 0:
                    client.weight_vec += mad.delta_from_dims(current_step)

                # send weight vector for inference
                inference_q.put(("train", client.weight_vec, client.batch_idx))

                # return through websocket (add obj)
                res = await asyncio.to_thread(results_q.get)

                # TODO need to be ok with getting errors here
                try:
                    await client.websocket.send(
                        hudes_pb2.Control(
                            type=hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
                            train_loss_and_preds=hudes_pb2.TrainLossAndPreds(
                                train_loss=res["train_loss"],
                                preds=pickle.dumps(res["train_preds"]),
                                confusion_matrix=pickle.dumps(res["confusion_matrix"]),
                            ),
                            request_idx=current_request_idx,
                        ).SerializeToString()
                    )
                except websockets.exceptions.ConnectionClosedOK:
                    pass
            elif client.request_full_val:
                # send weight vector for inference
                inference_q.put(("val", client.weight_vec, None))

                # return through websocket (add obj)
                res = await asyncio.to_thread(results_q.get)
                try:
                    await client.websocket.send(
                        hudes_pb2.Control(
                            type=hudes_pb2.Control.CONTROL_VAL_LOSS,
                            val_loss=hudes_pb2.ValLoss(
                                val_loss=res["val_loss"], minimize=mad.minimize
                            ),
                            request_idx=current_request_idx,
                        ).SerializeToString()
                    )
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                client.request_full_val = False
        await asyncio.sleep(0.01)


async def process_client(websocket):
    global client_idx, active_clients
    current_client = client_idx
    client_idx += 1
    client = Client(
        last_seen=time.time(),
        next_step={},
        batch_idx=0,
        websocket=websocket,
        request_idx=0,
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
        #


async def run_server(stop):
    async with serve(process_client, "localhost", 8765):
        await stop


async def run_wrapper():
    mad = mnist_model_data_and_subpace(model=MNISTFFNN(), loss_fn=indexed_loss)
    stop = asyncio.get_running_loop().create_future()
    await asyncio.gather(run_server(stop), inference_runner(mad, run_in="thread"))


if __name__ == "__main__":
    asyncio.run(run_wrapper())
