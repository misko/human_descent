#!/usr/bin/env python

import asyncio
import json
from multiprocessing import Queue
import multiprocessing
from threading import Thread
from model_data_and_subspace import dot_loss
from websockets.asyncio.server import serve
import hudes_pb2
from hudes.mnist import MNISTFFNN, ModelDataAndSubspace, mnist_model_data_and_subpace

import hudes_pb2

import logging
import time

client_idx = 0
active_clients = {}

# TODO
# move to protobuf https://protobuf.dev/getting-started/pythontutorial/
# TODO need to do brew install brew install protobuf


def listen_and_run(in_q, out_q, mad):
    mad.fuse()

    while True:
        v = in_q.get()  # blocking
        if isinstance(v, str):
            if v == "quit":
                return
        else:
            # assume its a tensor
            weight_tensor, batch_idx = v
            # st = time.time()
            res = mad.model_inference_with_delta_weights(weight_tensor, batch_idx)
            # print("INF", time.time() - st)
            out_q.put(res)


async def inference_runner(mad, run_in="process"):
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
        for client_id in range(len(active_clients)):
            client = active_clients[client_id]
            # TODO should be some kind of select not poll()
            if len(client["next step"]) > 0:
                current_step = client["next step"]
                current_request_idx = client["request idx"]
                client["next step"] = {}

                if "weight vec" not in client:
                    client["weight vec"] = mad.blank_weight_vec()
                # update weight vector for client
                weight_vec = mad.delta_from_dims(current_step)

                # send weight vector for inference
                inference_q.put((weight_vec, client["batch idx"]))

                # return through websocket (add obj)
                res = await asyncio.to_thread(results_q.get)

                await client["websocket"].send(
                    json.dumps(
                        {
                            "type": "result",
                            "request idx": current_request_idx,
                            "result": res,
                        }
                    )
                )
        await asyncio.sleep(0.1)


async def process_client(websocket):
    global client_idx, active_clients
    current_client = client_idx
    client_idx += 1
    client = {
        "last seen": time.time(),
        "next step": {},
        "batch idx": 0,
        "websocket": websocket,
        "request idx": 0,
    }
    active_clients[current_client] = client
    dims_offset = 0
    max_dim = 0
    async for message in websocket:
        print(message)
        msg = hudes_pb2.Control()
        msg.ParseFromString(message)
        print("GOT", msg)
        msg = json.loads(message)
        if msg["type"] == "control":
            for dim, step in msg["dims"].items():
                dim = int(dim)
                max_dim = max(max_dim, dim)
                dim += dims_offset
                if dim in client["next step"]:
                    client["next step"][dim] += step
                else:
                    client["next step"][dim] = step
            client["request idx"] += 1
        elif msg["type"] == "next batch":
            client["batch idx"] += 1
        elif msg["type"] == "next dims":
            dims_offset = max_dim + 1  # make sure we dont re-use dims
        elif msg["type"] == "quit":
            await websocket.send(message)
            break
        else:
            logging.warning("received invalid type from client")
        #


async def run_server(stop):
    async with serve(process_client, "localhost", 8765):
        await stop
        # await asyncio.get_running_loop().create_future()  # run forever


async def run_wrapper():
    mad = mnist_model_data_and_subpace(model=MNISTFFNN(), loss_fn=dot_loss)
    stop = asyncio.get_running_loop().create_future()
    await asyncio.gather(run_server(stop), inference_runner(mad))
    # await


if __name__ == "__main__":
    # ModelDataAndSubspace(MNISTFFNN(), train_batch_size=512, val_batch_size=1024)
    # x, y = mad.train_data_batcher[0]
    # mad.model(x)

    # stop.set_result(True) # to stop server
    asyncio.run(run_wrapper())
