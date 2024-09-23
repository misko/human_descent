#!/usr/bin/env python

import asyncio
import json
import logging
import multiprocessing
import pickle
import time
from multiprocessing import Queue
from threading import Thread

import websockets
from websockets.asyncio.server import serve

from hudes import hudes_pb2
from hudes.mnist import MNISTFFNN, ModelDataAndSubspace, mnist_model_data_and_subpace
from model_data_and_subspace import dot_loss

client_idx = 0
active_clients = {}

# TODO
# move to protobuf https://protobuf.dev/getting-started/pythontutorial/
# TODO need to do brew install brew install protobuf


# TODO cache?
def prepare_batch_example_message(batch_idx, mad, n=4):
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


def listen_and_run(in_q, out_q, mad):
    mad.fuse()

    while True:
        v = in_q.get()  # blocking
        # breakpoint()
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


async def inference_runner(mad, stop=None, run_in="process"):
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
            if client["sent batch"] != client["batch idx"]:
                await client["websocket"].send(
                    prepare_batch_example_message(
                        client["batch idx"], mad
                    ).SerializeToString()
                )
                client["sent batch"] = client["batch idx"]
                force_update = True

            if force_update or len(client["next step"]) > 0:
                current_step = client["next step"]
                current_request_idx = client["request idx"]
                client["next step"] = {}

                if "weight vec" not in client:
                    client["weight vec"] = mad.blank_weight_vec()
                # update weight vector for client
                if len(current_step) > 0:
                    client["weight vec"] += mad.delta_from_dims(current_step)

                print(client["weight vec"].abs().mean())

                # send weight vector for inference
                inference_q.put((client["weight vec"], client["batch idx"]))

                # return through websocket (add obj)
                res = await asyncio.to_thread(results_q.get)

                # TODO need to be ok with getting errors here
                try:
                    # print(pickle.loads(pickle.dumps(res["train_preds"])))
                    await client["websocket"].send(
                        hudes_pb2.Control(
                            type=hudes_pb2.Control.CONTROL_LOSS_AND_PREDS,
                            loss_and_preds=hudes_pb2.LossAndPreds(
                                train_loss=res["train_loss"],
                                val_loss=res["val_loss"],
                                preds=pickle.dumps(res["train_preds"]),
                                request_idx=current_request_idx,
                            ),
                        ).SerializeToString()
                    )

                    #     json.dumps(
                    #         {
                    #             "type": "result",
                    #             "request idx": current_request_idx,
                    #             "result": res,
                    #         }
                    #     )
                    # )
                except websockets.exceptions.ConnectionClosedOK:
                    pass

                # (Pdb) batch['train'][0].shape
                # torch.Size([512, 28, 28])
                # (Pdb) batch['train'][1].shape
                # torch.Size([512])
        await asyncio.sleep(0.01)


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
        "sent batch": -1,
    }
    active_clients[current_client] = client
    dims_offset = 0

    config = {}
    async for message in websocket:
        msg = hudes_pb2.Control()
        msg.ParseFromString(message)
        print("GOT MESSAGE!", msg)
        if msg.type == hudes_pb2.Control.CONTROL_DIMS:
            for dim_and_step in msg.dims_and_steps:
                # max_dim = max(max_dim, dim_and_step.dim)
                dim = dim_and_step.dim + dims_offset
                if dim in client["next step"]:
                    client["next step"][dim] += dim_and_step.step
                else:
                    client["next step"][dim] = dim_and_step.step
            client["request idx"] += 1
        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_BATCH:
            client["batch idx"] += 1
            # send back batch examples

        elif msg.type == hudes_pb2.Control.CONTROL_NEXT_DIMS:
            dims_offset += config["dims_at_a_time"]  # make sure we dont re-use dims
        elif msg.type == hudes_pb2.Control.CONTROL_CONFIG:
            config["dims_at_a_time"] = msg.config.dims_at_a_time
            config["seed"] = msg.config.seed
            # send back batch examples
            # websocket.send(message)
        elif msg.type == hudes_pb2.Control.CONTROL_QUIT:
            # await websocket.send(message)
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
