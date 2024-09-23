#!/usr/bin/env python

import asyncio
import json
import time
from functools import cache
from multiprocessing import Queue
from threading import Thread

from websockets.asyncio.client import connect
from websockets.sync.client import connect as sync_connect

from hudes import hudes_pb2

"""
Websocket client has a queue in and out
can push many messages down into ws client, but only sends as fast as possible

ws client loop (thread)


While True:
    #prepare and send
    while get(timeout=0.1) from queue():
        check how many events
        goup as many as possible

        send 

    #recv and pipe back up
    recv(timeout=0)



"""


@cache
def next_batch_message():
    return hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_NEXT_BATCH,
    )


@cache
def next_dims_message():
    return hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_NEXT_DIMS,
    )


def dims_and_steps_to_control_message(dims_and_steps, request_idx):
    return hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_DIMS,
        dims_and_steps=[
            hudes_pb2.DimAndStep(dim=dim, step=step)
            for dim, step in dims_and_steps.items()
        ],
        request_idx=request_idx,
    )


# used to aggregate control signals before sending to server
# dont overwhelm the connection with small instructions, istead
# buffer client side to reduce latency between result and control
class HudesWebsocketClient:
    def __init__(self, remote_addr):
        self.remote_addr = remote_addr
        self.thread = Thread(target=self.run_loop)  # , args=(send_q, recv_q))
        self.thread.daemon = True
        self.thread.start()
        self.send_q = Queue()
        self.recv_q = Queue()

    def send_config(self, dims_at_a_time, seed):
        self.send_q.put(
            hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_CONFIG,
                config=hudes_pb2.Config(seed=seed, dims_at_a_time=dims_at_a_time),
            ).SerializeToString()
        )

    def recv_ready(self):
        return not self.recv_q.empty()

    def recv_msg(self):
        return self.recv_q.get(timeout=0.0)

    def run_loop(self):
        with sync_connect(self.remote_addr) as websocket:
            while True:
                # figure out what if we should send
                send_or_recv = False
                request_idx = -1
                dims_and_steps = {}
                while not self.send_q.empty():
                    raw_msg = self.send_q.get()
                    msg = hudes_pb2.Control()
                    msg.ParseFromString(raw_msg)
                    if msg.type == hudes_pb2.Control.CONTROL_DIMS:
                        request_idx = max(request_idx, msg.request_idx)
                        for dim_and_step in msg.dims_and_steps:
                            dim = dim_and_step.dim
                            if dim not in dims_and_steps:
                                dims_and_steps[dim] = dim_and_step.step
                            else:
                                dims_and_steps[dim] += dim_and_step.step
                    else:
                        if len(dims_and_steps) != 0:
                            websocket.send(
                                dims_and_steps_to_control_message(
                                    dims_and_steps=dims_and_steps,
                                    request_idx=request_idx,
                                ).SerializeToString()
                            )
                            dims_and_steps = {}
                        websocket.send(raw_msg)
                        send_or_recv = True
                if len(dims_and_steps) != 0:
                    websocket.send(
                        dims_and_steps_to_control_message(
                            dims_and_steps=dims_and_steps, request_idx=request_idx
                        ).SerializeToString()
                    )
                    dims_and_steps = {}
                    send_or_recv = True

                # figure out what we are recv'ing if anything
                try:
                    msg = websocket.recv(timeout=0.0)
                    send_or_recv = True
                    self.recv_q.put(msg)
                except TimeoutError:
                    pass

                # when there is no interaction give the system a break(?)
                if not send_or_recv:
                    time.sleep(0.05)


async def send_dims(n=10):
    async with connect("ws://localhost:8765") as websocket:
        for _ in range(n):

            msg = hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_DIMS,
                # dims_and_steps=[hudes_pb2.DimAndStep(dim=1, step=0.01)],
            )
            # msg = {"type": "control", "dims": {1: 0.1, 2: 0.3}}
            print(msg.SerializeToString())
            await websocket.send(msg.SerializeToString())
            await asyncio.sleep(0.01)


if __name__ == "__main__":

    asyncio.run(send_dims())
