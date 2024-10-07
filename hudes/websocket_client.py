#!/usr/bin/env python

import asyncio
import json
import logging
import time
from functools import cache
from multiprocessing import Queue
from threading import Thread

import websockets
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


# def mesh_grid_config_message(dimA, dimB, grid_size, step_size):
#     return hudes_pb2.Control(
#         type=hudes_pb2.Control.CONTROL_MESHGRID_CONFIG,
#         mesh_grid_config=hudes_pb2.MeshGridConfig(
#             dimA=dimA, dimB=dimB, grid_size=grid_size, step_size=step_size
#         ),
#     )


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


def dims_and_steps_to_control_message(
    dims_and_steps: dict[int, float], request_idx: int
):
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
    def __init__(self, remote_addr: str):
        self.remote_addr = remote_addr
        self.send_q = Queue()
        self.recv_q = Queue()
        self.running = True

        self.thread = Thread(target=self.run_loop)  # , args=(send_q, recv_q))
        self.thread.daemon = True
        self.thread.start()

    def send_config(
        self, dims_at_a_time, seed, mesh_grid_size, mesh_step_size, mesh_grids
    ):
        self.send_q.put(
            hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_CONFIG,
                config=hudes_pb2.Config(
                    seed=seed,
                    dims_at_a_time=dims_at_a_time,
                    mesh_grid_size=mesh_grid_size,
                    mesh_step_size=mesh_step_size,
                    mesh_grids=mesh_grids,
                ),
            ).SerializeToString()
        )

    def recv_ready(self):
        return not self.recv_q.empty()

    def recv_msg(self):
        return self.recv_q.get(timeout=0.0)

    def run_loop(self):
        try:
            with sync_connect(self.remote_addr) as websocket:
                while self.running:
                    # figure out what if we should send
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
                    if len(dims_and_steps) != 0:
                        websocket.send(
                            dims_and_steps_to_control_message(
                                dims_and_steps=dims_and_steps, request_idx=request_idx
                            ).SerializeToString()
                        )
                        dims_and_steps = {}

                    # figure out what we are recv'ing if anything
                    try:
                        msg = websocket.recv(timeout=0.0)
                        self.recv_q.put(msg)
                    except TimeoutError:
                        pass
                    except (
                        websockets.exceptions.ConnectionClosedError,
                        websockets.exceptions.ConnectionClosedOK,
                    ) as e:
                        self.running = False

                    # when there is no interaction give the system a break(?)
                    # if not send_or_recv:
                    time.sleep(0.01)
        except ConnectionRefusedError:
            self.running = False


async def send_dims(n: int = 10):
    async with connect("ws://localhost:8765") as websocket:
        for _ in range(n):
            msg = hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_DIMS,
                # dims_and_steps=[hudes_pb2.DimAndStep(dim=1, step=0.01)],
            )
            # msg = {"type": "control", "dims": {1: 0.1, 2: 0.3}}
            logging.info(msg.SerializeToString())
            await websocket.send(msg.SerializeToString())
            await asyncio.sleep(0.01)


if __name__ == "__main__":

    asyncio.run(send_dims())
