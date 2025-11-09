#!/usr/bin/env python

import asyncio
import logging
import os
import socket
import time
from functools import cache
from multiprocessing import Queue
from threading import Thread

import websockets
from websockets.asyncio.client import connect
from websockets.sync.client import connect as sync_connect

from hudes import hudes_pb2


@cache
def sgd_step_message(steps, request_idx):
    return hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_SGD_STEP,
        sgd_steps=steps,
        request_idx=request_idx,
    )


@cache
def next_batch_message(request_idx):
    return hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_NEXT_BATCH, request_idx=request_idx
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
        self,
        dims_at_a_time,
        seed,
        mesh_grid_size,
        mesh_step_size,
        mesh_grids,
        batch_size,
        dtype,
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
                    batch_size=batch_size,
                    dtype=dtype,
                    resume_supported=False,
                ),
            ).SerializeToString()
        )

    def recv_ready(self):
        return not self.recv_q.empty()

    def recv_msg(self):
        return self.recv_q.get(timeout=0.0)

    def run_loop(self):
        backoff = 1
        while self.running:
            try:
                with sync_connect(self.remote_addr) as websocket:
                    logging.info("Connected to %s", self.remote_addr)
                    backoff = 1
                    while self.running:
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
                                    dims_and_steps=dims_and_steps,
                                    request_idx=request_idx,
                                ).SerializeToString()
                            )
                            dims_and_steps = {}

                        try:
                            msg = websocket.recv(timeout=0.00001)
                            self.recv_q.put(msg)
                        except TimeoutError:
                            pass
                        except (
                            websockets.exceptions.ConnectionClosedError,
                            websockets.exceptions.ConnectionClosedOK,
                        ):
                            logging.info(
                                "Remote closed connection; attempting reconnect"
                            )
                            break

                        time.sleep(0.01)
            except (ConnectionRefusedError, socket.gaierror) as e:
                logging.error(f"Connection issue with {self.remote_addr}, {e}")
            except Exception as e:
                logging.error(
                    f"Unexpected connection issue with {self.remote_addr}, {e}"
                )

            if not self.running:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)


async def send_dims(n: int = 10):
    async with connect("ws://localhost:8767") as websocket:
        for _ in range(n):
            msg = hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_DIMS,
            )
            logging.info(msg.SerializeToString())
            await websocket.send(msg.SerializeToString())
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(send_dims())
