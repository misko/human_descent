#!/usr/bin/env python

import asyncio
import json
import time

from websockets.asyncio.client import connect

from hudes import hudes_pb2


async def send_dims(n=10):
    async with connect("ws://localhost:8765") as websocket:
        for _ in range(n):

            msg = hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_DIMS,
                dims_and_steps=hudes_pb2.DimAndSteps(
                    dims_and_steps=[hudes_pb2.DimAndSteps.DimAndStep(dim=1, step=0.01)]
                ),
            )
            # msg = {"type": "control", "dims": {1: 0.1, 2: 0.3}}
            print(msg.SerializeToString())
            await websocket.send(msg.SerializeToString())
            await asyncio.sleep(0.01)


if __name__ == "__main__":

    asyncio.run(send_dims())
