#!/usr/bin/env python

import json
from websockets.sync.client import connect
import time
import hudes_pb2


def send_dims(n=10):
    with connect("ws://localhost:8765") as websocket:
        for _ in range(n):

            msg = hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_DIMS,
                dims_and_steps=hudes_pb2.DimAndSteps(
                    dims_and_steps=[hudes_pb2.DimAndSteps.DimAndStep(dim=1, step=0.01)]
                ),
            )
            # msg = {"type": "control", "dims": {1: 0.1, 2: 0.3}}
            print(msg.SerializeToString())
            websocket.send(msg.SerializeToString())
            time.sleep(0.05)
        while True:
            message = json.loads(websocket.recv())
            print(f"Received: {message['request idx']}")
            if int(message["request idx"]) == n:
                break


if __name__ == "__main__":
    send_dims()
