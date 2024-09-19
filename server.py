#!/usr/bin/env python

import asyncio
from websockets.asyncio.server import serve

from mnist import MNISTFFNN, ModelAndData


async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)


async def main():
    async with serve(echo, "localhost", 8765):
        await asyncio.get_running_loop().create_future()  # run forever


mad = ModelAndData(MNISTFFNN(), train_batch_size=512, val_batch_size=1024)
x, y = mad.train_data_batcher[0]
mad.model(x)
asyncio.run(main())
