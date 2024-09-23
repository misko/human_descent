import asyncio
import json
import time

import pytest
import pytest_asyncio
import websockets

from hudes.client import send_dims
from hudes.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.server import inference_runner, process_client
from model_data_and_subspace import dot_loss


async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)


@pytest_asyncio.fixture
async def run_server_thread():
    mad = mnist_model_data_and_subpace(model=MNISTFFNN(), loss_fn=dot_loss)
    stop = asyncio.get_running_loop().create_future()
    inference_task = asyncio.create_task(
        inference_runner(mad, stop=stop, run_in="thread")
    )
    async with websockets.serve(process_client, "localhost", 8765):
        yield
    stop.set_result(True)
    # inference_task.cancel()
    await inference_task


@pytest_asyncio.fixture
async def run_server_process():
    mad = mnist_model_data_and_subpace(model=MNISTFFNN(), loss_fn=dot_loss)
    stop = asyncio.get_running_loop().create_future()
    inference_task = asyncio.create_task(
        inference_runner(mad, stop=stop, run_in="process")
    )
    async with websockets.serve(process_client, "localhost", 8765):
        yield
    stop.set_result(True)
    # inference_task.cancel()
    await inference_task


@pytest.mark.asyncio
async def test_single_websocket_thread(run_server_thread):
    await send_dims(5)


@pytest.mark.asyncio
async def test_single_websocket_process(run_server_process):
    await send_dims(5)


@pytest.mark.asyncio
async def test_multi_websocket_process(run_server_process):
    await asyncio.gather(*[asyncio.create_task(send_dims(2)) for x in range(10)])
