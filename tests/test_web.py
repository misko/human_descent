import asyncio
import logging
from functools import partial

import pytest
import pytest_asyncio
import torch.multiprocessing as mp
import websockets

from hudes.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import indexed_loss
from hudes.websocket_client import send_dims
from hudes.websocket_server import inference_runner, process_client


async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)


@pytest_asyncio.fixture
async def run_server_thread():
    mad = mnist_model_data_and_subpace(model=MNISTFFNN(), loss_fn=indexed_loss)
    client_runner_q = mp.Queue()
    stop = asyncio.get_running_loop().create_future()
    inference_task = asyncio.create_task(
        inference_runner(
            mad, stop=stop, run_in="thread", client_runner_q=client_runner_q
        )
    )

    async with websockets.serve(
        partial(process_client, client_runner_q=client_runner_q), "localhost", 8765
    ):
        yield

    stop.set_result(True)
    client_runner_q.put(True)

    await inference_task


@pytest_asyncio.fixture
async def run_server_process():

    mad = mnist_model_data_and_subpace(
        model=MNISTFFNN(),
        loss_fn=indexed_loss,
    )
    stop = asyncio.get_running_loop().create_future()
    client_runner_q = mp.Queue()
    inference_task = asyncio.create_task(
        inference_runner(
            mad, stop=stop, run_in="process", client_runner_q=client_runner_q
        )
    )
    async with websockets.serve(
        partial(process_client, client_runner_q=client_runner_q), "localhost", 8765
    ):
        yield
    stop.set_result(True)
    await inference_task


@pytest.mark.timeout(45)
@pytest.mark.asyncio
async def test_single_websocket_thread(run_server_thread):
    await send_dims(5)


@pytest.mark.timeout(45)
@pytest.mark.asyncio
async def test_single_websocket_process(run_server_process):
    await send_dims(5)


@pytest.mark.timeout(45)
@pytest.mark.asyncio
async def test_multi_websocket_process(run_server_process):
    await asyncio.gather(*[asyncio.create_task(send_dims(2)) for x in range(10)])
