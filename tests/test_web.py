import asyncio
import uuid
from functools import partial

import pytest
import pytest_asyncio
import torch.multiprocessing as mp
import websockets

from hudes.models_and_datasets.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.websocket_client import send_dims
from hudes.websocket_server import inference_runner, process_client
from hudes import hudes_pb2
from hudes.hudes_pb2 import Control, DimAndStep, Config


async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)


@pytest_asyncio.fixture
async def run_server_thread():
    mad = mnist_model_data_and_subpace(model=MNISTFFNN())
    client_runner_q = mp.Queue()
    stop = asyncio.get_running_loop().create_future()
    inference_task = asyncio.create_task(
        inference_runner(
            mad, stop=stop, run_in="thread", client_runner_q=client_runner_q
        )
    )

    async with websockets.serve(
        partial(process_client, client_runner_q=client_runner_q),
        "localhost",
        8767,
    ):
        yield

    stop.set_result(True)
    client_runner_q.put(True)

    await inference_task


@pytest_asyncio.fixture
async def run_server_process():
    mad = mnist_model_data_and_subpace(
        model=MNISTFFNN(),
    )
    stop = asyncio.get_running_loop().create_future()
    client_runner_q = mp.Queue()
    inference_task = asyncio.create_task(
        inference_runner(
            mad, stop=stop, run_in="process", client_runner_q=client_runner_q
        )
    )
    async with websockets.serve(
        partial(process_client, client_runner_q=client_runner_q),
        "localhost",
        8767,
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


async def _open_configured_client(
    uri: str,
) -> tuple[websockets.WebSocketClientProtocol, str, str]:
    websocket = await websockets.connect(uri)
    session_token = f"session-{uuid.uuid4().hex}"
    config = hudes_pb2.Control(
        type=hudes_pb2.Control.CONTROL_CONFIG,
        config=hudes_pb2.Config(
            seed=42,
            dims_at_a_time=6,
            mesh_grid_size=31,
            mesh_step_size=0.1,
            mesh_grids=0,
            batch_size=32,
            dtype="float32",
            mesh_enabled=False,
            loss_lines=0,
            resume_supported=True,
            client_session_token=session_token,
        ),
    )
    await websocket.send(config.SerializeToString())
    resume_token = None
    for _ in range(20):
        data = await asyncio.wait_for(websocket.recv(), timeout=5)
        msg = hudes_pb2.Control()
        msg.ParseFromString(data)
        if msg.resume_token:
            resume_token = msg.resume_token
            break
    assert resume_token
    return websocket, resume_token, session_token


@pytest.mark.timeout(45)
@pytest.mark.asyncio
async def test_resume_handshake_success(run_server_thread):
    uri = "ws://localhost:8767"
    websocket, token, session_token = await _open_configured_client(uri)
    await websocket.close()

    async with websockets.connect("ws://localhost:8767") as resumed_ws:
        new_session_token = f"{session_token}-next"
        resume = hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_RESUME,
            resume=hudes_pb2.Control.Resume(
                token=token,
                last_request_idx=0,
                client_session_token=session_token,
                new_client_session_token=new_session_token,
            ),
        )
        await resumed_ws.send(resume.SerializeToString())
        data = await asyncio.wait_for(resumed_ws.recv(), timeout=5)
        msg = hudes_pb2.Control()
        msg.ParseFromString(data)
        assert msg.type == hudes_pb2.Control.CONTROL_RESUME
        assert msg.resume.status == hudes_pb2.Control.Resume.RESUME_OK
        assert msg.resume.client_session_token == new_session_token
        rotated_token = msg.resume.token

    # Old token should no longer be valid
    async with websockets.connect(uri) as another_ws:
        resume = hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_RESUME,
            resume=hudes_pb2.Control.Resume(
                token=token,
                last_request_idx=0,
                client_session_token=new_session_token,
                new_client_session_token=f"{new_session_token}-fail",
            ),
        )
        await another_ws.send(resume.SerializeToString())
        data = await asyncio.wait_for(another_ws.recv(), timeout=5)
        msg = hudes_pb2.Control()
        msg.ParseFromString(data)
        assert msg.type == hudes_pb2.Control.CONTROL_RESUME
        assert msg.resume.status == hudes_pb2.Control.Resume.RESUME_NOT_FOUND

    # Ensure new token can resume
    async with websockets.connect(uri) as final_ws:
        resume = hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_RESUME,
            resume=hudes_pb2.Control.Resume(
                token=rotated_token,
                last_request_idx=0,
                client_session_token=new_session_token,
                new_client_session_token=f"{new_session_token}-final",
            ),
        )
        await final_ws.send(resume.SerializeToString())
        data = await asyncio.wait_for(final_ws.recv(), timeout=5)
        msg = hudes_pb2.Control()
        msg.ParseFromString(data)
        assert msg.resume.status == hudes_pb2.Control.Resume.RESUME_OK


@pytest.mark.timeout(45)
@pytest.mark.asyncio
async def test_resume_invalid_token(run_server_thread):
    uri = "ws://localhost:8767"
    async with websockets.connect(uri) as websocket:
        resume = hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_RESUME,
            resume=hudes_pb2.Control.Resume(
                token="invalid-token",
                last_request_idx=0,
                client_session_token="nope",
                new_client_session_token="nope-next",
            ),
        )
        await websocket.send(resume.SerializeToString())
        data = await asyncio.wait_for(websocket.recv(), timeout=5)
        msg = hudes_pb2.Control()
        msg.ParseFromString(data)
        assert msg.type == hudes_pb2.Control.CONTROL_RESUME
        assert msg.resume.status == hudes_pb2.Control.Resume.RESUME_NOT_FOUND


async def _run_scripted_session(port: int, dims_sequence: list[dict[int, float]]):
    url = f"ws://localhost:{port}"
    request_idx = 0

    def next_request_idx():
        nonlocal request_idx
        current = request_idx
        request_idx += 1
        return current

    async with websockets.connect(url) as websocket:
        config_msg = Control(
            type=Control.CONTROL_CONFIG,
            request_idx=next_request_idx(),
            config=Config(
                seed=123,
                dims_at_a_time=6,
                mesh_grid_size=31,
                mesh_step_size=0.1,
                mesh_grids=1,
                batch_size=32,
                dtype="float32",
                mesh_enabled=False,
                loss_lines=0,
                resume_supported=False,
            ),
        )
        await websocket.send(config_msg.SerializeToString())

        for dims_map in dims_sequence:
            dims_msg = Control(
                type=Control.CONTROL_DIMS,
                request_idx=next_request_idx(),
                dims_and_steps=[
                    DimAndStep(dim=dim, step=step) for dim, step in dims_map.items()
                ],
            )
            await websocket.send(dims_msg.SerializeToString())

        next_batch_msg = Control(
            type=Control.CONTROL_NEXT_BATCH,
            request_idx=next_request_idx(),
        )
        await websocket.send(next_batch_msg.SerializeToString())

        expected_train = max(1, len(dims_sequence) + 2)
        train_losses = []
        val_losses = []
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline:
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            msg = Control()
            msg.ParseFromString(data)
            if msg.type == Control.CONTROL_TRAIN_LOSS_AND_PREDS:
                train_losses.append(msg.train_loss_and_preds.train_loss)
                if len(train_losses) >= expected_train and val_losses:
                    break
            elif msg.type == Control.CONTROL_VAL_LOSS:
                val_losses.append(msg.val_loss.val_loss)
                if len(train_losses) >= expected_train and val_losses:
                    break
        if not val_losses:
            raise AssertionError("No validation loss received during scripted session")
        return {
            "train_losses": train_losses[:expected_train],
            "val_loss": val_losses[-1],
        }


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_repeatable_scripted_session(run_server_thread):
    dims_sequence = [
        {0: 0.15},
        {1: -0.05, 3: 0.025},
        {2: 0.01},
    ]
    session_one = await _run_scripted_session(8767, dims_sequence)
    session_two = await _run_scripted_session(8767, dims_sequence)

    assert len(session_one["train_losses"]) == len(session_two["train_losses"])
    for a, b in zip(session_one["train_losses"], session_two["train_losses"]):
        assert a == pytest.approx(b, rel=0, abs=1e-9)
    assert session_one["val_loss"] == pytest.approx(
        session_two["val_loss"], rel=0, abs=1e-9
    )


@pytest.mark.timeout(45)
@pytest.mark.asyncio
async def test_resume_rejected_wrong_session_token(run_server_thread):
    uri = "ws://localhost:8767"
    websocket, token, session_token = await _open_configured_client(uri)
    await websocket.close()

    async with websockets.connect(uri) as resumed_ws:
        resume = hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_RESUME,
            resume=hudes_pb2.Control.Resume(
                token=token,
                last_request_idx=0,
                client_session_token=f"{session_token}-wrong",
                new_client_session_token=f"{session_token}-new",
            ),
        )
        await resumed_ws.send(resume.SerializeToString())
        data = await asyncio.wait_for(resumed_ws.recv(), timeout=5)
        msg = hudes_pb2.Control()
        msg.ParseFromString(data)
        assert msg.type == hudes_pb2.Control.CONTROL_RESUME
        assert msg.resume.status == hudes_pb2.Control.Resume.RESUME_REJECTED
