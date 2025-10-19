import asyncio
import os
import struct
import time
from functools import partial

import pytest
import pytest_asyncio
import torch.multiprocessing as mp
import websockets

from hudes import hudes_pb2
from hudes.high_scores import get_all_scores, get_log_by_id, get_top_scores, init_db
from hudes.models_and_datasets.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.websocket_server import inference_runner, process_client


def _decode_len_prefixed_log(blob: bytes):
    messages = []
    offset = 0
    n = len(blob)
    while offset + 4 <= n:
        (length,) = struct.unpack(">I", blob[offset : offset + 4])
        offset += 4
        if length < 0 or offset + length > n:
            break
        raw = blob[offset : offset + length]
        offset += length
        msg = hudes_pb2.Control()
        msg.ParseFromString(raw)
        messages.append(msg)
    return messages


@pytest_asyncio.fixture
async def run_speed_server_thread(tmp_path):
    # Configure short speed run and isolated DB path
    os.environ["HUDES_SPEED_RUN_SECONDS"] = "3"
    db_path = tmp_path / "scores.sqlite3"
    os.environ["HIGH_SCORES_PATH"] = str(db_path)
    init_db(str(db_path))

    mad = mnist_model_data_and_subpace(model=MNISTFFNN())
    # Use process/threaded runner as in other tests
    client_runner_q = mp.Queue()
    stop = asyncio.get_running_loop().create_future()
    inference_task = asyncio.create_task(
        inference_runner(
            mad, stop=stop, run_in="thread", client_runner_q=client_runner_q
        )
    )

    # Start websocket server on a dedicated port
    port = 8769
    async with websockets.serve(
        partial(process_client, client_runner_q=client_runner_q),
        "localhost",
        port,
    ):
        yield port, str(db_path)

    stop.set_result(True)
    await inference_task


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_speed_run_flow_and_db(run_speed_server_thread):
    port, db_path = run_speed_server_thread

    async with websockets.connect(f"ws://localhost:{port}") as websocket:
        # Send a config to initialize
        cfg = hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_CONFIG,
            config=hudes_pb2.Config(
                seed=123,
                dims_at_a_time=6,
                mesh_grid_size=21,
                mesh_step_size=0.1,
                mesh_grids=1,
                batch_size=32,
                dtype="float32",
            ),
        )
        await websocket.send(cfg.SerializeToString())

        # Start speed run
        await websocket.send(
            hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_SPEED_RUN_START
            ).SerializeToString()
        )

        # Interact during the run: dims, next batch, and an SGD (ignored)
        last_total_sgd = None
        saw_timer = False
        t_start = time.time()
        while time.time() - t_start < 1.2:
            dims = hudes_pb2.Control(
                type=hudes_pb2.Control.CONTROL_DIMS,
                dims_and_steps=[
                    hudes_pb2.DimAndStep(dim=0, step=0.1),
                    hudes_pb2.DimAndStep(dim=1, step=-0.05),
                ],
                request_idx=1,
            )
            await websocket.send(dims.SerializeToString())
            await asyncio.sleep(0.02)
            await websocket.send(
                hudes_pb2.Control(
                    type=hudes_pb2.Control.CONTROL_NEXT_BATCH
                ).SerializeToString()
            )
            # Attempt SGD (server should ignore during speed run)
            await websocket.send(
                hudes_pb2.Control(
                    type=hudes_pb2.Control.CONTROL_SGD_STEP, sgd_steps=1
                ).SerializeToString()
            )

            # Try to read any messages and inspect fields
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                msg = hudes_pb2.Control()
                msg.ParseFromString(raw)
                if msg.type == hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS:
                    # total SGD may be reported here
                    if msg.HasField("total_sgd_steps"):
                        last_total_sgd = msg.total_sgd_steps
                elif msg.type == hudes_pb2.Control.CONTROL_MESHGRID_RESULTS:
                    # When active, server includes seconds remaining at
                    # Control level
                    if msg.HasField("speed_run_seconds_remaining"):
                        srs = msg.speed_run_seconds_remaining
                        if srs > 0:
                            saw_timer = True
            except asyncio.TimeoutError:
                pass

        # Ensure SGD was not applied
        if last_total_sgd is not None:
            assert last_total_sgd == 0, "SGD steps should not increase during speed run"

        # Wait for run to end and submit score
        await asyncio.sleep(2.2)
        hs = hudes_pb2.Control(
            type=hudes_pb2.Control.CONTROL_HIGH_SCORE_LOG,
            high_score=hudes_pb2.Control.HighScore(name="TEST"),
        )
        await websocket.send(hs.SerializeToString())

        # Drain any remaining messages
        await asyncio.sleep(0.2)

    # Verify DB has entry
    rows = get_top_scores(path=db_path)
    assert rows, "No high score rows found"
    assert any(r[0] == "TEST" for r in rows), "Submitted name not found"

    # Fetch full rows to get an id
    all_rows = get_all_scores(path=db_path)
    assert all_rows, "No rows in high_scores table"
    row_id = all_rows[0][0]
    blob = get_log_by_id(row_id, path=db_path)
    assert blob is not None and len(blob) > 0, "Missing or empty log blob"

    decoded = _decode_len_prefixed_log(blob)
    assert len(decoded) >= 1, "Log did not decode into messages"
    # Expect the log to include our DIMS or NEXT_BATCH messages at least
    assert any(
        m.type
        in (
            hudes_pb2.Control.CONTROL_DIMS,
            hudes_pb2.Control.CONTROL_NEXT_BATCH,
        )
        for m in decoded
    ), "Expected interaction messages in log"
    # Timer should have been included during run
    assert saw_timer, "Did not observe speed run countdown in responses"
