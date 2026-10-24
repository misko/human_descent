import asyncio
from contextlib import AsyncExitStack
from functools import partial
from queue import Queue

import pytest
import torch
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from hudes import hudes_pb2 as pb
from hudes.game.engine import RealTeamEngine
from hudes.game.service import TeamGameService
from hudes.game.websocket import TeamWebSocketHandler
from hudes.websocket_server import dispatch_connection
from tests.test_game_scoring_real_model import TinyClassifier, source_mad
from tests.test_team_game_session import (
    Engine,
    Socket,
    eventually,
    move,
    spec,
    vote,
    wire_admission,
)


async def receive(websocket, kind, predicate=lambda _: True):
    async with asyncio.timeout(10):
        while True:
            message = pb.Control.FromString(await websocket.recv())
            if message.type == kind and predicate(message):
                return message


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", [(1, 3), (3, 3)])
async def test_real_worker_shared_websockets_and_authoritative_final_scores(sizes):
    now = [10.0]
    source = source_mad(37)
    engine = RealTeamEngine.from_mad(source, model_factory=TinyClassifier)
    service = TeamGameService(engine, clock=lambda: now[0])
    prepared = await service.prepare_match(spec(*sizes))
    handler = partial(
        dispatch_connection, client_runner_q=Queue(), game_service=service
    )
    try:
        async with serve(handler, "127.0.0.1", 0) as server, AsyncExitStack() as stack:
            endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws"
            sockets = []
            for admission in prepared.admissions:
                ws = await stack.enter_async_context(connect(endpoint))
                sockets.append(ws)
                await ws.send(
                    pb.Control(
                        type=pb.Control.CONTROL_TEAM_ADMIT,
                        team_admission=wire_admission(admission),
                    ).SerializeToString()
                )
                snapshot = await receive(ws, pb.Control.CONTROL_TEAM_STATE)
                assert snapshot.team_state.config.mesh_grids == 3
            await service.start_match("match")
            states = [
                await receive(
                    ws,
                    pb.Control.CONTROL_TEAM_STATE,
                    lambda m: m.team_state.phase == "running",
                )
                for ws in sockets
            ]
            assert len({m.team_state.start_at for m in states}) == 1
            assert len({m.team_state.deadline for m in states}) == 1
            session_id = prepared.admissions[-1].session_id
            other_id = prepared.admissions[0].session_id
            baseline = await engine.weights_snapshot(session_id)
            other_baseline = await engine.weights_snapshot(other_id)
            # Each of three players moves its own panel in one actual shared model.
            for panel, ws in enumerate(sockets[sizes[0] :]):
                await ws.send(move(panel, step=0.05 * (panel + 1)).SerializeToString())
            for ws in sockets[sizes[0] :]:
                await receive(
                    ws,
                    pb.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
                    lambda m: m.model_revision == 3,
                )
            moved = await engine.weights_snapshot(session_id)
            expected = await engine._run(
                lambda: (
                    engine._mads[0]
                    .delta_from_dims({0: 0.05, 2: 0.1, 4: 0.15}, torch.float32)
                    .cpu()
                )
            )
            assert torch.allclose(moved, baseline + expected, atol=1e-7)
            assert torch.equal(await engine.weights_snapshot(other_id), other_baseline)
            for ws in sockets[sizes[0] :]:
                await ws.send(vote().SerializeToString())
            for ws in sockets[sizes[0] :]:
                state = await receive(
                    ws, pb.Control.CONTROL_TEAM_STATE, lambda m: m.generation == 1
                )
                assert not any(p.voted for p in state.team_state.roster)
                await receive(
                    ws, pb.Control.CONTROL_MESHGRID_RESULTS, lambda m: m.generation == 1
                )
            assert torch.equal(await engine.weights_snapshot(session_id), moved)
            # Compute an independent full-population score from frozen vectors.
            expected_losses = []
            for weights in (other_baseline, moved):
                model = TinyClassifier()
                torch.nn.utils.vector_to_parameters(weights, model.parameters())
                data = source.batchers["val"].ds
                with torch.no_grad():
                    loss = (
                        -model(data.data.squeeze(1))[
                            torch.arange(len(data)), data.labels
                        ]
                        .mean()
                        .item()
                    )
                expected_losses.append(loss)
            # Disconnected teams still drain and finalize under the same cutoff.
            await asyncio.gather(*(ws.close() for ws in sockets))
            now[0] = 41
            await service.tick()
            await eventually(lambda: service.matches["match"].phase == "finished")
            result = await service.get_match_status("match")
            assert [r.loss for r in result.results] == pytest.approx(
                expected_losses, abs=1e-6
            )
            assert result.connected_player_ids == ()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_replaced_socket_buffered_input_cannot_mutate_active_slot():
    """A takeover can race a frame already buffered on the old connection."""
    engine = Engine()
    service = TeamGameService(engine)
    prepared = await service.prepare_match(spec(1))
    old = Socket()
    admitted = asyncio.Event()
    release_frame = asyncio.Event()
    raw_admit = pb.Control(
        type=pb.Control.CONTROL_TEAM_ADMIT,
        team_admission=wire_admission(prepared.admissions[0]),
    ).SerializeToString()

    class BufferedSocket(Socket):
        async def recv(self):
            return raw_admit

        def __aiter__(self):
            return self

        async def __anext__(self):
            admitted.set()
            await release_frame.wait()
            return move(0, step=10).SerializeToString()

    old = BufferedSocket()
    task = asyncio.create_task(TeamWebSocketHandler(service)(old))
    try:
        await admitted.wait()
        await service.start_match("match")
        rebound = await service.admit(wire_admission(prepared.admissions[0]), Socket())
        release_frame.set()
        await asyncio.wait_for(task, 2)
        session = rebound[0]
        assert session.revision == 0 and engine.calls == []
        assert session.connections["p0-0"] is rebound[2]
        assert old.closed
    finally:
        release_frame.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.close()
