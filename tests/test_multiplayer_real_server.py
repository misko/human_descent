"""Real MNIST/CPU smoke through the production run_wrapper composition."""

import asyncio
import socket
from functools import partial
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect

from hudes import hudes_pb2 as pb
from hudes.game_control import MatchPreset
from hudes.lobby import LobbyService
from hudes.websocket_server import run_wrapper
from tests.test_multiplayer_integration import LobbyPeer, game_event, launch, make_team


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.timeout(45)
@pytest.mark.parametrize("sizes", [(1, 3), (3, 3)])
async def test_production_composition_real_mnist_shared_refresh_and_final_results(
    monkeypatch, tmp_path, sizes
):
    port, health_port = free_port(), free_port()
    monkeypatch.setenv("HUDES_HEALTH_PORT", str(health_port))
    monkeypatch.setenv("HIGH_SCORES_PATH", str(tmp_path / "scores.sqlite3"))
    # Test-only policy injection keeps the actual production composition and
    # inference adapter intact while bounding the smoke test's duration.
    monkeypatch.setattr(
        "hudes.lobby.LobbyService",
        partial(
            LobbyService,
            preset=MatchPreset(
                duration_seconds=2,
                countdown_seconds=0.05,
                batch_size=4,
                mesh_grid_size=5,
            ),
        ),
    )
    stop = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        run_wrapper(
            SimpleNamespace(
                device="cpu",
                model="ffnn",
                run_in="thread",
                port=port,
                ssl_pem=None,
                max_batch_size=512,
                max_grids=3,
                max_grid_size=5,
                download_dataset_and_exit=False,
            ),
            stop_future=stop,
        )
    )
    sockets = []
    endpoint = f"ws://127.0.0.1:{port}"
    try:
        async with asyncio.timeout(15):
            while True:
                if task.done():
                    await task
                try:
                    reader, writer = await asyncio.open_connection(
                        "127.0.0.1", health_port
                    )
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    await asyncio.sleep(0.02)

        async def player(name):
            websocket = await connect(endpoint + "/lobby/ws")
            sockets.append(websocket)
            peer = LobbyPeer(websocket)
            peer.identity = await peer.command("hello", {"name": name})
            return peer

        async def admission(descriptor):
            websocket = await connect(endpoint + "/ws")
            sockets.append(websocket)
            await websocket.send(
                pb.Control(
                    type=pb.Control.CONTROL_TEAM_ADMIT,
                    team_admission=pb.TeamAdmission(
                        match_id=descriptor["matchId"],
                        session_id=descriptor["sessionId"],
                        player_id=descriptor["playerId"],
                        token=descriptor["token"],
                    ),
                ).SerializeToString()
            )
            return websocket

        left_id, left = await make_team(player, sizes[0], "Real left")
        right_id, right = await make_team(player, sizes[1], "Real right")
        challenge = await left[0].command("challenge.create", {"teamId": right_id})
        match = await right[0].command(
            "challenge.accept", {"challengeId": challenge["challengeId"]}
        )
        descriptors, game_sockets, states = await launch(
            left + right, match["matchId"], admission
        )
        right_offset = len(left)
        right_session = descriptors[right_offset]["sessionId"]
        for member_index in range(len(right)):
            await game_sockets[right_offset + member_index].send(
                pb.Control(
                    type=pb.Control.CONTROL_DIMS,
                    generation=0,
                    request_idx=1,
                    team_session_id=right_session,
                    team_match_id=match["matchId"],
                    dims_and_steps=[pb.DimAndStep(dim=member_index * 2, step=0.025)],
                ).SerializeToString()
            )
        for member_index in range(len(right)):
            await game_sockets[right_offset + member_index].send(
                pb.Control(
                    type=pb.Control.CONTROL_NEXT_DIMS,
                    generation=0,
                    team_session_id=right_session,
                    team_match_id=match["matchId"],
                ).SerializeToString()
            )
        for websocket in game_sockets[right_offset:]:
            refreshed = await game_event(
                websocket,
                pb.Control.CONTROL_MESHGRID_RESULTS,
                lambda message: message.generation == 1,
            )
            assert refreshed.model_revision == 3
            assert list(refreshed.mesh_grid_shape) == [3, 5, 5]
        final = await game_event(
            game_sockets[0],
            pb.Control.CONTROL_TEAM_STATE,
            lambda message: message.team_state.phase == "finished",
        )
        assert {result.team_id for result in final.team_state.results} == {
            left_id,
            right_id,
        }
        assert all(0 < result.loss < 100 for result in final.team_state.results)
        assert len({state.team_state.start_at for state in states}) == 1
        snapshot = await left[0].event(
            "snapshot",
            lambda data: any(
                item["id"] == match["matchId"] and item["phase"] == "finished"
                for item in data["matches"]
            ),
        )
        assert all(team["status"] == "available" for team in snapshot["teams"])
        assert len(snapshot["teams"]) == 2
    finally:
        await asyncio.gather(
            *(websocket.close() for websocket in sockets), return_exceptions=True
        )
        if not stop.done():
            stop.set_result(True)
        await asyncio.wait_for(task, 10)
