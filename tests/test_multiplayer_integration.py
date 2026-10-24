"""Real social/game sockets composed with deterministic numerical inference."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from hudes import hudes_pb2 as pb
from hudes.game import TeamGameService
from hudes.game.websocket import TeamWebSocketHandler
from hudes.game_control import MatchPreset
from hudes.lobby import LobbyPolicy, LobbyService, LobbyWebSocketHandler
from tests.multiplayer_support import DeterministicEngine


class LobbyPeer:
    def __init__(self, websocket):
        self.websocket = websocket
        self.counter = 0
        self.events = []
        self.identity = None

    async def command(self, kind, payload=None, *, ok=True):
        self.counter += 1
        request_id = f"request-{self.counter}"
        await self.websocket.send(
            json.dumps(
                {
                    "v": 1,
                    "type": kind,
                    "requestId": request_id,
                    "payload": payload or {},
                }
            )
        )
        async with asyncio.timeout(5):
            while True:
                envelope = json.loads(await self.websocket.recv())
                if (
                    envelope.get("type") == "response"
                    and envelope.get("requestId") == request_id
                ):
                    assert envelope["ok"] is ok, envelope
                    return envelope.get("payload") if ok else envelope["error"]
                self.events.append(envelope)

    async def event(self, kind, predicate=lambda _: True):
        async with asyncio.timeout(5):
            while True:
                for index, envelope in enumerate(self.events):
                    if envelope.get("type") == kind and predicate(envelope["payload"]):
                        return self.events.pop(index)["payload"]
                self.events.append(json.loads(await self.websocket.recv()))

    async def snapshot(self):
        await self.command("snapshot")
        # The requested full snapshot follows its correlated response.
        return await self.event(
            "snapshot", lambda data: data["you"]["id"] == self.identity["playerId"]
        )


@asynccontextmanager
async def running_services(*, duration=0.25):
    engine = DeterministicEngine()
    game = TeamGameService(engine)
    lobby = LobbyService(
        game,
        preset=MatchPreset(
            duration_seconds=duration,
            countdown_seconds=0.01,
            mesh_grid_size=5,
        ),
        policy=LobbyPolicy(maintenance_seconds=0.01),
    )
    social = LobbyWebSocketHandler(lobby)
    gameplay = TeamWebSocketHandler(game)
    peers, sockets = [], []

    async def dispatch(websocket):
        if websocket.request.path == "/lobby/ws":
            await social(websocket)
        elif websocket.request.path == "/ws":
            await gameplay(websocket)
        else:
            await websocket.close(code=1008, reason="unknown endpoint")

    await lobby.start()
    async with serve(dispatch, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"

        async def player(name):
            websocket = await connect(endpoint + "/lobby/ws")
            sockets.append(websocket)
            peer = LobbyPeer(websocket)
            peer.identity = await peer.command("hello", {"name": name})
            peers.append(peer)
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

        try:
            yield engine, game, lobby, player, admission, endpoint
        finally:
            await asyncio.gather(
                *(socket.close() for socket in sockets), return_exceptions=True
            )
            await lobby.close()
            await game.close()


async def game_event(websocket, kind, predicate=lambda _: True):
    async with asyncio.timeout(5):
        while True:
            message = pb.Control.FromString(await websocket.recv())
            if message.type == kind and predicate(message):
                return message


async def make_team(player, size, prefix):
    members = [await player(f"{prefix}-captain")]
    result = await members[0].command("team.create", {"name": prefix})
    for number in range(1, size):
        member = await player(f"{prefix}-{number}")
        invite = await members[0].command(
            "team.invite", {"playerId": member.identity["playerId"]}
        )
        await member.command("invite.accept", {"inviteId": invite["inviteId"]})
        members.append(member)
    return result["teamId"], members


async def launch(members, match_id, admission):
    # Wait for preparation to finish before acknowledging the fixed roster.
    for member in members:
        await member.event(
            "snapshot",
            lambda data: any(
                match["id"] == match_id and match["phase"] == "ready"
                for match in data["matches"]
            ),
        )
    for member in members:
        await member.command("match.ready", {"matchId": match_id, "ready": True})
    descriptors = [
        await member.event("game.launch", lambda data: data["matchId"] == match_id)
        for member in members
    ]
    sockets = [await admission(descriptor) for descriptor in descriptors]
    states = [
        await game_event(
            socket,
            pb.Control.CONTROL_TEAM_STATE,
            lambda message: message.team_state.phase == "running",
        )
        for socket in sockets
    ]
    return descriptors, sockets, states


@pytest.mark.parametrize(
    "left_size,right_size", [(left, right) for left in (1, 2, 3) for right in (1, 2, 3)]
)
async def test_all_ordered_team_sizes_complete_through_individual_admission(
    left_size, right_size
):
    async with running_services() as (engine, game, lobby, player, admission, endpoint):
        left_id, left = await make_team(player, left_size, "Left")
        right_id, right = await make_team(player, right_size, "Right")
        challenge = await left[0].command("challenge.create", {"teamId": right_id})
        match = await right[0].command(
            "challenge.accept", {"challengeId": challenge["challengeId"]}
        )
        members = left + right
        descriptors, sockets, states = await launch(
            members, match["matchId"], admission
        )
        assert (
            len({descriptor["token"] for descriptor in descriptors})
            == left_size + right_size
        )
        assert len({descriptor["sessionId"] for descriptor in descriptors}) == 2
        assert len({state.team_state.start_at for state in states}) == 1
        assert len({state.team_state.deadline for state in states}) == 1
        for state in states:
            size = left_size if state.team_state.team_id == left_id else right_size
            assert len(state.team_state.roster) == size
            assert state.team_state.config.dims_at_a_time == 6
            assert state.team_state.config.mesh_grids == 3
        final = await game_event(
            sockets[0],
            pb.Control.CONTROL_TEAM_STATE,
            lambda message: message.team_state.phase == "finished",
        )
        assert {result.team_id for result in final.team_state.results} == {
            left_id,
            right_id,
        }
        assert {result.loss for result in final.team_state.results} == {2.3}
        snapshot = await members[0].event(
            "snapshot",
            lambda data: any(
                match["id"] == descriptors[0]["matchId"]
                and match["phase"] == "finished"
                for match in data["matches"]
            ),
        )
        assert all(team["status"] == "available" for team in snapshot["teams"])
        assert {tuple(team["memberIds"]) for team in snapshot["teams"]} == {
            tuple(member.identity["playerId"] for member in left),
            tuple(member.identity["playerId"] for member in right),
        }
        public = json.dumps(snapshot)
        assert all(descriptor["token"] not in public for descriptor in descriptors)


@pytest.mark.parametrize("size", [1, 2, 3])
async def test_speedrun_can_repeat_without_changing_team_membership(size):
    async with running_services() as (_, game, _, player, admission, _):
        team_id, members = await make_team(player, size, "Together")
        match_ids = []
        for _cycle in range(2):
            match = await members[0].command("team.start_speedrun")
            descriptors, sockets, states = await launch(
                members, match["matchId"], admission
            )
            match_ids.append(match["matchId"])
            await game_event(
                sockets[0],
                pb.Control.CONTROL_TEAM_STATE,
                lambda message: message.team_state.phase == "finished",
            )
            snapshot = await members[0].event(
                "snapshot",
                lambda data: any(
                    item["id"] == match["matchId"] and item["phase"] == "finished"
                    for item in data["matches"]
                ),
            )
            team = next(item for item in snapshot["teams"] if item["id"] == team_id)
            assert team["memberIds"] == [
                member.identity["playerId"] for member in members
            ]
            assert team["status"] == "available"
            await asyncio.gather(*(socket.close() for socket in sockets))
        assert len(set(match_ids)) == 2


async def test_votes_shared_moves_reconnect_and_independent_social_connection():
    async with running_services(duration=2) as (
        engine,
        game,
        _,
        player,
        admission,
        endpoint,
    ):
        team_id, members = await make_team(player, 3, "Shared")
        match = await members[0].command("team.start_speedrun")
        descriptors, sockets, states = await launch(
            members, match["matchId"], admission
        )
        session_id = descriptors[0]["sessionId"]
        jobs_before = len(engine.jobs)

        async def send(index, kind, generation=0, **extra):
            await sockets[index].send(
                pb.Control(
                    type=kind,
                    generation=generation,
                    team_session_id=session_id,
                    team_match_id=match["matchId"],
                    request_idx=10,
                    **extra,
                ).SerializeToString()
            )

        await send(0, pb.Control.CONTROL_NEXT_DIMS)
        voted = await game_event(
            sockets[1],
            pb.Control.CONTROL_TEAM_STATE,
            lambda msg: msg.team_state.roster[0].voted,
        )
        assert voted.team_state.generation == 0
        assert len(engine.jobs) == jobs_before
        await send(0, pb.Control.CONTROL_NEXT_DIMS)
        await send(
            1, pb.Control.CONTROL_DIMS, dims_and_steps=[pb.DimAndStep(dim=2, step=0.05)]
        )
        update = await game_event(
            sockets[0],
            pb.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
            lambda msg: msg.model_revision == 1,
        )
        assert update.generation == 0
        assert engine.sessions[session_id]["weights"] == {2: pytest.approx(0.05)}

        # Chat and a game connection have independent lifetimes.
        await sockets[2].close()
        await members[2].command("chat.send", {"text": "Still in the chat"})
        await members[0].event(
            "snapshot",
            lambda data: any(
                item["text"] == "Still in the chat" for item in data["chat"]
            ),
        )
        sockets[2] = await admission(descriptors[2])
        restored = await game_event(sockets[2], pb.Control.CONTROL_TEAM_STATE)
        assert restored.team_state.panel_index == 2
        assert restored.team_state.roster[0].voted
        await members[0].websocket.close()
        await send(1, pb.Control.CONTROL_NEXT_DIMS)
        await send(2, pb.Control.CONTROL_NEXT_DIMS)
        for socket in sockets:
            advanced = await game_event(
                socket,
                pb.Control.CONTROL_TEAM_STATE,
                lambda msg: msg.team_state.generation == 1,
            )
            assert not any(member.voted for member in advanced.team_state.roster)
            assert advanced.team_state.dims_offset == 6
        await game_event(
            sockets[1],
            pb.Control.CONTROL_MESHGRID_RESULTS,
            lambda msg: msg.generation == 1,
        )
        assert engine.sessions[session_id]["weights"] == {2: pytest.approx(0.05)}
        await send(
            1,
            pb.Control.CONTROL_DIMS,
            generation=0,
            dims_and_steps=[pb.DimAndStep(dim=2, step=10)],
        )
        error = await game_event(sockets[1], pb.Control.CONTROL_TEAM_ERROR)
        assert "generation" in error.error_message
        assert engine.sessions[session_id]["weights"] == {2: pytest.approx(0.05)}
