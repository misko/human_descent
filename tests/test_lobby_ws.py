import asyncio
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from hudes.game_control import MatchStatus, TeamResult
from hudes.lobby.fake import FakeGameControl
from hudes.lobby.server import serve_lobby


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.counter = 0
        self.events = []

    async def read_until(self, predicate):
        async with asyncio.timeout(3):
            while True:
                message = json.loads(await self.ws.recv())
                if message["type"] != "response":
                    self.events.append(message)
                if predicate(message):
                    return message

    async def command(self, name, payload=None):
        self.counter += 1
        rid = f"request-{self.counter}"
        await self.ws.send(
            json.dumps(
                {"v": 1, "type": name, "requestId": rid, "payload": payload or {}}
            )
        )
        result = await self.read_until(
            lambda m: m["type"] == "response" and m["requestId"] == rid
        )
        assert result["ok"], result
        return result["payload"]

    async def snapshot(self, predicate=lambda p: True):
        message = await self.read_until(
            lambda m: m["type"] == "snapshot" and predicate(m["payload"])
        )
        return message["payload"]


async def test_real_websocket_chat_invites_launch_results_and_resume():
    game = FakeGameControl()
    async with serve_lobby(game) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}/lobby/ws"
        async with connect(url) as a_ws, connect(url) as b_ws:
            a, b = Client(a_ws), Client(b_ws)
            a_identity = await a.command("hello", {"name": "Captain"})
            a_first = await a.snapshot()
            assert a_first["you"]["id"] == a_identity["playerId"]
            b_identity = await b.command("hello", {"name": "Member"})
            await b.snapshot()
            tid = (await a.command("team.create", {"name": "Only team"}))["teamId"]
            invite = await a.command(
                "team.invite", {"playerId": b_identity["playerId"]}
            )
            await b.command("invite.accept", invite)
            await b.command("chat.send", {"text": "Ready to go"})
            chat_snapshot = await a.snapshot(lambda p: len(p["chat"]) == 1)
            assert chat_snapshot["chat"][0]["text"] == "Ready to go"
            mid = (await a.command("team.start_speedrun"))["matchId"]
            await a.snapshot(
                lambda p: p["matches"] and p["matches"][0]["phase"] == "ready"
            )
            await a.command("match.ready", {"matchId": mid, "ready": True})
            await b.command("match.ready", {"matchId": mid, "ready": True})
            launch_a = (await a.read_until(lambda m: m["type"] == "game.launch"))[
                "payload"
            ]
            launch_b = (await b.read_until(lambda m: m["type"] == "game.launch"))[
                "payload"
            ]
            assert launch_a["sessionId"] == launch_b["sessionId"]
            assert launch_a["token"] != launch_b["token"]
            assert (launch_a["panelIndex"], launch_b["panelIndex"]) == (0, 1)
            assert game.start_calls == []
            await game.connect_player(mid, a_identity["playerId"])
            await game.connect_player(mid, b_identity["playerId"])
            countdown = await a.snapshot(
                lambda p: p["matches"][0]["phase"] == "countdown"
            )
            assert countdown["matches"][0]["startAt"] is not None
            assert game.start_calls == [mid]
            await game.emit(
                MatchStatus(mid, "finished", results=(TeamResult(tid, 0.123),))
            )
            terminal = await b.snapshot(
                lambda p: p["matches"][0]["phase"] == "finished"
            )
            assert terminal["teams"][0]["memberIds"] == [
                a_identity["playerId"],
                b_identity["playerId"],
            ]
            assert terminal["teams"][0]["status"] == "available"
            assert terminal["matches"][0]["results"] == [{"teamId": tid, "loss": 0.123}]
            for client in (a, b):
                seqs = [m["seq"] for m in client.events]
                assert seqs == sorted(set(seqs))
                for event in client.events:
                    if event["type"] == "snapshot":
                        assert "resumeToken" not in json.dumps(event)
                        assert '"token"' not in json.dumps(event)
            # Replace the original socket using the private resume credential.
            async with connect(url) as resumed_ws:
                resumed = Client(resumed_ws)
                identity = await resumed.command(
                    "hello",
                    {"name": "Ignored", "resumeToken": b_identity["resumeToken"]},
                )
                assert identity == b_identity
                snap = await resumed.snapshot()
                assert snap["you"]["teamId"] == tid and snap["you"]["name"] == "Member"
                await b_ws.wait_closed()


@pytest.mark.parametrize("frame", [b"\x08\x01", "{", '{"v":1,"v":2}'])
async def test_binary_and_malformed_frames_are_rejected_on_separate_lobby_socket(frame):
    async with serve_lobby(FakeGameControl()) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}/lobby/ws") as ws:
            await ws.send(frame)
            result = json.loads(await ws.recv())
            assert (
                result["ok"] is False and result["error"]["code"] == "invalid_request"
            )
            await ws.wait_closed()
            assert ws.close_code == 1008


async def test_unknown_websocket_path_does_not_enter_lobby():
    game = FakeGameControl()
    async with serve_lobby(game) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            with pytest.raises(ConnectionClosed):
                await ws.recv()
            assert ws.close_code == 1008
        assert server.lobby_service.state.players == {}
        assert game.prepare_calls == []


async def test_oversize_frames_are_closed_before_social_dispatch():
    async with serve_lobby(FakeGameControl()) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}/lobby/ws") as ws:
            await ws.send("x" * (server.lobby_service.policy.max_frame_bytes + 1))
            with pytest.raises(ConnectionClosed):
                await ws.recv()
            assert ws.close_code == 1009
