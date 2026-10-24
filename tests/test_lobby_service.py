"""Social authorization/lifecycle tests. No model, dataset, or protobuf imports."""

import asyncio
import json
from dataclasses import replace

import pytest

from hudes.game_control import MatchStatus, TeamResult
from hudes.lobby import LobbyPolicy, LobbyService
from hudes.lobby.fake import FakeGameControl


class Clock:
    def __init__(self):
        self.value = 10000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class Peer:
    def __init__(self, service, conn):
        self.service = service
        self.conn = conn
        self.counter = 0
        self.inbox = []
        self.player_id = None
        self.token = None

    async def send(self, command, payload=None, request_id=None):
        self.counter += 1
        request = {
            "v": 1,
            "type": command,
            "requestId": request_id or f"request-{self.counter}",
            "payload": payload or {},
        }
        result = await self.service.handle(self.conn, request)
        self.pull()
        return result

    def pull(self):
        while not self.conn.outgoing.empty():
            event = self.conn.outgoing.get_nowait()
            if event is not None:
                self.inbox.append(event)
        return self.inbox

    async def ok(self, command, payload=None, request_id=None):
        result = await self.send(command, payload, request_id)
        assert result["ok"], result
        return result["payload"]

    async def error(self, command, payload, code):
        result = await self.send(command, payload)
        assert result.get("error", {}).get("code") == code, result

    async def snapshot(self):
        return await self.service.snapshot(self.conn)


@pytest.fixture
async def lobby():
    clock = Clock()
    game = FakeGameControl(clock=clock)
    policy = LobbyPolicy(command_burst=1000, chat_burst=1000, maintenance_seconds=3600)
    service = LobbyService(game, clock=clock, policy=policy)
    await service.start()
    peers = []

    async def peer(name, token=None):
        client = Peer(service, await service.connect())
        hello = {"name": name}
        if token:
            hello["resumeToken"] = token
        identity = await client.ok("hello", hello)
        client.player_id, client.token = identity["playerId"], identity["resumeToken"]
        peers.append(client)
        return client

    async def team(size, name):
        captain = await peer(f"{name} captain")
        team_id = (await captain.ok("team.create", {"name": name}))["teamId"]
        members = [captain]
        for number in range(1, size):
            member = await peer(f"{name} player {number}")
            invitation = await captain.ok("team.invite", {"playerId": member.player_id})
            await member.ok("invite.accept", invitation)
            members.append(member)
        return team_id, members

    service.test_peer = peer
    service.test_team = team
    service.test_clock = clock
    service.test_peers = peers
    yield service, game
    await service.close()


async def test_identity_resume_replaces_connection_and_keeps_captain(lobby):
    service, _ = lobby
    team_id, (captain,) = await service.test_team(1, "Minima")
    token = captain.token
    resumed = await service.test_peer("Untrusted rename", token)
    assert resumed.player_id == captain.player_id
    assert captain.conn.closed and captain.conn.close_reason == "replaced"
    await service.disconnect(captain.conn)
    snap = await resumed.snapshot()
    assert snap["you"] == {
        "id": captain.player_id,
        "name": "Minima captain",
        "teamId": team_id,
    }
    assert snap["teams"][0]["captainId"] == captain.player_id
    assert next(p for p in snap["players"] if p["id"] == captain.player_id)["connected"]
    bad = Peer(service, await service.connect())
    await bad.error(
        "hello", {"name": "Impostor", "resumeToken": "not-valid"}, "unauthorized"
    )
    assert bad.conn.player_id is None
    await bad.ok("hello", {"name": "New person"})


async def test_captain_permissions_and_no_public_join(lobby):
    service, _ = lobby
    team_id, (captain, member) = await service.test_team(2, "Team")
    outsider = await service.test_peer("Outside")
    for command, payload in [
        ("team.rename", {"name": "Stolen"}),
        ("team.invite", {"playerId": outsider.player_id}),
        ("team.start_speedrun", {}),
        ("challenge.create", {"teamId": team_id}),
    ]:
        await member.error(command, payload, "not_captain")
    await outsider.error("team.join", {"teamId": team_id}, "invalid_request")
    await captain.error(
        "team.invite",
        {"playerId": outsider.player_id, "captainId": captain.player_id},
        "invalid_request",
    )
    invitation = await captain.ok("team.invite", {"playerId": outsider.player_id})
    await member.error("invite.accept", invitation, "unauthorized")
    await outsider.ok("invite.decline", invitation)
    await outsider.error("invite.accept", invitation, "conflict")


async def test_capacity_and_one_team_are_atomic_during_racing_acceptances(lobby):
    service, _ = lobby
    team_id, (captain, _) = await service.test_team(2, "Team")
    first, second = await service.test_peer("First"), await service.test_peer("Second")
    invitations = [
        await captain.ok("team.invite", {"playerId": p.player_id})
        for p in (first, second)
    ]
    replies = await asyncio.gather(
        *(
            p.send("invite.accept", invitation)
            for p, invitation in zip((first, second), invitations)
        )
    )
    assert sorted(r["ok"] for r in replies) == [False, True]
    assert next(r for r in replies if not r["ok"])["error"]["code"] == "team_full"
    assert len(service.state.teams[team_id].member_ids) == 3
    winner = first if replies[0]["ok"] else second
    await winner.error("team.create", {"name": "Second team"}, "already_in_team")


async def test_racing_invites_from_two_teams_never_double_join(lobby):
    service, _ = lobby
    _, (a,) = await service.test_team(1, "A")
    _, (b,) = await service.test_team(1, "B")
    target = await service.test_peer("Target")
    invitations = [
        await p.ok("team.invite", {"playerId": target.player_id}) for p in (a, b)
    ]
    replies = await asyncio.gather(
        *(target.send("invite.accept", i) for i in invitations)
    )
    assert sum(r["ok"] for r in replies) == 1
    assert (
        sum(target.player_id in t.member_ids for t in service.state.teams.values()) == 1
    )


async def test_private_invites_credentials_and_ordered_snapshots(lobby):
    service, _ = lobby
    _, (captain,) = await service.test_team(1, "Team")
    recipient, bystander = (
        await service.test_peer("Recipient"),
        await service.test_peer("Bystander"),
    )
    invitation = await captain.ok("team.invite", {"playerId": recipient.player_id})
    assert (await bystander.snapshot())["invitations"] == []
    assert (await recipient.snapshot())["invitations"][0]["id"] == invitation[
        "inviteId"
    ]
    match_id = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    await captain.ok("match.ready", {"matchId": match_id, "ready": True})
    await service.wait_idle()
    for peer in service.test_peers:
        events = peer.pull()
        public = [e for e in events if e["type"] == "snapshot"]
        assert all(
            "resumeToken" not in json.dumps(e) and '"token"' not in json.dumps(e)
            for e in public
        )
        assert [e["seq"] for e in public] == sorted({e["seq"] for e in public})
        launches = [e for e in events if e["type"] == "game.launch"]
        assert len(launches) == (1 if peer is captain else 0)
        if launches:
            assert launches[0]["payload"]["playerId"] == captain.player_id


async def test_request_retries_replay_response_and_conflicting_reuse_fails(lobby):
    service, game = lobby
    _, (captain,) = await service.test_team(1, "Team")
    payload = await captain.ok("chat.send", {"text": "hello"}, "chat-id")
    assert await captain.ok("chat.send", {"text": "hello"}, "chat-id") == payload
    assert len(service.state.chat) == 1
    result = await captain.send("chat.send", {"text": "different"}, "chat-id")
    assert result["error"]["code"] == "conflict"
    first = await captain.ok("team.start_speedrun", request_id="start-id")
    assert await captain.ok("team.start_speedrun", request_id="start-id") == first
    await service.wait_idle()
    assert game.prepare_calls == [first["matchId"]]


@pytest.mark.parametrize("challenger_size", (1, 2, 3))
@pytest.mark.parametrize("target_size", (1, 2, 3))
async def test_all_ordered_team_sizes_ready_launch_connect_countdown_results(
    lobby, challenger_size, target_size
):
    service, game = lobby
    a_id, a = await service.test_team(challenger_size, "A")
    b_id, b = await service.test_team(target_size, "B")
    challenge = await a[0].ok("challenge.create", {"teamId": b_id})
    answer = await b[0].ok("challenge.accept", challenge)
    mid = answer["matchId"]
    await service.wait_idle()
    match = service.state.matches[mid]
    assert match.phase == "ready"
    assert [len(t.players) for t in game.specs[mid].teams] == [
        challenger_size,
        target_size,
    ]
    assert [p.panel_index for p in game.specs[mid].teams[1].players] == list(
        range(target_size)
    )
    players = a + b
    for peer in players[:-1]:
        await peer.ok("match.ready", {"matchId": mid, "ready": True})
    assert not match.launched and not game.start_calls
    await players[-1].ok("match.ready", {"matchId": mid, "ready": True})
    await service.wait_idle()
    assert match.launched and not game.start_calls
    for peer in players:
        own_launches = [e["payload"] for e in peer.pull() if e["type"] == "game.launch"]
        assert len(own_launches) == 1
        assert own_launches[0]["token"] == match.admissions[peer.player_id].token
    for peer in players[:-1]:
        await game.connect_player(mid, peer.player_id)
    assert not game.start_calls
    await game.connect_player(mid, players[-1].player_id)
    await service.wait_idle()
    assert game.start_calls == [mid]
    assert match.phase == "countdown"
    status = game.statuses[mid]
    await game.emit(status)  # Duplicate/replayed events cannot start twice.
    await game.emit(replace(status, phase="running"))
    await game.emit(replace(status, phase="prepared"))  # Obsolete status ignored.
    assert match.phase == "running"
    results = (TeamResult(a_id, 0.5), TeamResult(b_id, 0.5))
    await game.emit(replace(status, phase="finished", results=results))
    await game.emit(replace(status, phase="failed", reason="late_failure"))
    assert match.phase == "finished" and match.results == results
    assert all(t.match_id is None for t in service.state.teams.values())
    assert service.state.teams[a_id].member_ids == [p.player_id for p in a]
    snap = await a[0].snapshot()
    assert snap["matches"][0]["results"] == [
        {"teamId": a_id, "loss": 0.5},
        {"teamId": b_id, "loss": 0.5},
    ]


async def test_reservations_block_roster_changes_and_parallel_match_start(lobby):
    service, game = lobby
    _, members = await service.test_team(2, "Team")
    captain, member = members
    replies = await asyncio.gather(
        captain.send("team.start_speedrun"), captain.send("team.start_speedrun")
    )
    assert sum(r["ok"] for r in replies) == 1
    assert next(r for r in replies if not r["ok"])["error"]["code"] == "team_busy"
    await member.error("team.leave", {}, "team_busy")
    await captain.error("team.rename", {"name": "Other"}, "team_busy")
    await service.wait_idle()
    assert len(game.prepare_calls) == 1


async def test_roster_changes_invalidate_challenges_and_captain_leave_disbands(lobby):
    service, _ = lobby
    a_id, (captain, member) = await service.test_team(2, "A")
    b_id, (other,) = await service.test_team(1, "B")
    challenge = await captain.ok("challenge.create", {"teamId": b_id})
    await member.ok("team.leave")
    await other.error("challenge.accept", challenge, "conflict")
    await captain.ok("team.leave")
    assert a_id not in service.state.teams
    assert (await member.snapshot())["you"]["teamId"] is None


async def test_expiry_grace_and_token_retention_use_injected_clock(lobby):
    service, _ = lobby
    team_id, (captain,) = await service.test_team(1, "Team")
    target = await service.test_peer("Target")
    invitation = await captain.ok("team.invite", {"playerId": target.player_id})
    service.test_clock.advance(service.policy.invitation_seconds)
    await target.error("invite.accept", invitation, "expired")
    await service.disconnect(captain.conn)
    service.test_clock.advance(service.policy.reconnect_seconds - 1)
    await service.maintenance()
    assert team_id in service.state.teams
    service.test_clock.advance(1)
    await service.maintenance()
    assert team_id not in service.state.teams
    service.test_clock.advance(service.policy.identity_seconds)
    await service.maintenance()
    bad = Peer(service, await service.connect())
    await bad.error(
        "hello", {"name": "Old", "resumeToken": captain.token}, "unauthorized"
    )


async def test_readiness_and_admission_timeouts_release_team(lobby):
    service, game = lobby
    team_id, (captain,) = await service.test_team(1, "Team")
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    service.test_clock.advance(service.policy.readiness_seconds)
    await service.maintenance()
    await service.wait_idle()
    assert service.state.matches[mid].reason == "readiness_timeout"
    assert service.state.teams[team_id].match_id is None
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    await captain.ok("match.ready", {"matchId": mid, "ready": True})
    await service.wait_idle()
    service.test_clock.advance(service.policy.admission_seconds)
    await service.maintenance()
    await service.wait_idle()
    assert service.state.matches[mid].reason == "admission_timeout"
    assert game.cancel_calls.count(mid) == 1


async def test_resume_resends_own_launch_and_disconnect_does_not_change_fixed_roster(
    lobby,
):
    service, game = lobby
    team_id, (captain, member) = await service.test_team(2, "Team")
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    for peer in (captain, member):
        await peer.ok("match.ready", {"matchId": mid, "ready": True})
    await service.wait_idle()
    await service.disconnect(member.conn)
    service.test_clock.advance(service.policy.reconnect_seconds + 1)
    # Extend admission deadline only to examine membership independently.
    service.state.matches[mid].expires_at = service.test_clock() + 100
    await service.maintenance()
    assert service.state.teams[team_id].member_ids == [
        captain.player_id,
        member.player_id,
    ]
    resumed = await service.test_peer("Ignored", member.token)
    launch = next(e["payload"] for e in resumed.pull() if e["type"] == "game.launch")
    assert launch["playerId"] == member.player_id and launch["panelIndex"] == 1
    assert not game.start_calls


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_results_fail_without_winner(lobby, loss):
    service, game = lobby
    tid, (captain,) = await service.test_team(1, "Team")
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    await game.emit(MatchStatus(mid, "finished", results=(TeamResult(tid, loss),)))
    match = service.state.matches[mid]
    assert (
        match.phase == "failed"
        and match.reason == "invalid_results"
        and match.results == ()
    )


async def test_missed_game_event_recovers_from_status_snapshot(lobby):
    service, game = lobby
    tid, (captain,) = await service.test_team(1, "Team")
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    game.statuses[mid] = MatchStatus(mid, "finished", results=(TeamResult(tid, 0.25),))
    await service.maintenance()
    assert service.state.matches[mid].phase == "finished"
    assert service.state.teams[tid].match_id is None


async def test_history_request_cache_and_slow_subscribers_are_bounded(lobby):
    service, _ = lobby
    service.policy = replace(
        service.policy, chat_history=3, request_history=4, max_outgoing=3
    )
    fast = await service.test_peer("Fast")
    slow = await service.test_peer("Slow")
    for i in range(10):
        await fast.ok("chat.send", {"text": f"Message {i}"})
    assert len(service.state.chat) == 3
    assert len(service.state.players[fast.player_id].requests) == 4
    assert slow.conn.closed and slow.conn.close_reason == "slow_consumer"
    assert len(service.state.chat) == 3


async def test_chat_rate_limit_and_html_remain_plain_data(lobby):
    service, _ = lobby
    service.policy = replace(service.policy, chat_burst=1)
    peer = await service.test_peer("<b>name</b>")
    await peer.ok("chat.send", {"text": "<script>alert(1)</script>"})
    await peer.error("chat.send", {"text": "Flood"}, "rate_limited")
    assert service.state.chat[0]["text"] == "<script>alert(1)</script>"
    service.test_clock.advance(1)
    await peer.ok("chat.send", {"text": "Allowed"})


async def test_failed_partial_preparation_releases_both_teams_and_cancels_game(lobby):
    service, game = lobby
    a_id, (a,) = await service.test_team(1, "A")
    b_id, b = await service.test_team(3, "B")
    prepare = game.prepare_match

    async def fail_after_partial_preparation(spec):
        await prepare(spec)
        raise RuntimeError("worker unavailable")

    game.prepare_match = fail_after_partial_preparation
    challenge = await a.ok("challenge.create", {"teamId": b_id})
    mid = (await b[0].ok("challenge.accept", challenge))["matchId"]
    await service.wait_idle()
    assert service.state.matches[mid].phase == "failed"
    assert service.state.matches[mid].reason == "game_unavailable"
    assert service.state.teams[a_id].match_id is None
    assert service.state.teams[b_id].match_id is None
    assert game.cancel_calls == [mid]
    assert game.statuses[mid].phase == "cancelled"


async def test_late_preparation_cannot_resurrect_cancelled_reservation(lobby):
    service, game = lobby
    tid, (captain,) = await service.test_team(1, "Team")
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = game.prepare_match

    async def delayed_preparation(spec):
        result = await prepare(spec)
        entered.set()
        await release.wait()
        return result

    game.prepare_match = delayed_preparation
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await entered.wait()
    await captain.ok("match.cancel", {"matchId": mid})
    assert service.state.teams[tid].match_id is None
    # A new reservation can start before the old prepare response arrives.
    second = (await captain.ok("team.start_speedrun"))["matchId"]
    assert second != mid
    release.set()
    await service.wait_idle()
    assert service.state.matches[mid].phase == "cancelled"
    assert service.state.matches[second].phase == "ready"
    assert service.state.teams[tid].match_id == second
    assert not service.state.matches[mid].admissions


async def test_preparation_deadline_releases_reservation_before_blocked_adapter_returns(
    lobby,
):
    service, game = lobby
    tid, (captain,) = await service.test_team(1, "Team")
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = game.prepare_match

    async def delayed_preparation(spec):
        result = await prepare(spec)
        entered.set()
        await release.wait()
        return result

    game.prepare_match = delayed_preparation
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await entered.wait()
    service.test_clock.advance(service.policy.preparation_seconds)
    await service.maintenance()
    assert service.state.matches[mid].reason == "preparation_timeout"
    assert service.state.teams[tid].match_id is None
    release.set()
    await service.wait_idle()
    assert service.state.matches[mid].phase == "cancelled"
    assert game.statuses[mid].phase == "cancelled"


async def test_invalid_prepared_credentials_fail_and_never_leak(lobby):
    service, game = lobby
    _, (captain, member) = await service.test_team(2, "Team")
    prepare = game.prepare_match

    async def shared_secret(spec):
        result = await prepare(spec)
        first = result.admissions[0]
        return replace(
            result,
            admissions=tuple(replace(a, token=first.token) for a in result.admissions),
        )

    game.prepare_match = shared_secret
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    assert service.state.matches[mid].phase == "failed"
    assert not any(
        e["type"] == "game.launch" for p in (captain, member) for e in p.pull()
    )


async def test_start_failure_rolls_back_and_lost_start_response_recovers(lobby):
    service, game = lobby
    _, (captain,) = await service.test_team(1, "Team")
    start = game.start_match

    async def failed_start(match_id):
        raise RuntimeError("cannot start")

    game.start_match = failed_start
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    await captain.ok("match.ready", {"matchId": mid, "ready": True})
    await game.connect_player(mid, captain.player_id)
    await service.wait_idle()
    assert service.state.matches[mid].phase == "failed"

    async def lost_response(match_id):
        await start(match_id)
        raise RuntimeError("lost response after successful start")

    game.start_match = lost_response
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    await captain.ok("match.ready", {"matchId": mid, "ready": True})
    await game.connect_player(mid, captain.player_id)
    await service.wait_idle()
    assert service.state.matches[mid].phase == "countdown"
    assert game.start_calls == [mid]


async def test_unready_and_cancel_permissions_are_authoritative(lobby):
    service, _ = lobby
    _, (captain, member) = await service.test_team(2, "Team")
    outsider = await service.test_peer("Outside")
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    await member.error("match.cancel", {"matchId": mid}, "not_captain")
    await outsider.error("match.ready", {"matchId": mid, "ready": True}, "unauthorized")
    await captain.ok("match.ready", {"matchId": mid, "ready": True})
    await captain.ok("match.ready", {"matchId": mid, "ready": False})
    await member.ok("match.ready", {"matchId": mid, "ready": True})
    assert not service.state.matches[mid].launched
    await captain.ok("match.ready", {"matchId": mid, "ready": True})
    await member.error("match.ready", {"matchId": mid, "ready": False}, "conflict")
    await captain.ok("match.cancel", {"matchId": mid})
    await captain.ok("match.cancel", {"matchId": mid})
    await service.wait_idle()


async def test_challenge_recipient_permissions_decline_cancel_and_expiry(lobby):
    service, _ = lobby
    _, (a,) = await service.test_team(1, "A")
    b_id, (b, member) = await service.test_team(2, "B")
    _, (outsider,) = await service.test_team(1, "Outside")
    challenge = await a.ok("challenge.create", {"teamId": b_id})
    await a.error("challenge.accept", challenge, "unauthorized")
    await member.error("challenge.accept", challenge, "not_captain")
    await outsider.error("challenge.decline", challenge, "unauthorized")
    await b.error("challenge.cancel", challenge, "unauthorized")
    await b.ok("challenge.decline", challenge)
    await b.ok("challenge.decline", challenge)
    await a.error("challenge.accept", challenge, "unauthorized")
    challenge = await a.ok("challenge.create", {"teamId": b_id})
    await a.ok("challenge.cancel", challenge)
    await b.error("challenge.accept", challenge, "conflict")
    challenge = await a.ok("challenge.create", {"teamId": b_id})
    service.test_clock.advance(service.policy.challenge_seconds)
    await b.error("challenge.accept", challenge, "expired")


async def test_terminal_match_history_bounded_teams_retained_across_runs(lobby):
    service, game = lobby
    service.policy = replace(service.policy, match_history=2)
    tid, (captain,) = await service.test_team(1, "Team")
    mids = []
    for i in range(5):
        mid = (await captain.ok("team.start_speedrun"))["matchId"]
        mids.append(mid)
        await service.wait_idle()
        await game.emit(
            MatchStatus(mid, "finished", results=(TeamResult(tid, i + 0.1),))
        )
    assert list(service.state.matches) == mids[-2:]
    assert service.state.teams[tid].captain_id == captain.player_id
    assert service.state.teams[tid].match_id is None
    assert len((await captain.snapshot())["matches"]) == 2


async def test_reconnect_reuses_identity_request_cache(lobby):
    service, _ = lobby
    peer = await service.test_peer("Player")
    answer = await peer.ok("chat.send", {"text": "Once"}, "global-message-id")
    resumed = await service.test_peer("Ignored", peer.token)
    replay = await resumed.ok("chat.send", {"text": "Once"}, "global-message-id")
    assert replay == answer and len(service.state.chat) == 1


async def test_idle_captain_departure_disbands_without_succession(lobby):
    service, _ = lobby
    tid, (captain, member) = await service.test_team(2, "Team")
    await captain.ok("team.leave")
    assert tid not in service.state.teams
    assert (await member.snapshot())["you"]["teamId"] is None


async def test_presence_disconnect_does_not_cancel_running_game(lobby):
    service, game = lobby
    tid, (captain,) = await service.test_team(1, "Team")
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    await service.wait_idle()
    await captain.ok("match.ready", {"matchId": mid, "ready": True})
    await game.connect_player(mid, captain.player_id)
    await service.wait_idle()
    await game.emit(replace(game.statuses[mid], phase="running"))
    await service.disconnect(captain.conn)
    service.test_clock.advance(service.policy.identity_seconds * 2)
    await service.maintenance()
    assert service.state.matches[mid].phase == "running"
    assert service.state.teams[tid].member_ids == [captain.player_id]
    assert game.cancel_calls == []


async def test_operation_timeout_cancels_partial_preparation(lobby):
    service, game = lobby
    service.policy = replace(service.policy, operation_seconds=0.01)
    tid, (captain,) = await service.test_team(1, "Team")
    prepare = game.prepare_match

    async def forever(spec):
        await prepare(spec)
        await asyncio.Event().wait()

    game.prepare_match = forever
    mid = (await captain.ok("team.start_speedrun"))["matchId"]
    async with asyncio.timeout(1):
        await service.wait_idle()
    assert service.state.matches[mid].phase == "failed"
    assert service.state.teams[tid].match_id is None
    assert game.cancel_calls == [mid]


async def test_default_maximum_snapshot_fits_browser_frame_and_collection_bounds(lobby):
    from hudes.lobby.models import Challenge, Player, Team

    service, _ = lobby
    policy = LobbyPolicy()
    name = "🌋" * policy.max_name_length
    player_ids = []
    team_ids = []
    for index in range(policy.max_players):
        pid, tid = f"player-{index:017}", f"team-{index:019}"
        player_ids.append(pid)
        team_ids.append(tid)
        service.state.players[pid] = Player(pid, name, f"token-{index}", team_id=tid)
        service.state.teams[tid] = Team(tid, name, pid, [pid])
    # Count each directed challenge against both endpoint teams' pending cap.
    for index in range(policy.max_players):
        for offset in range(1, policy.max_pending_per_player // 2 + 1):
            cid = f"challenge-{index:013}-{offset}"
            target = (index + offset) % policy.max_players
            service.state.challenges[cid] = Challenge(
                cid, team_ids[index], team_ids[target], 1, 1, 20000
            )
    for index in range(policy.social_history):
        cid = f"terminal-{index:015}"
        service.state.challenges[cid] = Challenge(
            cid, team_ids[0], team_ids[1], 1, 1, 0, "expired"
        )
    for index in range(policy.chat_history):
        service.state.chat.append(
            {
                "id": f"chat-{index:019}",
                "playerId": player_ids[0],
                "name": name,
                "text": "🌋" * policy.max_chat_length,
                "sentAt": 10000,
            }
        )
    snapshot = service._snapshot(player_ids[0])
    encoded = json.dumps(
        {"v": 1, "type": "snapshot", "seq": 1, "payload": snapshot}, ensure_ascii=False
    )
    assert len(encoded.encode("utf-8")) < 8 * 1024 * 1024
    assert len(snapshot["challenges"]) == 4352 < 8192
    assert (
        max(
            len(snapshot[key])
            for key in ("players", "teams", "invitations", "chat", "matches")
        )
        < 4096
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"batchSize": True},
        {"batchSize": 0},
        {"batchSize": 513},
        {"batchSize": 2.5},
        {"maxDimensions": 0},
        {"maxDimensions": 7},
        {"maxDimensions": 6007},
        {"maxDimensions": False},
        {"maxDimensions": "12"},
        {"duration": 2},
        None,
    ],
)
async def test_invalid_match_settings_do_not_reserve_a_match(lobby, settings):
    service, _ = lobby
    _, (captain,) = await service.test_team(1, "Settings")
    await captain.error(
        "team.start_speedrun", {"settings": settings}, "invalid_request"
    )
    assert not service.state.matches


async def test_settings_are_frozen_in_challenge_and_visible_before_acceptance(lobby):
    service, _ = lobby
    _, (left, member) = await service.test_team(2, "Left")
    right_id, (right,) = await service.test_team(1, "Right")
    settings = {"batchSize": 64, "maxDimensions": 18}
    await member.error(
        "challenge.create", {"teamId": right_id, "settings": settings}, "not_captain"
    )
    challenge = await left.ok(
        "challenge.create", {"teamId": right_id, "settings": settings}
    )
    assert (await right.snapshot())["challenges"][0]["settings"] == settings
    await left.error(
        "challenge.create",
        {"teamId": right_id, "settings": {"batchSize": 128}},
        "conflict",
    )
    accepted = await right.ok("challenge.accept", challenge)
    match = service.state.matches[accepted["matchId"]]
    assert match.spec.preset.batch_size == 64
    assert match.spec.preset.max_dimensions == 18
    assert len(match.spec.teams) == 2
    assert service.preset.batch_size == 32 and service.preset.max_dimensions is None
    for peer in (left, right, member):
        assert (await peer.snapshot())["matches"][0]["settings"] == settings


async def test_speedrun_custom_settings_and_default_preservation(lobby):
    service, _ = lobby
    _, (captain,) = await service.test_team(1, "Custom")
    result = await captain.ok(
        "team.start_speedrun", {"settings": {"batchSize": 17, "maxDimensions": 6}}
    )
    preset = service.state.matches[result["matchId"]].spec.preset
    assert preset.batch_size == 17 and preset.max_dimensions == 6
    _, (other,) = await service.test_team(1, "Default")
    result = await other.ok("team.start_speedrun")
    assert service.state.matches[result["matchId"]].spec.preset == service.preset
