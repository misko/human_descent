"""Authoritative in-memory social lobby and GameControl orchestration.

All social transitions happen under one short lock. External calls, including
GameControl notifications that can re-enter this service, never hold that lock.
Outgoing queues are bounded and are written without awaiting network I/O.
"""

import asyncio
import logging
import math
import secrets
import time

from hudes.game_control import MatchPreset, MatchSpec, PlayerSlot, TeamRoster
from hudes.match_settings import configure_preset, public_settings

from .models import (
    Challenge,
    Invitation,
    LobbyConnection,
    LobbyMatch,
    LobbyPolicy,
    Player,
    State,
    Team,
)
from .protocol import LobbyError, fingerprint, response, validate_request

log = logging.getLogger(__name__)
TERMINAL = frozenset({"finished", "cancelled", "failed"})
GAME_PHASES = {
    "prepared": 0,
    "countdown": 1,
    "running": 2,
    "finalizing": 3,
    "finished": 4,
}


class LobbyService:
    def __init__(self, game_control, *, clock=time.time, preset=None, policy=None):
        self.game_control = game_control
        self.clock = clock
        self.preset = preset or MatchPreset()
        self.policy = policy or LobbyPolicy()
        self.state = State()
        self._lock = asyncio.Lock()
        self._connections = {}
        self._tasks = set()
        self._effects = []
        self._launches = []
        self._polling = set()
        self._unsubscribe = None
        self._maintenance_task = None
        self._closed = False

    async def start(self):
        """Subscribe and start maintenance. Safe to call repeatedly."""
        if self._closed:
            raise RuntimeError("Lobby is closed")
        if self._unsubscribe is None:
            self._unsubscribe = self.game_control.subscribe(self.on_game_status)
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        return self

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        async with self._lock:
            live_ids = [
                m.id for m in self.state.matches.values() if m.phase not in TERMINAL
            ]
            for match_id in live_ids:
                self._terminate(
                    self.state.matches[match_id], "cancelled", "lobby_shutdown"
                )
            self._effects.clear()
            for conn in self._connections.values():
                self._close_connection(conn, "lobby_shutdown")
        tasks = list(self._tasks)
        if self._maintenance_task is not None:
            tasks.append(self._maintenance_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Cancel only matches this service owns, outside its lock.
        await asyncio.gather(
            *(self._cancel_game(mid, "lobby_shutdown") for mid in live_ids)
        )

    async def connect(self):
        async with self._lock:
            if self._closed or len(self._connections) >= self.policy.max_connections:
                raise LobbyError("rate_limited", "Lobby connection limit reached")
            conn = LobbyConnection(
                self._id(),
                asyncio.Queue(self.policy.max_outgoing),
                command_tokens=self.policy.command_burst,
                command_at=self.clock(),
            )
            self._connections[conn.id] = conn
            return conn

    async def disconnect(self, conn):
        async with self._lock:
            self._connections.pop(conn.id, None)
            conn.closed = True
            player = self.state.players.get(conn.player_id)
            if player and player.connection is conn:
                player.connection = None
                player.disconnected_at = self.clock()
                self._broadcast()

    async def handle(self, conn, request):
        """Execute a decoded request and enqueue its response before new snapshots.

        The returned dict is also queued; transports must not send it a second
        time. Match preparation is asynchronous: acceptance returns its reserved
        matchId and later snapshots describe ready/failure transitions.
        """
        try:
            command, request_id, payload = validate_request(request)
            signature = fingerprint(request)
        except LobbyError as exc:
            rid = request.get("requestId") if isinstance(request, dict) else None
            if not isinstance(rid, str) or len(rid) > 80:
                rid = None
            result = response(rid, error=exc)
            self._enqueue(conn, result)
            return result
        async with self._lock:
            if conn.closed or conn.id not in self._connections:
                return response(
                    request_id, error=LobbyError("unauthorized", "Connection is closed")
                )
            player = self.state.players.get(conn.player_id)
            cache = player.requests if player and command != "hello" else conn.requests
            previous = cache.get(request_id)
            if previous is not None:
                old_signature, result = previous
                if signature != old_signature:
                    result = response(
                        request_id,
                        error=LobbyError(
                            "conflict",
                            "requestId was already used with different content",
                        ),
                    )
                self._enqueue(conn, result)
                return result
            changed = self._expire()
            try:
                self._command_rate(conn)
                if command == "hello":
                    result_payload = self._hello(conn, payload)
                    changed = True
                else:
                    if player is None or player.connection is not conn:
                        raise LobbyError("unauthorized", "Send hello first")
                    result_payload, mutation = self._dispatch(player, command, payload)
                    changed |= mutation
                result = response(request_id, result_payload)
            except LobbyError as exc:
                result = response(request_id, error=exc)
            cache[request_id] = (signature, result)
            while len(cache) > self.policy.request_history:
                cache.popitem(last=False)
            self._enqueue(conn, result)
            if changed:
                self._broadcast()
            elif command == "snapshot" and result["ok"]:
                self._snapshot_event(conn)
            self._flush_launches()
        self._run_effects()
        return result

    async def snapshot(self, conn):
        async with self._lock:
            if conn.player_id is None:
                raise LobbyError("unauthorized", "Send hello first")
            return self._snapshot(conn.player_id)

    async def maintenance(self):
        """Expire social state and recover missed game events; injectable-clock safe."""
        async with self._lock:
            changed = self._expire()
            match_ids = [
                m.id
                for m in self.state.matches.values()
                if m.phase not in TERMINAL
                and m.phase != "preparing"
                and m.id not in self._polling
            ]
            self._polling.update(match_ids)
            if changed:
                self._broadcast()
        self._run_effects()
        await asyncio.gather(*(self._poll_status(mid) for mid in match_ids))

    async def _maintenance_loop(self):
        try:
            while True:
                await asyncio.sleep(self.policy.maintenance_seconds)
                try:
                    await self.maintenance()
                except Exception:
                    log.exception("Lobby maintenance failed")
        except asyncio.CancelledError:
            pass

    def _hello(self, conn, payload):
        if conn.player_id is not None:
            raise LobbyError("conflict", "Connection already has an identity")
        name = self._text(payload["name"], self.policy.max_name_length, "name")
        token = payload.get("resumeToken")
        if token is not None:
            player_id = self.state.tokens.get(token)
            player = self.state.players.get(player_id)
            if player is None:
                raise LobbyError("unauthorized", "Invalid or expired resume token")
            if player.connection is not None:
                self._close_connection(player.connection, "replaced")
            # Resume restores the authoritative display name and captain role.
        else:
            if len(self.state.players) >= self.policy.max_players:
                raise LobbyError("rate_limited", "Lobby identity limit reached")
            player = Player(
                self._id(),
                name,
                secrets.token_urlsafe(32),
                chat_tokens=self.policy.chat_burst,
                chat_at=self.clock(),
            )
            self.state.players[player.id] = player
            self.state.tokens[player.token] = player.id
        conn.player_id = player.id
        player.connection = conn
        player.disconnected_at = None
        for match in self.state.matches.values():
            if (
                match.launched
                and match.phase not in TERMINAL
                and player.id in match.player_ids
            ):
                self._launches.append((match.id, player.id))
        return {"playerId": player.id, "resumeToken": player.token}

    def _dispatch(self, player, command, payload):
        state = self.state
        if command == "snapshot":
            return {}, False
        if command == "chat.send":
            text = self._text(
                payload["text"], self.policy.max_chat_length, "text", multiline=True
            )
            now = self.clock()
            player.chat_tokens = min(
                self.policy.chat_burst,
                player.chat_tokens
                + max(0, now - player.chat_at) * self.policy.chats_per_second,
            )
            player.chat_at = now
            if player.chat_tokens < 1:
                raise LobbyError("rate_limited", "Chat rate limit reached")
            player.chat_tokens -= 1
            entry = {
                "id": self._id(),
                "playerId": player.id,
                "name": player.name,
                "text": text,
                "sentAt": now,
            }
            state.chat.append(entry)
            while len(state.chat) > self.policy.chat_history:
                state.chat.popleft()
            return {"messageId": entry["id"]}, True
        if command == "team.create":
            if player.team_id:
                raise LobbyError("already_in_team", "Leave your team first")
            team = Team(
                self._id(),
                self._text(payload["name"], self.policy.max_name_length, "name"),
                player.id,
                [player.id],
            )
            state.teams[team.id] = team
            player.team_id = team.id
            self._cancel_player_invitations(player.id)
            return {"teamId": team.id}, True
        if command == "team.leave":
            team = self._own_team(player)
            self._available(team)
            self._remove_member(team, player.id)
            return {}, True
        if command == "team.rename":
            team = self._captain(player)
            self._available(team)
            team.name = self._text(payload["name"], self.policy.max_name_length, "name")
            team.revision += 1
            self._invalidate_challenges(team.id)
            return {"teamId": team.id}, True
        if command == "team.invite":
            team = self._captain(player)
            self._available(team)
            if len(team.member_ids) >= 3:
                raise LobbyError("team_full", "Team already has three members")
            target = state.players.get(payload["playerId"])
            if target is None:
                raise LobbyError("not_found", "Player does not exist")
            if target.team_id:
                raise LobbyError("already_in_team", "Player already has a team")
            for invitation in state.invitations.values():
                if (
                    invitation.team_id == team.id
                    and invitation.recipient_id == target.id
                    and invitation.status == "pending"
                ):
                    return {"inviteId": invitation.id}, False
            pending = [i for i in state.invitations.values() if i.status == "pending"]
            if (
                sum(i.team_id == team.id for i in pending)
                >= self.policy.max_pending_per_player
                or sum(i.recipient_id == target.id for i in pending)
                >= self.policy.max_pending_per_player
            ):
                raise LobbyError("rate_limited", "Too many pending invitations")
            invitation = Invitation(
                self._id(),
                team.id,
                player.id,
                target.id,
                self.clock() + self.policy.invitation_seconds,
            )
            state.invitations[invitation.id] = invitation
            self._prune_social(state.invitations)
            return {"inviteId": invitation.id}, True
        if command in {"invite.accept", "invite.decline"}:
            invitation = state.invitations.get(payload["inviteId"])
            if invitation is None:
                raise LobbyError("not_found", "Invitation does not exist")
            if invitation.recipient_id != player.id:
                raise LobbyError("unauthorized", "Invitation belongs to another player")
            desired = "accepted" if command == "invite.accept" else "declined"
            if invitation.status == desired:
                return {"inviteId": invitation.id}, False
            self._pending(invitation)
            if command == "invite.decline":
                invitation.status = "declined"
            else:
                if player.team_id:
                    raise LobbyError("already_in_team", "Leave your team first")
                team = state.teams.get(invitation.team_id)
                if team is None:
                    raise LobbyError("not_found", "Team no longer exists")
                self._available(team)
                if len(team.member_ids) >= 3:
                    raise LobbyError("team_full", "Team already has three members")
                invitation.status = "accepted"
                team.member_ids.append(player.id)
                team.revision += 1
                player.team_id = team.id
                self._invalidate_challenges(team.id)
                self._cancel_player_invitations(player.id)
            return {"inviteId": invitation.id}, True
        if command == "team.start_speedrun":
            team = self._captain(player)
            self._available(team)
            match = self._reserve(
                (team,), configure_preset(self.preset, payload.get("settings", {}))
            )
            return {"matchId": match.id}, True
        if command == "challenge.create":
            team = self._captain(player)
            preset = configure_preset(self.preset, payload.get("settings", {}))
            target = state.teams.get(payload["teamId"])
            if target is None:
                raise LobbyError("not_found", "Team does not exist")
            if target.id == team.id:
                raise LobbyError("conflict", "Choose another team")
            self._available(team)
            self._available(target)
            pending = [c for c in state.challenges.values() if c.status == "pending"]
            for challenge in pending:
                if (challenge.challenger_team_id, challenge.target_team_id) == (
                    team.id,
                    target.id,
                ):
                    if challenge.preset != preset:
                        raise LobbyError(
                            "conflict",
                            "Cancel the pending challenge before changing its settings",
                        )
                    return {"challengeId": challenge.id}, False
            if (
                sum(
                    team.id in (c.challenger_team_id, c.target_team_id) for c in pending
                )
                >= self.policy.max_pending_per_player
                or sum(
                    target.id in (c.challenger_team_id, c.target_team_id)
                    for c in pending
                )
                >= self.policy.max_pending_per_player
            ):
                raise LobbyError("rate_limited", "Too many pending challenges")
            challenge = Challenge(
                self._id(),
                team.id,
                target.id,
                team.revision,
                target.revision,
                self.clock() + self.policy.challenge_seconds,
                preset=preset,
            )
            state.challenges[challenge.id] = challenge
            self._prune_social(state.challenges)
            return {"challengeId": challenge.id}, True
        if command in {"challenge.accept", "challenge.decline", "challenge.cancel"}:
            challenge = state.challenges.get(payload["challengeId"])
            if challenge is None:
                raise LobbyError("not_found", "Challenge does not exist")
            team = self._captain(player)
            expected = (
                challenge.challenger_team_id
                if command == "challenge.cancel"
                else challenge.target_team_id
            )
            if team.id != expected:
                raise LobbyError("unauthorized", "Challenge belongs to another captain")
            desired = {
                "challenge.accept": "accepted",
                "challenge.decline": "declined",
                "challenge.cancel": "cancelled",
            }[command]
            if challenge.status == desired:
                answer = {"challengeId": challenge.id}
                if challenge.match_id:
                    answer["matchId"] = challenge.match_id
                return answer, False
            self._pending(challenge)
            if command != "challenge.accept":
                challenge.status = desired
                return {"challengeId": challenge.id}, True
            challenger = state.teams.get(challenge.challenger_team_id)
            if challenger is None or (challenger.revision, team.revision) != (
                challenge.challenger_revision,
                challenge.target_revision,
            ):
                challenge.status = "cancelled"
                raise LobbyError("conflict", "Team roster changed")
            self._available(challenger)
            self._available(team)
            challenge.status = "accepted"
            match = self._reserve((challenger, team), challenge.preset)
            challenge.match_id = match.id
            return {"challengeId": challenge.id, "matchId": match.id}, True
        if command in {"match.ready", "match.cancel"}:
            match = state.matches.get(payload["matchId"])
            if match is None:
                raise LobbyError("not_found", "Match does not exist")
            if player.id not in match.player_ids:
                raise LobbyError("unauthorized", "You are not in this match")
            if command == "match.cancel":
                self._captain(player)
                if match.phase in {"cancelled", "failed"}:
                    return {"matchId": match.id}, False
                if match.phase not in {"preparing", "ready"}:
                    raise LobbyError("team_busy", "Match has already started")
                self._terminate(match, "cancelled", "captain_cancelled")
                self._effects.append(("cancel", match.id, "captain_cancelled"))
                return {"matchId": match.id}, True
            if match.phase != "ready":
                raise LobbyError("not_ready", "Match is not waiting for readiness")
            if match.launched and not payload["ready"]:
                raise LobbyError("conflict", "Gameplay admission has begun")
            if payload["ready"]:
                match.ready.add(player.id)
            else:
                match.ready.discard(player.id)
            if match.ready == match.player_ids and not match.launched:
                match.launched = True
                match.expires_at = self.clock() + self.policy.admission_seconds
                self._launches.extend((match.id, pid) for pid in match.player_ids)
                self._effects.append(("poll", match.id, None))
            return {"matchId": match.id}, True
        raise LobbyError("invalid_request", "Unknown command")

    def _reserve(self, teams, preset):
        rosters = tuple(
            TeamRoster(
                t.id,
                t.name,
                tuple(
                    PlayerSlot(pid, self.state.players[pid].name, index)
                    for index, pid in enumerate(t.member_ids)
                ),
            )
            for t in teams
        )
        mid = self._id()
        match = LobbyMatch(
            mid,
            MatchSpec(mid, rosters, preset),
            self.clock() + self.policy.preparation_seconds,
        )
        self.state.matches[mid] = match
        for team in teams:
            team.match_id = mid
            self._invalidate_challenges(team.id)
            for invitation in self.state.invitations.values():
                if invitation.team_id == team.id and invitation.status == "pending":
                    invitation.status = "cancelled"
        self._effects.append(("prepare", mid, None))
        self._prune_matches()
        return match

    def _own_team(self, player):
        team = self.state.teams.get(player.team_id)
        if team is None:
            raise LobbyError("not_found", "You do not have a team")
        return team

    def _captain(self, player):
        team = self._own_team(player)
        if team.captain_id != player.id:
            raise LobbyError("not_captain", "Only the captain can do that")
        return team

    @staticmethod
    def _available(team):
        if team.match_id is not None:
            raise LobbyError("team_busy", "Team is reserved for a match")

    @staticmethod
    def _pending(item):
        if item.status == "expired":
            raise LobbyError("expired", "Offer has expired")
        if item.status != "pending":
            raise LobbyError("conflict", "Offer is no longer pending")

    def _invalidate_challenges(self, team_id):
        for challenge in self.state.challenges.values():
            if challenge.status == "pending" and team_id in (
                challenge.challenger_team_id,
                challenge.target_team_id,
            ):
                challenge.status = "cancelled"

    def _cancel_player_invitations(self, player_id):
        for invitation in self.state.invitations.values():
            if invitation.recipient_id == player_id and invitation.status == "pending":
                invitation.status = "cancelled"

    def _remove_member(self, team, player_id):
        self._invalidate_challenges(team.id)
        if team.captain_id == player_id:
            for member_id in team.member_ids:
                self.state.players[member_id].team_id = None
            del self.state.teams[team.id]
            for invitation in self.state.invitations.values():
                if invitation.team_id == team.id and invitation.status == "pending":
                    invitation.status = "cancelled"
        else:
            team.member_ids.remove(player_id)
            self.state.players[player_id].team_id = None
            team.revision += 1

    def _expire(self):
        now = self.clock()
        changed = False
        for items in (self.state.invitations, self.state.challenges):
            for item in items.values():
                if item.status == "pending" and item.expires_at <= now:
                    item.status = "expired"
                    changed = True
            self._prune_social(items)
        for match in list(self.state.matches.values()):
            if match.phase in {"preparing", "ready"} and match.expires_at <= now:
                self._terminate(
                    match,
                    "cancelled",
                    "preparation_timeout"
                    if match.phase == "preparing"
                    else "admission_timeout"
                    if match.launched
                    else "readiness_timeout",
                )
                self._effects.append(("cancel", match.id, match.reason))
                changed = True
        for player in list(self.state.players.values()):
            if player.connection is not None or player.disconnected_at is None:
                continue
            age = now - player.disconnected_at
            team = self.state.teams.get(player.team_id)
            if age >= self.policy.reconnect_seconds and team and team.match_id is None:
                self._remove_member(team, player.id)
                changed = True
            if age >= self.policy.identity_seconds and player.team_id is None:
                self._cancel_player_invitations(player.id)
                self.state.tokens.pop(player.token, None)
                del self.state.players[player.id]
                changed = True
        self._prune_matches()
        return changed

    def _terminate(self, match, phase, reason=None):
        if match.phase in TERMINAL:
            return False
        match.phase = phase
        match.reason = reason
        match.admissions.clear()
        for team_id in match.team_ids:
            team = self.state.teams.get(team_id)
            if team and team.match_id == match.id:
                team.match_id = None
        self._prune_matches()
        return True

    def _prune_social(self, items):
        # Pending offers are separately limited per identity. Only terminal
        # history is discarded, so pruning cannot revive or strand an offer.
        terminal = [key for key, item in items.items() if item.status != "pending"]
        for key in terminal[: max(0, len(terminal) - self.policy.social_history)]:
            del items[key]

    def _prune_matches(self):
        terminal = [
            key for key, match in self.state.matches.items() if match.phase in TERMINAL
        ]
        for key in terminal[: max(0, len(terminal) - self.policy.match_history)]:
            del self.state.matches[key]

    def _command_rate(self, conn):
        now = self.clock()
        conn.command_tokens = min(
            self.policy.command_burst,
            conn.command_tokens
            + max(0, now - conn.command_at) * self.policy.commands_per_second,
        )
        conn.command_at = now
        if conn.command_tokens < 1:
            raise LobbyError("rate_limited", "Command rate limit reached")
        conn.command_tokens -= 1

    @staticmethod
    def _id():
        return secrets.token_urlsafe(18)

    @staticmethod
    def _text(value, limit, field, multiline=False):
        value = value.strip()
        if (
            not value
            or len(value) > limit
            or any(
                (ord(c) < 32 and not (multiline and c in "\n\t")) or ord(c) == 127
                for c in value
            )
        ):
            raise LobbyError("invalid_request", f"Invalid {field}")
        return value

    def _close_connection(self, conn, reason):
        conn.closed = True
        conn.close_reason = reason
        # Wake a writer blocked on an empty queue without unbounded buffers.
        if conn.outgoing.empty():
            conn.outgoing.put_nowait(None)

    def _enqueue(self, conn, event):
        if conn.closed:
            return
        try:
            conn.outgoing.put_nowait(event)
        except asyncio.QueueFull:
            self._close_connection(conn, "slow_consumer")
            player = self.state.players.get(conn.player_id)
            if player and player.connection is conn:
                player.connection = None
                player.disconnected_at = self.clock()

    def _event(self, conn, kind, payload):
        conn.seq += 1
        self._enqueue(conn, {"v": 1, "type": kind, "seq": conn.seq, "payload": payload})

    def _snapshot_event(self, conn):
        if conn.player_id in self.state.players and not conn.closed:
            self._event(conn, "snapshot", self._snapshot(conn.player_id))

    def _broadcast(self):
        for conn in list(self._connections.values()):
            self._snapshot_event(conn)

    def _snapshot(self, player_id):
        state = self.state
        you = state.players[player_id]
        invitations = [
            i
            for i in state.invitations.values()
            if player_id in (i.inviter_id, i.recipient_id)
        ]
        matches = []
        for match in state.matches.values():
            # The player's own fixed roster alone receives readiness details.
            if player_id not in match.player_ids:
                continue
            matches.append(
                {
                    "id": match.id,
                    "teamIds": list(match.team_ids),
                    "phase": match.phase,
                    "readyPlayerIds": sorted(match.ready),
                    "settings": public_settings(match.spec.preset),
                    "startAt": match.start_at,
                    "deadline": match.deadline,
                    "results": [
                        {"teamId": r.team_id, "loss": r.loss} for r in match.results
                    ],
                    "reason": match.reason,
                }
            )
        return {
            "you": {"id": you.id, "name": you.name, "teamId": you.team_id},
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "connected": p.connection is not None and not p.connection.closed,
                    "teamId": p.team_id,
                }
                for p in state.players.values()
            ],
            "teams": [
                {
                    "id": t.id,
                    "name": t.name,
                    "captainId": t.captain_id,
                    "memberIds": list(t.member_ids),
                    "revision": t.revision,
                    "status": "busy" if t.match_id else "available",
                }
                for t in state.teams.values()
            ],
            "invitations": [
                {
                    "id": i.id,
                    "teamId": i.team_id,
                    "inviterId": i.inviter_id,
                    "recipientId": i.recipient_id,
                    "status": i.status,
                    "expiresAt": i.expires_at,
                }
                for i in invitations
            ],
            "challenges": [
                {
                    "id": c.id,
                    "challengerTeamId": c.challenger_team_id,
                    "settings": public_settings(c.preset),
                    "targetTeamId": c.target_team_id,
                    "status": c.status,
                    "expiresAt": c.expires_at,
                }
                for c in state.challenges.values()
            ],
            "matches": matches,
            "chat": list(state.chat),
        }

    def _flush_launches(self):
        launches, self._launches = self._launches, []
        for mid, pid in launches:
            match = self.state.matches.get(mid)
            player = self.state.players.get(pid)
            admission = match.admissions.get(pid) if match else None
            if (
                match is None
                or match.phase in TERMINAL
                or admission is None
                or player is None
                or player.connection is None
            ):
                continue
            self._event(
                player.connection,
                "game.launch",
                {
                    "matchId": mid,
                    "teamId": admission.team_id,
                    "playerId": pid,
                    "sessionId": admission.session_id,
                    "panelIndex": admission.panel_index,
                    "token": admission.token,
                    "wsUrl": "/ws",
                },
            )

    def _run_effects(self):
        effects, self._effects = self._effects, []
        for operation, match_id, reason in effects:
            if self._closed:
                continue
            if operation == "prepare":
                coroutine = self._prepare(match_id)
            elif operation == "cancel":
                coroutine = self._cancel_game(match_id, reason)
            elif operation == "start":
                coroutine = self._start_match(match_id)
            else:
                coroutine = self._poll_status(match_id)
            task = asyncio.create_task(coroutine)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def wait_idle(self):
        """Drain currently scheduled control operations (useful for embedders/tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks))

    async def _prepare(self, match_id):
        match = self.state.matches.get(match_id)
        if match is None or match.phase != "preparing":
            return
        try:
            prepared = await asyncio.wait_for(
                self.game_control.prepare_match(match.spec),
                self.policy.operation_seconds,
            )
            async with self._lock:
                current = self.state.matches.get(match_id)
                if current is None or current.phase != "preparing":
                    self._effects.append(("cancel", match_id, "obsolete_preparation"))
                else:
                    expected = {
                        p.player_id: (t.team_id, p.panel_index)
                        for t in match.spec.teams
                        for p in t.players
                    }
                    actual = {
                        a.player_id: (a.team_id, a.panel_index)
                        for a in prepared.admissions
                    }
                    sessions = {a.team_id: a.session_id for a in prepared.admissions}
                    valid = (
                        prepared.match_id == match_id
                        and actual == expected
                        and len(prepared.admissions) == len(expected)
                        and all(
                            a.session_id
                            and a.token
                            and sessions[a.team_id] == a.session_id
                            for a in prepared.admissions
                        )
                        and len(set(sessions.values())) == len(match.team_ids)
                        and len({a.token for a in prepared.admissions}) == len(expected)
                    )
                    if not valid:
                        raise ValueError("Invalid game preparation credentials")
                    match.admissions = {a.player_id: a for a in prepared.admissions}
                    match.phase = "ready"
                    match.expires_at = self.clock() + self.policy.readiness_seconds
                    self._broadcast()
            self._run_effects()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("Game preparation failed for %s", match_id, exc_info=True)
            await self._fail(match_id, "game_unavailable")

    async def _start_match(self, match_id):
        async with self._lock:
            match = self.state.matches.get(match_id)
            if match is None or match.phase in TERMINAL or not match.start_requested:
                return
        try:
            status = await asyncio.wait_for(
                self.game_control.start_match(match_id), self.policy.operation_seconds
            )
            await self.on_game_status(status)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("Game start failed for %s", match_id, exc_info=True)
            # An operation can succeed before its response is lost. Recover the
            # authoritative clock instead of cancelling an already running game.
            try:
                status = await asyncio.wait_for(
                    self.game_control.get_match_status(match_id),
                    self.policy.operation_seconds,
                )
                if status.phase in {
                    "countdown",
                    "running",
                    "finalizing",
                    "finished",
                    "failed",
                    "cancelled",
                }:
                    await self.on_game_status(status)
                    return
            except Exception:
                pass
            await self._fail(match_id, "game_unavailable")

    async def _cancel_game(self, match_id, reason):
        try:
            await asyncio.wait_for(
                self.game_control.cancel_match(match_id, reason or "cancelled"),
                self.policy.operation_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("Game cancellation failed for %s", match_id, exc_info=True)

    async def _poll_status(self, match_id):
        try:
            status = await asyncio.wait_for(
                self.game_control.get_match_status(match_id),
                self.policy.operation_seconds,
            )
            await self.on_game_status(status)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A temporary status failure does not stop a game. Readiness and
            # admission still have bounded deadlines maintained independently.
            log.debug("Game status unavailable for %s", match_id, exc_info=True)
        finally:
            self._polling.discard(match_id)

    async def _fail(self, match_id, reason):
        async with self._lock:
            match = self.state.matches.get(match_id)
            if match and self._terminate(match, "failed", reason):
                self._effects.append(("cancel", match_id, reason))
                self._broadcast()
        self._run_effects()

    async def on_game_status(self, status):
        """GameControl listener; safe even when called from an awaited operation."""
        if self._closed:
            return
        async with self._lock:
            match = self.state.matches.get(status.match_id)
            if match is None or match.phase in TERMINAL:
                return
            phase = status.phase
            if phase not in GAME_PHASES and phase not in TERMINAL:
                return
            current_rank = GAME_PHASES.get(match.phase, 0)
            if phase in GAME_PHASES and GAME_PHASES[phase] < current_rank:
                return
            changed = False
            if phase == "finished":
                match.start_at, match.deadline = status.start_at, status.deadline
                results = status.results
                if (
                    len(results) != len(match.team_ids)
                    or {r.team_id for r in results} != set(match.team_ids)
                    or any(not math.isfinite(r.loss) for r in results)
                ):
                    self._terminate(match, "failed", "invalid_results")
                else:
                    match.results = tuple(results)
                    self._terminate(match, "finished", status.reason)
                changed = True
            elif phase in {"cancelled", "failed"}:
                self._terminate(match, phase, status.reason)
                changed = True
            elif phase != "prepared":
                changed = (match.phase, match.start_at, match.deadline) != (
                    phase,
                    status.start_at,
                    status.deadline,
                )
                match.phase, match.start_at, match.deadline = (
                    phase,
                    status.start_at,
                    status.deadline,
                )
            if (
                phase == "prepared"
                and match.phase == "ready"
                and match.launched
                and not match.start_requested
                and match.player_ids <= set(status.connected_player_ids)
            ):
                match.start_requested = True
                self._effects.append(("start", match.id, None))
            if changed:
                self._broadcast()
        self._run_effects()
