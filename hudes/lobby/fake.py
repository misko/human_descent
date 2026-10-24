"""Deterministic, Torch-free GameControl for lobby tests and standalone demos.

This is explicitly a fake: it performs no numerical gameplay or scoring. A
composed application injects its real GameControl instead.
"""

import secrets
import time
from dataclasses import replace

from hudes.game_control import Admission, MatchStatus, PreparedMatch


class FakeGameControl:
    def __init__(self, *, clock=time.time, auto_connect=False):
        self.clock = clock
        self.auto_connect = auto_connect
        self.specs = {}
        self.prepared = {}
        self.statuses = {}
        self.listeners = set()
        self.prepare_calls = []
        self.start_calls = []
        self.cancel_calls = []

    async def prepare_match(self, spec):
        self.prepare_calls.append(spec.match_id)
        if spec.match_id in self.specs:
            if self.specs[spec.match_id] != spec:
                raise ValueError("Conflicting match specification")
            return self.prepared[spec.match_id]
        if spec.match_id in self.statuses:
            raise ValueError("Match was cancelled before preparation")
        self.specs[spec.match_id] = spec
        admissions = []
        for team in spec.teams:
            session_id = secrets.token_urlsafe(18)
            admissions.extend(
                Admission(
                    player.player_id,
                    team.team_id,
                    session_id,
                    secrets.token_urlsafe(32),
                    player.panel_index,
                )
                for player in team.players
            )
        prepared = PreparedMatch(spec.match_id, tuple(admissions))
        self.prepared[spec.match_id] = prepared
        connected = tuple(a.player_id for a in admissions) if self.auto_connect else ()
        await self.emit(MatchStatus(spec.match_id, "prepared", connected))
        return prepared

    async def start_match(self, match_id):
        self.start_calls.append(match_id)
        status = self.statuses[match_id]
        if status.phase != "prepared":
            return status
        spec = self.specs[match_id]
        players = {p.player_id for t in spec.teams for p in t.players}
        if not players <= set(status.connected_player_ids):
            raise ValueError("Not every player is connected")
        start_at = self.clock() + spec.preset.countdown_seconds
        status = replace(
            status,
            phase="countdown",
            start_at=start_at,
            deadline=start_at + spec.preset.duration_seconds,
        )
        await self.emit(status)
        return status

    async def cancel_match(self, match_id, reason="cancelled"):
        self.cancel_calls.append(match_id)
        status = self.statuses.get(match_id)
        if status is not None and status.phase in {"finished", "cancelled", "failed"}:
            return status
        status = MatchStatus(match_id, "cancelled", reason=reason)
        await self.emit(status)
        return status

    async def get_match_status(self, match_id):
        return self.statuses[match_id]

    def subscribe(self, listener):
        self.listeners.add(listener)

        def unsubscribe():
            self.listeners.discard(listener)

        return unsubscribe

    async def emit(self, status):
        self.statuses[status.match_id] = status
        for listener in tuple(self.listeners):
            await listener(status)

    async def connect_player(self, match_id, player_id):
        status = self.statuses[match_id]
        players = {a.player_id for a in self.prepared[match_id].admissions}
        if player_id not in players:
            raise ValueError("Unknown player")
        await self.emit(
            replace(
                status,
                connected_player_ids=tuple(
                    sorted(set(status.connected_player_ids) | {player_id})
                ),
            )
        )

    async def disconnect_player(self, match_id, player_id):
        status = self.statuses[match_id]
        await self.emit(
            replace(
                status,
                connected_player_ids=tuple(
                    p for p in status.connected_player_ids if p != player_id
                ),
            )
        )
