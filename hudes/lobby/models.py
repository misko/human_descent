"""Private, tensor-free social state. Wire snapshots are built by the service."""

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from hudes.game_control import Admission, MatchPreset, MatchSpec


@dataclass(frozen=True)
class LobbyPolicy:
    max_players: int = 512
    max_connections: int = 1024
    max_frame_bytes: int = 16384
    max_outgoing: int = 128
    chat_history: int = 100
    match_history: int = 64
    request_history: int = 128
    social_history: int = 256
    max_pending_per_player: int = 16
    max_name_length: int = 48
    max_chat_length: int = 1000
    invitation_seconds: float = 120.0
    challenge_seconds: float = 120.0
    preparation_seconds: float = 30.0
    readiness_seconds: float = 120.0
    admission_seconds: float = 30.0
    operation_seconds: float = 15.0
    reconnect_seconds: float = 300.0
    identity_seconds: float = 86400.0
    maintenance_seconds: float = 1.0
    command_burst: int = 60
    commands_per_second: float = 10.0
    chat_burst: int = 5
    chats_per_second: float = 1.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(eq=False)
class LobbyConnection:
    id: str
    outgoing: asyncio.Queue
    player_id: str | None = None
    seq: int = 0
    closed: bool = False
    close_reason: str = "closed"
    command_tokens: float = 0
    command_at: float = 0
    requests: OrderedDict = field(default_factory=OrderedDict)


@dataclass
class Player:
    id: str
    name: str
    token: str
    team_id: str | None = None
    connection: LobbyConnection | None = None
    disconnected_at: float | None = None
    requests: OrderedDict = field(default_factory=OrderedDict)
    chat_tokens: float = 0
    chat_at: float = 0


@dataclass
class Team:
    id: str
    name: str
    captain_id: str
    member_ids: list[str]
    revision: int = 1
    match_id: str | None = None


@dataclass
class Invitation:
    id: str
    team_id: str
    inviter_id: str
    recipient_id: str
    expires_at: float
    status: str = "pending"


@dataclass
class Challenge:
    id: str
    challenger_team_id: str
    target_team_id: str
    challenger_revision: int
    target_revision: int
    expires_at: float
    status: str = "pending"
    match_id: str | None = None
    preset: MatchPreset = MatchPreset()


@dataclass
class LobbyMatch:
    id: str
    spec: MatchSpec
    expires_at: float
    phase: str = "preparing"
    ready: set[str] = field(default_factory=set)
    admissions: dict[str, Admission] = field(default_factory=dict)
    launched: bool = False
    start_requested: bool = False
    start_at: float | None = None
    deadline: float | None = None
    results: tuple = ()
    reason: str | None = None

    @property
    def player_ids(self):
        return {p.player_id for t in self.spec.teams for p in t.players}

    @property
    def team_ids(self):
        return tuple(t.team_id for t in self.spec.teams)


@dataclass
class State:
    players: dict[str, Player] = field(default_factory=dict)
    tokens: dict[str, str] = field(default_factory=dict)
    teams: dict[str, Team] = field(default_factory=dict)
    invitations: OrderedDict[str, Invitation] = field(default_factory=OrderedDict)
    challenges: OrderedDict[str, Challenge] = field(default_factory=OrderedDict)
    matches: OrderedDict[str, LobbyMatch] = field(default_factory=OrderedDict)
    chat: deque[dict[str, Any]] = field(default_factory=deque)
