"""Torch-free contract between the social lobby and gameplay.

Only immutable values cross this boundary. Gameplay owns timing and scoring;
the lobby owns identities, invitations and social readiness.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PlayerSlot:
    player_id: str
    name: str
    panel_index: int


@dataclass(frozen=True)
class TeamRoster:
    team_id: str
    name: str
    players: tuple[PlayerSlot, ...]


@dataclass(frozen=True)
class MatchPreset:
    preset_id: str = "standard"
    duration_seconds: float = 120.0
    countdown_seconds: float = 3.0
    batch_size: int = 32
    mesh_grid_size: int = 31
    # Same initial sampling scale as solo ClientState(-0.05, 12).
    mesh_step_size: float = 10 ** (-0.05 * 12)
    dtype: str = "float32"
    seed: int = 0
    scoring: str = "final_validation_loss_v2"
    max_dimensions: int | None = None


@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    teams: tuple[TeamRoster, ...]
    preset: MatchPreset = MatchPreset()


@dataclass(frozen=True)
class Admission:
    player_id: str
    team_id: str
    session_id: str
    token: str
    panel_index: int


@dataclass(frozen=True)
class PreparedMatch:
    match_id: str
    admissions: tuple[Admission, ...]


@dataclass(frozen=True)
class TeamResult:
    team_id: str
    loss: float


@dataclass(frozen=True)
class MatchStatus:
    match_id: str
    phase: str  # prepared, countdown, running, finalizing, finished, cancelled, failed
    connected_player_ids: tuple[str, ...] = ()
    start_at: float | None = None  # Unix seconds, display only
    deadline: float | None = None  # Unix seconds, display only
    results: tuple[TeamResult, ...] = ()
    reason: str | None = None


MatchListener = Callable[[MatchStatus], Awaitable[None]]


class GameControl(Protocol):
    async def prepare_match(self, spec: MatchSpec) -> PreparedMatch:
        """Idempotent by match_id; a different spec for an existing ID is invalid."""
        ...

    async def start_match(self, match_id: str) -> MatchStatus:
        """Start once after every roster member is admitted; game owns the clock."""
        ...

    async def cancel_match(
        self, match_id: str, reason: str = "cancelled"
    ) -> MatchStatus: ...

    async def get_match_status(self, match_id: str) -> MatchStatus: ...

    def subscribe(self, listener: MatchListener) -> Callable[[], None]:
        """Subscribe to status changes; return an idempotent unsubscribe function."""
        ...
