"""Authoritative team sessions, independent of the legacy solo scheduler.

State transitions run synchronously on the asyncio loop. Engine jobs capture all
identity and direction values at acceptance. Socket queues and listener delivery
never hold a state lock or block a vote, deadline, or lobby callback.
"""

import asyncio
import logging
import math
import secrets
import time
from collections import deque
from dataclasses import dataclass, field, replace

from hudes import hudes_pb2 as pb
from hudes.match_settings import switches_remaining, valid_dimension_limit
from hudes.game_control import (
    Admission,
    MatchSpec,
    MatchStatus,
    PreparedMatch,
    TeamResult,
)

TERMINAL = frozenset({"finished", "cancelled", "failed"})
LANDSCAPE_PANELS = 3


@dataclass(frozen=True)
class GameJob:
    match_id: str
    session_id: str
    job_id: int
    generation: int
    model_revision: int
    dims_offset: int
    panels: int
    preset: object
    kind: str = "surface"
    steps: tuple[tuple[int, float], ...] = ()
    batch_idx: int = 0


@dataclass
class _Connection:
    websocket: object
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(64))
    task: asyncio.Task | None = None
    closing: bool = False

    def send(self, message):
        if self.closing:
            return
        try:
            self.queue.put_nowait(message.SerializeToString())
        except asyncio.QueueFull:
            self.closing = True
            asyncio.create_task(self.websocket.close(code=1013, reason="slow reader"))


@dataclass
class _Session:
    session_id: str
    match_id: str
    roster: object
    preset: object
    tokens: dict
    generation: int = 0
    revision: int = 0
    votes: set = field(default_factory=set)
    connections: dict = field(default_factory=dict)
    last_request: dict = field(default_factory=dict)
    jobs: deque = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    worker: asyncio.Task | None = None
    job_counter: int = 0
    last_output: tuple | None = None
    batch_idx: int = 0

    @property
    def dims_offset(self):
        return self.generation * LANDSCAPE_PANELS * 2


@dataclass
class _Match:
    spec: MatchSpec
    sessions: tuple
    phase: str = "prepared"
    start_at: float | None = None
    deadline: float | None = None
    start_mono: float | None = None
    deadline_mono: float | None = None
    results: tuple = ()
    reason: str | None = None
    clock_task: asyncio.Task | None = None
    final_losses: dict = field(default_factory=dict)
    admissions: PreparedMatch | None = None
    released: bool = False


class TeamGameService:
    """GameControl implementation.

    Engine API: async prepare(session_id, preset, panels) -> output dict;
    async execute(GameJob) -> output dict; async drop(session_id). Real engines
    must serialize private model state; tests can inject a deterministic engine.
    Outputs contain optional train/mesh/batch and final jobs require val_loss.
    """

    def __init__(
        self,
        engine,
        *,
        clock=time.monotonic,
        wall_clock=time.time,
        terminal_grace_seconds=60.0,
        max_terminal_matches=128,
    ):
        self.engine = engine
        self.clock = clock
        self.wall_clock = wall_clock
        self.matches = {}
        self.sessions = {}
        self._preparing = {}
        self._listeners = set()
        self._events = asyncio.Queue()
        self._event_task = None
        self._closed = False
        self._terminal_grace = terminal_grace_seconds
        self._max_terminal = max_terminal_matches
        self._cleanup_tasks = set()

    def subscribe(self, listener):
        self._listeners.add(listener)
        active = True

        def unsubscribe():
            nonlocal active
            if active:
                self._listeners.discard(listener)
                active = False

        return unsubscribe

    def _emit(self, match):
        self._events.put_nowait(self._status(match))
        if self._event_task is None or self._event_task.done():
            self._event_task = asyncio.create_task(self._dispatch_events())

    async def _dispatch_events(self):
        while not self._events.empty():
            status = self._events.get_nowait()
            for listener in tuple(self._listeners):
                try:
                    await listener(status)
                except Exception:
                    logging.exception("Game status listener failed")

    def _status(self, match):
        connected = tuple(
            p.player_id
            for s in match.sessions
            for p in s.roster.players
            if p.player_id in s.connections
        )
        return MatchStatus(
            match.spec.match_id,
            match.phase,
            connected,
            match.start_at,
            match.deadline,
            match.results,
            match.reason,
        )

    @staticmethod
    def _validate_spec(spec):
        if not spec.match_id or not 1 <= len(spec.teams) <= 2:
            raise ValueError("one or two teams required")
        seen_players, seen_teams = set(), set()
        for team in spec.teams:
            if not team.team_id or team.team_id in seen_teams:
                raise ValueError("duplicate team")
            seen_teams.add(team.team_id)
            if not 1 <= len(team.players) <= 3:
                raise ValueError("team requires one to three players")
            if sorted(p.panel_index for p in team.players) != list(
                range(len(team.players))
            ):
                raise ValueError("invalid fixed panel assignment")
            for player in team.players:
                if not player.player_id or player.player_id in seen_players:
                    raise ValueError("duplicate player")
                seen_players.add(player.player_id)
        p = spec.preset
        if not (
            math.isfinite(p.duration_seconds)
            and 0 < p.duration_seconds <= 3600
            and math.isfinite(p.countdown_seconds)
            and 0 <= p.countdown_seconds <= 60
            and 1 <= p.batch_size <= 512
            and valid_dimension_limit(p.max_dimensions)
            and 5 <= p.mesh_grid_size <= 41
            and p.mesh_grid_size % 2 == 1
            and math.isfinite(p.mesh_step_size)
            and 0 < p.mesh_step_size <= 100
            and p.dtype in {"float32", "float16"}
            and p.scoring == "final_validation_loss_v2"
        ):
            raise ValueError("invalid server match preset")

    async def prepare_match(self, spec):
        self._validate_spec(spec)
        if self._closed:
            raise RuntimeError("game service closed")
        existing = self.matches.get(spec.match_id)
        if existing is not None:
            if existing.spec != spec:
                raise ValueError("match ID already has another spec")
            return self._admissions(existing)
        pending = self._preparing.get(spec.match_id)
        if pending:
            old_spec, task = pending
            if old_spec != spec:
                raise ValueError("match ID already preparing another spec")
            return await asyncio.shield(task)
        if (
            len(self._preparing) + sum(not m.released for m in self.matches.values())
            >= 128
        ):
            raise ValueError("game service at capacity")
        task = asyncio.create_task(self._prepare(spec))
        self._preparing[spec.match_id] = (spec, task)

        def completed(done):
            pending = self._preparing.get(spec.match_id)
            if pending and pending[1] is done:
                self._preparing.pop(spec.match_id, None)
            if not done.cancelled():
                done.exception()  # a disconnected prepare caller need not leak an error

        task.add_done_callback(completed)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._preparing.pop(spec.match_id, None)

    async def _prepare(self, spec):
        sessions = []
        try:
            for roster in spec.teams:
                session = _Session(
                    secrets.token_urlsafe(18),
                    spec.match_id,
                    roster,
                    spec.preset,
                    {p.player_id: secrets.token_urlsafe(32) for p in roster.players},
                )
                sessions.append(session)
                output = await self.engine.prepare(
                    session.session_id, spec.preset, LANDSCAPE_PANELS
                )
                job = self._new_job(session)
                if not isinstance(output, dict):
                    raise ValueError("invalid engine preparation result")
                for message in self._output_messages(job, output):
                    message.SerializeToString()
                session.last_output = (job, output)
            if self._closed:
                raise RuntimeError("game service closed")
            match = _Match(spec, tuple(sessions))
            match.admissions = self._admissions(match)
            self.matches[spec.match_id] = match
            for session in sessions:
                self.sessions[session.session_id] = session
                session.worker = asyncio.create_task(self._run_session(session))
            self._emit(match)
            return self._admissions(match)
        except BaseException:
            for session in sessions:
                try:
                    await self.engine.drop(session.session_id)
                except Exception:
                    logging.exception("Failed to drop partially prepared team")
            raise

    def _admissions(self, match):
        if match.admissions is not None:
            return match.admissions
        return PreparedMatch(
            match.spec.match_id,
            tuple(
                Admission(
                    p.player_id,
                    s.roster.team_id,
                    s.session_id,
                    s.tokens[p.player_id],
                    p.panel_index,
                )
                for s in match.sessions
                for p in s.roster.players
            ),
        )

    async def get_match_status(self, match_id):
        return self._status(self.matches[match_id])

    async def start_match(self, match_id):
        match = self.matches[match_id]
        if match.phase != "prepared":
            return self._status(match)
        expected = sum(len(s.roster.players) for s in match.sessions)
        if len(self._status(match).connected_player_ids) != expected:
            raise ValueError("all fixed players must be admitted before start")
        preset = match.spec.preset
        match.start_mono = self.clock() + preset.countdown_seconds
        match.deadline_mono = match.start_mono + preset.duration_seconds
        match.start_at = self.wall_clock() + preset.countdown_seconds
        match.deadline = match.start_at + preset.duration_seconds
        match.phase = "countdown" if preset.countdown_seconds else "running"
        self._broadcast_match(match)
        self._emit(match)
        match.clock_task = asyncio.create_task(self._run_clock(match))
        return self._status(match)

    async def _run_clock(self, match):
        try:
            await asyncio.sleep(max(0, match.start_mono - self.clock()))
            if match.phase == "countdown":
                match.phase = "running"
                self._broadcast_match(match)
                self._emit(match)
            await asyncio.sleep(max(0, match.deadline_mono - self.clock()))
            self._cutoff(match)
        except asyncio.CancelledError:
            pass

    def _cutoff(self, match):
        if match.phase not in {"countdown", "running"}:
            return
        match.phase = "finalizing"
        # All accepted work is already in each session's ordered queue. A final
        # job is the fixed barrier; later inputs cannot append behind it.
        for session in match.sessions:
            if match.phase in TERMINAL:
                break
            self._enqueue(session, self._new_job(session, kind="final"))
        if match.phase in TERMINAL:
            return
        self._broadcast_match(match)
        self._emit(match)

    async def tick(self):
        """Reconcile injected clocks, also useful for deterministic tests."""
        for match in self.matches.values():
            if match.phase in {"running", "countdown"}:
                if self.clock() >= match.deadline_mono:
                    self._cutoff(match)
                elif match.phase == "countdown" and self.clock() >= match.start_mono:
                    match.phase = "running"
                    self._broadcast_match(match)
                    self._emit(match)

    async def cancel_match(self, match_id, reason="cancelled"):
        pending = self._preparing.get(match_id)
        if pending and match_id not in self.matches:
            pending[1].cancel()
            try:
                await pending[1]
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.exception("Preparation cleanup failed during cancellation")
            self._preparing.pop(match_id, None)
            match = _Match(pending[0], (), phase="cancelled", reason=reason)
            self.matches[match_id] = match
            self._emit(match)
            self._schedule_cleanup(match)
            return self._status(match)
        match = self.matches[match_id]
        if match.phase not in TERMINAL:
            self._terminate(match, "cancelled", reason)
        return self._status(match)

    def _terminate(self, match, phase, reason=None):
        match.phase, match.reason = phase, reason
        if match.clock_task and match.clock_task is not asyncio.current_task():
            match.clock_task.cancel()
        for session in match.sessions:
            session.jobs.clear()
            session.wake.set()
        self._broadcast_match(match)
        self._emit(match)
        self._schedule_cleanup(match)

    def _schedule_cleanup(self, match):
        task = asyncio.create_task(self._release_terminal(match))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _release_terminal(self, match):
        # In-flight work may finish, but terminal checks prevent it publishing or
        # changing scores. Wait before dropping its private weight vector.
        workers = [s.worker for s in match.sessions if s.worker]
        await asyncio.gather(*workers, return_exceptions=True)
        for session in match.sessions:
            session.last_output = None
            await self.engine.drop(session.session_id)
        await asyncio.sleep(self._terminal_grace)
        for session in match.sessions:
            session.tokens.clear()
            self.sessions.pop(session.session_id, None)
            for connection in tuple(session.connections.values()):
                connection.task.cancel()
                try:
                    await connection.websocket.close(code=1000, reason="match complete")
                except Exception:
                    pass
            session.connections.clear()
        match.released = True
        # A session remains addressable during its result-reconnect grace. Do
        # not evict its match while admit/disconnect may still reference it.
        terminal = [m for m in self.matches.values() if m.released]
        for old in terminal[: -self._max_terminal] if self._max_terminal else terminal:
            self.matches.pop(old.spec.match_id, None)

    def _new_job(self, session, *, kind="surface", steps=()):
        session.job_counter += 1
        return GameJob(
            session.match_id,
            session.session_id,
            session.job_counter,
            session.generation,
            session.revision,
            session.dims_offset,
            LANDSCAPE_PANELS,
            session.preset,
            kind,
            steps,
            session.batch_idx,
        )

    def _enqueue(self, session, job):
        # Coalesce only waiting movements in the same captured direction set.
        # In-flight work and generation/final barriers remain strictly ordered.
        if job.kind == "move" and session.jobs:
            last = session.jobs[-1]
            if last.kind == "move" and last.generation == job.generation:
                steps = dict(last.steps)
                for dim, step in job.steps:
                    steps[dim] = steps.get(dim, 0.0) + step
                session.jobs[-1] = replace(job, steps=tuple(steps.items()))
                session.wake.set()
                return
        if len(session.jobs) >= 256:
            self._terminate(
                self.matches[session.match_id],
                "failed",
                "inference queue capacity exceeded",
            )
            return
        session.jobs.append(job)
        session.wake.set()

    async def _run_session(self, session):
        try:
            while True:
                await session.wake.wait()
                session.wake.clear()
                if self.matches[session.match_id].phase in TERMINAL:
                    return
                while session.jobs:
                    job = session.jobs.popleft()
                    match = self.matches[session.match_id]
                    if match.phase in TERMINAL:
                        return
                    try:
                        output = await self.engine.execute(job)
                        if not isinstance(output, dict):
                            raise ValueError("invalid engine result")
                        if job.kind != "final":
                            messages = self._output_messages(job, output)
                            for message in messages:
                                message.SerializeToString()
                    except Exception:
                        logging.exception(
                            "Team inference failed for %s", session.session_id
                        )
                        if match.phase not in TERMINAL:
                            self._terminate(match, "failed", "inference failed")
                        break
                    if match.phase in TERMINAL:
                        return
                    if job.kind == "final":
                        try:
                            loss = float(output["val_loss"])
                        except (KeyError, TypeError, ValueError, OverflowError):
                            self._terminate(
                                match, "failed", "invalid final validation result"
                            )
                            return
                        if not math.isfinite(loss):
                            self._terminate(
                                match, "failed", "nonfinite final validation loss"
                            )
                            return
                        match.final_losses[session.roster.team_id] = loss
                        if len(match.final_losses) == len(match.sessions):
                            match.results = tuple(
                                TeamResult(
                                    s.roster.team_id,
                                    match.final_losses[s.roster.team_id],
                                )
                                for s in match.sessions
                            )
                            self._terminate(match, "finished")
                        return
                    elif job.generation == session.generation:
                        session.last_output = (job, output)
                        for message in messages:
                            for connection in tuple(session.connections.values()):
                                connection.send(message)
        except asyncio.CancelledError:
            pass

    async def admit(self, admission, websocket):
        session = self.sessions.get(admission.session_id)
        if session is None or admission.match_id != session.match_id:
            raise ValueError("invalid admission")
        token = session.tokens.get(admission.player_id)
        if not token or not secrets.compare_digest(token, admission.token):
            raise ValueError("invalid admission")
        match = self.matches[session.match_id]
        if match.phase in TERMINAL:
            # Credentials still allow the owner to retrieve immutable results.
            pass
        old = session.connections.get(admission.player_id)
        connection = _Connection(websocket)
        connection.task = asyncio.create_task(self._send_loop(connection))
        session.connections[admission.player_id] = connection
        if old:
            old.task.cancel()
            asyncio.create_task(
                old.websocket.close(code=1000, reason="session resumed")
            )
        self._broadcast_state(session)
        if (
            session.last_output
            and session.last_output[0].generation == session.generation
        ):
            job, output = session.last_output
            for message in self._output_messages(job, output):
                connection.send(message)
        self._emit(match)
        return session, admission.player_id, connection

    async def _send_loop(self, connection):
        try:
            while True:
                await connection.websocket.send(await connection.queue.get())
        except asyncio.CancelledError:
            pass
        except Exception:
            logging.debug("Team socket disconnected")
            try:
                await connection.websocket.close()
            except Exception:
                pass

    def disconnect(self, session, player_id, connection):
        connection.task.cancel()
        if session.connections.get(player_id) is connection:
            session.connections.pop(player_id, None)
            self._broadcast_state(session)
            self._emit(self.matches[session.match_id])

    def handle_control(self, session, player_id, message, *, received_at=None):
        now = self.clock() if received_at is None else received_at
        match = self.matches[session.match_id]
        if match.phase in {"countdown", "running"} and now >= match.deadline_mono:
            self._cutoff(match)
        if match.phase not in {"running", "countdown"} or now < match.start_mono:
            raise ValueError("match is not accepting input")
        if (
            not message.HasField("generation")
            or message.generation != session.generation
        ):
            raise ValueError("stale direction generation")
        if (
            message.HasField("team_session_id")
            and message.team_session_id != session.session_id
        ):
            raise ValueError("wrong team session")
        if (
            message.HasField("team_match_id")
            and message.team_match_id != session.match_id
        ):
            raise ValueError("wrong match")
        if message.type in {
            pb.Control.CONTROL_DIMS,
            pb.Control.CONTROL_NEXT_BATCH,
            pb.Control.CONTROL_CONFIG,
        }:
            if not message.HasField("request_idx") or message.request_idx <= 0:
                raise ValueError("positive request index required")
            if message.request_idx <= session.last_request.get(player_id, 0):
                return False
        if message.type == pb.Control.CONTROL_DIMS:
            panel = next(
                p.panel_index
                for p in session.roster.players
                if p.player_id == player_id
            )
            # A teammate retains their assigned panel even while disconnected.
            # Vacant panels are shared; accepted moves use the same ordered queue.
            assigned = {p.panel_index for p in session.roster.players}
            allowed_panels = {panel} | (set(range(LANDSCAPE_PANELS)) - assigned)
            if (
                not message.dims_and_steps
                or len(message.dims_and_steps) > 2
                or len({d.dim // 2 for d in message.dims_and_steps}) != 1
                or any(
                    d.dim // 2 not in allowed_panels
                    or not math.isfinite(d.step)
                    or abs(d.step) > 1000
                    for d in message.dims_and_steps
                )
            ):
                raise ValueError("player does not own these directions")
            absolute_steps = {}
            for d in message.dims_and_steps:
                dim = d.dim + session.dims_offset
                absolute_steps[dim] = absolute_steps.get(dim, 0.0) + d.step
            steps = tuple(absolute_steps.items())
            session.last_request[player_id] = message.request_idx
            session.revision += 1
            self._enqueue(session, self._new_job(session, kind="move", steps=steps))
            return True
        if message.type == pb.Control.CONTROL_NEXT_BATCH:
            session.last_request[player_id] = message.request_idx
            session.batch_idx += 1
            self._enqueue(session, self._new_job(session))
            self._broadcast_state(session)
            return True
        if message.type == pb.Control.CONTROL_CONFIG:
            if (
                not message.HasField("team_mesh_step_size")
                or message.HasField("config")
                or not (
                    math.isfinite(message.team_mesh_step_size)
                    and 0 < message.team_mesh_step_size <= 100
                )
            ):
                raise ValueError("only the shared mesh sampling scale can change")
            session.last_request[player_id] = message.request_idx
            session.preset = replace(
                session.preset, mesh_step_size=message.team_mesh_step_size
            )
            self._enqueue(session, self._new_job(session))
            self._broadcast_state(session)
            return True
        if message.type == pb.Control.CONTROL_NEXT_DIMS:
            if switches_remaining(session.preset, session.generation) == 0:
                raise ValueError(
                    "No dimension switches remaining; keep optimizing the current landscapes"
                )
            if player_id in session.votes:
                return False
            session.votes.add(player_id)
            if len(session.votes) == len(session.roster.players):
                session.generation += 1
                session.votes.clear()
                self._enqueue(session, self._new_job(session))
            self._broadcast_state(session)
            return True
        raise ValueError("team settings are fixed; control not allowed")

    def _config(self, session):
        p = session.preset
        return pb.Config(
            seed=p.seed,
            dims_at_a_time=LANDSCAPE_PANELS * 2,
            mesh_grid_size=p.mesh_grid_size,
            mesh_step_size=p.mesh_step_size,
            mesh_grids=LANDSCAPE_PANELS,
            batch_size=p.batch_size,
            dtype=p.dtype,
            mesh_enabled=True,
            loss_lines=0,
            resume_supported=False,
        )

    def _state(self, session, player_id):
        match = self.matches[session.match_id]
        player = next(p for p in session.roster.players if p.player_id == player_id)
        state = pb.TeamState(
            match_id=session.match_id,
            session_id=session.session_id,
            team_id=session.roster.team_id,
            player_id=player_id,
            panel_index=player.panel_index,
            generation=session.generation,
            dims_offset=session.dims_offset,
            model_revision=session.revision,
            phase=match.phase,
            config=self._config(session),
            batch_idx=session.batch_idx,
        )
        if match.start_at is not None:
            state.start_at, state.deadline = match.start_at, match.deadline
        remaining = switches_remaining(session.preset, session.generation)
        if remaining is not None:
            state.dimension_switches_remaining = remaining
        if match.reason:
            state.reason = match.reason
        for p in session.roster.players:
            state.roster.add(
                player_id=p.player_id,
                name=p.name,
                panel_index=p.panel_index,
                voted=p.player_id in session.votes,
                connected=p.player_id in session.connections,
            )
        for result in match.results:
            state.results.add(team_id=result.team_id, loss=result.loss)
        return pb.Control(
            type=pb.Control.CONTROL_TEAM_STATE,
            team_state=state,
            team_match_id=session.match_id,
            team_session_id=session.session_id,
            generation=session.generation,
            model_revision=session.revision,
            request_idx=session.last_request.get(player_id, 0),
            job_id=session.job_counter,
            team_batch_idx=session.batch_idx,
        )

    def _broadcast_state(self, session):
        for player_id, connection in tuple(session.connections.items()):
            connection.send(self._state(session, player_id))

    def _broadcast_match(self, match):
        for session in match.sessions:
            self._broadcast_state(session)

    @staticmethod
    def _output_messages(job, output):
        tags = dict(
            team_match_id=job.match_id,
            team_session_id=job.session_id,
            generation=job.generation,
            model_revision=job.model_revision,
            job_id=job.job_id,
            team_batch_idx=job.batch_idx,
        )
        messages = []
        train = output.get("train")
        if train is not None:
            messages.append(
                pb.Control(
                    type=pb.Control.CONTROL_TRAIN_LOSS_AND_PREDS,
                    train_loss_and_preds=pb.TrainLossAndPreds(
                        train_loss=train["train_loss"],
                        preds=train.get("preds", []),
                        preds_shape=train.get("preds_shape", []),
                        confusion_matrix=train.get("confusion_matrix", []),
                        confusion_matrix_shape=train.get("confusion_matrix_shape", []),
                    ),
                    **tags,
                )
            )
        mesh = output.get("mesh")
        if mesh is not None:
            messages.append(
                pb.Control(
                    type=pb.Control.CONTROL_MESHGRID_RESULTS,
                    mesh_grid_results=mesh["values"],
                    mesh_grid_shape=mesh["shape"],
                    **tags,
                )
            )
        batch = output.get("batch")
        if batch is not None:
            messages.append(
                pb.Control(
                    type=pb.Control.CONTROL_BATCH_EXAMPLES,
                    batch_examples=pb.BatchExamples(
                        type=pb.BatchExamples.IMG_BW,
                        n=batch["n"],
                        train_data=batch["data"],
                        train_data_shape=batch["data_shape"],
                        train_labels=batch["labels"],
                        train_labels_shape=batch["labels_shape"],
                        batch_idx=job.batch_idx,
                    ),
                    **tags,
                )
            )
        return messages

    def _broadcast_output(self, session, job, output):
        for message in self._output_messages(job, output):
            for connection in tuple(session.connections.values()):
                connection.send(message)

    async def close(self):
        self._closed = True
        for _, task in tuple(self._preparing.values()):
            task.cancel()
        await asyncio.gather(
            *(task for _, task in self._preparing.values()), return_exceptions=True
        )
        tasks = []
        for match in self.matches.values():
            if match.clock_task:
                match.clock_task.cancel()
                tasks.append(match.clock_task)
            for session in match.sessions:
                if session.worker:
                    session.worker.cancel()
                    tasks.append(session.worker)
                for connection in session.connections.values():
                    connection.task.cancel()
                    tasks.append(connection.task)
                await self.engine.drop(session.session_id)
        if self._event_task:
            self._event_task.cancel()
            tasks.append(self._event_task)
        for task in tuple(self._cleanup_tasks):
            task.cancel()
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        close = getattr(self.engine, "close", None)
        if close:
            await close()
