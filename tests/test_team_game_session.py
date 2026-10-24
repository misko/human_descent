import asyncio
from dataclasses import replace

import pytest

from hudes import hudes_pb2 as pb
from hudes.game.service import TeamGameService
from hudes.game_control import MatchPreset, MatchSpec, PlayerSlot, TeamRoster


class Socket:
    def __init__(self):
        self.messages = []
        self.closed = False

    async def send(self, raw):
        self.messages.append(pb.Control.FromString(raw))

    async def close(self, **kwargs):
        self.closed = True


class Engine:
    def __init__(self):
        self.calls = []
        self.weights = {}
        self.block = None
        self.entered = asyncio.Event()
        self.final_output = None
        self.preparing = None
        self.dropped = []

    async def prepare(self, session_id, preset, panels):
        self.weights[session_id] = {}
        if self.preparing:
            self.entered.set()
            await self.preparing.wait()
        return {"train": {"train_loss": 2.0}}

    async def execute(self, job):
        self.calls.append(job)
        if self.block and job.kind == "move":
            self.entered.set()
            await self.block.wait()
        for dim, value in job.steps:
            w = self.weights[job.session_id]
            w[dim] = w.get(dim, 0) + value
        if job.kind == "final":
            if self.final_output is not None:
                return self.final_output
            return {"val_loss": 2 - sum(self.weights[job.session_id].values())}
        return {"train": {"train_loss": 2.0}}

    async def drop(self, session_id):
        self.dropped.append(session_id)
        self.weights.pop(session_id, None)


def spec(*sizes, match_id="match", duration=30):
    return MatchSpec(
        match_id,
        tuple(
            TeamRoster(
                f"t{t}",
                f"Team {t}",
                tuple(PlayerSlot(f"p{t}-{p}", f"Player {p}", p) for p in range(size)),
            )
            for t, size in enumerate(sizes)
        ),
        MatchPreset(
            duration_seconds=duration,
            countdown_seconds=0,
            mesh_grid_size=5,
            batch_size=4,
        ),
    )


async def eventually(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.001)


def wire_admission(admission, match_id="match"):
    return pb.TeamAdmission(
        match_id=match_id,
        session_id=admission.session_id,
        player_id=admission.player_id,
        token=admission.token,
    )


async def ready(service, match_spec):
    prepared = await service.prepare_match(match_spec)
    members = []
    for admission in prepared.admissions:
        socket = Socket()
        bound = await service.admit(
            wire_admission(admission, match_spec.match_id), socket
        )
        members.append((admission, socket, bound))
    await service.start_match(match_spec.match_id)
    return members


def move(panel, *, generation=0, index=1, step=0.1):
    return pb.Control(
        type=pb.Control.CONTROL_DIMS,
        generation=generation,
        request_idx=index,
        dims_and_steps=[pb.DimAndStep(dim=panel * 2, step=step)],
    )


def vote(generation=0):
    return pb.Control(type=pb.Control.CONTROL_NEXT_DIMS, generation=generation)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 2, 3])
async def test_fixed_quorum_and_partial_vote_without_inference(size):
    engine = Engine()
    service = TeamGameService(engine)
    try:
        members = await ready(service, spec(size))
        session = members[0][2][0]
        config = service._config(session)
        assert config.mesh_grids == 3 and config.dims_at_a_time == 6
        for i in range(size):
            service.handle_control(session, f"p0-{i}", vote())
            if i + 1 < size:
                assert engine.calls == []
                assert session.votes == {f"p0-{j}" for j in range(i + 1)}
                assert not service.handle_control(session, f"p0-{i}", vote())
                await asyncio.sleep(0)
                assert any(
                    m.team_state.roster[i].voted
                    for m in members[0][1].messages
                    if m.HasField("team_state")
                )
        assert session.generation == 1
        assert session.votes == set()
        assert session.dims_offset == 6
        await eventually(lambda: len(engine.calls) == 1)
        with pytest.raises(ValueError, match="stale"):
            service.handle_control(session, "p0-0", vote())
        assert session.generation == 1
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 2, 3])
async def test_own_and_unassigned_panels_follow_roster_not_connections(size):
    engine = Engine()
    service = TeamGameService(engine)
    try:
        members = await ready(service, spec(size))
        session = members[0][2][0]
        # A disconnected member's panel remains private to that member.
        if size > 1:
            service.disconnect(*members[-1][2])
        accepted = 0
        for owner in range(size):
            for panel in range(4):
                message = move(panel, index=panel + 1)
                if panel == owner or size <= panel < 3:
                    assert service.handle_control(session, f"p0-{owner}", message)
                    accepted += 1
                else:
                    before = session.revision
                    with pytest.raises(ValueError, match="own"):
                        service.handle_control(session, f"p0-{owner}", message)
                    assert session.revision == before
        await eventually(lambda: session.last_output[0].model_revision == accepted)
        assert session.generation == 0 and not session.votes
        for panel in range(3):
            expected_moves = 1 if panel < size else size
            assert engine.weights[session.session_id][panel * 2] == pytest.approx(
                expected_moves * 0.1
            )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_shared_panel_moves_preserve_order_generation_and_budget():
    engine = Engine()
    engine.block = asyncio.Event()
    service = TeamGameService(engine)
    try:
        match_spec = spec(2, 1)
        match_spec = replace(
            match_spec, preset=replace(match_spec.preset, max_dimensions=12)
        )
        members = await ready(service, match_spec)
        session = members[0][2][0]
        other = members[2][2][0]
        assert service.handle_control(session, "p0-0", move(2, step=0.2))
        await engine.entered.wait()
        service.handle_control(session, "p0-0", vote())
        assert service.handle_control(session, "p0-1", move(2, step=-0.1))
        service.handle_control(session, "p0-1", vote())
        assert session.generation == 1
        assert (
            service._state(session, "p0-0").team_state.dimension_switches_remaining == 0
        )
        # Selecting/moving existing panels is still permitted at zero budget.
        assert service.handle_control(
            session, "p0-0", move(2, generation=1, index=2, step=0.3)
        )
        with pytest.raises(ValueError, match="stale"):
            service.handle_control(session, "p0-0", move(2, index=3))
        mixed = move(0, generation=1, index=3)
        mixed.dims_and_steps.add(dim=4, step=0.1)
        with pytest.raises(ValueError, match="own"):
            service.handle_control(session, "p0-0", mixed)
        engine.block.set()
        await eventually(lambda: len(engine.calls) == 4)
        assert engine.weights[session.session_id] == {
            4: pytest.approx(0.1),
            10: pytest.approx(0.3),
        }
        assert engine.weights[other.session_id] == {}
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pending_moves_survive_unanimity_and_votes_survive_disconnect():
    engine = Engine()
    engine.block = asyncio.Event()
    service = TeamGameService(engine)
    try:
        members = await ready(service, spec(2, 1))
        session = members[0][2][0]
        other = members[2][2][0]
        service.handle_control(session, "p0-0", move(0))
        await engine.entered.wait()
        service.handle_control(session, "p0-0", vote())
        service.handle_control(session, "p0-1", move(1, step=0.2))
        assert session.votes == {"p0-0"}
        service.disconnect(*members[0][2])
        assert session.votes == {"p0-0"}
        assert len(session.roster.players) == 2
        service.handle_control(session, "p0-1", vote())
        assert session.generation == 1 and not session.votes
        assert other.generation == 0
        # The earlier accepted moves retain their absolute old-generation dims.
        engine.block.set()
        await eventually(lambda: len(engine.calls) == 3)
        assert engine.weights[session.session_id] == {
            0: pytest.approx(0.1),
            2: pytest.approx(0.2),
        }
        assert engine.weights[other.session_id] == {}
        assert [j.generation for j in engine.calls] == [0, 0, 1]
        assert [j.model_revision for j in engine.calls] == [1, 2, 2]
        assert not any(
            m.type == pb.Control.CONTROL_TRAIN_LOSS_AND_PREDS
            and m.model_revision > 0
            and m.generation == 0
            for m in members[1][1].messages
        )
        admission = members[0][0]
        new_socket = Socket()
        rebound = await service.admit(wire_admission(admission), new_socket)
        await asyncio.sleep(0)
        snapshot = next(m for m in new_socket.messages if m.HasField("team_state"))
        assert snapshot.team_state.generation == 1
        assert snapshot.request_idx == 1
        # Replayed accepted input cannot move the model twice after reconnect.
        assert not service.handle_control(
            session, admission.player_id, move(0, generation=1)
        )
        service.disconnect(*rebound)
    finally:
        engine.block.set()
        await service.close()


@pytest.mark.asyncio
async def test_admission_ownership_and_shared_settings_navigation():
    engine = Engine()
    service = TeamGameService(engine)
    try:
        prepared = await service.prepare_match(spec(2, 1))
        with pytest.raises(ValueError, match="all fixed"):
            await service.start_match("match")
        own = prepared.admissions[0]
        for forged in [
            wire_admission(replace(own, player_id="p0-1")),
            wire_admission(replace(own, session_id=prepared.admissions[-1].session_id)),
            wire_admission(own, "other"),
        ]:
            with pytest.raises(ValueError, match="invalid admission"):
                await service.admit(forged, Socket())
        members = await ready(service, spec(2, 1))
        session = members[0][2][0]
        with pytest.raises(ValueError, match="own"):
            service.handle_control(session, "p0-0", move(1))
        assert session.revision == 0
        service.handle_control(session, "p0-0", vote())
        change = pb.Control(
            type=pb.Control.CONTROL_CONFIG,
            generation=0,
            request_idx=1,
            team_mesh_step_size=0.2,
        )
        service.handle_control(session, "p0-0", change)
        batch = pb.Control(
            type=pb.Control.CONTROL_NEXT_BATCH, generation=0, request_idx=2
        )
        service.handle_control(session, "p0-0", batch)
        assert not service.handle_control(session, "p0-0", batch)
        assert (
            session.batch_idx == 1
            and session.preset.mesh_step_size == pytest.approx(0.2)
        )
        assert session.votes == {"p0-0"}
        assert session.preset.batch_size == 4 and session.preset.dtype == "float32"
        await eventually(lambda: len(engine.calls) == 2)
        assert engine.calls[-1].batch_idx == 1
        assert engine.calls[-1].preset.mesh_step_size == pytest.approx(0.2)
        state = service._state(session, "p0-0")
        assert (
            state.team_state.batch_idx == 1 and state.job_id == engine.calls[-1].job_id
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_deadline_drains_fixed_barrier_disconnected_and_immutable_result():
    now = [100.0]
    engine = Engine()
    engine.block = asyncio.Event()
    service = TeamGameService(engine, clock=lambda: now[0])
    try:
        members = await ready(service, spec(2, 1, duration=2))
        session = members[0][2][0]
        service.handle_control(session, "p0-0", move(0, step=0.25), received_at=101.99)
        await engine.entered.wait()
        now[0] = 102.0
        with pytest.raises(ValueError, match="not accepting"):
            service.handle_control(session, "p0-1", move(1, step=20), received_at=102)
        assert (await service.get_match_status("match")).phase == "finalizing"
        for _, _, bound in members:
            service.disconnect(*bound)
        assert (await service.get_match_status("match")).connected_player_ids == ()
        engine.block.set()
        await eventually(lambda: service.matches["match"].phase == "finished")
        result = await service.get_match_status("match")
        assert [r.loss for r in result.results] == pytest.approx([1.75, 2.0])
        assert (await service.start_match("match")).results == result.results
        assert (await service.cancel_match("match")).phase == "finished"
        await service.tick()
        assert (await service.get_match_status("match")).results == result.results
        assert len([j for j in engine.calls if j.kind == "final"]) == 2
        await eventually(lambda: len(engine.dropped) == 2)
        assert not engine.weights
    finally:
        engine.block.set()
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [{}, {"val_loss": "bad"}, {"val_loss": float("nan")}, {"val_loss": float("inf")}],
)
async def test_invalid_final_results_fail_instead_of_hanging(output):
    now = [0.0]
    engine = Engine()
    engine.final_output = output
    service = TeamGameService(engine, clock=lambda: now[0])
    try:
        await ready(service, spec(1, duration=1))
        now[0] = 2
        await service.tick()
        await eventually(lambda: service.matches["match"].phase == "failed")
        assert not (await service.get_match_status("match")).results
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_prepare_cancel_tombstone_and_terminal_cleanup():
    engine = Engine()
    engine.preparing = asyncio.Event()
    service = TeamGameService(engine, terminal_grace_seconds=0)
    try:
        task = asyncio.create_task(service.prepare_match(spec(1)))
        await engine.entered.wait()
        status = await service.cancel_match("match")
        assert status.phase == "cancelled"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await service.cancel_match("match")).phase == "cancelled"
        assert (await service.start_match("match")).phase == "cancelled"
        assert (await service.prepare_match(spec(1))).admissions == ()
        with pytest.raises(ValueError, match="another spec"):
            await service.prepare_match(spec(2))
        assert engine.weights == {}
    finally:
        engine.preparing.set()
        await service.close()


@pytest.mark.asyncio
async def test_listener_can_reenter_start_and_status_without_deadlock():
    engine = Engine()
    service = TeamGameService(engine)
    statuses = []

    async def listener(status):
        statuses.append(status.phase)
        await service.get_match_status(status.match_id)
        if status.phase == "prepared" and len(status.connected_player_ids) == 2:
            await service.start_match(status.match_id)

    service.subscribe(listener)
    try:
        prepared = await service.prepare_match(spec(2))
        for admission in prepared.admissions:
            await service.admit(wire_admission(admission), Socket())
        await eventually(lambda: "running" in statuses)
        start = await service.get_match_status("match")
        assert (await service.start_match("match")).start_at == start.start_at
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_shared_countdown_is_idempotent_and_rejects_early_input():
    now = [100.0]
    engine = Engine()
    service = TeamGameService(engine, clock=lambda: now[0], wall_clock=lambda: 1000)
    match_spec = spec(1, 3)
    match_spec = replace(
        match_spec, preset=replace(match_spec.preset, countdown_seconds=3)
    )
    try:
        members = await ready(service, match_spec)
        session = members[0][2][0]
        status = await service.get_match_status("match")
        assert (
            status.phase == "countdown"
            and status.start_at == 1003
            and status.deadline == 1033
        )
        assert (await service.start_match("match")) == status
        assert len(engine.weights) == 2
        with pytest.raises(ValueError, match="not accepting"):
            service.handle_control(session, "p0-0", move(0))
        now[0] = 103
        await service.tick()
        assert (await service.get_match_status("match")).phase == "running"
        assert service.handle_control(session, "p0-0", move(0))
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [[], {"train": {"train_loss": "bad"}}, {"mesh": {"values": [], "shape": ["bad"]}}],
)
async def test_malformed_engine_outputs_fail_match(invalid):
    class BadEngine(Engine):
        async def execute(self, job):
            return invalid

    service = TeamGameService(BadEngine())
    try:
        members = await ready(service, spec(1))
        service.handle_control(members[0][2][0], "p0-0", move(0))
        await eventually(lambda: service.matches["match"].phase == "failed")
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_terminal_eviction_cannot_orphan_a_still_retained_session():
    engine = Engine()
    engine.block = asyncio.Event()
    service = TeamGameService(engine, terminal_grace_seconds=0, max_terminal_matches=1)
    try:
        members = await ready(service, spec(1, match_id="old"))
        service.handle_control(members[0][2][0], "p0-0", move(0))
        await engine.entered.wait()
        await service.cancel_match("old")
        # The first cancelled worker is still in flight while newer terminal
        # history fills and expires. Its retained admission remains addressable.
        for match_id in ("new", "newer"):
            await service.prepare_match(spec(1, match_id=match_id))
            await service.cancel_match(match_id)
            await eventually(lambda: service.matches[match_id].released)
        assert "old" in service.matches
        session, player, connection = await service.admit(
            wire_admission(members[0][0], "old"), Socket()
        )
        assert service._state(session, player).team_state.phase == "cancelled"
        service.disconnect(session, player, connection)
        engine.block.set()
        await eventually(lambda: len(service.matches) <= 1 and not service.sessions)
        assert engine.weights == {}
    finally:
        engine.block.set()
        await service.close()


@pytest.mark.asyncio
async def test_overflow_at_cutoff_does_not_append_work_after_failure():
    engine = Engine()
    service = TeamGameService(engine)
    try:
        members = await ready(service, spec(1, 1))
        session = members[0][2][0]
        session.jobs.extend(service._new_job(session) for _ in range(256))
        service._cutoff(service.matches["match"])
        assert service.matches["match"].phase == "failed"
        assert all(not s.jobs for s in service.matches["match"].sessions)
        await eventually(lambda: not engine.weights)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_partial_preparation_failure_drops_every_team_and_close_waits_prepare():
    class PartialEngine(Engine):
        async def prepare(self, session_id, preset, panels):
            output = await super().prepare(session_id, preset, panels)
            if len(self.weights) == 2:
                raise RuntimeError("second team failed")
            return output

    engine = PartialEngine()
    service = TeamGameService(engine)
    with pytest.raises(RuntimeError, match="second team"):
        await service.prepare_match(spec(1, 3))
    assert not engine.weights and len(engine.dropped) == 2
    await service.close()

    engine = Engine()
    engine.preparing = asyncio.Event()
    service = TeamGameService(engine)
    pending = asyncio.create_task(service.prepare_match(spec(1)))
    await engine.entered.wait()
    await service.close()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not engine.weights


@pytest.mark.asyncio
async def test_duplicate_direction_entries_are_preserved_when_waiting_moves_coalesce():
    engine = Engine()
    engine.block = asyncio.Event()
    service = TeamGameService(engine)
    try:
        members = await ready(service, spec(1))
        session = members[0][2][0]
        service.handle_control(session, "p0-0", move(0, index=1))
        await engine.entered.wait()
        duplicate = pb.Control(
            type=pb.Control.CONTROL_DIMS,
            generation=0,
            request_idx=2,
            dims_and_steps=[
                pb.DimAndStep(dim=0, step=0.2),
                pb.DimAndStep(dim=0, step=0.3),
            ],
        )
        service.handle_control(session, "p0-0", duplicate)
        service.handle_control(session, "p0-0", move(0, index=3, step=0.4))
        assert len(session.jobs) == 1
        engine.block.set()
        await eventually(lambda: len(engine.calls) == 2)
        assert engine.weights[session.session_id][0] == pytest.approx(1.0)
    finally:
        engine.block.set()
        await service.close()


@pytest.mark.asyncio
async def test_bad_prepared_surface_cannot_admit_a_partially_valid_slot():
    class BadPrepareEngine(Engine):
        async def prepare(self, session_id, preset, panels):
            await super().prepare(session_id, preset, panels)
            return {"train": {"train_loss": "invalid"}}

    engine = BadPrepareEngine()
    service = TeamGameService(engine)
    try:
        with pytest.raises((TypeError, ValueError)):
            await service.prepare_match(spec(1))
        assert not engine.weights and not service.matches and not service.sessions
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("limit", [6, 18, None])
async def test_dimension_budget_is_per_team_and_only_spent_on_unanimity(size, limit):
    service = TeamGameService(Engine())
    try:
        match = spec(size, 1)
        match = replace(
            match, preset=replace(match.preset, max_dimensions=limit, batch_size=17)
        )
        members = await ready(service, match)
        session, opponent = members[0][2][0], members[-1][2][0]
        budget = None if limit is None else limit // 6 - 1

        def remaining(s):
            state = service._state(s, s.roster.players[0].player_id).team_state
            assert state.config.batch_size == 17
            return (
                state.dimension_switches_remaining
                if state.HasField("dimension_switches_remaining")
                else None
            )

        assert remaining(session) == budget
        for generation in range(budget if budget is not None else 3):
            for player in range(size):
                assert remaining(session) == (
                    None if budget is None else budget - generation
                )
                service.handle_control(session, f"p0-{player}", vote(generation))
            assert session.generation == generation + 1
            assert not session.votes
            assert remaining(opponent) == budget
        if budget is not None:
            assert remaining(session) == 0
            with pytest.raises(ValueError, match="No dimension switches"):
                service.handle_control(session, "p0-0", vote(budget))
            assert not session.votes
            assert service.handle_control(session, "p0-0", move(0, generation=budget))
            # A reconnect receives the authoritative exhausted budget.
            socket = Socket()
            await service.admit(wire_admission(members[0][0]), socket)
            await eventually(
                lambda: any(m.HasField("team_state") for m in socket.messages)
            )
            snapshot = next(
                m.team_state for m in socket.messages if m.HasField("team_state")
            )
            assert snapshot.HasField("dimension_switches_remaining")
            assert snapshot.dimension_switches_remaining == 0
        else:
            assert remaining(session) is None
    finally:
        await service.close()
