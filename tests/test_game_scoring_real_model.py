import pytest
import torch
from torch import nn
from dataclasses import replace

from hudes.game.engine import RealTeamEngine
from hudes.game.service import TeamGameService
from hudes.model_data_and_subspace import (
    DatasetBatcher,
    ModelDataAndSubspace,
    indexed_loss,
)
from tests.test_team_game_session import eventually, move, ready, spec, vote


class TinyClassifier(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(2, 10), nn.LogSoftmax(dim=1))

    def forward(self, data):
        return self.net(data)

    def probs(self, output):
        return output.exp()


class SyntheticDataset:
    def __init__(self, size):
        self.data = torch.linspace(-2, 3, size * 2).reshape(size, 1, 2)
        self.labels = torch.arange(size) % 10
        self.reads = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = int(index)
        self.reads.append(index)
        return self.data[index], int(self.labels[index])


def source_mad(val_size=1057):
    return ModelDataAndSubspace(
        TinyClassifier(),
        indexed_loss,
        DatasetBatcher(SyntheticDataset(33)),
        DatasetBatcher(SyntheticDataset(val_size)),
        val_batch_size=1024,
        max_batch_size=512,
        max_grid_size=5,
    )


def test_ragged_validation_covers_every_example_with_actual_model():
    mad = source_mad()
    mad.move_to_device()
    mad.fuse()
    dataset = mad.batchers["val"].ds
    with torch.no_grad():
        expected = (
            -indexed_loss(mad._model(dataset.data.squeeze(1)), dataset.labels)
            .mean()
            .item()
        )
    evaluated = []
    mad.models[torch.float32].register_forward_pre_hook(
        lambda model, args: evaluated.append(args[0].detach().clone())
    )
    actual = mad.val_model_inference_with_delta_weights(
        mad.saved_weights[torch.float32], torch.float32
    )["val_loss"]
    assert actual == pytest.approx(expected, abs=1e-6)
    # DatasetBatcher fetches data and labels separately; model evaluation itself
    # must include each distinct example once, across 512/512/33 chunks.
    assert [len(batch) for batch in evaluated] == [512, 512, 33]
    actual_data = torch.cat(evaluated)
    assert torch.equal(actual_data[:, 0].sort().values, dataset.data[:, 0, 0])
    assert sorted(dataset.reads) == sorted(list(range(1057)) * 2)
    # The old implementation fetched only ceil(1057/1024) * 512 examples.
    assert len(actual_data) == 1057


@pytest.mark.asyncio
async def test_actual_engine_combines_teammate_weights_and_isolates_opponents():
    engine = RealTeamEngine.from_mad(source_mad(37), model_factory=TinyClassifier)
    service = TeamGameService(engine)
    try:
        custom = spec(2, 1)
        custom = replace(
            custom, preset=replace(custom.preset, batch_size=17, max_dimensions=12)
        )
        members = await ready(service, custom)
        session = members[0][2][0]
        opponent = members[2][2][0]
        original = await engine.weights_snapshot(session.session_id)
        opponent_original = await engine.weights_snapshot(opponent.session_id)
        assert torch.equal(original, opponent_original)
        service.handle_control(session, "p0-0", move(0, step=0.15))
        service.handle_control(session, "p0-0", vote())
        service.handle_control(session, "p0-1", move(1, step=-0.23))
        service.handle_control(session, "p0-0", move(2, index=2, step=0.12))
        service.handle_control(session, "p0-1", move(2, index=2, step=-0.07))
        service.handle_control(session, "p0-1", vote())
        await eventually(lambda: session.last_output[0].generation == 1)
        # Queue a diagnostic on the same real worker after all accepted jobs.
        weights = await engine.weights_snapshot(session.session_id)
        expected_delta = await engine._run(
            lambda: (
                engine._mads[0]
                .delta_from_dims({0: 0.15, 2: -0.23, 4: 0.05}, torch.float32)
                .cpu()
            )
        )
        assert torch.allclose(weights, original + expected_delta, atol=1e-7)
        assert torch.equal(
            await engine.weights_snapshot(opponent.session_id), opponent_original
        )
        job, output = session.last_output
        assert job.generation == 1 and job.dims_offset == 6 and job.model_revision == 4
        assert output["mesh"]["shape"] == [3, 5, 5]
        batcher = engine._mads[0].batchers["train"]
        indices = batcher.idxs[:17]
        expected_model = TinyClassifier()
        torch.nn.utils.vector_to_parameters(weights, expected_model.parameters())
        with torch.no_grad():
            expected_loss = (
                -indexed_loss(
                    expected_model(batcher.ds.data[indices].squeeze(1)),
                    batcher.ds.labels[indices],
                )
                .mean()
                .item()
            )
        assert output["train"]["train_loss"] == pytest.approx(expected_loss, abs=1e-6)
        assert (
            service._state(session, "p0-0").team_state.dimension_switches_remaining == 0
        )
        # The center of every actual landscape is the team's common batch loss.
        grid = torch.tensor(output["mesh"]["values"]).reshape(3, 5, 5)
        assert grid[:, 2, 2].tolist() == pytest.approx(
            [output["train"]["train_loss"]] * 3, abs=1e-5
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_preset_seed_initializes_actual_model_batches_and_basis():
    source = source_mad(37)
    engine = RealTeamEngine.from_mad(source, model_factory=TinyClassifier)
    match_spec = spec(1, 1)
    match_spec = replace(match_spec, preset=replace(match_spec.preset, seed=42))
    service = TeamGameService(engine)
    random_state = torch.random.get_rng_state().clone()
    try:
        prepared = await service.prepare_match(match_spec)
        model = TinyClassifier(seed=42)
        expected = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        for admission in prepared.admissions:
            assert torch.equal(
                await engine.weights_snapshot(admission.session_id), expected
            )
        generator = torch.Generator().manual_seed(42)
        expected_batches = torch.randperm(33, generator=generator)
        generator = torch.Generator().manual_seed(42)
        expected_basis_seeds = torch.randint(
            size=(2**16,), high=2**32, generator=generator
        )
        batches, seeds = await engine._run(
            lambda: (
                engine._mads[42].batchers["train"].idxs.clone(),
                engine._mads[42].seeds_for_dims.clone(),
            )
        )
        assert torch.equal(batches, expected_batches)
        assert torch.equal(seeds, expected_basis_seeds)
        assert torch.equal(torch.random.get_rng_state(), random_state)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_team_surface_matches_solo_for_the_same_starting_settings():
    from hudes.game_control import MatchPreset

    source = source_mad(37)
    source.move_to_device()
    source.fuse()
    source.init_param_model()
    preset = replace(MatchPreset(), batch_size=17, mesh_grid_size=5)
    assert preset.mesh_step_size == pytest.approx(10 ** (-0.05 * 12))
    engine = RealTeamEngine.from_mad(source, model_factory=TinyClassifier)
    try:
        output = await engine.prepare("parity", preset, 3)
        solo = source.get_loss_grid(
            base_weights=source.saved_weights[torch.float32],
            grid_size=5,
            step_size=preset.mesh_step_size / 2,
            grids=3,
            dims_offset=0,
            batch_idx=0,
            batch_size=17,
            dtype=torch.float32,
        )
        team = torch.tensor(output["mesh"]["values"]).reshape(output["mesh"]["shape"])
        assert torch.allclose(team, solo.cpu(), atol=1e-6)
    finally:
        await engine.close()
