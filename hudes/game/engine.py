"""Real ModelDataAndSubspace inference on one private serialized worker.

Team weight vectors live only in this worker. The MAD evaluator can temporarily
load different vectors without exposing mutable parameters to the event loop or
the legacy solo worker. Every opponent starts from the same prepared baseline.
"""

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor

import torch

from hudes.model_data_and_subspace import DatasetBatcher, ModelDataAndSubspace


class RealTeamEngine:
    def __init__(self, mad_factory):
        self._factory = mad_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="team-inference"
        )
        self._mads = {}
        self._weights = {}
        self._sessions = {}

    @classmethod
    def from_mad(cls, source, *, model_factory=None):
        """Construct fresh batchers/basis/models instead of sharing solo state.

        A supplied factory must actually initialize its model with the requested
        seed. Without one only the source's seed-zero baseline is supported.
        """

        def factory(preset):
            if preset.seed == 0:
                model = copy.deepcopy(source._model)
            else:
                if model_factory is None:
                    raise ValueError("a seeded model factory is required")
                with torch.random.fork_rng():
                    model = model_factory(preset.seed)
            return ModelDataAndSubspace(
                model,
                source.loss_fn,
                DatasetBatcher(source.batchers["train"].ds, seed=preset.seed),
                DatasetBatcher(source.batchers["val"].ds, seed=preset.seed),
                seed=preset.seed,
                minimize=source.minimize,
                device=source.device,
                val_batch_size=source.val_batch_size,
                max_batch_size=source.max_batch_size,
                max_grids=max(3, source.max_grids),
                max_grid_size=source.max_grid_size,
            )

        return cls(factory)

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, fn, *args
        )

    async def prepare(self, session_id, preset, panels):
        return await self._run(self._prepare, session_id, preset, panels)

    @torch.no_grad()
    def _prepare(self, session_id, preset, panels):
        if preset.seed not in self._mads:
            mad = self._factory(preset)
            mad.move_to_device()
            mad.fuse()
            mad.init_param_model()
            self._mads[preset.seed] = mad
        mad = self._mads[preset.seed]
        if (
            preset.batch_size > mad.max_batch_size
            or preset.mesh_grid_size > mad.max_grid_size
        ):
            raise ValueError("preset exceeds inference engine limits")
        self._sessions[session_id] = preset.seed
        self._weights[session_id] = mad.saved_weights[torch.float32].clone()
        return self._surface(session_id, preset, panels, 0, 0)

    async def execute(self, job):
        return await self._run(self._execute, job)

    @torch.no_grad()
    def _execute(self, job):
        mad = self._mads[self._sessions[job.session_id]]
        weights = self._weights[job.session_id]
        if job.steps:
            # Preserve duplicate dimensions from one command by summing them.
            steps = {}
            for dim, step in job.steps:
                steps[dim] = steps.get(dim, 0.0) + step
            weights.add_(mad.delta_from_dims(steps, dtype=torch.float32))
        if job.kind == "final":
            return mad.val_model_inference_with_delta_weights(
                weights, getattr(torch, job.preset.dtype)
            )
        return self._surface(
            job.session_id, job.preset, job.panels, job.dims_offset, job.batch_idx
        )

    @torch.no_grad()
    def _surface(self, session_id, preset, panels, dims_offset, batch_idx):
        mad = self._mads[self._sessions[session_id]]
        dtype = getattr(torch, preset.dtype)
        weights = self._weights[session_id].to(dtype)
        train = mad.train_model_inference_with_delta_weights(
            weights,
            preset.batch_size,
            batch_idx,
            dtype,
        )
        preds = train["train_preds"].cpu()
        confusion = train["confusion_matrix"].cpu()
        mesh = mad.get_loss_grid(
            base_weights=weights,
            batch_idx=batch_idx,
            dims_offset=dims_offset,
            grids=panels,
            grid_size=preset.mesh_grid_size,
            step_size=preset.mesh_step_size / 2,
            batch_size=preset.batch_size,
            dtype=dtype,
        ).cpu()
        data, labels = mad.get_batch(preset.batch_size, batch_idx, dtype, "train")
        data, labels = data[:4].cpu(), labels[:4].cpu()
        return {
            "train": {
                "train_loss": train["train_loss"],
                "preds": preds.flatten().tolist(),
                "preds_shape": list(preds.shape),
                "confusion_matrix": confusion.flatten().tolist(),
                "confusion_matrix_shape": list(confusion.shape),
            },
            "mesh": {"values": mesh.flatten().tolist(), "shape": list(mesh.shape)},
            "batch": {
                "n": len(labels),
                "data": data.flatten().tolist(),
                "data_shape": list(data.shape),
                "labels": labels.flatten().tolist(),
                "labels_shape": list(labels.shape),
            },
        }

    async def drop(self, session_id):
        await self._run(self._drop, session_id)

    def _drop(self, session_id):
        self._weights.pop(session_id, None)
        self._sessions.pop(session_id, None)

    async def weights_snapshot(self, session_id):
        """Copied diagnostic value used by numerical integration tests."""
        return await self._run(lambda: self._weights[session_id].detach().cpu().clone())

    async def close(self):
        await self._run(self._weights.clear)
        self._executor.shutdown(wait=True)
