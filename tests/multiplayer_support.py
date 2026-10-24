"""Deterministic numerical engine for transport and browser integration tests."""

import asyncio


class DeterministicEngine:
    def __init__(self):
        self.sessions = {}
        self.jobs = []

    async def prepare(self, session_id, preset, panels):
        self.sessions[session_id] = {"weights": {}, "panels": panels}
        return self.output(session_id, preset, panels, 0)

    async def execute(self, job):
        state = self.sessions[job.session_id]
        for dimension, step in job.steps:
            state["weights"][dimension] = state["weights"].get(dimension, 0.0) + step
        self.jobs.append(
            {
                "sessionId": job.session_id,
                "generation": job.generation,
                "kind": job.kind,
                "revision": job.model_revision,
            }
        )
        await asyncio.sleep(0)
        if job.kind == "final":
            return {"val_loss": self.loss(job.session_id)}
        return self.output(job.session_id, job.preset, job.panels, job.dims_offset)

    def loss(self, session_id):
        weights = self.sessions[session_id]["weights"]
        return 2.3 + sum(value * value - value * 0.1 for value in weights.values())

    def output(self, session_id, preset, panels, offset):
        size = preset.mesh_grid_size
        weights = self.sessions[session_id]["weights"]
        baseline = self.loss(session_id)
        values = []
        for panel in range(panels):
            x_shift = weights.get(offset + panel * 2, 0)
            y_shift = weights.get(offset + panel * 2 + 1, 0)
            for row in range(size):
                for column in range(size):
                    x = (column - size // 2) * preset.mesh_step_size + x_shift
                    y = (row - size // 2) * preset.mesh_step_size + y_shift
                    values.append(baseline + x * x + y * y)
        return {
            "train": {
                "train_loss": baseline,
                "preds": [0.1] * 40,
                "preds_shape": [4, 10],
                "confusion_matrix": [int(i % 11 == 0) for i in range(100)],
                "confusion_matrix_shape": [10, 10],
            },
            "mesh": {"values": values, "shape": [panels, size, size]},
            "batch": {
                "n": 4,
                "data": [0.0] * (4 * 28 * 28),
                "data_shape": [4, 1, 28, 28],
                "labels": [0, 1, 2, 3],
                "labels_shape": [4],
            },
        }

    async def drop(self, session_id):
        self.sessions.pop(session_id, None)
