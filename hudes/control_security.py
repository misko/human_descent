"""Validate public game input before scheduling work in the inference process."""

import math

from hudes import hudes_pb2

MAX_CLIENTS = 64
MAX_PENDING_DIMS = 128


def validate_control(msg):
    if not msg.IsInitialized():
        raise ValueError("incomplete control message")
    if msg.type == hudes_pb2.Control.CONTROL_CONFIG:
        if msg.HasField("team_mesh_step_size"):
            if msg.HasField("config") or not (
                math.isfinite(msg.team_mesh_step_size)
                and 0 < msg.team_mesh_step_size <= 100
            ):
                raise ValueError("invalid team mesh step size")
            return
        c = msg.config
        if not msg.HasField("config") or c.dtype not in {"float16", "float32"}:
            raise ValueError("invalid inference configuration")
        if not (
            1 <= c.dims_at_a_time <= 128
            and 5 <= c.mesh_grid_size <= 401
            and c.mesh_grid_size % 2 == 1
            and 0 <= c.mesh_grids <= 32
            and 1 <= c.batch_size <= 512
            and 0 <= c.loss_lines <= 128
            and math.isfinite(c.mesh_step_size)
            and 0 <= c.mesh_step_size <= 1000
            and len(c.client_session_token) <= 256
        ):
            raise ValueError("inference configuration exceeds limits")
    elif msg.type == hudes_pb2.Control.CONTROL_DIMS:
        if len(msg.dims_and_steps) > MAX_PENDING_DIMS or any(
            not (0 <= d.dim <= 1_000_000)
            or not math.isfinite(d.step)
            or abs(d.step) > 1_000_000
            for d in msg.dims_and_steps
        ):
            raise ValueError("invalid dimension update")
    elif msg.type == hudes_pb2.Control.CONTROL_SGD_STEP:
        if not 0 <= msg.sgd_steps <= 32:
            raise ValueError("SGD request exceeds limits")
    elif msg.type == hudes_pb2.Control.CONTROL_TEAM_ADMIT:
        if not msg.HasField("team_admission") or any(
            not value or len(value) > 256
            for value in (
                msg.team_admission.match_id,
                msg.team_admission.session_id,
                msg.team_admission.player_id,
                msg.team_admission.token,
            )
        ):
            raise ValueError("invalid team admission")
