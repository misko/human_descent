import pytest

from hudes import hudes_pb2 as pb
from hudes.control_security import validate_control


def configured_message(**overrides):
    values = {
        "seed": 42,
        "dims_at_a_time": 6,
        "mesh_grid_size": 31,
        "mesh_step_size": 0.1,
        "mesh_grids": 3,
        "batch_size": 32,
        "dtype": "float32",
    }
    values.update(overrides)
    return pb.Control(type=pb.Control.CONTROL_CONFIG, config=pb.Config(**values))


@pytest.mark.parametrize(
    "values",
    [
        {"dtype": "cuda"},
        {"batch_size": -1},
        {"batch_size": 2**30},
        {"mesh_grid_size": 30},
        {"loss_lines": 2**30},
        {"mesh_step_size": float("nan")},
        {"dims_at_a_time": -1},
    ],
)
def test_rejects_configuration_that_could_crash_shared_inference(values):
    with pytest.raises(ValueError):
        validate_control(configured_message(**values))


def test_accepts_normal_browser_configuration():
    validate_control(configured_message())


def test_rejects_nonfinite_dimension_steps():
    msg = pb.Control(
        type=pb.Control.CONTROL_DIMS,
        dims_and_steps=[pb.DimAndStep(dim=0, step=float("inf"))],
    )
    with pytest.raises(ValueError):
        validate_control(msg)


def test_rejects_unbounded_sgd_work():
    with pytest.raises(ValueError):
        validate_control(pb.Control(type=pb.Control.CONTROL_SGD_STEP, sgd_steps=2**30))
