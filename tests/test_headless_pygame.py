import asyncio
import importlib
import inspect
import os
import sys
import types
from types import SimpleNamespace

import pytest
import websockets

from hudes.controllers.keyboard_client import KeyboardClient
from hudes.websocket_server import run_wrapper


def _get_keyboard_client_gl():
    module = importlib.import_module(
        "hudes.controllers.keyboard_client_openGL"
    )
    return module.KeyboardClientGL


def test_keyboard_client_gl_run_loop_signature_accepts_limits():
    KeyboardClientGL = _get_keyboard_client_gl()
    signature = inspect.signature(KeyboardClientGL.run_loop)
    params = signature.parameters
    assert "max_frames" in params, (
        "KeyboardClientGL.run_loop missing max_frames"
    )
    assert "timeout_s" in params, (
        "KeyboardClientGL.run_loop missing timeout_s"
    )


def test_keyboard_client_gl_import_without_opengl(monkeypatch):
    module_name = "hudes.controllers.keyboard_client_openGL"
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "OpenGL", types.ModuleType("OpenGL"))
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop("OpenGL", None)
    assert hasattr(module, "KeyboardClientGL")


class DummyViewBase:
    def __init__(self):
        self.client_state = None
        self.examples_updated = 0
        self.predictions_updated = 0
        self.confusion_updated = 0
        self.dims_updated = 0
        self.step_updates = 0
        self.train_plot_updates = 0
        self.draw_calls = 0

    def update_step_size(self):
        self.step_updates += 1

    def plot_train_and_val(self, *_args, **_kwargs):
        self.train_plot_updates += 1

    def update_examples(self, *_args, **_kwargs):
        self.examples_updated += 1

    def update_example_preds(self, *_args, **_kwargs):
        self.predictions_updated += 1

    def update_confusion_matrix(self, *_args, **_kwargs):
        self.confusion_updated += 1

    def update_dims_since_last_update(self, *_args, **_kwargs):
        self.dims_updated += 1

    def draw(self):
        self.draw_calls += 1


class DummyView(DummyViewBase):
    pass


class DummyOpenGLView(DummyViewBase):
    def __init__(self, grid_size, grids):
        super().__init__()
        self.grid_size = grid_size
        self.grids = grids
        self.selected_grid = grids // 2
        self.angle_h = 0.0
        self.angle_v = 0.0
        self.mesh_updates = 0
        self.points_updates = 0
        self.last_mesh_shape = None

    def update_mesh_grids(self, mesh_grids):
        self.mesh_updates += 1
        self.last_mesh_shape = getattr(mesh_grids, "shape", None)

    def update_points_and_colors(self):
        self.points_updates += 1

    def get_angles(self):
        return self.angle_h, self.angle_v

    def get_selected_grid(self):
        return self.selected_grid

    def increment_selected_grid(self):
        self.selected_grid = (self.selected_grid + 1) % max(1, self.grids)

    def decrement_selected_grid(self):
        self.selected_grid = (self.selected_grid - 1) % max(1, self.grids)

    def adjust_angles(self, angle_h, angle_v):
        self.angle_h += angle_h
        self.angle_v += angle_v

    def reset_angle(self):
        self.angle_h = 0.0
        self.angle_v = 0.0


async def wait_for_server(host: str, port: int, timeout: float = 20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            async with websockets.connect(f"ws://{host}:{port}"):
                return
        except OSError:
            if asyncio.get_running_loop().time() > deadline:
                message = f"Server {host}:{port} did not become ready"
                raise TimeoutError(message)
            await asyncio.sleep(0.1)


def run_clients_sequence(port: int):
    KeyboardClientGL = _get_keyboard_client_gl()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame as pg

    controller_gl = None
    controller_keyboard = None

    pg.init()
    try:
        controller_gl = KeyboardClientGL(
            addr="localhost",
            port=port,
            seed=0,
            step_size_resolution=-0.005,
            inital_step_size_idx=100,
            mesh_grids=3,
            mesh_grid_size=41,
            joystick_controller_key="wireless_osxA",
            skip_help=True,
        )
        view_gl = DummyOpenGLView(grid_size=41, grids=3)
        controller_gl.attach_view(view_gl)
        controller_gl.run_loop(max_frames=600, timeout_s=10.0)

        controller_keyboard = KeyboardClient(
            addr="localhost",
            port=port,
            seed=0,
            joystick_controller_key="wireless_osxA",
            skip_help=True,
        )
        view_keyboard = DummyView()
        controller_keyboard.attach_view(view_keyboard)
        controller_keyboard.run_loop(max_frames=600, timeout_s=10.0)

        return {
            "keyboardGL": {
                "train_losses": list(controller_gl.train_losses),
                "train_steps": list(controller_gl.train_steps),
                "examples_updated": view_gl.examples_updated,
                "mesh_updates": view_gl.mesh_updates,
                "draw_calls": view_gl.draw_calls,
            },
            "keyboard": {
                "train_losses": list(controller_keyboard.train_losses),
                "train_steps": list(controller_keyboard.train_steps),
                "examples_updated": view_keyboard.examples_updated,
                "draw_calls": view_keyboard.draw_calls,
            },
        }
    finally:
        for controller in (controller_gl, controller_keyboard):
            if controller is not None:
                controller.quit()
                controller.hudes_websocket_client.thread.join(timeout=2.0)
        pg.quit()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_sh_headless_pygame():
    port = 8765

    download_args = SimpleNamespace(
        device="cpu",
        model="ffnn",
        max_batch_size=512,
        max_grids=5,
        port=port,
        max_grid_size=41,
        ssl_pem=None,
        run_in="thread",
        download_dataset_and_exit=True,
    )
    await run_wrapper(download_args)

    server_args = SimpleNamespace(
        device="mps",
        model="ffnn",
        max_batch_size=512,
        max_grids=5,
        port=port,
        max_grid_size=41,
        ssl_pem=None,
        run_in="thread",
        download_dataset_and_exit=False,
    )
    loop = asyncio.get_running_loop()
    stop_future = loop.create_future()
    server_task = asyncio.create_task(
        run_wrapper(server_args, stop_future=stop_future)
    )

    try:
        await wait_for_server("localhost", port)

        results = await asyncio.to_thread(run_clients_sequence, port)

        assert results["keyboardGL"][
            "train_losses"
        ], "keyboardGL client received no train data"
        assert results["keyboardGL"]["examples_updated"] > 0

        assert results["keyboard"][
            "train_losses"
        ], "keyboard client received no train data"
        assert results["keyboard"]["examples_updated"] > 0
    finally:
        if not stop_future.done():
            stop_future.set_result(True)
        await server_task
