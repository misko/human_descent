import logging
import os
from typing import List

# import cairo # uncomment if trying cairo
import matplotlib

matplotlib.use("Agg")  # also kinda works with Cairo , else its ?
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pygame as pg
import pygame.midi
import torch
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *

from hudes.hud_overlay import render_bottom_hud
from hudes.opengl_func import (
    create_grid_indices,
    create_grid_points_with_colors,
    create_matplotlib_texture_rgba,
    create_surface_grid_indices,
    create_surface_grid_points,
    create_texture_rgba,
    draw_red_sphere,
    load_texture,
    render_text_2d,
    render_texture_rgba,
    update_grid_cbo,
    update_grid_vbo,
)

mplstyle.use("fast")

plt_backend = matplotlib.get_backend()


def norm_mesh(mesh_grid):
    # self.mesh_grid /= 3
    # self.origin_loss = mesh_grid[self.grid_size // 2, self.grid_size // 2]
    grid_size = mesh_grid.shape[1]
    origin_loss = mesh_grid[grid_size // 2, grid_size // 2].item()
    mesh_grid -= origin_loss  # mesh_grid.mean()
    mesh_grid /= (mesh_grid.std() + 1e-3) * 4
    # mesh_grid = mesh_grid.sign() * (mesh_grid.abs() + 1).log()
    return mesh_grid


class View:
    def __init__(self, use_midi=False):
        pg.midi.init()

        pg.key.set_repeat(100)
        if use_midi:
            device_count = pygame.midi.get_count()
            logging.info(f"Number of MIDI devices found: {device_count}")
            # List available MIDI devices
            for device_id in range(device_count):
                device_info = pygame.midi.get_device_info(device_id)
                # device_info returns a tuple (interface, name, input, output, opened)
                interface, name, is_input, is_output, opened = device_info
                if is_input:
                    name = name.decode("utf-8")
                    logging.info(f"Device ID {device_id}: {name}")
                    if "x-touch" in name.lower() or "xtouch" in name.lower():
                        self.midi_input_id = device_id
            logging.info(f"using input_id :{self.midi_input_id}:")
            self.midi_input = pygame.midi.Input(self.midi_input_id)

        logging.info(f"Matplotlib backend: {plt.get_backend()}")

        self.window_size = (1200, 800)
        self.fig = plt.figure(figsize=(12, 8), facecolor="white")

        self.window = pg.display.set_mode(self.window_size)

        self.canvas = self.fig.canvas
        self.renderer = self.canvas.get_renderer()
        if "cairo" in plt_backend.lower():
            self.surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, self.window_size[0], self.window_size[1]
            )
            ctx = cairo.Context(self.surface)  # pg.display.get_surface())
            self.renderer.set_context(ctx)
        self.fig.subplots_adjust(
            left=0.07,
            right=0.95,
            hspace=0.8,
            top=0.92,
            bottom=0.07,
            wspace=0.5,  # 0.5
        )
        self.axd = self.fig.subplot_mosaic(
            #    AAAAAA
            """
                BBDDII
                BBDDOO
                EEFGHM
                EEJKLN
                """
        )

        self.screen = pg.display.get_surface()
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.plt_colors = prop_cycle.by_key()["color"]

        self.redraw_confusion = True
        self.redraw_dims = True
        self.redraw_preds = True
        self.redraw_step_size = True
        self.redraw_top = True
        self.redraw_train_and_val = True

        self.confusion_matrix_init = False
        self.example_im_show_init = False
        self.example_imshow_init = False
        self.dims_chart_init = False
        self.init_step_size_plot = False

        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 46)
        for _ax in self.axd:
            self.axd[_ax].redraw = True

    def update_examples(self, train_data: torch.Tensor):
        if not self.example_im_show_init:
            for idx, _ax in enumerate(("F", "G", "H", "M")):
                ax = self.axd[_ax]
                ax.cla()
                ax.im = ax.imshow(train_data[idx])
                ax.set_title(f"Ex. {idx} img")
                ax.redraw = True
            self.example_im_show_init = True
        else:
            for idx, _ax in enumerate(("F", "G", "H", "M")):
                ax = self.axd[_ax]
                if idx < len(train_data):
                    ax.im.set_data(train_data[idx])
                ax.redraw = True

    def update_top(self):
        time = pg.time.get_ticks() / 1000
        render_str = f"Hudes: MNIST Score: {self.client_state.best_score:.3f} BSize: {self.client_state.batch_size} {self.client_state.dtype} SGD: {self.client_state.sgd_steps} Time: {time:0.1f} Dims: {self.client_state.dims_used}"
        self.top_title_rendered = self.font.render(render_str, False, (0, 0, 0))

    def update_step_size(self):
        if not self.init_step_size_plot:
            self.axd["I"].cla()
            self.axd["I"].bars = self.axd["I"].barh(
                [0], self.client_state.log_step_size
            )
            self.axd["I"].set_xlim(
                self.client_state.min_log_step_size, self.client_state.max_log_step_size
            )
            self.axd["I"].set_title("log(Step size)")
            self.axd["I"].set_yticks([])
            self.init_step_size_plot = True
        else:
            self.axd["I"].bars[0].set_width(self.client_state.log_step_size)
        self.axd["I"].redraw = True

    def update_confusion_matrix(self, confusion_matrix: torch.Tensor):
        if not self.confusion_matrix_init:
            self.axd["E"].cla()
            self.axd["E"].im = self.axd["E"].imshow(confusion_matrix)
            self.axd["E"].set_yticks(range(10))
            self.axd["E"].set_xticks(range(10))
            self.axd["E"].set_ylabel("Ground truth")
            self.axd["E"].set_xlabel("Prediction")
            self.axd["E"].set_title("Confusion matrix")
            self.confusion_matrix_init = True
        else:
            self.axd["E"].im.set_data(confusion_matrix)
        self.axd["E"].redraw = True

    def update_dims_since_last_update(self, dims_and_steps_on_current_dims):
        if not self.dims_chart_init:
            self.axd["O"].cla()
            colors = [
                self.plt_colors[idx % len(self.plt_colors)]
                for idx in range(dims_and_steps_on_current_dims.shape[0])
            ]
            self.axd["O"].bars = self.axd["O"].bar(
                range(dims_and_steps_on_current_dims.shape[0]),
                dims_and_steps_on_current_dims,
                color=colors,
            )
            self.axd["O"].set_xlabel("dimension #")
            self.axd["O"].set_ylabel("cumulative step")
            self.axd["O"].set_title("Dims and Steps")
            self.axd["O"].set_yticks([])
            self.dims_chart_init = True
        else:
            max_mag = np.abs(dims_and_steps_on_current_dims).max()
            self.axd["O"].set_ylim([-max_mag, max_mag])
            for bar, new_height in zip(
                self.axd["O"].bars, dims_and_steps_on_current_dims
            ):
                bar.set_height(new_height)
        self.axd["O"].redraw = True

    def update_example_preds(self, train_preds: List[float]):
        if not self.example_imshow_init:
            for idx, _ax in enumerate(("J", "K", "L", "N")):
                ax = self.axd[_ax]
                ax.cla()
                ax.bars = ax.bar(torch.arange(10), train_preds[idx])
                ax.set_xlim([0, 9])
                ax.set_ylim([0, 1.0])
                ax.set_title(f"Ex. {idx} pr(y)")
                ax.redraw = True
            self.example_imshow_init = True
        else:
            for idx, _ax in enumerate(("J", "K", "L", "N")):
                ax = self.axd[_ax]
                if idx < train_preds.shape[0]:
                    for bar, new_height in zip(ax.bars, train_preds[idx]):
                        bar.set_height(new_height)
                else:
                    for bar in ax.bars:
                        bar.set_height(0)
                ax.redraw = True

    def plot_train_and_val(
        self,
        train_losses: List[float],
        train_steps: List[int],
        val_losses: List[float],
        val_steps: List[int],
    ):
        self.update_top()

        self.axd["B"].cla()
        self.axd["B"].plot(train_steps, train_losses, label="train")
        self.axd["B"].plot(val_steps, val_losses, label="val")
        # self.axd["B"].axhline(y=0.5, color="m", linestyle="--", label="Amazing")
        self.axd["B"].axhline(y=1.5, color="r", linestyle="--", label="Warmer")
        self.axd["B"].axhline(y=2.3, color="black", linestyle="--", label="Random")
        self.axd["B"].legend(loc="upper right")
        self.axd["B"].set_title("Loss")
        self.axd["B"].set_xlabel("Step")
        self.axd["B"].set_ylabel("Loss")
        self.axd["B"].set_ylim([0.0, 2.5])
        self.axd["B"].redraw = True

        self.axd["D"].cla()
        self.axd["D"].plot(train_steps[-8:], train_losses[-8:], label="train")
        self.axd["D"].set_title("Loss [last 8steps]")
        self.axd["D"].set_yticks([])
        self.axd["D"].set_xlabel("Step")
        self.axd["D"].redraw = True

    def draw_or_restore(self):
        for _, ax in self.axd.items():
            if ax.redraw:
                ax.draw(self.renderer)
                ax.cache = self.fig.canvas.copy_from_bbox(
                    ax.get_tightbbox(self.renderer)  # TODO cache bbox?
                )
                ax.redraw = False
            else:
                self.fig.canvas.restore_region(ax.cache)

    def next_help_screen(self):
        self.client_state.help_screen_idx += 1
        if self.client_state.help_screen_idx >= len(self.help_screen_fns):
            self.client_state.help_screen_idx = -1  # disable help screen mode
            return

    def draw(self):
        logging.debug("hudes_client: redraw ")
        if self.client_state.help_screen_idx == -1:
            if "cairo" in plt_backend.lower():
                self.draw_or_restore()

                surf = pygame.image.frombuffer(
                    self.surface.get_data(), self.window_size, "RGBA"
                )
                self.screen.blit(surf, (0, 0))
                self.screen.blit(self.top_title_rendered, (0, 0))
            else:
                self.renderer.clear()
                self.draw_or_restore()

                surf = pg.image.frombytes(
                    self.renderer.tostring_rgb(),
                    self.window_size,
                    "RGB",
                )
                self.screen.blit(surf, (0, 0))
                self.screen.blit(self.top_title_rendered, (0, 0))
        else:
            help_screen_fn = self.client_state.help_screen_fns[
                self.client_state.help_screen_idx
            ]
            logging.debug(f"help_screen: {help_screen_fn}")
            image = pygame.image.load(
                help_screen_fn
            ).convert()  # Replace "your_image.jpg" with the path to your image
            image = pygame.transform.scale(image, self.window_size)
            self.screen.blit(image, (0, 0))

        pg.display.flip()  # draws whole screen vs update that draws a parts

        logging.debug("hudes_client: redraw done")


def norm_deg(x):
    return (x + 180) % 360 - 180


class OpenGLView:
    def __init__(self, grid_size, grids):
        # pg.init()
        pg.font.init()
        pg.key.set_repeat(70)

        self.window_size = (1200, 800)

        self.grids = grids
        self.effective_grids = grids  # 1 if grids > 1 else grids
        self.spacing = 0.1
        self.grid_size = grid_size

        self.alpha = 0.7
        self.alpha_dim = 0.2
        self.grid_colors = (
            (0.0, 1.0, 1.0, self.alpha),  # Cyan
            (1.0, 0.0, 1.0, self.alpha),  # Magenta
            (1.0, 1.0, 0.0, self.alpha),  # Yellow
            (0.0, 1.0, 0.0, self.alpha),  # Green
            (1.0, 0.5, 0.0, self.alpha),  # Orange
            (1.0, 0.0, 0.0, self.alpha),  # Red
            (0.0, 0.0, 1.0, self.alpha),  # Blue
            (0.5, 0.0, 1.0, self.alpha),  # Purple
            (0.0, 0.5, 1.0, self.alpha),  # Sky Blue
            (1.0, 0.0, 0.5, self.alpha),  # Pink
            (0.5, 1.0, 0.0, self.alpha),  # Lime
            (1.0, 0.75, 0.8, self.alpha),  # Light Pink
        )

        self.grid_dim_colors = ((*x[:-1], self.alpha_dim) for x in self.grid_colors)

        self.selected_grid = self.grids // 2

        self.grid_spacing = (
            1.0  # Adjust this value for more or less space between grids
        )
        self.grid_width = (
            self.grid_size * self.spacing
        )  # Actual width of each grid, taking point spacing into account

        self.selected_grid_multiplier = 2
        # Calculate total width of all grids with spacing between them
        self.total_width = (
            self.effective_grids + self.selected_grid_multiplier
        ) * self.grid_width + (self.effective_grids - 1) * self.grid_spacing

        # Calculate the camera distance based on total width (adjust scale factor as needed)
        if self.effective_grids == 1:
            self.scale_factor = 3.0
        else:
            self.scale_factor = 1.0  # Adjust this value to control zoom level

        # Assuming the new create_grid_vbo function and height maps are set
        # to have 'grids' number of height maps of size 'grid_size x grid_size'

        pg.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)

        # Set up the viewport, projection matrix, and modelview matrix
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            45, self.window_size[0] / self.window_size[1], 0.1, 50.0
        )  # Set up perspective

        # Switch to model view for object rendering
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.use_surface = False

        if self.use_surface:
            self.points = create_surface_grid_points(
                np.zeros((self.effective_grids, grid_size, grid_size)), self.spacing
            )

            self.indices = create_surface_grid_indices(
                np.zeros((self.effective_grids, grid_size, grid_size)), self.spacing
            )
        else:
            # Create and bind the VBO and EBO for multiple height maps
            # Create and bind Vertex Buffer Object (VBO)
            self.points, self.colors = create_grid_points_with_colors(
                np.zeros((self.effective_grids, grid_size, grid_size)),
                self.spacing,
                self.grid_colors,
                selected_grid=self.selected_grid,
            )

            self.indices = create_grid_indices(
                np.zeros((self.effective_grids, grid_size, grid_size)), self.spacing
            )

        # Create and bind VBO for vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes, self.points, GL_STATIC_DRAW)

        # Create and bind CBO (Color Buffer Object) for colors
        self.cbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
        glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_STATIC_DRAW)

        # Create and bind Element Buffer Object (EBO) for indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW
        )

        # Enable vertex arrays and color arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        self.running = True
        self.default_angleV = 20
        self.max_angleV = 25
        self.angleH = 0.0
        self.angleV = self.default_angleV
        self.origin_loss = 0.0

        # init plt
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(12, 2), facecolor="none")
        self.fig.subplots_adjust(
            left=0.07,
            right=0.95,
            hspace=0.8,
            top=0.80,
            bottom=0.1,
            wspace=0.5,  # 0.5
        )
        self.axd = self.fig.subplot_mosaic(
            #    AAAAAA
            """
                BBDDEEFG
                BBDDEEJK
                """
        )

        self.bottom_bar_data = None
        self.bottom_bar_width = 0
        self.bottom_bar_height = 0

        self.client_state = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.help_screen_fns = [
            os.path.join(current_dir, x)
            for x in [
                "help_screens/hudes_help_start.png",
                "help_screens/hudes_1.png",
                "help_screens/hudes_2.png",
                "help_screens/hudes_3.png",
                "help_screens/hudes_2d_keyboard_controls.png",
                "help_screens/hudes_2d_xbox_controls.png",
            ]
        ]

        self.old_batch_size = 0
        self.old_dtype = 0
        self.old_dims_used = 0
        self.old_best_score = 0
        self.large_text_start = 0
        self.old_sgd = 0

        self.text_str = ""

    def update_examples(self, train_data: torch.Tensor):
        self.axd["F"].cla()
        self.axd["F"].imshow(train_data[0])
        self.axd["F"].set_title("Ex. 1 img")

        self.axd["G"].cla()
        self.axd["G"].imshow(train_data[1])
        self.axd["G"].set_title("Ex. 2 img")

    def update_step_size(self):
        pass

    def update_confusion_matrix(self, confusion_matrix: torch.Tensor):
        self.axd["E"].cla()
        self.axd["E"].imshow(confusion_matrix)
        self.axd["E"].set_yticks(range(10))
        self.axd["E"].set_xticks(range(10))
        self.axd["E"].set_ylabel("Ground truth")
        self.axd["E"].set_xlabel("Prediction")
        self.axd["E"].set_title("Confusion matrix")

    def update_example_preds(self, train_preds: List[float]):
        self.axd["J"].cla()
        self.axd["J"].bar(torch.arange(10), train_preds[0])
        self.axd["J"].set_title("Ex. 1 pr(y)")

        self.axd["K"].cla()
        self.axd["K"].bar(torch.arange(10), train_preds[1])
        self.axd["K"].set_title("Ex. 2 pr(y)")

    def update_dims_since_last_update(self, dims_and_steps):
        pass

    def plot_train_and_val(
        self,
        train_losses: List[float],
        train_steps: List[int],
        val_losses: List[float],
        val_steps: List[int],
    ):
        self.axd["B"].cla()
        self.axd["B"].plot(train_steps, train_losses, label="train")
        self.axd["B"].plot(val_steps, val_losses, label="val")
        self.axd["B"].legend(loc="upper right")
        self.axd["B"].set_title("Loss")
        self.axd["B"].set_xlabel("Step")
        self.axd["B"].set_ylabel("Loss")

        self.axd["D"].cla()
        self.axd["D"].plot(train_steps[-8:], train_losses[-8:], label="train")
        self.axd["D"].set_title("Loss [last 8steps]")
        self.axd["D"].set_yticks([])
        self.axd["D"].set_xlabel("Step")

    def update_mesh_grids(self, mesh_grids):
        self.raw_mesh_grids = mesh_grids

        origin_loss = mesh_grids[0, self.grid_size // 2, self.grid_size // 2].item()
        mesh_grids -= origin_loss

        _mx = np.abs(mesh_grids).max()
        eps = 1e-3
        mesh_grids *= self.grid_width / (_mx + eps)

        self.normalized_grids = mesh_grids
        self.update_points_and_colors()

    def update_points_and_colors(self):
        if self.use_surface:
            new_points = create_surface_grid_points(self.normalized_grids, self.spacing)
        else:
            new_points, new_colors = create_grid_points_with_colors(
                self.normalized_grids,
                self.spacing,
                self.grid_colors,
                selected_grid=self.selected_grid,
            )

        start_idx = self.selected_grid * self.grid_size * self.grid_size

        new_points[
            start_idx : start_idx + self.grid_size * self.grid_size, [0, 2]
        ] *= self.selected_grid_multiplier

        update_grid_vbo(self.vbo, new_points)
        update_grid_cbo(self.cbo, new_colors)

    def get_angles(self):
        return self.angleH, self.angleV

    def get_selected_grid(self):
        return self.selected_grid

    def increase_zoom(self):
        self.scale_factor = max(0.7, self.scale_factor - 0.05)

    def decrease_zoom(self):
        self.scale_factor = min(5.0, self.scale_factor + 0.05)

    def increment_selected_grid(self):
        self.selected_grid = (self.selected_grid + 1) % self.effective_grids

    def decrement_selected_grid(self):
        self.selected_grid = (self.selected_grid - 1) % self.effective_grids

    def set_selected_grid(self, grid_idx):
        self.selected_grid = grid_idx

    def adjust_angles(self, angle_H, angle_V):
        self.angleH += 2 * angle_H
        self.angleV += 2 * angle_V
        self.angleV = norm_deg(self.angleV)  # % 360
        self.angleH = norm_deg(self.angleH)  # % 360
        self.angleV = np.sign(self.angleV) * min(np.abs(self.angleV), self.max_angleV)

    def reset_angle(self):
        self.angleH = 0
        self.angleV = self.default_angleV

    def draw_all_text(self):
        if self.client_state is not None:
            if (
                self.old_batch_size != self.client_state.batch_size
                or self.old_dtype != self.client_state.dtype
                or self.old_dims_used != self.client_state.dims_used
                or self.old_best_score != self.client_state.best_score
                or self.old_sgd != self.client_state.sgd_steps
            ):
                self.large_text_start = pg.time.get_ticks()
                self.old_batch_size = self.client_state.batch_size
                self.old_dtype = self.client_state.dtype
                self.old_dims_used = self.client_state.dims_used
                self.old_best_score = self.client_state.best_score
                self.old_sgd = self.client_state.sgd_steps
            time = pg.time.get_ticks()
            text_parts = [
                f"val:{self.client_state.best_score:.3f}",
                "bs:"
                + f"{self.client_state.batch_size} "
                + f"({self.client_state.dtype.replace('float', 'f')})",
                f"t:{time/1000:.1f}s",
                f"dims:{self.client_state.dims_used}",
                f"sgd:{self.client_state.sgd_steps}",
            ]
            text_str = " ".join(text_parts)
            if self.text_str != text_str or self.bottom_bar_data is None:
                self.text_str = text_str
                hud_surface = render_bottom_hud(text_str)
                self.bottom_bar_data = pg.image.tostring(
                    hud_surface,
                    "RGBA",
                    True,
                )
                self.bottom_bar_width = hud_surface.get_width()
                self.bottom_bar_height = hud_surface.get_height()

            if self.bottom_bar_data is not None:
                render_text_2d(
                    text_data=self.bottom_bar_data,
                    text_width=self.bottom_bar_width,
                    text_height=self.bottom_bar_height,
                    screen_width=self.window_size[0],
                    screen_height=self.window_size[1],
                )

    def next_help_screen(self):
        self.client_state.help_screen_idx += 1
        if self.client_state.help_screen_idx >= len(self.help_screen_fns):
            self.client_state.help_screen_idx = -1  # disable help screen mode
            return

    def draw(self):
        if self.client_state.help_screen_idx == -1:
            self.draw_gl()

        else:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            help_screen_fn = self.help_screen_fns[self.client_state.help_screen_idx]

            logging.debug(f"help_screen: {help_screen_fn}")

            texture_id, image_width, image_height = load_texture(
                help_screen_fn, *self.window_size
            )

            render_texture_rgba(texture_id, image_width, image_height, self.window_size)
            pg.display.flip()
            pg.time.wait(10)

    def draw_gl(self):
        # Rendering Loop
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        self.camera_distance = self.total_width * self.scale_factor
        glTranslatef(0, 0.0, -self.camera_distance)

        # Translate to center the grids
        # -3 for now, moves it up
        translate_offset = -(self.grid_size - 11) / 10
        glTranslatef(
            -self.total_width / 2.0 + self.grid_width / 2, translate_offset, 0.0
        )

        # Enable vertex arrays and color arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Bind the vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))

        # Bind the color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
        glColorPointer(4, GL_FLOAT, 0, ctypes.c_void_p(0))
        # Draw each grid individually by offsetting the points
        for k in range(self.effective_grids):
            # Save the current matrix state
            glPushMatrix()

            # Translate the grid by 'grid_spacing' along the X-axis, accounting for grid width
            offset = 0
            if k > self.selected_grid:
                offset = self.grid_width * self.selected_grid_multiplier
            elif k == self.selected_grid:
                offset = self.grid_width * self.selected_grid_multiplier / 2

            glTranslatef(
                k * (self.grid_width + self.grid_spacing) + offset,
                0.0 if k == self.selected_grid else -2,
                0.0,
            )

            glRotatef(self.angleV, 1, 0, 0)  # Rotate to visualize the grid
            glRotatef(self.angleH, 0, 1, 0)  # Rotate to visualize the grid
            # Calculate the offset for the current grid's points in the VBO

            # Calculate the offset for the current grid's points and color data
            grid_idx_at_position_k = (
                k  # (k - self.selected_grid) % self.effective_grids
            )
            vertex_offset = (
                grid_idx_at_position_k * self.grid_size * self.grid_size * 3 * 4
            )  # For 3 floats per vertex
            color_offset = (
                grid_idx_at_position_k * self.grid_size * self.grid_size * 4 * 4
            )  # For 4 floats (RGBA) per color

            # Bind the vertex buffer and specify the offset for the current grid
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(vertex_offset))

            # Bind the color buffer and specify the offset for the current grid
            glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
            glColorPointer(4, GL_FLOAT, 0, ctypes.c_void_p(color_offset))

            # Bind the VBO containing the vertex data
            glBindBuffer(GL_ARRAY_BUFFER, self.ebo)

            if self.use_surface:
                glDrawElements(
                    GL_TRIANGLES,
                    len(self.indices) // self.effective_grids,
                    GL_UNSIGNED_INT,
                    ctypes.c_void_p(0),
                )
            else:
                glDrawElements(
                    GL_LINES, len(self.indices), GL_UNSIGNED_INT, ctypes.c_void_p(0)
                )

            # Draw the grid as a surface using triangles
            draw_red_sphere(0.0)

            # Restore the previous matrix state
            glPopMatrix()

        # Disable vertex and color arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        self.draw_all_text()
        # # Render the texture from the Matplotlib figure in 2D
        window_size = pg.display.get_surface().get_size()  # Get window size

        raw_data, self.fig_width, self.fig_height = create_matplotlib_texture_rgba(
            self.fig
        )
        self.texture_id = create_texture_rgba(raw_data, self.fig_width, self.fig_height)
        render_texture_rgba(
            self.texture_id, self.fig_width, self.fig_height, window_size
        )

        pg.display.flip()
        pg.time.wait(10)
