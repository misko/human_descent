import logging
import math
from typing import List

import cairo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import pygame.midi
import torch
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *

from hudes.opengl_func import (
    create_grid_indices,
    create_grid_points_with_colors,
    create_matplotlib_texture_rgba,
    create_surface_grid_indices,
    create_surface_grid_points,
    create_texture_rgba,
    draw_red_sphere,
    render_text_2d,
    render_texture_rgba,
    update_grid_cbo,
    update_grid_vbo,
)

# backend = "Agg"
# backend = "cairo"
# matplotlib.use(backend)
plt_backend = matplotlib.get_backend()

import matplotlib.style as mplstyle

mplstyle.use("fast")


def surface_to_npim(surface):
    """Transforms a Cairo surface into a numpy array."""
    im = +np.frombuffer(surface.get_data(), np.uint8)
    H, W = surface.get_height(), surface.get_width()
    im.shape = (H, W, 4)  # for RGBA
    return im[:, :, :3]


def svg_to_npim(svg_bytestring, dpi):
    """Renders a svg bytestring as a RGB image in a numpy array"""
    tree = cairosvg.parser.Tree(bytestring=svg_bytestring)
    surf = cairosvg.surface.PNGSurface(tree, None, dpi).cairo
    return surface_to_npim(surf)


# Shader creation helper functions
def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # Check for compilation errors
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation error: {error}")
    return shader


def create_shader_program(vertex_source, fragment_source):
    program = glCreateProgram()

    # Create vertex and fragment shaders
    vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_source)
    fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_source)

    # Attach shaders to the program
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)

    # Link the program
    glLinkProgram(program)

    # Check for linking errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking error: {error}")

    # Clean up shaders (they are now linked into the program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


# Vertex and fragment shader source code (compatible with OpenGL 2.1)
vertex_shader_src = """
#version 120
attribute vec3 vertexPosition;  // Position of each vertex (x, y, z)

void main() {
    gl_Position = vec4(vertexPosition, 1.0);  // Set the vertex position
}
"""

fragment_shader_src = """
#version 120
void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Set all fragments to red
}
"""


def norm_mesh(mesh_grid):
    # self.mesh_grid /= 3
    # self.origin_loss = mesh_grid[self.grid_size // 2, self.grid_size // 2]
    grid_size = mesh_grid.shape[1]
    origin_loss = mesh_grid[grid_size // 2, grid_size // 2].item()
    mesh_grid -= origin_loss  # mesh_grid.mean()
    mesh_grid /= (mesh_grid.std() + 1e-3) * 4
    # mesh_grid = mesh_grid.sign() * (mesh_grid.abs() + 1).log()
    return mesh_grid


# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

    # pygame.init()
    # pygame.display.set_mode(self.window_size)
    return pg.display.get_surface()


def _print_device_info():
    for i in range(pygame.midi.get_count()):
        r = pygame.midi.get_device_info(i)
        (interf, name, input, output, opened) = r

        in_out = ""
        if input:
            in_out = "(input)"
        if output:
            in_out = "(output)"

        print(
            "%2i: interface :%s:, name :%s:, opened :%s:  %s"
            % (i, interf, name, opened, in_out)
        )


class View:
    def __init__(self, use_midi=False):

        pg.init()

        pg.midi.init()

        if use_midi:
            self.midi_input_id = pygame.midi.get_default_input_id()
            print(f"using input_id :{self.midi_input_id}:")
            self.midi_input = pygame.midi.Input(self.midi_input_id)

        # dpi = 200  # plt.rcParams["figure.dpi"]
        logging.info(f"Matplotlib backend: {plt.get_backend()}")

        self.window_size = (1200, 800)
        # self.fig = plt.figure(
        #     figsize=(self.window_size[0] / dpi, self.window_size[1] / dpi), dpi=dpi
        # )
        self.fig = plt.figure(figsize=(12, 8), facecolor="white")

        # if self.fig.dpi != dpi:
        #     logging.warning(
        #         f"DPI flag not respected by matplotlib backend ({plt.get_backend()})! Should be {dpi} but is {self.fig.dpi} "
        #     )
        #     self.window_size = (
        #         int(self.fig.get_figwidth() * self.fig.dpi),
        #         int(self.fig.get_figheight() * self.fig.dpi),
        #     )

        self.window = pg.display.set_mode(self.window_size)

        # self.canvas = agg.FigureCanvasAgg(self.fig)
        self.canvas = self.fig.canvas
        self.renderer = self.canvas.get_renderer()
        if "cairo" in plt_backend.lower():
            self.surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, self.window_size[0], self.window_size[1]
            )
            ctx = cairo.Context(self.surface)  # pg.display.get_surface())
            self.renderer.set_context(ctx)
        self.fig.subplots_adjust(
            left=0.07, right=0.95, hspace=0.8, top=0.92, bottom=0.07, wspace=0.5  # 0.5
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

        self.best_score = math.inf

        self.redraw_confusion = True
        self.redraw_dims = True
        self.redraw_preds = True
        self.redraw_step_size = True
        self.redraw_top = True
        self.redraw_train_and_val = True

        for _ax in self.axd:
            self.axd[_ax].redraw = True

    def update_examples(self, train_data: torch.Tensor):
        for idx, _ax in enumerate(("F", "G", "H", "M")):
            ax = self.axd[_ax]
            ax.cla()
            ax.imshow(train_data[idx])
            ax.set_title(f"Ex. {idx} img")
            ax.redraw = True

        # self.axd["I"].cla()
        # self.axd["I"].imshow(train_data[3])

    def update_top(self, best_score):
        if best_score is None:
            self.fig.suptitle("Human Descent: MNIST      Top-score: ?")
        else:
            self.fig.suptitle(f"Human Descent: MNIST      Top-score: {best_score:.5e}")
        self.redraw_top = True
        # self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_step_size(
        self, log_step_size: float, max_log_step_size: float, min_log_step_size: float
    ):
        self.axd["I"].cla()
        self.axd["I"].barh([0], log_step_size)
        self.axd["I"].set_xlim(min_log_step_size, max_log_step_size)
        self.axd["I"].set_title("log(Step size)")
        self.axd["I"].set_yticks([])
        self.axd["I"].redraw = True

    def update_confusion_matrix(self, confusion_matrix: torch.Tensor):
        self.axd["E"].cla()
        self.axd["E"].imshow(confusion_matrix)
        self.axd["E"].set_yticks(range(10))
        self.axd["E"].set_xticks(range(10))
        self.axd["E"].set_ylabel("Ground truth")
        self.axd["E"].set_xlabel("Prediction")
        self.axd["E"].set_title("Confusion matrix")
        self.axd["E"].redraw = True

    def update_dims_since_last_update(self, dims_and_steps_on_current_dims):
        self.axd["O"].cla()
        colors = [
            self.plt_colors[idx % len(self.plt_colors)]
            for idx in range(dims_and_steps_on_current_dims.shape[0])
        ]
        self.axd["O"].bar(
            range(dims_and_steps_on_current_dims.shape[0]),
            dims_and_steps_on_current_dims,
            color=colors,
        )
        self.axd["O"].set_xlabel("dimension #")
        self.axd["O"].set_ylabel("cumulative step")
        self.axd["O"].set_title("Dims and Steps")
        self.axd["O"].set_yticks([])
        self.axd["O"].redraw = True

    def update_example_preds(self, train_preds: List[float]):
        for idx, _ax in enumerate(("J", "K", "L", "N")):
            ax = self.axd[_ax]
            ax.cla()
            ax.bar(torch.arange(10), train_preds[idx])
            ax.set_title(f"Ex. {idx} pr(y)")
            ax.redraw = True

    def plot_train_and_val(
        self,
        train_losses: List[float],
        train_steps: List[int],
        val_losses: List[float],
        val_steps: List[int],
    ):
        new_best_score = min(val_losses) if len(val_losses) > 0 else -math.inf
        if new_best_score < self.best_score:
            self.best_score = new_best_score
            self.update_top(self.best_score)

        n = len(train_losses)
        # x = torch.arange(n)
        self.axd["B"].cla()
        self.axd["B"].plot(train_steps, train_losses, label="train")
        self.axd["B"].plot(val_steps, val_losses, label="val")
        self.axd["B"].legend(loc="upper right")
        self.axd["B"].set_title("Loss")
        self.axd["B"].set_xlabel("Step")
        self.axd["B"].set_ylabel("Loss")
        self.axd["B"].redraw = True

        # self.axd["C"].cla()
        # self.axd["C"].plot(train_steps[n // 2 :], train_losses[n // 2 :], label="train")
        # self.axd["C"].set_title("Loss [half time]")
        # self.axd["C"].set_xlabel("Step")
        # self.axd["C"].set_yticks([])

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

    def draw(self):

        logging.debug("hudes_client: redraw")
        # cairo
        # np.frombuffer(self.canvas._get_printed_image_surface().get_data(),np.uint8)
        # https://www.pygame.org/wiki/CairoPygame
        if "cairo" in plt_backend.lower():

            # self.renderer.gc.ctx = fake_ctx()
            # self.canvas.draw()
            if True:
                self.draw_or_restore()
                surf = pygame.image.frombuffer(
                    self.surface.get_data(), self.window_size, "RGBA"
                )
                self.screen.blit(surf, (0, 0))
            # self.draw_or_restore()
            # self.renderer = self.canvas._renderer

            # self.draw_or_restore()
            if False:
                self.canvas.draw()
                surf = pg.image.frombuffer(
                    # self.renderer.tostring_rgb(),
                    self.canvas._get_printed_image_surface().get_data(),
                    self.window_size,
                    "RGBA",
                )
                self.screen.blit(surf, (0, 0))
        else:  # backend.lower()=='agg':
            # self.canvas.draw()
            # self.canvas.update()
            # breakpoint()
            self.renderer.clear()
            self.draw_or_restore()
            # if self.redraw_train_and_val:
            #     self.axd["B"].draw(self.renderer)
            #     self.axd["B"].cache = self.fig.canvas.copy_from_bbox(
            #         self.axd["B"].get_tightbbox(self.renderer)
            #     )
            #     self.axd["D"].draw(self.renderer)
            #     self.axd["D"].cache = self.fig.canvas.copy_from_bbox(self.axd["D"].bbox)
            #     self.redraw_train_and_val = False
            # else:
            #     self.fig.canvas.restore_region(self.axd["B"].cache)
            #     self.fig.canvas.restore_region(self.axd["D"].cache)
            # TODO TRY FORM BUFFER AND USE THE TOSTRING BUFFER!!!
            surf = pg.image.frombytes(
                self.renderer.tostring_rgb(),
                self.window_size,
                "RGB",
            )
            # surf = pg.image.frombuffer(
            #     self.renderer.buffer_rgba(),
            #     self.window_size,
            #     "RGBA",
            # )
            self.screen.blit(surf, (0, 0))
        # else:

        pg.display.flip()  # draws whole screen vs update that draws a parts

        logging.debug("hudes_client: redraw done")


def norm_deg(x):
    return (x + 180) % 360 - 180


class OpenGLView:
    def __init__(self, grid_size, grids):

        pg.init()
        pg.font.init()
        # display = (800, 600)

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
            0.0  # Adjust this value for more or less space between grids
        )
        self.grid_width = (
            self.grid_size * self.spacing
        )  # Actual width of each grid, taking point spacing into account

        # Calculate total width of all grids with spacing between them
        self.total_width = (
            self.effective_grids * self.grid_width
            + (self.effective_grids - 1) * self.grid_spacing
        )

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

        # # Enable vertex attribute 0 (positions)
        # glEnableClientState(GL_VERTEX_ARRAY)
        # glVertexPointer(3, GL_FLOAT, 0, None)

        # Enable vertex arrays and color arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Mouse control variables
        self.is_mouse_dragging = False
        self.last_mouse_pos = None
        self.rotation_x, self.rotation_y = 0.0, 0.0  # Initialize rotation angles
        self.sensitivity = 0.5  # Mouse sensitivity for controlling rotation

        self.running = True
        self.default_angleV = 20
        self.max_angleV = 40
        self.angleH = 0.0
        self.angleV = self.default_angleV
        self.origin_loss = 0.0
        self.target = (0.0, 0.0, 0.0)

        # init plt
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(12, 2), facecolor="none")
        self.fig.subplots_adjust(
            left=0.07, right=0.95, hspace=0.8, top=0.80, bottom=0.1, wspace=0.5  # 0.5
        )
        self.axd = self.fig.subplot_mosaic(
            #    AAAAAA
            """
                BBDDEEFG
                BBDDEEJK
                """
        )

        # self.screen = pg.display.get_surface()

        # # # Step 2: Create the Matplotlib figure
        # self.fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
        # x = np.linspace(0, 10, 100)
        # y = np.sin(x)
        # ax.plot(x, y)
        # ax.set_title("Sine Wave")

    def update_examples(self, train_data: torch.Tensor):
        self.axd["F"].cla()
        self.axd["F"].imshow(train_data[0])
        self.axd["F"].set_title("Ex. 1 img")

        self.axd["G"].cla()
        self.axd["G"].imshow(train_data[1])
        self.axd["G"].set_title("Ex. 2 img")

    def update_top(self, best_score):
        if best_score is None:
            self.fig.suptitle("Human Descent: MNIST      Top-score: ?")
        else:
            self.fig.suptitle(f"Human Descent: MNIST      Top-score: {best_score:.5e}")
        # self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_step_size(
        self, log_step_size: float, max_log_step_size: float, min_log_step_size: float
    ):
        # self.axd["I"].cla()
        # self.axd["I"].bar([0], log_step_size)
        # self.axd["I"].set_ylim(min_log_step_size, max_log_step_size)
        # self.axd["I"].set_title("log(Step size)")
        # self.axd["I"].set_xticks([])
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
        best_score = min(val_losses) if len(val_losses) > 0 else -math.inf
        self.update_top(best_score)

        n = len(train_losses)
        # x = torch.arange(n)
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

        # if self.grids > 1:
        #     normalized_grids.append(norm_mesh(mesh_grids.sum(axis=0)).unsqueeze(0))
        # for grid_idx in range(mesh_grids.shape[0]):
        #     normalized_grids.append(
        #         norm_mesh(mesh_grids[grid_idx].clone()).unsqueeze(0)
        #     )  # might not need clone if we are safe here

        # if self.grids > 1:
        #    normalized_grids.append(norm_mesh(mesh_grids.sum(axis=0)).unsqueeze(0))

        origin_loss = mesh_grids[0, self.grid_size // 2, self.grid_size // 2].item()
        mesh_grids -= origin_loss
        # _u, _std, _mx = mesh_grids.mean(), mesh_grids.var()
        _mx = mesh_grids.abs().max()
        eps = 1e-3
        mesh_grids *= self.grid_width / (_mx + eps)
        # for grid_idx in range(mesh_grids.shape[0]):
        #     normalized_grids.append(
        #         # ((mesh_grids[grid_idx] - _u) / (_std + eps)).unsqueeze(0)
        #         self.grid_width
        #         * ((mesh_grids[grid_idx] - _u) / (_mx + eps)).unsqueeze(0)
        #     )  # might not need clone if we are safe here

        # # origin_loss = mesh_grid[grid_size // 2, grid_size // 2].item()

        # self.normalized_grids = torch.concatenate(normalized_grids, dim=0)
        self.normalized_grids = mesh_grids
        # breakpoint()
        # Define the center point (where the red sphere is located) and relative target position (A, B)
        center_row, center_col = self.grid_size // 2, self.grid_size // 2

        # Relative target offset (A, B) in grid units
        A, B = 0, 5  # Example: One cell to the right of the red sphere

        # Convert relative target (A, B) to absolute grid coordinates
        target_row = center_row + A
        target_col = center_col + B

        self.target = (0, 1, 0)
        # # Ensure target is within bounds of the height_map
        # if 0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size:
        #     # Convert grid position to 3D coordinates
        #     target_x = (target_row - (self.grid_size - 1) / 2.0) * self.spacing
        #     target_y = self.origin_loss  # 3  # self.mesh_grid[-target_row, -target_col]
        #     target_z = (target_col - (self.grid_size - 1) / 2.0) * self.spacing
        #     self.target = (target_x, target_y, target_z)  # Target grid point
        # else:
        #     self.target = (0, self.origin_loss, 0)  # Default to center if out of bounds

        # self.target = (target_x, target_y, target_z)  # Target grid point
        self.update_points_and_colors()

    def update_points_and_colors(self):
        if self.use_surface:
            new_points = create_surface_grid_points(self.normalized_grids, self.spacing)
        else:
            # new_points_old = create_grid_points(self.normalized_grids, self.spacing)
            new_points, new_colors = create_grid_points_with_colors(
                self.normalized_grids,
                self.spacing,
                self.grid_colors,
                selected_grid=self.selected_grid,
            )
        # breakpoint()
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

    def adjust_angles(self, angle_H, angle_V):
        self.angleH += angle_H
        self.angleV += angle_V
        self.angleV = norm_deg(self.angleV)  # % 360
        self.angleH = norm_deg(self.angleH)  # % 360
        self.angleV = np.sign(self.angleV) * min(np.abs(self.angleV), self.max_angleV)

    def reset_angle(self):
        self.angleH = 0
        self.angleV = self.default_angleV

    def draw_all_text(self):

        render_text_2d(
            "Human Descent: MNIST", 36, self.window_size[0], self.window_size[1]
        )

    def draw(self):

        # glClearColor(1.0, 1.0, 1.0, 1.0)  # Set the clear color to white
        # Handle mouse motion for rotation
        if self.is_mouse_dragging:
            mouse_pos = pg.mouse.get_pos()
            if self.last_mouse_pos:

                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]

                # Update rotation angles based on mouse movement
                self.rotation_x += dy * self.sensitivity
                self.rotation_y += dx * self.sensitivity

            self.last_mouse_pos = mouse_pos

        # Rendering Loop
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        self.camera_distance = self.total_width * self.scale_factor
        glTranslatef(0, 0.0, -self.camera_distance)

        # Apply rotations
        glRotatef(self.angleV, 1, 0, 0)
        glRotatef(self.angleH, 0, 1, 0)

        # Translate to center the grids
        # -3 for now, moves it up
        glTranslatef(-self.total_width / 2.0 + self.grid_width / 2, -2, 0.0)

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
            glTranslatef(k * (self.grid_width + self.grid_spacing), 0.0, 0.0)

            glRotatef(self.angleV, 1, 0, 0)  # Rotate to visualize the grid
            glRotatef(self.angleH, 0, 1, 0)  # Rotate to visualize the grid
            # Calculate the offset for the current grid's points in the VBO

            # Calculate the offset for the current grid's points and color data
            vertex_offset = (
                k * self.grid_size * self.grid_size * 3 * 4
            )  # For 3 floats per vertex
            color_offset = (
                k * self.grid_size * self.grid_size * 4 * 4
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
            # draw_red_plane(
            #     0.0,
            #     grid_size=self.grid_size,
            #     spacing=self.spacing,
            # )

            # Restore the previous matrix state
            glPopMatrix()

        # Disable vertex and color arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # self.draw_all_text()
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
