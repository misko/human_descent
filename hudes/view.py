import math
from typing import List

import matplotlib

from hudes.opengl_func import (
    create_grid_indices,
    create_grid_points,
    create_surface_grid_indices,
    create_surface_grid_points,
    draw_arrow,
    draw_red_plane,
    draw_red_sphere,
    update_grid_vbo,
)

matplotlib.use("Agg")
import time

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import torch
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *


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
varying vec4 vertexColor;  // Output color to the fragment shader

void main() {
    gl_Position = vec4(vertexPosition, 1.0);  // Pass the vertex position to the pipeline

    // Color based on height (y value)
    if (vertexPosition.y < 0.0) {
        vertexColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red for negative heights
    } else {
        vertexColor = vec4(0.0, 1.0, 1.0, 1.0);  // Cyan for non-negative heights
    }
}
"""

fragment_shader_src = """
#version 120

varying vec4 vertexColor;  // Input color from the vertex shader

void main() {
    gl_FragColor = vertexColor;  // Simply pass the vertex color to the fragment
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


class View:
    def __init__(self):

        pg.init()
        self.window_size = (1400, 1000)
        self.window = pg.display.set_mode(self.window_size)
        self.fig = plt.figure(
            figsize=(12, 8),
        )

        self.fig.subplots_adjust(
            left=0.07, right=0.95, hspace=0.8, top=0.92, bottom=0.07, wspace=0.5  # 0.5
        )
        self.axd = self.fig.subplot_mosaic(
            #    AAAAAA
            """
                BBCCDD
                BBCCDD
                EEFGHI
                EEJKLI
                """
        )
        # self.fig.tight_layout()

        self.screen = pg.display.get_surface()

    def update_examples(self, train_data: torch.Tensor, val_data: torch.Tensor):
        self.axd["F"].cla()
        self.axd["F"].imshow(train_data[0])
        self.axd["F"].set_title("Ex. 1 img")

        self.axd["G"].cla()
        self.axd["G"].imshow(train_data[1])
        self.axd["G"].set_title("Ex. 2 img")

        self.axd["H"].cla()
        self.axd["H"].imshow(train_data[2])
        self.axd["H"].set_title("Ex. 3 img")

        # self.axd["I"].cla()
        # self.axd["I"].imshow(train_data[3])

    def update_top(self, best_score):
        if best_score is None:
            self.fig.suptitle("Human Descent: MNIST      Top-score: ?")
        else:
            self.fig.suptitle(f"Human Descent: MNIST      Top-score: {best_score:.5e}")
        # self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_step_size(
        self, log_step_size: float, max_log_step_size: float, min_log_step_size: float
    ):
        self.axd["I"].cla()
        self.axd["I"].bar([0], log_step_size)
        self.axd["I"].set_ylim(min_log_step_size, max_log_step_size)
        self.axd["I"].set_title("log(Step size)")
        self.axd["I"].set_xticks([])

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

        self.axd["L"].cla()
        self.axd["L"].bar(torch.arange(10), train_preds[2])
        self.axd["L"].set_title("Ex. 3 pr(y)")

        # self.axd["M"].cla()
        # self.axd["M"].bar(torch.arange(10), train_preds[3])

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

        self.axd["C"].cla()
        self.axd["C"].plot(train_steps[n // 2 :], train_losses[n // 2 :], label="train")
        self.axd["C"].set_title("Loss [half time]")
        self.axd["C"].set_xlabel("Step")
        self.axd["C"].set_yticks([])

        self.axd["D"].cla()
        self.axd["D"].plot(train_steps[-8:], train_losses[-8:], label="train")
        self.axd["D"].set_title("Loss [last 8steps]")
        self.axd["D"].set_yticks([])
        self.axd["D"].set_xlabel("Step")

    def draw(self):
        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        surf = pg.image.frombytes(
            canvas.get_renderer().tostring_rgb(), self.window_size, "RGB"
        )
        self.screen.blit(surf, (0, 0))
        pg.display.flip()  # draws whole screen vs update that draws a parts


def norm_deg(x):
    return (x + 180) % 360 - 180


class OpenGLView:
    def __init__(self, grid_size, grids):

        pg.init()
        display = (800, 600)
        self.grids = grids
        self.effective_grids = grids  # 1 if grids > 1 else grids
        self.spacing = 0.1
        self.grid_size = grid_size

        self.alpha = 0.7
        self.alpha_dim = 0.2
        self.colors = [
            (0.0, 1.0, 1.0, self.alpha),  # Cyan
            (1.0, 0.0, 1.0, self.alpha),  # Magenta
            (1.0, 1.0, 0.0, self.alpha),  # Yellow
            (0.0, 1.0, 0.0, self.alpha),  # Green
            (1.0, 0.5, 0.0, self.alpha),  # Orange
        ]
        self.dim_colors = [(*x[:-1], self.alpha_dim) for x in self.colors]

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
        self.scale_factor = 1.0  # Adjust this value to control zoom level
        self.camera_distance = self.total_width * self.scale_factor

        pg.display.set_mode(display, DOUBLEBUF | OPENGL)
        # Assuming the new create_grid_vbo function and height maps are set
        # to have 'grids' number of height maps of size 'grid_size x grid_size'

        # Set up the viewport, projection matrix, and modelview matrix
        glViewport(0, 0, display[0], display[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, display[0] / display[1], 0.1, 50.0)  # Set up perspective

        # Switch to model view for object rendering
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

        # Create shader program
        self.shader_program = create_shader_program(
            vertex_shader_src, fragment_shader_src
        )

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
            self.points = create_grid_points(
                np.zeros((self.effective_grids, grid_size, grid_size)), self.spacing
            )
            self.indices = create_grid_indices(
                np.zeros((self.effective_grids, grid_size, grid_size)), self.spacing
            )

        # Create and bind Vertex Buffer Object (VBO)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes, self.points, GL_STATIC_DRAW)

        # Create and bind Element Buffer Object (EBO)
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW
        )

        # Enable vertex attribute 0 (positions)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)

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

    def update_examples(self, train_data: torch.Tensor, val_data: torch.Tensor):
        pass

    def update_top(self, best_score):
        pass

    def update_step_size(
        self, log_step_size: float, max_log_step_size: float, min_log_step_size: float
    ):
        pass

    def update_confusion_matrix(self, confusion_matrix: torch.Tensor):
        pass

    def update_example_preds(self, train_preds: List[float]):
        pass

    def plot_train_and_val(
        self,
        train_losses: List[float],
        train_steps: List[int],
        val_losses: List[float],
        val_steps: List[int],
    ):
        pass

    def update_mesh_grids(self, mesh_grids):
        # breakpoint()
        # breakpoint()
        # print(mesh_grid)
        self.raw_mesh_grids = mesh_grids
        normalized_grids = []

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

        if self.use_surface:
            new_points = create_surface_grid_points(self.normalized_grids, self.spacing)
        else:
            new_points = create_grid_points(self.normalized_grids, self.spacing)

        update_grid_vbo(self.vbo, new_points)

    def get_angles(self):
        return self.angleH, self.angleV

    def get_selected_grid(self):
        return self.selected_grid

    def increment_selected_grid(self):
        self.selected_grid = (self.selected_grid + 1) % self.effective_grids

    def decrement_selected_grid(self):
        self.selected_grid = (self.selected_grid - 1) % self.effective_grids

    def adjust_angles(self, angle_H, angle_V):
        self.angleH += angle_H
        self.angleV += angle_V
        self.angleV = norm_deg(self.angleV)  # % 360
        self.angleH = norm_deg(self.angleH)  # % 360
        print(self.angleH, self.angleV)
        self.angleV = np.sign(self.angleV) * min(np.abs(self.angleV), self.max_angleV)

    def reset_angle(self):
        self.angleH = 0
        self.angleV = self.default_angleV

    def draw(self):

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
        # Define the spacing between grids (this is separate from point spacing in the grid)

        # Position the camera based on the calculated distance
        glTranslatef(0, 0.0, -self.camera_distance)

        # Apply rotations
        # glRotatef(20, 1.0, 0.0, 0.0)
        # glRotatef(self.rotation_x, 1, 0, 0)  # Rotate around the X-axis
        # glRotatef(self.rotation_y, 0, 1, 0)  # Rotate around the Y-axis
        # glRotatef(self.angleH, 0, 1, 0)  # Rotate to visualize the grid
        # glRotatef(self.angleV, 1, 0, 0)  # Rotate to visualize the grid

        # Set color for semi-transparent grids

        # Translate to center the grids
        glTranslatef(-self.total_width / 2.0 + self.grid_width / 2, 0.0, 0.0)

        # Draw each grid individually by offsetting the points
        for k in range(self.effective_grids):

            # Save the current matrix state
            glPushMatrix()

            if k == self.selected_grid:
                glColor4f(
                    *self.colors[k % len(self.colors)]
                )  # Semi-transparent white color
            else:
                glColor4f(
                    *self.dim_colors[k % len(self.colors)]
                )  # Semi-transparent white color

            # Translate the grid by 'grid_spacing' along the X-axis, accounting for grid width
            glTranslatef(k * (self.grid_width + self.grid_spacing), 0.0, 0.0)

            glRotatef(self.angleV, 1, 0, 0)  # Rotate to visualize the grid
            glRotatef(self.angleH, 0, 1, 0)  # Rotate to visualize the grid
            # Calculate the offset for the current grid's points in the VBO
            offset = (
                k * self.grid_size * self.grid_size * 3 * 4
            )  # Multiply by 4 to account for float32 size

            offset = k * self.grid_size * self.grid_size * 3 * 4

            # Set up the vertex pointer for the current grid's points
            glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(offset))

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
            # Use the shader program
            # glUseProgram(self.shader_program)
            # Draw the grid using the indices and the point offset

            # Draw the grid as a surface using triangles

            draw_red_sphere(0.0)
            draw_red_plane(
                0.0,
                grid_size=self.grid_size,
                spacing=self.spacing,
            )

            # Restore the previous matrix state
            glPopMatrix()

        # Draw the faint red plane
        # draw_red_sphere(self.origin_loss)
        # draw_arrow((0, self.origin_loss, 0), self.target)

        # Draw the coordinate axes
        # draw_axes()

        # # Render the axis labels
        # glMatrixMode(GL_PROJECTION)
        # glPushMatrix()
        # glLoadIdentity()
        # gluOrtho2D(0, display[0], 0, display[1])
        # glMatrixMode(GL_MODELVIEW)
        # glPushMatrix()
        # glLoadIdentity()

        # # Render X and Y axis labels
        # render_text("X", (750, 300), color=(0, 0, 255))  # Label for X axis
        # render_text("Y", (400, 550), color=(0, 0, 255))  # Label for Y axis

        # glPopMatrix()
        # glMatrixMode(GL_PROJECTION)
        # glPopMatrix()
        # glMatrixMode(GL_MODELVIEW)

        pg.display.flip()
        pg.time.wait(10)
