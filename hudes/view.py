import math
from typing import List

import matplotlib

from hudes.opengl_func import create_grid_vbo, update_grid_vbo

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
        self.window = pg.display.set_mode((1200, 900))
        self.window_size = (1200, 800)
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


class OpenGLView:
    def __init__(self, grid_size):

        pg.init()
        display = (800, 600)

        self.spacing = 0.1
        self.grid_size = grid_size

        pg.display.set_mode(display, DOUBLEBUF | OPENGL)

        # Set up the viewport, projection matrix, and modelview matrix
        glViewport(0, 0, display[0], display[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, display[0] / display[1], 0.1, 50.0)  # Set up perspective

        # Switch to model view for object rendering
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.points, self.indices = create_grid_vbo(
            np.zeros((grid_size, grid_size)), self.spacing
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
        self.angle = 0.0

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

    def update_mesh_grid(self, mesh_grid):
        # breakpoint()
        new_points, _ = create_grid_vbo(mesh_grid * 0, self.spacing)
        update_grid_vbo(self.vbo, new_points)

    def draw(self):

        # # Handle mouse button events
        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     if event.button == 1:  # Left mouse button pressed
        #         is_mouse_dragging = True
        #         last_mouse_pos = pygame.mouse.get_pos()

        # if event.type == pygame.MOUSEBUTTONUP:
        #     if event.button == 1:  # Left mouse button released
        #         is_mouse_dragging = False

        # if event.type == pygame.TEXTINPUT:  # event.type == pg.KEYDOWN or
        #     key = event.text

        #     if key == "[":
        #         print("ZOOM out")
        #     if key == "]":
        #         print("ZOOM in")
        # elif event.type == pygame.KEYDOWN:
        #     # batch_idx += 1
        #     if event.key == pygame.K_LEFT:
        #         print("move left")
        #     if event.key == pygame.K_RIGHT:
        #         print("move right")
        #     if event.key == pygame.K_UP:
        #         print("move up")
        #     if event.key == pygame.K_DOWN:
        #         print("move down")

        # # Handle mouse motion for rotation
        # if is_mouse_dragging:
        #     mouse_pos = pygame.mouse.get_pos()
        #     if last_mouse_pos:
        #         dx = mouse_pos[0] - last_mouse_pos[0]
        #         dy = mouse_pos[1] - last_mouse_pos[1]

        #         # Update rotation angles based on mouse movement
        #         rotation_x += dy * sensitivity
        #         rotation_y += dx * sensitivity

        #     last_mouse_pos = mouse_pos

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Position the camera (move it back along the z-axis)
        glTranslatef(0, 0.0, -15.0)  # Move the grid into the camera's view

        glRotatef(20, 1.0, 0.0, 0.0)
        # Apply the rotations based on mouse input
        glRotatef(self.rotation_x, 1, 0, 0)  # Rotate around the X-axis
        glRotatef(self.rotation_y, 0, 1, 0)  # Rotate around the Y-axis

        glRotatef(self.angle, 0, 1, 0)  # Rotate to visualize the grid
        self.angle += 0.3
        print("ANGLE", self.angle)

        # Set wireframe color
        # glColor3f(1.0, 1.0, 1.0)

        glColor4f(1.0, 1.0, 1.0, 0.5)  # Semi-transparent white color
        # Draw the wireframe grid using line segments
        glDrawElements(GL_LINES, len(self.indices), GL_UNSIGNED_INT, None)

        # Draw the faint red plane
        # draw_red_sphere(loss_np[grid_size // 2, grid_size // 2])

        # draw_red_plane(
        #     # , loss_np.shape[0], spacing
        #     loss_np[grid_size // 2, grid_size // 2],
        #     loss_np.shape[0],
        #     spacing,
        # )

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
