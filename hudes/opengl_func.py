import numpy as np
import pygame
import torch
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *

from hudes.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import indexed_loss

"""
I used chatGPT a lot for this, I have no idea how to use openGL
"""


def draw_red_sphere(z_position):
    """
    Draw a red sphere at the position (0, 0, z_position).
    """
    glPushMatrix()  # Save the current matrix state
    glTranslatef(0.0, z_position, 0.0)  # Move the sphere along the Z-axis by z_position
    glColor3f(1.0, 0.0, 0.0)  # Set color to red
    quadric = gluNewQuadric()  # Create a new quadric object
    gluSphere(
        quadric, 0.1, 32, 32
    )  # Draw a sphere with radius 0.5 and high resolution (32 slices, 32 stacks)
    gluDeleteQuadric(quadric)  # Delete the quadric object to clean up
    glPopMatrix()  # Restore the previous matrix state


def create_grid_vbo(height_map, spacing):
    """
    Creates a grid of points and indices for rendering a wireframe grid, with heights determined by a height_map.

    :param height_map: A 2D numpy array of shape (H, W) representing the heights (Z values) at each (h, w) grid point.
    :param spacing: The distance between adjacent points in the grid.
    :return: (points, indices) - The points and indices for the grid.
    """
    H, W = height_map.shape  # Get the dimensions of the height map

    # Create the grid points based on the height_map
    points = []
    offset_h = (H - 1) / 2.0  # Offset to center the grid along height (H)
    offset_w = (W - 1) / 2.0  # Offset to center the grid along width (W)
    for i in range(H):
        for j in range(W):
            x = (i - offset_h) * spacing  # X coordinate centered
            y = height_map[i, j]  # Z coordinate (height) from the height_map
            z = (j - offset_w) * spacing  # Y coordinate centered
            points.append((x, y, z))

    # Convert points to a numpy array (flatten for use with VBOs)
    points = np.array(points, dtype=np.float32).flatten()

    # Create line segments connecting the points (indices for wireframe grid)
    indices = []
    for i in range(H):
        for j in range(W - 1):  # Horizontal lines
            indices.append(i * W + j)
            indices.append(i * W + j + 1)

    for i in range(H - 1):
        for j in range(W):  # Vertical lines
            indices.append(i * W + j)
            indices.append((i + 1) * W + j)

    # Convert indices to a numpy array
    indices = np.array(indices, dtype=np.uint32)

    return points, indices


def t20_get_mad(device):
    mad = mnist_model_data_and_subpace(
        model=MNISTFFNN(),
        loss_fn=indexed_loss,
        device=device,
    )
    mad.move_to_device()
    mad.fuse()
    mad.init_param_model()

    return mad


def draw_red_plane(plane_z, grid_size, spacing):
    """
    Draws a faint red plane at a specified Z height.

    :param plane_z: The Z height of the plane.
    :param grid_size: The size of the grid (NxN).
    :param spacing: The distance between grid points.
    """
    half_size = (grid_size - 1) / 2.0 * spacing

    # Enable blending for transparency
    # glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Set the plane color (faint red with transparency)
    glColor4f(1.0, 0.0, 0.0, 0.2)

    # Draw the plane as a grid of quads or lines at the specified Z level
    glBegin(GL_QUADS)
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            x1 = (i - half_size / spacing) * spacing
            z1 = (j - half_size / spacing) * spacing
            x2 = ((i + 1) - half_size / spacing) * spacing
            z2 = ((j + 1) - half_size / spacing) * spacing

            # Define the four corners of each quad
            glVertex3f(x1, plane_z, z1)
            glVertex3f(x2, plane_z, z1)
            glVertex3f(x2, plane_z, z2)
            glVertex3f(x1, plane_z, z2)
    glEnd()

    # Disable blending to return to normal rendering
    # glDisable(GL_BLEND)


def draw_axes():
    """
    Draw blue X and Y axes.
    """
    glColor3f(0.0, 0.0, 1.0)  # Blue color

    # Draw X axis
    glBegin(GL_LINES)
    glVertex3f(-10, 0, 0)
    glVertex3f(10, 0, 0)
    glEnd()

    # Draw Y axis
    glBegin(GL_LINES)
    glVertex3f(0, 0, -10)
    glVertex3f(0, 0, 10)
    glEnd()


def render_text(text, position, color=(255, 255, 255)):
    """
    Render 2D text on the screen at the given position.
    """
    font = pygame.font.SysFont("Arial", 18)
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(*position)
    glDrawPixels(
        text_surface.get_width(),
        text_surface.get_height(),
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        text_data,
    )


def t20_get_loss(mad, batch, arange, brange, dims):

    mp = mad.dim_idxs_and_ranges_to_models_parms(dims, arange=arange, brange=brange)

    mp_reshaped = mp.reshape(-1, 26506)
    # batch = torch.rand(1, 512, 28, 28, device=device)
    data = batch[0].unsqueeze(0)
    batch_size = data.shape[1]
    label = batch[1]

    predictions = mad.param_model.forward(mp_reshaped, data)[1].reshape(
        *mp.shape[:2], batch_size, -1
    )
    loss = torch.gather(
        predictions,
        3,
        label.reshape(1, 1, -1, 1).expand(*mp.shape[:2], batch_size, 1),
    ).mean(axis=[2, 3])

    loss_np = loss.detach().cpu().numpy()
    loss_np -= loss_np.mean()
    loss_np /= loss_np.std()

    # invert loss_np
    loss_np = -loss_np
    return loss_np


def update_grid_vbo(vbo, points):
    """
    Efficiently update the VBO with new points.
    """
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, points.nbytes, points)
