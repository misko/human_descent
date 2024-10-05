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


def draw_arrow(start, end, color=(1.0, 0.0, 0.0), arrow_length=0.2, arrow_angle=20):
    """
    Draw an arrow from start to end.
    :param start: Starting point of the arrow (tuple of x, y, z)
    :param end: Ending point of the arrow (tuple of x, y, z)
    :param color: Color of the arrow as an (r, g, b) tuple
    :param arrow_length: Length of the arrowhead lines
    :param arrow_angle: Angle of the arrowhead in degrees
    """
    print(start, end)
    glColor3f(*color)
    glLineWidth(2)

    # Draw the main line of the arrow
    glBegin(GL_LINES)
    glVertex3f(*start)
    glVertex3f(*end)
    glEnd()

    # Calculate the direction vector of the arrow
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    direction /= length  # Normalize the direction vector

    # Find orthogonal vectors to the direction for the arrowhead
    up = np.array([0.0, 1.0, 0.0])
    if np.allclose(direction, up):  # If direction is parallel to up, use another axis
        up = np.array([1.0, 0.0, 0.0])

    # Cross product to get orthogonal vectors
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, direction)

    # Arrowhead points
    end = np.array(end)
    arrow_end1 = (
        end
        - (direction * arrow_length)
        + (right * arrow_length * np.tan(np.radians(arrow_angle)))
    )
    arrow_end2 = (
        end
        - (direction * arrow_length)
        - (right * arrow_length * np.tan(np.radians(arrow_angle)))
    )

    # Draw the arrowhead
    glBegin(GL_LINES)
    glVertex3f(*end)
    glVertex3f(*arrow_end1)
    glVertex3f(*end)
    glVertex3f(*arrow_end2)
    glEnd()


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


import numpy as np


def create_grid_points(height_maps, spacing):
    """
    Creates grid points and indices for rendering multiple wireframe grids,
    with heights determined by a set of height maps using NumPy operations.

    :param height_maps: A 3D numpy array of shape (n, H, W) representing the heights
                        (Z values) at each (h, w) grid point for each height map.
    :param spacing: The distance between adjacent points in the grid.
    :return: (points, indices) - The points and indices for all grids.
    """
    n, H, W = height_maps.shape  # Get the dimensions of the height maps

    # Calculate X and Z coordinates for the grid (centered around 0)
    offset_h = (H - 1) / 2.0
    offset_w = (W - 1) / 2.0

    x_coords = (np.arange(H) - offset_h) * spacing  # X coordinates for all rows
    z_coords = (np.arange(W) - offset_w) * spacing  # Z coordinates for all columns

    # Create a meshgrid of X and Z coordinates (Shape: (H, W) for both)
    x_grid, z_grid = np.meshgrid(x_coords, z_coords, indexing="ij")  # Shape (H, W)

    # Repeat x_grid and z_grid across the n height maps to match the shape (n, H, W)
    x_grid = np.repeat(x_grid[np.newaxis, :, :], n, axis=0)  # Shape (n, H, W)
    z_grid = np.repeat(z_grid[np.newaxis, :, :], n, axis=0)  # Shape (n, H, W)

    # Now x_grid, height_maps, and z_grid all have shape (n, H, W)

    # Combine X, Y (heights), and Z coordinates into a single (n, H, W, 3) array
    points = np.stack([x_grid, height_maps, z_grid], axis=-1)  # Shape (n, H, W, 3)

    # Flatten the points array to shape (n * H * W, 3) for use with VBOs
    return points.reshape(-1, 3).astype(np.float32)


def create_grid_indices(height_maps, spacing):
    n, H, W = height_maps.shape  # Get the dimensions of the height maps

    # Create line segments (indices) for one grid
    indices = []

    # Horizontal lines
    for i in range(H):
        for j in range(W - 1):
            indices.append(i * W + j)
            indices.append(i * W + j + 1)

    # Vertical lines
    for i in range(H - 1):
        for j in range(W):
            indices.append(i * W + j)
            indices.append((i + 1) * W + j)

    # Convert indices to numpy array
    return np.array(indices, dtype=np.uint32)


def create_surface_grid_points(height_maps, spacing):
    """
    Creates grid points for rendering multiple surfaces (triangle grids),
    with heights determined by a set of height maps using NumPy operations.

    :param height_maps: A 3D numpy array of shape (n, H, W) representing the heights
                        (Z values) at each (h, w) grid point for each height map.
    :param spacing: The distance between adjacent points in the grid.
    :return: points - The points for all grids, shape (n * H * W, 3).
    """
    n, H, W = height_maps.shape  # Get the dimensions of the height maps

    # Calculate X and Z coordinates for the grid (centered around 0)
    offset_h = (H - 1) / 2.0
    offset_w = (W - 1) / 2.0

    x_coords = (np.arange(H) - offset_h) * spacing  # X coordinates for all rows
    z_coords = (np.arange(W) - offset_w) * spacing  # Z coordinates for all columns

    # Create a meshgrid of X and Z coordinates (Shape: (H, W) for both)
    x_grid, z_grid = np.meshgrid(x_coords, z_coords, indexing="ij")  # Shape (H, W)

    # Repeat x_grid and z_grid across the n height maps to match the shape (n, H, W)
    x_grid = np.repeat(x_grid[np.newaxis, :, :], n, axis=0)  # Shape (n, H, W)
    z_grid = np.repeat(z_grid[np.newaxis, :, :], n, axis=0)  # Shape (n, H, W)

    # Now x_grid, height_maps, and z_grid all have shape (n, H, W)

    # Combine X, Y (heights), and Z coordinates into a single (n, H, W, 3) array
    points = np.stack([x_grid, height_maps, z_grid], axis=-1)  # Shape (n, H, W, 3)

    # Flatten the points array to shape (n * H * W, 3) for use with VBOs
    return points.reshape(-1, 3).astype(np.float32)


def create_surface_grid_indices(height_maps, spacing):
    """
    Creates indices for rendering multiple surface grids as triangle grids.

    :param height_maps: A 3D numpy array of shape (n, H, W) representing the heights
                        (Z values) at each (h, w) grid point for each height map.
    :param spacing: The distance between adjacent points in the grid.
    :return: indices - The indices for all grids to render triangles.
    """
    n, H, W = height_maps.shape  # Get the dimensions of the height maps

    indices = []

    for k in range(n):  # For each height map
        for i in range(H - 1):
            for j in range(W - 1):
                # For each grid cell, generate two triangles:

                # Triangle 1: Top-left, Bottom-left, Bottom-right
                indices.append(k * H * W + i * W + j)
                indices.append(k * H * W + (i + 1) * W + j)
                indices.append(k * H * W + (i + 1) * W + (j + 1))

                # Triangle 2: Top-left, Bottom-right, Top-right
                indices.append(k * H * W + i * W + j)
                indices.append(k * H * W + (i + 1) * W + (j + 1))
                indices.append(k * H * W + i * W + (j + 1))

    return np.array(indices, dtype=np.uint32)


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
