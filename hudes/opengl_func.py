from functools import cache

import matplotlib.backends.backend_agg as agg
import numpy as np
import pygame as pg
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from PIL import Image
from pygame.locals import *

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
    glColor3f(1.0, 1.0, 1.0)


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


@cache
def get_color_matrix(n, grid_size, grid_colors):
    colors = np.zeros((n, grid_size, grid_size, 4), dtype=np.float32)  # RGBA colors
    for i in range(n):
        colors[i] = grid_colors[i]
    return colors


# Step 2: Convert Pygame surface to OpenGL texture
def create_texture_from_surface(surf):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Convert surface to string data and upload to OpenGL texture
    texture_data = pg.image.tostring(surf, "RGB", 1)
    width, height = surf.get_size()

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        texture_data,
    )

    return texture_id, width, height


import pygame as pg
from OpenGL.GL import *


def load_texture(image_path, output_width, output_height):
    # Load the image using Pygame
    image = pg.image.load(
        image_path
    ).convert_alpha()  # Ensure the image has an alpha channel

    # Flip the image vertically to match OpenGL's coordinate system
    image = pg.transform.flip(image, False, True)

    # Resize the image to the specified output dimensions
    image = pg.transform.smoothscale(image, (output_width, output_height))

    # Get the resized image data
    image_data = pg.image.tostring(
        image, "RGBA", True
    )  # Convert the image to a string with RGBA format

    # Generate and bind texture
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Upload the resized image data to the texture
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        output_width,
        output_height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        image_data,
    )

    return texture_id, output_width, output_height


def render_texture_rgba(texture_id, width, height, window_size):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()

    # Set up orthographic projection to render the texture in 2D
    glOrtho(0, window_size[0], window_size[1], 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Enable transparency
    glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Bind the texture and draw a quad with it
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glEnable(GL_TEXTURE_2D)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(0, 0)  # Bottom-left corner
    glTexCoord2f(1, 0)
    glVertex2f(width, 0)  # Bottom-right corner
    glTexCoord2f(1, 1)
    glVertex2f(width, height)  # Top-right corner
    glTexCoord2f(0, 1)
    glVertex2f(0, height)  # Top-left corner
    glEnd()

    # Disable the texture and blending
    glDisable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    # glDisable(GL_BLEND)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def create_texture_rgba(image_data, width, height):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Upload the texture data with RGBA format
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        image_data,
    )

    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind the texture
    return texture_id


def create_texture(image_data, width, height):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Generate texture with proper format (GL_RGB)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        int(width),
        int(height),
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        image_data,
    )

    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind the texture
    return texture_id


def render_texture(texture_id, width, height, window_size):
    # Save the current matrix mode
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()

    # Use the window size for setting up the orthographic projection
    glOrtho(0, window_size[0], window_size[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Enable 2D texture rendering
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Draw the 2D texture quad at the top center of the screen
    glBegin(GL_QUADS)

    glTexCoord2f(0, 0)
    glVertex2f(100, 100)  # Bottom-left corner

    glTexCoord2f(1, 0)
    glVertex2f(width + 100, 100)  # Bottom-right corner

    glTexCoord2f(1, 1)
    glVertex2f(width + 100, height + 100)  # Top-right corner

    glTexCoord2f(0, 1)
    glVertex2f(100, height + 100)  # Top-left corner

    glEnd()

    glDisable(GL_TEXTURE_2D)

    # Restore previous matrix mode
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


# Create the Matplotlib plot and convert it to a raw image
def create_matplotlib_texture_rgba(fig):
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()

    raw_data = canvas.get_renderer().buffer_rgba()
    width, height = canvas.get_width_height()
    return raw_data, int(width), int(height)


# Create the Matplotlib plot and convert it to a raw image
def create_matplotlib_texture(fig):
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()

    raw_data = canvas.get_renderer().tostring_rgb()
    width, height = canvas.get_width_height()
    return raw_data, int(width), int(height)


def render_text_2d(text, font_size, screen_width, screen_height):
    font = pg.font.SysFont("Arial", font_size)
    text_surface = font.render(text, True, (255, 255, 255))  # White text
    text_data = pg.image.tostring(text_surface, "RGBA", True)

    # Get the dimensions of the text surface
    text_width = text_surface.get_width()
    text_height = text_surface.get_height()

    # Calculate the position to center the text at the top
    x_position = (screen_width - text_width) // 2
    y_position = screen_height - text_height

    # Disable depth testing for text rendering
    glDisable(GL_DEPTH_TEST)

    # Switch to orthographic projection to render text
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, screen_width, screen_height, 0, -1, 1)  # Set orthographic projection

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Set the raster position in 2D screen space
    glRasterPos2f(x_position, y_position)

    # Render the text as pixels
    glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Restore previous projection matrix
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    # Re-enable depth testing
    glEnable(GL_DEPTH_TEST)


def render_text(text, font_size, screen_width, screen_height):
    # Initialize the font system
    font = pg.font.SysFont("Arial", font_size)

    # Render the text to a surface
    text_surface = font.render(text, True, (255, 255, 255))  # White text
    text_data = pg.image.tostring(text_surface, "RGBA", True)

    # Get the dimensions of the text surface
    text_width = text_surface.get_width()
    text_height = text_surface.get_height()

    # Calculate the position to center the text at the top
    x_position = (screen_width - text_width) // 2
    y_position = screen_height - text_height  # Top of the screen
    # Set up OpenGL to render text
    glRasterPos2f(x_position / screen_width * 2 - 1, 1 - y_position / screen_height * 2)

    # Draw the text as pixels
    glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)


def create_grid_points_with_colors(
    height_maps, spacing, grid_colors, selected_grid, selected_alpha=0.6
):
    """
    Creates grid points and colors for rendering multiple wireframe grids,
    with heights determined by a set of height maps using NumPy operations.

    :param height_maps: A 3D numpy array of shape (n, H, W) representing the heights
                        (Z values) at each (h, w) grid point for each height map.
    :param spacing: The distance between adjacent points in the grid.
    :return: (points, colors) - The points and colors for all grids.
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

    # Combine X, Y (heights), and Z coordinates into a single (n, H, W, 3) array
    points = np.stack([x_grid, height_maps, z_grid], axis=-1)  # Shape (n, H, W, 3)

    # Normalize heights for color generation (assuming the height map has a wide range of values)
    min_height = height_maps.min()
    max_height = height_maps.max()
    normalized_heights = (height_maps - min_height) / (max_height - min_height)

    # Generate colors based on height (blue for low, red for high)
    colors = get_color_matrix(n, H, grid_colors=grid_colors).copy()  # RGBA colors
    colors[..., 3] = -height_maps

    colors[..., 3][colors[..., 3] >= 0] = 1.0 - selected_alpha
    colors[..., :][colors[..., 3] >= 0] += 0.3
    colors[..., 3][colors[..., 3] < 0] = 0.02

    colors[selected_grid, ..., 3] += selected_alpha

    # Flatten the points and colors arrays to shape (n * H * W, 3) and (n * H * W, 4)
    points = points.reshape(-1, 3).astype(np.float32)
    colors = colors.reshape(-1, 4).astype(np.float32)

    return points, colors


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


def draw_red_plane(plane_z, grid_size, spacing):
    """
    Draws a faint red plane at a specified Z height.

    :param plane_z: The Z height of the plane.
    :param grid_size: The size of the grid (NxN).
    :param spacing: The distance between grid points.
    """
    half_size = (grid_size - 1) / 2.0 * spacing

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


def update_grid_vbo(vbo, points):
    """
    Efficiently update the VBO with new points.
    """
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, points.nbytes, points)


def update_grid_cbo(cbo, colors):
    # Update the Color Buffer Object (CBO) with new colors
    glBindBuffer(GL_ARRAY_BUFFER, cbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, colors.nbytes, colors)
