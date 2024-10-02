import numpy as np
import pygame
import torch
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *

from hudes.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import ParamModelDataAndSubspace, indexed_loss

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


def get_mad(device):
    mad = mnist_model_data_and_subpace(
        model=MNISTFFNN(),
        loss_fn=indexed_loss,
        device=device,
        constructor=ParamModelDataAndSubspace,
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


def get_loss(mad, batch, arange, brange, dims):

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


def main(height_map, spacing=0.1):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

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

    # Create grid VBOs with the height map and spacing

    device = "mps"
    mad = get_mad(device)
    grid_size = 81
    arange = torch.linspace(-5, 5, grid_size, device=device)
    brange = torch.linspace(-5, 5, grid_size, device=device)

    dim_idx = 0
    batch_idx = 0
    batch = mad.get_batch(batch_idx)["train"]

    loss_np = get_loss(
        mad, batch, arange=arange, brange=brange, dims=[dim_idx, dim_idx + 1]
    )

    points, indices = create_grid_vbo(loss_np, spacing)

    # Create and bind Vertex Buffer Object (VBO)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)

    # Create and bind Element Buffer Object (EBO)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Enable vertex attribute 0 (positions)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)

    # Mouse control variables
    is_mouse_dragging = False
    last_mouse_pos = None
    rotation_x, rotation_y = 0.0, 0.0  # Initialize rotation angles
    sensitivity = 0.5  # Mouse sensitivity for controlling rotation

    running = True
    angle = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle mouse button events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button pressed
                    is_mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button released
                    is_mouse_dragging = False

            if event.type == pygame.KEYDOWN:
                batch_idx += 1
                batch = mad.get_batch(batch_idx)["train"]
                # dim_idx += 1
                loss_np = get_loss(
                    mad,
                    batch,
                    arange=arange,
                    brange=brange,
                    dims=[dim_idx, dim_idx + 1],
                )
                new_points, _ = create_grid_vbo(loss_np, spacing)
                update_grid_vbo(vbo, new_points)

        # Handle mouse motion for rotation
        if is_mouse_dragging:
            mouse_pos = pygame.mouse.get_pos()
            if last_mouse_pos:
                dx = mouse_pos[0] - last_mouse_pos[0]
                dy = mouse_pos[1] - last_mouse_pos[1]

                # Update rotation angles based on mouse movement
                rotation_x += dy * sensitivity
                rotation_y += dx * sensitivity

            last_mouse_pos = mouse_pos

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Position the camera (move it back along the z-axis)
        glTranslatef(0, 0.0, -15.0)  # Move the grid into the camera's view

        glRotatef(20, 1.0, 0.0, 0.0)
        # Apply the rotations based on mouse input
        glRotatef(rotation_x, 1, 0, 0)  # Rotate around the X-axis
        glRotatef(rotation_y, 0, 1, 0)  # Rotate around the Y-axis

        glRotatef(angle, 0, 1, 0)  # Rotate to visualize the grid
        angle += 0.3

        # Set wireframe color
        # glColor3f(1.0, 1.0, 1.0)

        glColor4f(1.0, 1.0, 1.0, 0.5)  # Semi-transparent white color
        # Draw the wireframe grid using line segments
        glDrawElements(GL_LINES, len(indices), GL_UNSIGNED_INT, None)

        # Draw the faint red plane
        draw_red_sphere(loss_np[grid_size // 2, grid_size // 2])

        draw_red_plane(
            # , loss_np.shape[0], spacing
            loss_np[grid_size // 2, grid_size // 2],
            loss_np.shape[0],
            spacing,
        )
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

        pygame.display.flip()
        pygame.time.wait(10)

    # Clean up
    glDisableClientState(GL_VERTEX_ARRAY)
    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [ebo])
    pygame.quit()


if __name__ == "__main__":
    # Example usage: A 10x10 grid with random height values
    height_map = (
        np.random.rand(10, 10) * 2
    )  # 10x10 grid with random Z values between 0 and 2
    main(height_map)  # , spacing=1.0)
