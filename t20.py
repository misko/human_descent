import numpy as np
import pygame
import torch
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *

from hudes.opengl_func import (
    create_grid_vbo,
    draw_red_plane,
    draw_red_sphere,
    t20_get_loss,
    t20_get_mad,
    update_grid_vbo,
)

"""
I used chatGPT a lot for this, I have no idea how to use openGL
"""


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
    mad = t20_get_mad(device)
    grid_size = 81

    step_size_resolution = 0.05
    log_step_size = -1
    arange = torch.linspace(-5, 5, grid_size, device=device)
    brange = torch.linspace(-5, 5, grid_size, device=device)

    dim_idx = 0
    batch_idx = 0
    batch = mad.get_batch(batch_idx)["train"]

    loss_np = t20_get_loss(
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

            if event.type == pygame.TEXTINPUT:  # event.type == pg.KEYDOWN or
                key = event.text

                if key == "[":
                    print("ZOOM out")
                if key == "]":
                    print("ZOOM in")
            elif event.type == pygame.KEYDOWN:
                # batch_idx += 1
                if event.key == pygame.K_LEFT:
                    print("move left")
                if event.key == pygame.K_RIGHT:
                    print("move right")
                if event.key == pygame.K_UP:
                    print("move up")
                if event.key == pygame.K_DOWN:
                    print("move down")

                batch = mad.get_batch(batch_idx)["train"]
                dim_idx += 1
                loss_np = t20_get_loss(
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
