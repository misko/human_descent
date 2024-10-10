import time
from time import sleep

import pygame as pg
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *
from pygame.math import Vector2

from hudes.keyboard_client import KeyboardClient

"""
I used chatGPT a lot for this, I have no idea how to use openGL
"""


class KeyboardClientGL(KeyboardClient):
    def init_input(self):
        super().init_input()  # setup keyboard

        self.joysticks = {}

    def process_key_press(self, event):
        redraw = super().process_key_press(event)  # use the keyboard interface

        if event.type == pg.JOYBUTTONDOWN:
            if event.button == 2:
                self.get_next_dims()

            if event.button == 0:
                joystick = self.joysticks[event.instance_id]
                joystick.rumble(0, 0.7, 500)
                self.get_next_batch()

            if event.button == 1:
                self.toggle_dtype()

            if event.button == 3:
                self.toggle_batch_size()

        if event.type == pg.JOYBUTTONUP:
            print("Joystick button released.")

        # Handle hotplugging
        if event.type == pg.JOYDEVICEADDED:
            # This event will be generated when the program starts for every
            # joystick, filling up the list without needing to create them manually.
            joy = pg.joystick.Joystick(event.device_index)
            self.joysticks[joy.get_instance_id()] = joy
            print(f"Joystick {joy.get_instance_id()} connencted")

        if event.type == pg.JOYDEVICEREMOVED:
            del self.joysticks[event.instance_id]
            print(f"Joystick {event.instance_id} disconnected")

        # Handle mouse button events
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button pressed
                self.view.is_mouse_dragging = True
                self.view.last_mouse_pos = pg.mouse.get_pos()
                redraw = True

        if event.type == pg.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button released
                self.view.is_mouse_dragging = False
                redraw = True
        return redraw

    def run_loop(self):
        self.before_first_loop()
        i = 0
        last_select_press = 0
        while self.hudes_websocket_client.running:
            # check and send local interactions(?)
            redraw = False
            for event in pg.event.get():
                redraw |= self.process_key_press(event)

            ct = time.time()
            for joystick in self.joysticks.values():
                # axes = joystick.get_numaxes()
                # print(f"Number of axes: {axes}")
                # for i in range(axes):
                #     axis = joystick.get_axis(i)
                #     print(f"Axis {i} value: {axis:>6.3f}")

                if joystick.get_button(8) > 0.5:
                    self.view.reset_angle()
                    redraw = True

                if joystick.get_button(9) > 0.5:
                    self.step_size_decrease(2)
                    self.send_config()

                if joystick.get_button(10) > 0.5:
                    self.step_size_increase(2)
                    self.send_config()

                # if joystick.get_button(11) > 0.5:
                #     self.view.increase_zoom()
                #     redraw = True

                # if joystick.get_button(12) > 0.5:
                #     self.view.decrease_zoom()
                #     redraw = True

                if joystick.get_axis(4) > 0.5 and (ct - last_select_press) > 0.2:
                    self.view.decrement_selected_grid()
                    self.view.update_points_and_colors()
                    redraw = True
                    last_select_press = ct
                if joystick.get_axis(5) > 0.5 and (ct - last_select_press) > 0.2:
                    self.view.increment_selected_grid()
                    self.view.update_points_and_colors()
                    redraw = True
                    last_select_press = ct

                A = Vector2(joystick.get_axis(0), joystick.get_axis(1)).rotate(
                    self.view.get_angles()[0]
                )
                B = Vector2(joystick.get_axis(2), joystick.get_axis(3))

                radius, angle = A.as_polar()
                if radius > 0.3:
                    selected_grid = self.view.get_selected_grid()
                    # math.atan2(A[0], A[1])
                    self.send_dims_and_steps(
                        {
                            1 + selected_grid * 2: A[1] * self.step_size * 0.25,
                            0 + selected_grid * 2: A[0] * self.step_size * 0.25,
                        }
                    )

                radius, angle = B.as_polar()
                if radius > 0.4:

                    adjustH = B[0] * 2
                    adjustV = B[1]
                    # if np.abs(np.abs(angle) - 90 / 2) < 40:
                    #     adjustH += np.sign(angle)
                    #     redraw = True

                    # if np.abs(angle) < 40:
                    #     adjustV += 1
                    #     redraw = True
                    # elif np.abs(np.abs(angle) - 180) < 40:
                    #     adjustV += -1
                    redraw = True
                    self.view.adjust_angles(adjustH, adjustV)
            redraw = redraw | self.view.is_mouse_dragging | self.receive_messages()
            if redraw:
                self.view.draw()
            else:
                sleep(0.01)
            i += 1
