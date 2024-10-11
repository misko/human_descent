import time
from dataclasses import dataclass
from time import sleep

import pygame as pg
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *
from pygame.math import Vector2

from hudes.controllers.keyboard_client import KeyboardClient

"""
I used chatGPT a lot for this, I have no idea how to use openGL
"""


@dataclass
class JoyStickController:
    left_trig_axis: int
    right_trig_axis: int
    left_js_down_axis: int
    left_js_right_axis: int
    right_js_down_axis: int
    right_js_right_axis: int
    right_js_press_button: int

    button_y: int
    button_b: int
    button_x: int
    button_a: int
    button_left: int
    button_right: int


controllers = {
    "wireless_osx": JoyStickController(
        left_trig_axis=4,
        right_trig_axis=5,
        left_js_down_axis=1,
        left_js_right_axis=0,
        right_js_down_axis=3,
        right_js_right_axis=2,
        button_y=2,
        button_b=0,
        button_x=3,
        button_a=1,
        button_left=9,
        button_right=10,
        right_js_press_button=8,
    ),
    "wireless_rpi": JoyStickController(
        left_trig_axis=2,
        right_trig_axis=5,
        left_js_down_axis=1,
        left_js_right_axis=0,
        right_js_down_axis=4,
        right_js_right_axis=3,
        button_y=3,
        button_b=1,
        button_x=2,
        button_a=0,
        button_left=4,
        button_right=5,
        right_js_press_button=10,
    ),
}


class KeyboardClientGL(KeyboardClient):
    def init_input(self):
        super().init_input()  # setup keyboard
        self.joysticks = {}
        self.joystick_controller: JoyStickController = controllers[
            self.joystick_controller_key
        ]

        self.set_batch_size(32)
        self.set_dtype("float16")
        self.set_n(self.mesh_grids * 2)

    def process_key_press(self, event):
        redraw = super().process_key_press(event)  # use the keyboard interface

        if event.type == pg.JOYBUTTONDOWN:
            if event.button == self.joystick_controller.button_y:
                self.get_next_dims()

            if event.button == self.joystick_controller.button_b:
                joystick = self.joysticks[event.instance_id]
                joystick.rumble(0, 0.7, 500)
                self.get_next_batch()

            if event.button == self.joystick_controller.button_a:
                self.toggle_dtype()

            if event.button == self.joystick_controller.button_x:
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
                if (
                    joystick.get_button(self.joystick_controller.right_js_press_button)
                    > 0.5
                ):
                    self.view.reset_angle()
                    redraw = True

                if joystick.get_button(self.joystick_controller.button_left) > 0.5:
                    self.step_size_decrease(2)

                if joystick.get_button(self.joystick_controller.button_right) > 0.5:
                    self.step_size_increase(2)

                if (
                    joystick.get_axis(self.joystick_controller.left_trig_axis) > 0.5
                    and (ct - last_select_press) > 0.2
                ):
                    self.view.decrement_selected_grid()
                    self.view.update_points_and_colors()
                    redraw = True
                    last_select_press = ct
                if (
                    joystick.get_axis(self.joystick_controller.right_trig_axis) > 0.5
                    and (ct - last_select_press) > 0.2
                ):
                    self.view.increment_selected_grid()
                    self.view.update_points_and_colors()
                    redraw = True
                    last_select_press = ct

                A = Vector2(
                    joystick.get_axis(self.joystick_controller.left_js_right_axis),
                    joystick.get_axis(self.joystick_controller.left_js_down_axis),
                ).rotate(self.view.get_angles()[0])
                B = Vector2(
                    joystick.get_axis(self.joystick_controller.right_js_right_axis),
                    joystick.get_axis(self.joystick_controller.right_js_down_axis),
                )

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
                    redraw = True
                    self.view.adjust_angles(adjustH, adjustV)
            redraw = redraw | self.view.is_mouse_dragging | self.receive_messages()
            if redraw:
                self.view.draw()
            else:
                sleep(0.01)
            i += 1
