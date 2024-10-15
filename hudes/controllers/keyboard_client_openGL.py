import logging
import time
from dataclasses import dataclass
from time import sleep

import pygame as pg
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *
from pygame.math import Vector2

from hudes.controllers.keyboard_client import KeyboardClient
from hudes.hudes_client import HudesClient

"""
I used chatGPT a lot for this, I have no idea how to use openGL
"""


@dataclass
class JoyStickController:
    left_js_down_axis: int
    left_js_right_axis: int
    right_js_down_axis: int
    right_js_right_axis: int
    right_js_press_button: int
    sgd_button: int
    quit_button: int

    button_y: int
    button_b: int
    button_x: int
    button_a: int

    button_left: int
    button_right: int

    button_key_left: int = -1
    button_key_right: int = -1
    button_key_up: int = -1
    button_key_down: int = -1

    right_js_accel: float = 1.0
    left_js_accel: float = 1.0

    left_trig_axis: int = -1
    right_trig_axis: int = -1

    left_trig_button: int = -1
    right_trig_button: int = -1


controllers = {
    "wireless_osx": JoyStickController(
        left_trig_button=6,
        right_trig_button=7,
        left_js_down_axis=1,
        left_js_right_axis=0,
        right_js_down_axis=3,
        right_js_right_axis=2,
        sgd_button=9,
        quit_button=8,
        button_y=3,
        button_b=2,
        button_x=0,
        button_a=1,
        button_left=4,
        button_right=5,
        right_js_press_button=11,
    ),
    "wireless_rpi": JoyStickController(
        left_trig_axis=2,
        right_trig_axis=5,
        left_js_down_axis=1,
        left_js_right_axis=0,
        right_js_down_axis=4,
        right_js_right_axis=3,
        sgd_button=7,
        quit_button=6,
        button_y=3,
        button_b=1,
        button_x=2,
        button_a=0,
        button_left=4,
        button_right=5,
        right_js_press_button=10,
        right_js_accel=5,
    ),
}


class KeyboardClientGL(HudesClient):
    def init_input(self):

        self.joysticks = {}
        self.joystick_controller: JoyStickController = controllers[
            self.joystick_controller_key
        ]

        self.client_state.set_batch_size(64)
        self.client_state.set_dtype("float16")
        self.set_n(self.mesh_grids * 2)

        self.last_select_press = 0
        self.last_batch_press = 0
        self.last_dims_press = 0
        self.last_batch_size_press = 0

        self.step_size_keyboard_multiplier = 2.5

        self.client_state.set_help_screen_fns(
            [
                "help_screens/hudes_help_start.png",
                "help_screens/hudes_1.png",
                "help_screens/hudes_2.png",
                "help_screens/hudes_3.png",
                "help_screens/hudes_2d_keyboard_controls.png",
                "help_screens/hudes_2d_xbox_controls.png",
            ]
        )

    def process_key_press(self, event):
        redraw = self.process_common_keys(event)

        if (
            event.type == pg.JOYBUTTONDOWN
            and event.button == self.joystick_controller.button_x
        ):
            logging.debug("calling next_help_screen...")
            self.client_state.next_help_screen()
            return True

        if self.client_state.help_screen_idx != -1:
            return redraw  # we are in help screen mode

        if self.client_state.help_screen_idx != -1:
            return redraw
        ct = pg.time.get_ticks()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_w:
                self.step_in_selected_grid(
                    *Vector2(
                        0,
                        -1,
                    ).rotate(self.view.get_angles()[0])
                )
            elif event.key == pg.K_s:
                self.step_in_selected_grid(
                    *Vector2(
                        0,
                        1,
                    ).rotate(self.view.get_angles()[0])
                )
            elif event.key == pg.K_a:
                self.step_in_selected_grid(
                    *Vector2(
                        -1,
                        0,
                    ).rotate(self.view.get_angles()[0])
                )
            elif event.key == pg.K_d:
                self.step_in_selected_grid(
                    *Vector2(
                        1,
                        0,
                    ).rotate(self.view.get_angles()[0])
                )
            elif self.check_cooldown_time(
                event=event,
                key=pg.K_LSHIFT,
                ct=ct,
                cool_down_time=self.key_seconds_pressed,
            ):
                self.view.decrement_selected_grid()
                self.view.update_points_and_colors()
                redraw = True
            elif self.check_cooldown_time(
                event=event,
                key=pg.K_RSHIFT,
                ct=ct,
                cool_down_time=self.key_seconds_pressed,
            ):
                self.view.increment_selected_grid()
                self.view.update_points_and_colors()
                redraw = True
            elif event.key == pg.K_RIGHT:
                self.view.adjust_angles(5, 0)
                redraw = True
            elif event.key == pg.K_LEFT:
                self.view.adjust_angles(-5, 0)
                redraw = True
            elif event.key == pg.K_UP:
                self.view.adjust_angles(0, 2.5)
                redraw = True
            elif event.key == pg.K_DOWN:
                self.view.adjust_angles(0, -2.5)
                redraw = True

        elif event.type == pg.JOYBUTTONDOWN:
            if event.button == self.joystick_controller.button_y:
                self.get_next_dims()

            elif event.button == self.joystick_controller.button_b:
                joystick = self.joysticks[event.instance_id]
                joystick.rumble(0, 0.7, 500)
                self.get_next_batch()

            elif event.button == self.joystick_controller.button_a:
                self.client_state.toggle_batch_size()
                self.send_config()

            elif event.button == self.joystick_controller.sgd_button:
                self.get_sgd()
            elif event.button == self.joystick_controller.quit_button:
                self.quit()
            elif (
                self.joystick_controller.left_trig_button >= 0
                and event.button == self.joystick_controller.left_trig_button
            ):
                self.view.decrement_selected_grid()
                self.view.update_points_and_colors()
                redraw = True
            elif (
                self.joystick_controller.right_trig_button >= 0
                and event.button == self.joystick_controller.right_trig_button
            ):
                self.view.increment_selected_grid()
                self.view.update_points_and_colors()
                redraw = True

            elif event.button in (
                self.joystick_controller.button_key_down,
                self.joystick_controller.button_key_up,
            ):
                self.client_state.toggle_dtype()
                self.send_config()

        return redraw

    def step_in_selected_grid(self, s0, s1):
        selected_grid = self.view.get_selected_grid()
        self.send_dims_and_steps(
            {
                1 + selected_grid * 2: s1 * self.client_state.step_size * 0.25,
                0 + selected_grid * 2: s0 * self.client_state.step_size * 0.25,
            }
        )

    def process_js(self):
        if self.client_state.help_screen_idx != -1:
            return False
        redraw = False
        ct = pg.time.get_ticks()
        for joystick in self.joysticks.values():
            if (
                joystick.get_button(self.joystick_controller.right_js_press_button)
                > 0.5
            ):
                self.view.reset_angle()
                redraw = True

            if joystick.get_button(self.joystick_controller.button_left) > 0.5:
                self.client_state.step_size_decrease(2)
                self.send_config()

            if joystick.get_button(self.joystick_controller.button_right) > 0.5:
                self.client_state.step_size_increase(2)
                self.send_config()

            # use that axis that sometimes shows up?
            if (
                self.joystick_controller.left_trig_axis >= 0
                and joystick.get_axis(self.joystick_controller.left_trig_axis) > 0.5
                and (ct - self.last_select_press) > 0.2 * 1000
            ):
                self.view.decrement_selected_grid()
                self.view.update_points_and_colors()
                redraw = True
                self.last_select_press = ct
            if (
                self.joystick_controller.right_trig_axis >= 0
                and joystick.get_axis(self.joystick_controller.right_trig_axis) > 0.5
                and (ct - self.last_select_press) > 0.2 * 1000
            ):
                self.view.increment_selected_grid()
                self.view.update_points_and_colors()
                redraw = True
                self.last_select_press = ct

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
                self.step_in_selected_grid(A[0], A[1])

            radius, angle = B.as_polar()
            if radius > 0.4:

                adjustH = B[0] * 2 * self.joystick_controller.right_js_accel
                adjustV = B[1] * self.joystick_controller.right_js_accel
                redraw = True
                self.view.adjust_angles(adjustH, adjustV)

            hats = joystick.get_numhats()
            if hats > 0:

                hat = joystick.get_hat(0)
                if abs(hat[1]) > 0.5 and (ct - self.last_batch_size_press) > 0.2 * 1000:
                    self.last_batch_size_press = ct
                    self.client_state.toggle_dtype()
                    self.send_config()

        return redraw

    def run_loop(self):
        self.before_first_loop()
        i = 0

        while self.hudes_websocket_client.running:
            # check and send local interactions(?)
            redraw = False
            for event in pg.event.get():

                # Handle hotplugging
                if event.type == pg.JOYDEVICEADDED:
                    # This event will be generated when the program starts for every
                    # joystick, filling up the list without needing to create them manually.
                    joy = pg.joystick.Joystick(event.device_index)
                    self.joysticks[joy.get_instance_id()] = joy
                    logging.info(f"Joystick {joy.get_instance_id()} connencted")

                elif event.type == pg.JOYDEVICEREMOVED:
                    del self.joysticks[event.instance_id]
                    logging.info(f"Joystick {event.instance_id} disconnected")

                if event.type == pg.KEYDOWN:
                    self.update_key_holds()
                redraw |= self.process_key_press(event)

            redraw |= self.process_js()
            redraw |= self.receive_messages()
            if redraw:
                self.view.draw()
            else:
                sleep(0.01)
            i += 1
