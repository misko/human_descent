import argparse
import math
import time
from time import sleep

import numpy as np
import pygame
import pygame as pg
import torch
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU for perspective functions
from pygame.locals import *
from pygame.math import Vector2

from hudes.hudes_client import HudesClient
from hudes.websocket_client import next_batch_message, next_dims_message

"""
I used chatGPT a lot for this, I have no idea how to use openGL
"""


class KeyboardClientGL(HudesClient):
    def init_input(self):

        self.paired_keys = [
            ("w", "s"),
            ("e", "d"),
            ("r", "f"),
            ("u", "j"),
            ("i", "k"),
            ("o", "l"),
        ]
        self.n = len(self.paired_keys)

        self.key_to_param_and_sign = {}
        for idx in range(self.n):
            u, d = self.paired_keys[idx]
            self.key_to_param_and_sign[u] = (idx, 1)
            self.key_to_param_and_sign[d] = (idx, -1)

        self.joysticks = {}

    def usage_str(self) -> str:
        return (
            f"""
Keyboard controller usage:

Hold (q) to quit
Use [ , ] to decrease/increase step size respectively
Tap _space_ to get a new random projection
Enter/Return , get a new training batch

This keybord controller configuration controlls a random {self.n} 
dimensional subspace of target model

To control each dimension use:
"""
            + "\n".join(
                [
                    f"dim{idx} : {self.paired_keys[idx][0]} +, {self.paired_keys[idx][1]} -"
                    for idx in range(self.n)
                ]
            )
            + "\nGOOD LUCK!\n"
        )

    def process_key_press(self, event):
        if event.type == pg.TEXTINPUT:  # event.type == pg.KEYDOWN or
            key = event.text

            if key == "q":
                self.quit_count += 1
                print("Keep holding to quit!")
                if self.quit_count > 4:
                    print("Quiting")
                    self.hudes_websocket_client.running = False
                return
            self.quit_count = 0

            if key in self.key_to_param_and_sign:
                dim, sign = self.key_to_param_and_sign[key]

                self.send_dims_and_steps({dim: self.step_size * sign})
                # lowrank_idx, sign = self.key_to_param_and_sign[key]
                # self.lowrank_state[lowrank_idx] += self.step_size * sign
                # TODO need to batch here before sending
                # send needs to be independent of this
            elif key == "[":
                self.step_size_decrease()
                self.send_config()
            elif key == "]":
                self.step_size_increase()
                self.send_config()
            elif key == " ":
                print("getting new set of vectors")
                self.hudes_websocket_client.send_q.put(
                    next_dims_message().SerializeToString()
                )
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                print("Getting new batch")
                self.hudes_websocket_client.send_q.put(
                    next_batch_message().SerializeToString()
                )

        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
            if event.button == 0:
                joystick = self.joysticks[event.instance_id]
                if joystick.rumble(0, 0.7, 500):
                    print(f"Rumble effect played on joystick {event.instance_id}")

        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")

        # Handle hotplugging
        if event.type == pygame.JOYDEVICEADDED:
            # This event will be generated when the program starts for every
            # joystick, filling up the list without needing to create them manually.
            joy = pygame.joystick.Joystick(event.device_index)
            self.joysticks[joy.get_instance_id()] = joy
            print(f"Joystick {joy.get_instance_id()} connencted")

        if event.type == pygame.JOYDEVICEREMOVED:
            del self.joysticks[event.instance_id]
            print(f"Joystick {event.instance_id} disconnected")

    def run_loop(self):
        i = 0
        last_dims_press = 0
        last_step_press = 0
        last_select_press = 0
        while self.hudes_websocket_client.running:
            # print("RUN")
            # check and send local interactions(?)
            redraw = False
            for event in pg.event.get():
                self.process_key_press(event)
                # Handle mouse button events
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button pressed
                        self.view.is_mouse_dragging = True
                        # print("DRAG")
                        self.view.last_mouse_pos = pygame.mouse.get_pos()
                        redraw = True

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button released
                        self.view.is_mouse_dragging = False
                        redraw = True
                        # print("STOP DRAG")

            angle_adjust = 0
            ct = time.time()
            for joystick in self.joysticks.values():
                axes = joystick.get_numaxes()
                # print(f"Number of axes: {axes}")
                # for i in range(axes):
                #     axis = joystick.get_axis(i)
                #     print(f"Axis {i} value: {axis:>6.3f}")

                if joystick.get_button(8) > 0.5:
                    self.view.reset_angle()
                    redraw = True

                if joystick.get_button(9) > 0.5:
                    self.step_size_decrease()
                    self.send_config()

                if joystick.get_button(10) > 0.5:
                    self.step_size_increase()
                    self.send_config()

                if joystick.get_button(2) > 0.5 and (ct - last_dims_press) > 1:
                    self.hudes_websocket_client.send_q.put(
                        next_dims_message().SerializeToString()
                    )
                    last_dims_press = ct

                if joystick.get_axis(4) > 0.5 and (ct - last_select_press) > 0.2:
                    self.view.decrement_selected_grid()
                    redraw = True
                    last_select_press = ct
                if joystick.get_axis(5) > 0.5 and (ct - last_select_press) > 0.2:
                    self.view.increment_selected_grid()
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
                            1 + selected_grid * 2: A[1] * self.step_size * 0.1,
                            0 + selected_grid * 2: A[0] * self.step_size * 0.1,
                        }
                    )

                radius, angle = B.as_polar()
                # print("B", radius, angle, B)
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
                # B_x = joystick.get_axis(2)
                # B_y = joystick.get_axis(3)
                # if math.sqrt(B_x**2 + B_y**2) > 0.2:
                #     print(math.atan2(B_x, B_y))
                # print(A_x, A_y, B_x, B_y)

            # print("VIEW DRAW")
            redraw = redraw | self.view.is_mouse_dragging | self.receive_messages()
            if redraw:
                self.view.draw()
            else:
                sleep(0.01)
            i += 1
