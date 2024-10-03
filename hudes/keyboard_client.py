import argparse
import math
from time import sleep

import pygame as pg

from hudes.hudes_client import HudesClient
from hudes.websocket_client import (
    mesh_grid_config_message,
    next_batch_message,
    next_dims_message,
)


class KeyboardClient(HudesClient):
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
            elif key == "]":
                self.step_size_increase()
            elif key == " ":
                print("getting new set of vectors")
                self.hudes_websocket_client.send_q.put(
                    next_dims_message().SerializeToString()
                )
            # elif key == "x":
            #     self.hudes_websocket_client.send_q.put(
            #         mesh_grid_config_message(
            #             dimA=0, dimB=1, grid_size=9, step_size=0.01
            #         ).SerializeToString()
            #     )
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                print("Getting new batch")
                self.hudes_websocket_client.send_q.put(
                    next_batch_message().SerializeToString()
                )
