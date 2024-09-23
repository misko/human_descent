import argparse
import math
import pickle

# if PLT_PYGAME:
#     matplotlib.use("module://pygame_matplotlib.backend_pygame")
from time import sleep

import matplotlib
import pygame as pg
import torch

from hudes import hudes_pb2
from hudes.client import (
    HudesWebsocketClient,
    dims_and_steps_to_control_message,
    next_batch_message,
    next_dims_message,
)
from hudes.view import View

# # matplotlib.use("Agg")
# PLT_PYGAME = True


class KeyboardClient:
    def __init__(self, log_step_size=-2, step_size_resolution=0.05):

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

        self.log_step_size = log_step_size
        self.max_log_step_size = 0
        self.min_log_step_size = -10
        self.step_size = math.pow(10, self.log_step_size)
        self.step_size_resolution = step_size_resolution

        self.quit_count = 0
        self.seed = 0

        self.hudes_client = HudesWebsocketClient("ws://localhost:8765")

        self.hudes_client.send_config(
            seed=self.seed, dims_at_a_time=len(self.paired_keys)
        )

        pg.init()
        self.window = pg.display.set_mode((1200, 900))
        self.running = True
        self.request_idx = 0

        self.train_losses = []
        self.val_losses = []

        self.view = View()

    def usage_str(self):
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

    def process_event(self, event):
        if event.type == pg.TEXTINPUT:  # event.type == pg.KEYDOWN or
            key = event.text

            if key == "q":
                self.quit_count += 1
                print("Keep holding to quit!")
                if self.quit_count > 10:
                    print("Quiting")
                    self.running = False
                return
            self.quit_count = 0

            if key in self.key_to_param_and_sign:
                dim, sign = self.key_to_param_and_sign[key]
                self.hudes_client.send_q.put(
                    dims_and_steps_to_control_message(
                        dims_and_steps={dim: self.step_size * sign},
                        request_idx=self.request_idx,
                    ).SerializeToString()
                )
                self.request_idx += 1
                # lowrank_idx, sign = self.key_to_param_and_sign[key]
                # self.lowrank_state[lowrank_idx] += self.step_size * sign
                # TODO need to batch here before sending
                # send needs to be independent of this
            elif key == "[":
                self.log_step_size = min(
                    max(
                        self.log_step_size - self.step_size_resolution,
                        self.min_log_step_size,
                    ),
                    self.max_log_step_size,
                )
                self.step_size = math.pow(10, self.log_step_size)
                print(
                    f"Step size: {self.step_size} , Log step size: {self.log_step_size}"
                )
                self.view.update_step_size(
                    self.log_step_size, self.max_log_step_size, self.min_log_step_size
                )
            elif key == "]":
                self.log_step_size = min(
                    max(
                        self.log_step_size + self.step_size_resolution,
                        self.min_log_step_size,
                    ),
                    self.max_log_step_size,
                )
                self.step_size = math.pow(10, self.log_step_size)
                print(
                    f"Step size: {self.step_size} , Log step size: {self.log_step_size}"
                )
                self.view.update_step_size(
                    self.log_step_size, self.max_log_step_size, self.min_log_step_size
                )
            elif key == " ":
                print("getting new set of vectors")
                self.hudes_client.send_q.put(next_dims_message().SerializeToString())
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                print("Getting new batch")
                self.hudes_client.send_q.put(next_batch_message().SerializeToString())
                # self.q.put(("new batch", None))

    def run_loop(self):
        self.view.update_top()
        self.view.update_step_size(
            self.log_step_size, self.max_log_step_size, self.min_log_step_size
        )
        self.view.plot_train_and_val(self.train_losses, self.val_losses)
        while self.running:
            # check and send local interactions(?)
            for event in pg.event.get():
                self.process_event(event)

            # listen from server?
            while self.hudes_client.recv_ready():
                # recv and process!
                raw_msg = self.hudes_client.recv_msg()
                msg = hudes_pb2.Control()
                msg.ParseFromString(raw_msg)

                if msg.type == hudes_pb2.Control.CONTROL_LOSS_AND_PREDS:

                    train_preds = pickle.loads(msg.loss_and_preds.preds)

                    self.train_losses.append(msg.loss_and_preds.train_loss)
                    self.val_losses.append(msg.loss_and_preds.val_loss)

                    self.view.plot_train_and_val(self.train_losses, self.val_losses)
                    self.view.update_example_preds(train_preds=train_preds)
                elif msg.type == hudes_pb2.Control.CONTROL_BATCH_EXAMPLES:
                    self.train_data = pickle.loads(msg.batch_examples.train_data)
                    self.val_data = pickle.loads(msg.batch_examples.val_data)
                    self.train_labels = pickle.loads(msg.batch_examples.train_labels)
                    self.val_labels = pickle.loads(msg.batch_examples.val_labels)
                    self.view.update_examples(
                        train_data=self.train_data,
                        val_data=self.val_data,
                    )

                # print(self.train_losses)

            # msg.ParseFromString(msg)
            # draw something?

            self.view.draw()
            sleep(0.001)  # give the model a chance
            # print("UPDATE")
            # pg.display.update()


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--input", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    controller = KeyboardClient()

    controller.run_loop()
    # pg.init()
    # screen = pg.display.set_mode((1200, 900))
    # while True:
    #     print("RUN")
    #     for event in pg.event.get():
    #         print(event)
    #         # elf.process_event(event)
    #     sleep(0.001)  # give the model a chance


if __name__ == "__main__":
    main()
