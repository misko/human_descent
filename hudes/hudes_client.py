import math
import pickle
from time import sleep

import pygame as pg

from hudes import hudes_pb2
from hudes.view import View
from hudes.websocket_client import (
    HudesWebsocketClient,
    dims_and_steps_to_control_message,
)


class HudesClient:
    def __init__(self, step_size_resolution=-0.05, seed=0):

        self.quit_count = 0
        self.seed = seed

        self.hudes_websocket_client = HudesWebsocketClient("ws://localhost:8765")

        pg.init()
        self.window = pg.display.set_mode((1200, 900))
        self.request_idx = 0

        self.train_steps = []
        self.train_losses = []

        self.val_steps = []
        self.val_losses = []

        self.init_input()

        self.hudes_websocket_client.send_config(seed=self.seed, dims_at_a_time=self.n)

        self.view = View()

        self.max_log_step_size = 0
        self.min_log_step_size = -10
        self.step_size_resolution = step_size_resolution

        self.set_step_size_idx(10)

        self.view.update_step_size(
            self.log_step_size, self.max_log_step_size, self.min_log_step_size
        )
        self.view.plot_train_and_val(
            self.train_losses,
            self.train_steps,
            self.val_losses,
            self.val_steps,
        )

    def step_size_increase(self):
        self.set_step_size_idx(self.step_size_idx + 1)

    def step_size_decrease(self):
        self.set_step_size_idx(self.step_size_idx - 1)

    def set_step_size_idx(self, idx):
        self.step_size_idx = max(idx, 0)
        self.log_step_size = min(
            max(
                self.step_size_idx * self.step_size_resolution,
                self.min_log_step_size,
            ),
            self.max_log_step_size,
        )
        self.step_size = math.pow(10, self.log_step_size)
        print(f"Step size: {self.step_size} , Log step size: {self.log_step_size}")
        self.view.update_step_size(
            self.log_step_size, self.max_log_step_size, self.min_log_step_size
        )

    def send_dims_and_steps(self, dims_and_steps):
        self.hudes_websocket_client.send_q.put(
            dims_and_steps_to_control_message(
                dims_and_steps=dims_and_steps,
                request_idx=self.request_idx,
            ).SerializeToString()
        )
        self.request_idx += 1

    def run_loop(self):
        while self.hudes_websocket_client.running:
            # check and send local interactions(?)
            for event in pg.event.get():
                self.process_key_press(event)

            redraw = self.receive_messages()
            if redraw:
                self.view.draw()
            else:
                sleep(0.01)

    def receive_messages(self):
        # listen from server?
        received_message = False
        while self.hudes_websocket_client.recv_ready():
            received_message = True
            # recv and process!
            raw_msg = self.hudes_websocket_client.recv_msg()
            msg = hudes_pb2.Control()
            msg.ParseFromString(raw_msg)
            if msg.type == hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS:

                train_preds = pickle.loads(msg.train_loss_and_preds.preds)
                confusion_matrix = pickle.loads(
                    msg.train_loss_and_preds.confusion_matrix
                )

                self.train_losses.append(msg.train_loss_and_preds.train_loss)
                self.train_steps.append(msg.request_idx)
                # self.val_losses.append(msg.loss_and_preds.val_loss)

                self.view.plot_train_and_val(
                    self.train_losses,
                    self.train_steps,
                    self.val_losses,
                    self.val_steps,
                )
                self.view.update_example_preds(train_preds=train_preds)
                self.view.update_confusion_matrix(confusion_matrix)
            elif msg.type == hudes_pb2.Control.CONTROL_BATCH_EXAMPLES:
                self.train_data = pickle.loads(msg.batch_examples.train_data)
                self.val_data = pickle.loads(msg.batch_examples.val_data)
                self.train_labels = pickle.loads(msg.batch_examples.train_labels)
                self.val_labels = pickle.loads(msg.batch_examples.val_labels)
                self.view.update_examples(
                    train_data=self.train_data,
                    val_data=self.val_data,
                )
            elif msg.type == hudes_pb2.Control.CONTROL_VAL_LOSS:
                self.val_losses.append(msg.val_loss.val_loss)
                self.val_steps.append(msg.request_idx)

                self.view.plot_train_and_val(
                    self.train_losses,
                    self.train_steps,
                    self.val_losses,
                    self.val_steps,
                )
        return received_message
