import math
import pickle
from time import sleep

import numpy as np
import pygame as pg

from hudes import hudes_pb2
from hudes.view import View
from hudes.websocket_client import (
    HudesWebsocketClient,
    dims_and_steps_to_control_message,
    next_batch_message,
    next_dims_message,
)


class HudesClient:
    def __init__(
        self,
        step_size_resolution=-0.05,
        inital_step_size_idx=10,
        seed=0,
        mesh_grids=0,
        mesh_grid_size=31,
    ):
        self.mesh_grids = mesh_grids
        self.mesh_grid_size = mesh_grid_size
        self.quit_count = 0
        self.seed = seed

        self.hudes_websocket_client = HudesWebsocketClient("ws://localhost:8765")

        self.request_idx = 0

        self.train_steps = []
        self.train_losses = []

        self.val_steps = []
        self.val_losses = []

        self.init_input()

        self.max_log_step_size = 10
        self.min_log_step_size = -10
        self.inital_step_size_idx = inital_step_size_idx
        self.step_size_resolution = step_size_resolution

    def get_next_batch(self):
        self.hudes_websocket_client.send_q.put(next_batch_message().SerializeToString())

    def get_next_dims(self):
        self.hudes_websocket_client.send_q.put(next_dims_message().SerializeToString())
        self.zero_dims_and_steps_on_current_dims()
        self.dims_and_steps_updated()

    def zero_dims_and_steps_on_current_dims(self):
        self.dims_and_steps_on_current_dims = np.zeros(self.n)

    def set_n(self, n):
        self.n = n
        self.zero_dims_and_steps_on_current_dims()

    def attach_view(self, view):
        self.view = view

    def send_config(self):
        self.hudes_websocket_client.send_config(
            seed=self.seed,
            dims_at_a_time=self.n,
            mesh_step_size=self.step_size,
            mesh_grids=self.mesh_grids,
            mesh_grid_size=self.mesh_grid_size,
        )

    def dims_and_steps_updated(self):
        self.view.update_dims_since_last_update(self.dims_and_steps_on_current_dims)

    def step_size_increase(self):
        self.set_step_size_idx(self.step_size_idx + 1)

    def step_size_decrease(self):
        self.set_step_size_idx(self.step_size_idx - 1)

    def set_step_size_idx(self, idx):
        self.step_size_idx = idx
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
        for dim, step in dims_and_steps.items():
            self.dims_and_steps_on_current_dims[dim] += step
        self.dims_and_steps_updated()
        self.request_idx += 1

    def before_first_loop(self):
        self.set_step_size_idx(self.inital_step_size_idx)
        self.send_config()
        self.view.update_step_size(
            self.log_step_size, self.max_log_step_size, self.min_log_step_size
        )
        self.view.plot_train_and_val(
            self.train_losses,
            self.train_steps,
            self.val_losses,
            self.val_steps,
        )
        self.dims_and_steps_updated()

    def before_pg_event(self):
        pass

    def run_loop(self):
        self.before_first_loop()
        while self.hudes_websocket_client.running:
            # check and send local interactions(?)
            self.before_pg_event()
            redraw = False
            for event in pg.event.get():
                redraw |= self.process_key_press(event)

            redraw |= self.receive_messages()
            if redraw:
                self.view.draw()
            else:
                sleep(0.01)

    def receive_messages(self):
        # listen from server?
        received_message = False
        received_train = False
        received_batch = False
        received_val = False
        while self.hudes_websocket_client.recv_ready():
            received_message = True
            # recv and process!
            raw_msg = self.hudes_websocket_client.recv_msg()
            msg = hudes_pb2.Control()
            msg.ParseFromString(raw_msg)
            if msg.type == hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS:
                received_train = True

                self.train_preds = pickle.loads(msg.train_loss_and_preds.preds)
                self.confusion_matrix = pickle.loads(
                    msg.train_loss_and_preds.confusion_matrix
                )

                self.train_losses.append(msg.train_loss_and_preds.train_loss)
                self.train_steps.append(msg.request_idx)
                # self.val_losses.append(msg.loss_and_preds.val_loss)

            elif msg.type == hudes_pb2.Control.CONTROL_BATCH_EXAMPLES:
                received_batch = True
                self.train_data = pickle.loads(msg.batch_examples.train_data)
                self.val_data = pickle.loads(msg.batch_examples.val_data)
                self.train_labels = pickle.loads(msg.batch_examples.train_labels)
                self.val_labels = pickle.loads(msg.batch_examples.val_labels)

            elif msg.type == hudes_pb2.Control.CONTROL_VAL_LOSS:
                received_val = True
                self.val_losses.append(msg.val_loss.val_loss)
                self.val_steps.append(msg.request_idx)

            # called if we only changed scale etc?
            elif msg.type == hudes_pb2.Control.CONTROL_MESHGRID_RESULTS:
                self.view.update_mesh_grids(pickle.loads(msg.mesh_grid_results))
                # print("GOT MESH GRID", self.mesh_grid.shape)

        if received_message:
            if received_train:
                self.view.plot_train_and_val(
                    self.train_losses,
                    self.train_steps,
                    self.val_losses,
                    self.val_steps,
                )
                self.view.update_example_preds(train_preds=self.train_preds)
                self.view.update_confusion_matrix(self.confusion_matrix)
            if received_batch:
                self.view.update_examples(
                    train_data=self.train_data,
                )
            if received_val:
                self.view.plot_train_and_val(
                    self.train_losses,
                    self.train_steps,
                    self.val_losses,
                    self.val_steps,
                )
        return received_message
