import logging
import math
import pickle
import time
from time import sleep

import numpy as np
import pygame as pg

from hudes import hudes_pb2
from hudes.websocket_client import (
    HudesWebsocketClient,
    dims_and_steps_to_control_message,
    next_batch_message,
    next_dims_message,
    sgd_step_message,
)


class HudesClient:
    def __init__(
        self,
        addr,
        port,
        step_size_resolution=-0.05,
        inital_step_size_idx=10,
        seed=0,
        mesh_grids=0,
        mesh_grid_size=31,
        joystick_controller_key=None,
    ):
        self.joystick_controller_key = joystick_controller_key

        self.mesh_grids = mesh_grids
        self.mesh_grid_size = mesh_grid_size
        self.quit_count = 0
        self.seed = seed

        self.hudes_websocket_client = HudesWebsocketClient(f"ws://{addr}:{port}")

        self.request_idx = 0

        self.train_steps = []
        self.train_losses = []

        self.val_steps = []
        self.val_losses = []

        self.max_log_step_size = 10
        self.min_log_step_size = -10
        self.inital_step_size_idx = inital_step_size_idx
        self.step_size_resolution = step_size_resolution

        self.batch_size = 128

        self.dtype_idx = -1
        self.dtypes = ("float16", "float32")

        self.batch_size_idx = 3 - 1
        self.batch_sizes = [2, 8, 32, 128, 512]
        self.view = None
        self.init_input()

    def get_sgd(self):
        self.hudes_websocket_client.send_q.put(
            sgd_step_message(1, request_idx=self.request_idx).SerializeToString()
        )
        self.request_idx += 1
        print("SGD REQ IDX", self.request_idx)

    def get_next_batch(self):
        self.hudes_websocket_client.send_q.put(
            next_batch_message(request_idx=self.request_idx).SerializeToString()
        )
        self.request_idx += 1

    def get_next_dims(self):
        self.hudes_websocket_client.send_q.put(next_dims_message().SerializeToString())
        self.zero_dims_and_steps_on_current_dims()
        self.dims_and_steps_updated()

    def zero_dims_and_steps_on_current_dims(self):
        self.dims_and_steps_on_current_dims = np.zeros(self.n)

    def set_n(self, n):
        n = max(6, n)
        logging.debug(f"hudes_client: set_n {n}")
        self.n = n
        self.zero_dims_and_steps_on_current_dims()

    def attach_view(self, view):
        self.view = view
        self.set_batch_size(self.batch_size)
        self.set_dtype(self.dtype)

    def set_dtype(self, dtype):
        self.dtype_idx = self.dtypes.index(dtype)
        self.dtype = dtype
        if self.view is not None:
            self.view.dtype = dtype

    def set_batch_size(self, batch_size):
        self.batch_size_idx = self.batch_sizes.index(batch_size)
        self.batch_size = batch_size
        if self.view is not None:
            self.view.batch_size = batch_size

    def toggle_dtype(self):
        self.dtype_idx = (self.dtype_idx + 1) % len(self.dtypes)
        self.dtype = self.dtypes[self.dtype_idx]
        self.view.dtype = self.dtype
        self.send_config()

    def toggle_batch_size(self):
        self.batch_size_idx = (self.batch_size_idx + 1) % len(self.batch_sizes)
        self.batch_size = self.batch_sizes[self.batch_size_idx]
        self.view.batch_size = self.batch_size
        self.send_config()

    def send_config(self):
        self.hudes_websocket_client.send_config(
            seed=self.seed,
            dims_at_a_time=self.n,
            mesh_step_size=self.step_size,
            mesh_grids=self.mesh_grids,
            mesh_grid_size=self.mesh_grid_size,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

    def dims_and_steps_updated(self):
        self.view.update_dims_since_last_update(self.dims_and_steps_on_current_dims)

    def step_size_increase(self, mag: int = 1):
        self.set_step_size_idx(self.step_size_idx + 1 * mag)
        self.send_config()

    def step_size_decrease(self, mag: int = 1):
        self.set_step_size_idx(self.step_size_idx - 1 * mag)
        self.send_config()

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
        time.sleep(0.01)
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
        self.get_next_batch()

    def before_pg_event(self):
        pass

    def run_loop(self):
        self.before_first_loop()
        while self.hudes_websocket_client.running:
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
        received_message = False
        received_train = False
        received_batch = False
        received_val = False
        while self.hudes_websocket_client.recv_ready():
            received_message = True
            raw_msg = self.hudes_websocket_client.recv_msg()
            msg = hudes_pb2.Control()
            msg.ParseFromString(raw_msg)
            if msg.type == hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS:
                logging.debug("hudes_client: recieve message : loss and preds")
                received_train = True

                self.train_preds = pickle.loads(msg.train_loss_and_preds.preds)
                self.confusion_matrix = pickle.loads(
                    msg.train_loss_and_preds.confusion_matrix
                )

                if len(self.train_steps) == 0 or self.train_steps[-1] < msg.request_idx:
                    self.train_losses.append(msg.train_loss_and_preds.train_loss)
                    self.train_steps.append(msg.request_idx)
                logging.debug("hudes_client: recieve message : loss and preds : done")

            elif msg.type == hudes_pb2.Control.CONTROL_BATCH_EXAMPLES:
                logging.debug("hudes_client: recieve message : examples")
                received_batch = True
                self.train_data = pickle.loads(msg.batch_examples.train_data)
                self.train_labels = pickle.loads(msg.batch_examples.train_labels)
                logging.debug("hudes_client: recieve message : done")

            elif msg.type == hudes_pb2.Control.CONTROL_VAL_LOSS:
                logging.debug("hudes_client: recieve message : val loss")
                received_val = True
                self.val_losses.append(msg.val_loss.val_loss)
                self.val_steps.append(msg.request_idx)
                logging.debug("hudes_client: recieve message : val loss : done")

            # called if we only changed scale etc?
            elif msg.type == hudes_pb2.Control.CONTROL_MESHGRID_RESULTS:
                self.view.update_mesh_grids(pickle.loads(msg.mesh_grid_results))

        if received_message:
            if received_train:
                logging.debug("hudes_client: recieve message : render train")
                self.view.plot_train_and_val(
                    self.train_losses,
                    self.train_steps,
                    self.val_losses,
                    self.val_steps,
                )
                self.view.update_example_preds(train_preds=self.train_preds)
                self.view.update_confusion_matrix(self.confusion_matrix)
                logging.debug("hudes_client: recieve message : render train done")
            if received_batch:
                logging.debug("hudes_client: recieve message : render batch")
                self.view.update_examples(
                    train_data=self.train_data,
                )
                logging.debug("hudes_client: recieve message : render batch done")
            if received_val:
                logging.debug("hudes_client: recieve message : render val")
                self.view.plot_train_and_val(
                    self.train_losses,
                    self.train_steps,
                    self.val_losses,
                    self.val_steps,
                )
                logging.debug("hudes_client: recieve message : render val done")
        return received_message
