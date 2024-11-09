import logging
import math
import os
import pickle
import time
from dataclasses import dataclass
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


class ClientState:

    def __init__(self, step_size_resolution, initial_step_size_idx, help_screen_idx=0):
        self.best_score = math.inf
        self.sgd_steps = 0

        self.dtype_idx = -1
        self.dtypes = ("float16", "float32")

        self.batch_size_idx = 3 - 1
        self.batch_sizes = [2, 8, 32, 64, 128, 256, 512]

        self.n = 6

        self.max_log_step_size = 2
        self.min_log_step_size = -4
        self.inital_step_size_idx = initial_step_size_idx
        self.step_size_resolution = step_size_resolution

        self.set_step_size_idx(initial_step_size_idx)

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.help_screen_idx = help_screen_idx
        self.help_screen_fns = []
        self.dims_used = 0

    def set_help_screen_fns(self, help_screen_fns):
        self.help_screen_fns = [
            os.path.join(self.current_dir, fn) for fn in help_screen_fns
        ]

    def set_dtype(self, dtype):
        self.dtype_idx = self.dtypes.index(dtype)
        self.dtype = dtype

    def set_batch_size(self, batch_size):
        self.batch_size_idx = self.batch_sizes.index(batch_size)
        self.batch_size = batch_size

    def toggle_dtype(self):
        self.dtype_idx = (self.dtype_idx + 1) % len(self.dtypes)
        self.dtype = self.dtypes[self.dtype_idx]

    def toggle_batch_size(self):
        self.batch_size_idx = (self.batch_size_idx + 1) % len(self.batch_sizes)
        self.batch_size = self.batch_sizes[self.batch_size_idx]

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
        logging.debug(
            f"Step size: {self.step_size} , Log step size: {self.log_step_size}"
        )

    def set_n(self, n):
        n = max(6, n)
        logging.debug(f"client_state: set_n {n}")
        self.n = n

    def step_size_increase(self, mag: int = 1):
        self.set_step_size_idx(self.step_size_idx + 1 * mag)

    def step_size_decrease(self, mag: int = 1):
        self.set_step_size_idx(self.step_size_idx - 1 * mag)

    def next_help_screen(self):
        self.help_screen_idx += 1
        if self.help_screen_idx == len(self.help_screen_fns):
            self.help_screen_idx = -1


@dataclass
class KeyHold:
    first_press: float = 0
    last_exec: float = 0


class HudesClient:
    def __init__(
        self,
        addr,
        port,
        seed=0,
        mesh_grids=0,
        mesh_grid_size=31,
        joystick_controller_key=None,
        step_size_resolution=-0.05,
        inital_step_size_idx=12,
        skip_help=False,
    ):
        self.joystick_controller_key = joystick_controller_key

        self.mesh_grids = mesh_grids
        self.mesh_grid_size = mesh_grid_size
        self.seed = seed

        self.hudes_websocket_client = HudesWebsocketClient(f"ws://{addr}:{port}")

        self.request_idx = 0

        self.train_steps = []
        self.train_losses = []

        self.val_steps = []
        self.val_losses = []

        self.dims_used = 0

        self.client_state = ClientState(
            step_size_resolution=step_size_resolution,
            initial_step_size_idx=inital_step_size_idx,
            help_screen_idx=-1 if skip_help else 0,
        )

        self.view = None
        self.step_size_keyboard_multiplier = 1

        self.key_holds: dict[int, KeyHold] = {}
        for key in (
            pg.K_LSHIFT,
            pg.K_RSHIFT,
            pg.K_SPACE,
            pg.K_QUOTE,
            pg.K_RETURN,
            pg.K_SEMICOLON,
            pg.K_SLASH,
            pg.K_x,
            pg.K_q,
        ):
            self.add_key_to_watch(key=key)

        self.key_seconds_pressed = 0.2

        self.init_input()

    def add_key_to_watch(self, key):
        self.key_holds[key] = KeyHold()

    def get_sgd(self):
        self.hudes_websocket_client.send_q.put(
            sgd_step_message(1, request_idx=self.request_idx).SerializeToString()
        )
        self.request_idx += 1

    def get_next_batch(self):
        self.hudes_websocket_client.send_q.put(
            next_batch_message(request_idx=self.request_idx).SerializeToString()
        )
        self.request_idx += 1

    def get_next_dims(self):
        self.hudes_websocket_client.send_q.put(next_dims_message().SerializeToString())
        self.zero_dims_and_steps_on_current_dims()
        self.dims_and_steps_updated()
        self.client_state.dims_used += self.client_state.n

    def zero_dims_and_steps_on_current_dims(self):
        self.dims_and_steps_on_current_dims = np.zeros(self.client_state.n)

    def set_n(self, n):
        self.client_state.set_n(n)
        self.zero_dims_and_steps_on_current_dims()

    def attach_view(self, view):
        self.view = view
        self.view.client_state = self.client_state

    def send_config(self):
        self.hudes_websocket_client.send_config(
            seed=self.seed,
            dims_at_a_time=self.client_state.n,
            mesh_step_size=self.client_state.step_size,
            mesh_grids=self.mesh_grids,
            mesh_grid_size=self.mesh_grid_size,
            batch_size=self.client_state.batch_size,
            dtype=self.client_state.dtype,
        )

    def dims_and_steps_updated(self):
        self.view.update_dims_since_last_update(self.dims_and_steps_on_current_dims)

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
        self.get_next_batch()
        self.send_config()
        self.view.update_step_size()
        self.view.plot_train_and_val(
            self.train_losses,
            self.train_steps,
            self.val_losses,
            self.val_steps,
        )
        self.dims_and_steps_updated()

    def before_pg_event(self):
        pass

    def check_cooldown_time(self, event, key, ct, cool_down_time):
        cool_down_time *= 1000  # convert to ms
        if event.key != key:
            return False
        if ct - self.key_holds[key].last_exec > cool_down_time:
            self.key_holds[key].last_exec = ct
            return True
        return False

    # check if time held down exceeds wait time, if so reurn True and reset
    def check_keyhold_time(self, event, key, ct, wait_time):
        wait_time *= 1000  # convert to ms
        if event.key != key:
            return False
        if (
            self.key_holds[key].first_press >= 0
            and (ct - self.key_holds[key].first_press) > wait_time
        ):
            return True
        return False

    def update_key_holds(self):
        ct = pg.time.get_ticks()
        currently_pressing = pg.key.get_pressed()
        for key in self.key_holds:
            if currently_pressing[key]:
                if self.key_holds[key].first_press == -1:
                    self.key_holds[key].first_press = ct
            elif key in self.key_holds:
                self.key_holds[key].first_press = -1

    def quit(self):
        logging.debug("quitting...")
        self.hudes_websocket_client.running = False

    def process_common_keys(self, event):
        ct = pg.time.get_ticks()
        if event.type == pg.KEYDOWN:

            if self.check_cooldown_time(
                event=event, key=pg.K_x, ct=ct, cool_down_time=0.5
            ):
                logging.debug("calling next_help_screen...")
                self.client_state.next_help_screen()
                return True

            if self.client_state.help_screen_idx != -1:
                return False  # we are in help screen mode

            if self.check_keyhold_time(event=event, key=pg.K_q, ct=ct, wait_time=1.0):
                self.quit()
                return False

            if self.check_cooldown_time(
                event=event, key=pg.K_SLASH, ct=ct, cool_down_time=0.1
            ):
                self.get_sgd()
            elif self.check_cooldown_time(
                event=event,
                key=pg.K_SPACE,
                ct=ct,
                cool_down_time=self.key_seconds_pressed,
            ):
                logging.debug("getting new set of vectors")
                self.get_next_dims()
            elif event.key == pg.K_LEFTBRACKET:
                self.client_state.step_size_increase(self.step_size_keyboard_multiplier)
                self.view.update_step_size()
                self.send_config()
                logging.debug("keyboard_client: increase step size")
                return True
            elif event.key == pg.K_RIGHTBRACKET:
                self.client_state.step_size_decrease(self.step_size_keyboard_multiplier)
                self.view.update_step_size()
                self.send_config()
                logging.debug("keyboard_client: decrease step size")
                return True
            elif self.check_cooldown_time(
                event=event,
                key=pg.K_QUOTE,
                ct=ct,
                cool_down_time=self.key_seconds_pressed,
            ):
                self.client_state.toggle_dtype()
                self.send_config()
            elif self.check_cooldown_time(
                event=event,
                key=pg.K_SEMICOLON,
                ct=ct,
                cool_down_time=0.5,
            ):
                self.client_state.toggle_batch_size()
                self.send_config()
            elif self.check_cooldown_time(
                event=event,
                key=pg.K_RETURN,
                ct=ct,
                cool_down_time=self.key_seconds_pressed,
            ):
                self.get_next_batch()
        return False

    def run_loop(self):
        self.before_first_loop()
        while self.hudes_websocket_client.running:
            self.before_pg_event()
            redraw = False
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    self.update_key_holds()
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
                logging.debug("hudes_client: receive message : loss and preds")
                received_train = True
                # Reconstruct the train predictions and confusion matrix from the protobuf message
                self.train_preds = np.array(msg.train_loss_and_preds.preds).reshape(
                    msg.train_loss_and_preds.preds_shape
                )
                logging.debug(f"hudes_client: len train preds {len(self.train_preds)}")

                self.confusion_matrix = np.array(
                    msg.train_loss_and_preds.confusion_matrix
                ).reshape(msg.train_loss_and_preds.confusion_matrix_shape)
                logging.debug(
                    f"hudes_client: confusion {msg.train_loss_and_preds.confusion_matrix_shape}"
                )

                self.client_state.sgd_steps = msg.total_sgd_steps

                # Update the train losses and steps
                if len(self.train_steps) == 0 or self.train_steps[-1] < msg.request_idx:
                    self.train_losses.append(msg.train_loss_and_preds.train_loss)
                    self.train_steps.append(msg.request_idx)

                logging.debug("hudes_client: receive message : loss and preds : done")

            elif msg.type == hudes_pb2.Control.CONTROL_BATCH_EXAMPLES:
                logging.debug("hudes_client: receive message : examples")
                received_batch = True

                # Reshape train data and labels using numpy and convert them to PyTorch tensors
                self.train_data = np.array(msg.batch_examples.train_data).reshape(
                    list(msg.batch_examples.train_data_shape)
                )
                self.train_labels = np.array(msg.batch_examples.train_labels).reshape(
                    list(msg.batch_examples.train_labels_shape)
                )

                logging.debug("hudes_client: receive message : done")

            elif msg.type == hudes_pb2.Control.CONTROL_VAL_LOSS:
                logging.debug("hudes_client: recieve message : val loss")
                received_val = True
                self.val_losses.append(msg.val_loss.val_loss)
                self.val_steps.append(msg.request_idx)
                if self.val_losses[-1] < self.client_state.best_score:
                    self.client_state.best_score = self.val_losses[-1]
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
