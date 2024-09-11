from threading import Thread
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib

# import matplotlib.backends.backend_agg as agg
import argparse

# matplotlib.use("Qt5Agg")
from akai_pro_py import controllers

# matplotlib.use("Agg")
matplotlib.use("module://pygame_matplotlib.backend_pygame")
from torch import nn
import torch
import torch
import pygame as pg
import math
import matplotlib.pyplot as plt
import time
from time import sleep

from multiprocessing import Process, Queue

running = True

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def get_basic_mnist_model():
    mnist_width_height = 28
    mnist_classes = 10
    hidden = 32
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(mnist_width_height * mnist_width_height, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, mnist_classes),
        nn.LogSoftmax(dim=1),
    )


def subspace_basis(model, seed, dim):
    torch.manual_seed(seed)
    return [torch.rand(dim, *x.shape) - 0.5 for x in model.parameters()]


class View:
    def __init__(self):
        # plt.ion()
        pg.init()
        self.screen = pg.display.set_mode((1200, 900))

        my_dpi = 96
        self.loss_fig, self.loss_axs = plt.subplots(
            1, 2, figsize=(800 / my_dpi, 400 / my_dpi), dpi=my_dpi
        )
        self.n_digits = 2
        self.digit_fig, self.digit_axs = plt.subplots(
            2, self.n_digits, figsize=(800 / my_dpi, 400 / my_dpi), dpi=my_dpi
        )

        self.loss_fig.set_tight_layout(True)
        self.digit_fig.set_tight_layout(True)

        self.loss_fig.canvas.draw()
        self.digit_fig.canvas.draw()
        self.screen.blit(self.loss_fig, (10, 10))
        self.screen.blit(self.digit_fig, (10, 410))

        self.last_imshow = None

    def render_loss(self, name, loss, ax):
        x = [_x[0] for _x in loss]
        y = [_x[1] for _x in loss]
        ax.plot(x, y, label=name)
        self.loss_fig.canvas.draw()
        self.screen.blit(self.loss_fig, (10, 10))

    def render_digits(self, data, probs):
        render_imshow = False
        if self.last_imshow is None or self.last_imshow != data[0].mean():
            render_imshow = True
        for idx in range(self.n_digits):
            self.digit_axs[1, idx].cla()
            self.digit_axs[1, idx].plot(probs[idx])
            if render_imshow:
                print("RENDER IMSHOW")
                self.digit_axs[0, idx].cla()
                self.digit_axs[0, idx].imshow(data[idx])
        self.last_imshow = data[0].mean()
        self.digit_fig.canvas.draw()
        self.screen.blit(self.digit_fig, (10, 410))

    def render_losses(self, losses, new_batch_steps):
        for a_idx in [0, 1]:
            self.loss_axs[a_idx].cla()
            self.loss_axs[a_idx].set_xlabel("Step")
            self.loss_axs[a_idx].set_ylabel("Loss")
            self.loss_axs[a_idx].set_title("Loss vs step")
            self.loss_axs[a_idx].legend()

        for idx in new_batch_steps:
            self.loss_axs[0].axvline(x=idx, color="red")

        self.render_loss("train", losses["train"], self.loss_axs[0])
        self.render_loss("val", losses["val"], self.loss_axs[0])

        def subset_losses(losses, start_idx, end_idx):
            return [x for x in losses if x[0] >= start_idx and x[0] <= end_idx]

        last_train_step_idx = losses["train"][-1][0]
        self.render_loss(
            "train",
            subset_losses(
                losses["train"],
                start_idx=last_train_step_idx - 5,
                end_idx=last_train_step_idx,
            ),
            self.loss_axs[1],
        )
        # self.render_loss("val", losses["val"], self.axs[1])
        # self.axs.draw()
        # plt.draw()
        # plt.show()
        # plt.pause(0.1)
        # plt.show()

        # time.sleep(0.1)


class DescentModel:

    def get_batch(self, ds, batch_idx):
        start_idx = batch_idx * self.train_batch_size
        end_idx = min(len(ds), start_idx + self.train_batch_size)
        idxs = torch.arange(start_idx, end_idx)
        data_tensor = torch.vstack([ds[idx][0] for idx in idxs])
        label_tensor = torch.tensor([ds[idx][1] for idx in idxs])
        return data_tensor, label_tensor

    def fetch_and_set_train_and_val_batch_tensors(self):
        self.train_data_tensor, self.train_label_tensor = self.get_batch(
            self.train_raw, self.current_batch_idx
        )
        self.val_data_tensor, self.val_label_tensor = self.get_batch(
            self.val_raw, self.current_batch_idx
        )

    def __init__(self, pytorch_model, view, train_batch_size, val_batch_size, seed):
        torch.manual_seed(0)
        self.train_batch_size = train_batch_size  # number of samples to use
        self.val_batch_size = val_batch_size  # number of samples to use
        self.view = view
        self.pytorch_model = pytorch_model
        self.saved_weights = [x.clone() for x in pytorch_model.parameters()]

        self.train_raw = MNIST(".", train=True, transform=transform, download=True)
        self.train_batches = math.ceil(len(self.train_raw) / self.train_batch_size)

        self.val_raw = MNIST(".", train=False, transform=transform, download=True)
        self.val_batches = math.ceil(len(self.val_raw) / self.val_batch_size)

        self.current_batch_idx = 0
        self.fetch_and_set_train_and_val_batch_tensors()

        self.step = 0
        self.train_losses = []
        self.val_losses = []
        self.new_batch_steps = []

        self.input_q = Queue()

    def set_parameters(self, weights):
        with torch.no_grad():
            idx = 0
            for param in self.pytorch_model.parameters():
                # assert param.shape == weights[idx].shape
                # param.copy_(weights[idx])
                param *= 0
                param += weights[idx]
                idx += 1

    def apply_update(self, seed, lowrank_params):
        weights = [x.clone() for x in self.saved_weights]
        basis = subspace_basis(self.pytorch_model, seed, lowrank_params.shape[0])
        # print(basis[0])
        delta = [
            (
                basis[idx].reshape(lowrank_params.shape[0], -1)
                * lowrank_params.reshape(-1, 1)
            )
            .sum(axis=0)
            .reshape(basis[idx].shape[1:])
            for idx in range(len(basis))
        ]
        weights = [weights[idx] + delta[idx] for idx in range(len(weights))]
        self.set_parameters(weights)

    def get_loss_and_pred(self, data_tensor, label_tensor):
        pred = self.pytorch_model(data_tensor)
        return -pred[torch.arange(data_tensor.shape[0]), label_tensor].mean(), pred

    def get_losses(self):
        val_loss, val_pred = self.get_loss_and_pred(
            self.val_data_tensor, self.val_label_tensor
        )
        train_loss, train_pred = self.get_loss_and_pred(
            self.train_data_tensor, self.train_label_tensor
        )
        return {
            "val": val_loss,
            "val_pred": val_pred,
            "train": train_loss,
            "train_pred": train_pred,
        }

    def record_losses(self, losses):
        self.train_losses.append((self.step, losses["train"].item()))
        self.val_losses.append((self.step, losses["val"].item()))
        self.view.render_losses(
            {"train": self.train_losses, "val": self.val_losses}, self.new_batch_steps
        )

    def print_losses(self, losses):
        print(
            f"Loss (-LogP): Train ( {losses['train'].item():0.8f} ) , Val ( {losses['val'].item():0.8f} )"
        )

    def do_losses(self):
        losses = self.get_losses()
        self.record_losses(losses)
        self.view.render_digits(
            self.train_data_tensor, losses["train_pred"].detach().numpy()
        )
        # plt.pause(0.1)
        self.print_losses(losses)

    def update(self, cmd, data):
        print("UPDATE")
        # create a coroutine for a blocking function
        self.apply_update(data[0], data[1])
        if cmd == "save":
            self.saved_weights = [x.clone() for x in self.pytorch_model.parameters()]

    def run_loop(self):
        while True:
            cmd, data = self.input_q.get()
            if cmd == "quit":
                print("return from model run loop")
                return
            elif cmd == "update" or cmd == "save":
                self.update(cmd, data)
                if self.input_q.empty():
                    self.step += 1
                    self.do_losses()
            elif cmd == "new batch":
                print("New batch")
                self.new_batch_steps.append(self.step)
                self.current_batch_idx += 1
                self.fetch_and_set_train_and_val_batch_tensors()
                losses = self.get_losses()
                self.record_losses(losses)
                self.print_losses(losses)


def key_to_pg_key(key):
    return getattr(pg, f"K_{key.lower()}")


class KeyboardController:
    def __init__(self, q, step_size=0.01, step_size_resolution=0.0005):
        self.paired_keys = [
            ("w", "s"),
            ("e", "d"),
            ("r", "f"),
            ("u", "j"),
            ("i", "k"),
            ("o", "l"),
        ]
        self.n = len(self.paired_keys)

        self.q = q
        self.key_to_param_and_sign = {}
        for idx in range(self.n):
            u, d = self.paired_keys[idx]
            self.key_to_param_and_sign[u] = (idx, 1)
            self.key_to_param_and_sign[d] = (idx, -1)
        self.step_size = step_size
        self.step_size_resolution = step_size_resolution

        self.lowrank_state = torch.zeros(self.n)
        self.quit_count = 0
        self.seed = 0

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
        global running
        if event.type == pg.TEXTINPUT:  # event.type == pg.KEYDOWN or
            key = event.text

            if key == "q":
                self.quit_count += 1
                print("Keep holding to quit!")
                if self.quit_count > 10:
                    print("Quiting")
                    self.q.put(("quit", None))
                    running = False
                return
            self.quit_count = 0

            if key in self.key_to_param_and_sign:
                lowrank_idx, sign = self.key_to_param_and_sign[key]
                self.lowrank_state[lowrank_idx] += self.step_size * sign
                self.q.put(("update", (self.seed, self.lowrank_state.clone())))
            elif key == "[":
                self.step_size = max(self.step_size - self.step_size_resolution, 0)
                print(f"Step size: {self.step_size}")
            elif key == "]":
                self.step_size = self.step_size + self.step_size_resolution
                print(f"Step size: {self.step_size}")
            elif key == " ":
                print("getting new set of vectors")
                self.q.put(("save", (self.seed, self.lowrank_state.clone())))
                self.lowrank_state *= 0.0
                self.seed += 1337
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                print("Getting new batch")
                self.q.put(("new batch", None))

    def run_loop(self):
        global running
        while running:
            for event in pg.event.get():
                self.process_event(event)
            sleep(0.05)  # give the model a chance
            pg.display.update()


class AkaiMixerController:
    def __init__(self, q, step_size=0.01, step_size_resolution=0.0005):
        print("STARTED!!!")

        self.n = 32
        self.step_size = step_size
        self.step_size_resolution = step_size_resolution

        self.q = q
        self.first_boot = True

        self.quit_count = 0
        self.seed = 0

        # Define the MIDI Mix and APC Mini, first argument: MIDI in, second argument: MIDI out
        self.midi_mix = controllers.MIDIMix("MIDI Mix", "MIDI Mix")

        self.midi_mix.reset()

        self.midi_mix.on_event(self.process_midi_event)

        # @midi_mix.on_event
        # def test(x):
        #     print(x)
        self.fader_id_step_size = 8
        self.lowrank_state = torch.zeros(self.n)
        self.board_state = {}
        print("DONE INIT")

    def usage_str(self):
        return f"""
AKAI controller usage:

"""

    def board_state_to_vector(self):
        return torch.tensor(
            [self.board_state[k] for k in sorted(self.board_state.keys())]
        )

    def process_midi_event(self, event):
        print(event)
        board_state_vector = self.board_state_to_vector()
        if isinstance(event, controllers.MIDIMix.Knob):
            self.board_state[f"knob_{event.x}.{event.y}"] = event.value
            print("WTF")
        if isinstance(event, controllers.MIDIMix.Fader):
            if event.fader_id == self.fader_id_step_size:
                self.step_size = (event.value + 1) / 127
            else:
                self.board_state[f"fader_{event.fader_id}"] = event.value
            print("WTF2")
        # print(step_size)
        if isinstance(event, controllers.MIDIMix.BlankButton):
            print("BLANK!!")
            print("getting new set of vectors")
            # try:
            #     asyncio.get_running_loop().run_in_executor(
            #         None,
            #         self.akai_q.put(("save", (self.seed, self.lowrank_state.clone()))),
            #     )
            # except Exception as e:
            #     print(e)
            print("got new set of vectors")

            self.lowrank_state *= 0.0
            self.seed += 1337
        # print(board_state_vector.shape)
        print(board_state_vector)
        if board_state_vector.shape[0] == 32:
            if self.first_boot:
                self.offset_board_state_vector = board_state_vector.clone()
                self.first_boot = False
            else:
                board_state_vector -= self.offset_board_state_vector
                self.offset_board_state_vector += board_state_vector
                self.lowrank_state[board_state_vector != 0] += (
                    board_state_vector[board_state_vector != 0] * self.step_size
                )
                self.q.put(("update", (self.seed, self.lowrank_state.clone())))
        print("DONE DONE DONE DONE")

    def run_loop(self):
        global running
        while running:
            # print("CONTR LOOP")
            for event in pg.event.get():
                # await self.process_event(event)
                pass
            sleep(0.05)  # give the model a chance
            pg.display.update()


# class AkaiMixerControllerX:
#     def __init__(self):
#         last_command = 0
#         current_seed = 100
#         updates = []
#         offset_board_state_vector = None

#         first_boot = True
#         fader_id_step_size = 8
#         step_size = 1.0

#         board_state = {}


#     # @midi_mix.on_event TODO fix this
#     def process_event(self, event):
#         # global last_command, data_tensor, label_tensor, current_seed, offset_board_state_vector, first_boot, initial_weights, fader_id_step_size, step_size


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--input", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # init()
    v = View()
    dm = DescentModel(
        pytorch_model=get_basic_mnist_model(),
        view=v,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        seed=args.seed,
    )
    if args.input == "keyboard":
        controller = KeyboardController(q=dm.input_q)
    elif args.input == "akai":
        controller = AkaiMixerController(q=dm.input_q)
        # p = Process(target=AkaiMixerController, args=(dm.input_q,))
        # p.start()
        # p.join()
    else:
        raise ValueError("No such input as ", args.input)
    # print(controller.usage_str())
    # await kc.run_loop()
    thread = Thread(target=dm.run_loop)
    # thread2 = Thread(target=controller.run_loop)
    thread.start()
    # thread2.start()
    controller.run_loop()
    # time.sleep(100000)
    # await midi_mix.start()  # Start event loop


if __name__ == "__main__":
    main()
    # loop = asyncio.new_event_loop()
    # loop.run_forever()
    # pass


"""
MVC

Controller 
mixer multiple keys, events on async
on event push update onto global state

model listens on queue , processes and outputs loss
has delta gathering ~0.1s

controller
shares dim N with model
needs controller state knobs + sliders
needs mapping of knobs + sliders -> step size, which dims
pushes info to model

model
self.N = N
main_loop()
update(seed, params, save)
loss()


controller
process_event(e)



"""
