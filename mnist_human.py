from torchvision.datasets import MNIST
from torchvision import transforms
from torch import nn
import torch
import time
import asyncio
from akai_pro_py import controllers
import torch
import pygame as pg
import math
import matplotlib.pyplot as plt
import time

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
        self.loss_fig, self.loss_axs = plt.subplots(1, 2, figsize=(8, 4))
        self.n_digits = 2
        self.digit_fig, self.digit_axs = plt.subplots(2, self.n_digits, figsize=(8, 4))
        plt.ioff()

    def render_loss(self, name, loss, ax):
        x = [_x[0] for _x in loss]
        y = [_x[1] for _x in loss]
        ax.plot(x, y, label=name)

    def render_digits(self, data, probs):
        for idx in range(self.n_digits):
            self.digit_axs[1, idx].cla()
            self.digit_axs[1, idx].plot(probs[idx])
            self.digit_axs[0, idx].cla()
            self.digit_axs[0, idx].imshow(data[idx])
        plt.pause(0.1)

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
        plt.pause(0.1)

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

    def __init__(self, pytorch_model, view, seed=0):
        torch.manual_seed(0)
        self.train_batch_size = 512  # number of samples to use
        self.val_batch_size = 1024  # number of samples to use
        self.view = view
        self.pytorch_model = pytorch_model
        self.saved_weights = [x.clone() for x in pytorch_model.parameters()]

        self.train_raw = MNIST(".", train=True, transform=transform)
        self.train_batches = math.ceil(len(self.train_raw) / self.train_batch_size)

        self.val_raw = MNIST(".", train=False, transform=transform)
        self.val_batches = math.ceil(len(self.val_raw) / self.val_batch_size)

        self.current_batch_idx = 0
        self.fetch_and_set_train_and_val_batch_tensors()

        self.step = 0
        self.train_losses = []
        self.val_losses = []
        self.new_batch_steps = []

        self.input_q = asyncio.Queue()

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

    async def run_loop(self):
        while True:
            cmd, data = await self.input_q.get()
            if cmd == "quit":
                print("return from model run loop")
                return
            elif cmd == "update" or cmd == "save":
                self.apply_update(data[0], data[1])
                self.step += 1
                if cmd == "save":
                    self.saved_weights = [
                        x.clone() for x in self.pytorch_model.parameters()
                    ]
                losses = self.get_losses()
                self.record_losses(losses)
                self.view.render_digits(
                    self.train_data_tensor, losses["train_pred"].detach().numpy()
                )
                self.print_losses(losses)
            elif cmd == "new batch":
                print("New batch")
                self.new_batch_steps.append(self.step)
                self.current_batch_idx += 1
                self.fetch_and_set_train_and_val_batch_tensors()
                losses = self.get_losses()
                self.record_losses(losses)
                self.print_losses(losses)


def init():
    pg.init()
    pg.display.set_mode((300, 300))


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

    async def process_event(self, event):
        global running
        if event.type == pg.TEXTINPUT:  # event.type == pg.KEYDOWN or
            key = event.text

            if key == "q":
                self.quit_count += 1
                print("Keep holding to quit!")
                if self.quit_count > 10:
                    print("Quiting")
                    await self.q.put(("quit", None))
                    running = False
                return
            self.quit_count = 0

            if key in self.key_to_param_and_sign:
                lowrank_idx, sign = self.key_to_param_and_sign[key]
                self.lowrank_state[lowrank_idx] += self.step_size * sign
                await self.q.put(("update", (self.seed, self.lowrank_state.clone())))
            elif key == "[":
                self.step_size = max(self.step_size - self.step_size_resolution, 0)
                print(f"Step size: {self.step_size}")
            elif key == "]":
                self.step_size = self.step_size + self.step_size_resolution
                print(f"Step size: {self.step_size}")
            elif key == " ":
                print("getting new set of vectors")
                await self.q.put(("save", (self.seed, self.lowrank_state.clone())))
                self.lowrank_state *= 0.0
                self.seed += 1337
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                print("Getting new batch")
                await self.q.put(("new batch", None))

    async def run_loop(self):
        global running
        while running:
            for event in pg.event.get():
                await self.process_event(event)
            await asyncio.sleep(0.05)  # give the model a chance


class AkaiMixerController:
    def __init__(self):
        last_command = 0
        current_seed = 100
        updates = []
        offset_board_state_vector = None

        first_boot = True
        fader_id_step_size = 8
        step_size = 1.0

        # Define the MIDI Mix and APC Mini, first argument: MIDI in, second argument: MIDI out
        midi_mix = controllers.MIDIMix("MIDI Mix", "MIDI Mix")

        midi_mix.reset()

        board_state = {}

    def board_state_to_vector(self, scale=0.2):
        return torch.tensor(
            [scale * (board_state[k] - 64) / 64 for k in sorted(board_state.keys())]
        )

    # @midi_mix.on_event TODO fix this
    def process_event(self, event):
        # global last_command, data_tensor, label_tensor, current_seed, offset_board_state_vector, first_boot, initial_weights, fader_id_step_size, step_size

        board_state_vector = self.board_state_to_vector()
        if isinstance(event, controllers.MIDIMix.Knob):
            board_state[f"knob_{event.x}.{event.y}"] = event.value
        if isinstance(event, controllers.MIDIMix.Fader):
            if event.fader_id == self.fader_id_step_size:
                step_size = (event.value + 1) / 127
            else:
                board_state[f"fader_{event.fader_id}"] = event.value
        # print(step_size)
        if isinstance(event, controllers.MIDIMix.BlankButton):
            print("BLANK!!")
            # updates.append((current_seed, board_state_vector - offset_board_state_vector))
            initial_weights = [
                x.clone() for x in model.parameters()
            ]  # instead of updates , just update initalweights
            offset_board_state_vector = board_state_vector.clone()
            current_seed += 1337
        # print(board_state_vector.shape)
        if board_state_vector.shape[0] == 32:
            if first_boot:
                offset_board_state_vector = board_state_vector.clone()
                first_boot = False
            board_state_vector -= offset_board_state_vector
            # print(board_state_vector)
            _updates = updates + [[current_seed, step_size * board_state_vector]]
            loss = get_loss(model, data_tensor, label_tensor, _updates)
            print("LOSS", -loss.item())
            # print(_updates)
            last_command = time.time()


async def main():
    init()
    v = View()
    dm = DescentModel(pytorch_model=get_basic_mnist_model(), view=v)
    kc = KeyboardController(q=dm.input_q)
    print(kc.usage_str())
    # await kc.run_loop()
    await asyncio.gather(dm.run_loop(), kc.run_loop())
    # await midi_mix.start()  # Start event loop


if __name__ == "__main__":
    asyncio.run(main())
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
