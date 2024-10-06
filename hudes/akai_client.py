from time import sleep

import torch
from akai_pro_py import controllers

from hudes.hudes_client import HudesClient


class AkaiClient(HudesClient):
    def init_input(self):
        self.set_n(32)

        self.first_boot = True

        # Define the MIDI Mix and APC Mini, first argument: MIDI in, second argument: MIDI out
        self.midi_mix = controllers.MIDIMix("MIDI Mix", "MIDI Mix")

        self.midi_mix.reset()

        self.midi_mix.on_event(self.process_midi_event)

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
        board_state_vector_initial = self.board_state_to_vector()
        if isinstance(event, controllers.MIDIMix.Knob):
            self.board_state[f"knob_{event.x}.{event.y}"] = event.value

        if isinstance(event, controllers.MIDIMix.Fader):
            if event.fader_id == self.fader_id_step_size:
                self.set_step_size_idx(event.value)
            else:
                self.board_state[f"fader_{event.fader_id}"] = event.value

        board_state_vector_current = self.board_state_to_vector()

        if isinstance(event, controllers.MIDIMix.BlankButton) and event.state:
            if event.button_id == 1:
                print("getting new set of vectors")
                self.get_next_dims()
            elif event.button_id == 0:
                print("Getting new batch")
                self.get_next_batch()

        if board_state_vector_initial.shape[0] == 32:
            diff = board_state_vector_current - board_state_vector_initial
            dims_and_steps = {}
            for dim in torch.where(diff != 0)[0]:
                dims_and_steps[dim] = self.step_size * diff[dim]
            self.send_dims_and_steps(dims_and_steps)
        # board_state_vector = self.offset_board_state_vector
        # self.offset_board_state_vector += board_state_vector
        # self.lowrank_state[board_state_vector != 0] += (
        #     board_state_vector[board_state_vector != 0] * self.step_size
        # )
        # self.q.put(("update", (self.seed, self.lowrank_state.clone())))
        # print("DONE DONE DONE DONE")

    def process_key_press(self, event):
        pass
