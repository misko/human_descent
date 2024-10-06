import argparse
import math
from time import sleep

import pygame as pg
import pygame.midi

from hudes.keyboard_client import KeyboardClient
from hudes.websocket_client import next_batch_message, next_dims_message


class XTouchClient(KeyboardClient):
    def init_input(self):
        super().init_input()  # setup keyboard

        self.n = 8
        # pg.init()
        # pg.midi.init()
        # _print_device_info()
        # self.midi_input_id = pygame.midi.get_default_input_id()
        # print(f"using input_id :{self.midi_input_id}:")
        # self.midi_input = pygame.midi.Input(self.midi_input_id)
        # pg.display.set_mode((1, 1))

        # for e in events:
        #     if e.type in [pg.QUIT]:
        #         going = False
        #     if e.type in [pg.KEYDOWN]:
        #         going = False
        #     if e.type in [pygame.midi.MIDIIN]:
        #         print(e)

        # if i.poll():
        #     midi_events = i.read(10)
        #     # convert them into pygame events.
        #     midi_evs = pygame.midi.midis2events(midi_events, i.device_id)

        #     for m_e in midi_evs:
        #         pygame.event.post(m_e)

    def before_pg_event(self):
        if self.view.midi_input.poll():
            midi_events = self.view.midi_input.read(10)
            # convert them into pygame events.
            midi_evs = pygame.midi.midis2events(
                midi_events, self.view.midi_input.device_id
            )

            for m_e in midi_evs:
                pygame.event.post(m_e)

    def process_key_press(self, event):
        redraw = super().process_key_press(event)
        if event.type in [pygame.midi.MIDIIN]:
            if event.data1 >= 1 and event.data1 <= 8:
                dim = event.data1 - 1
                scale = event.data2 - 64
                self.send_dims_and_steps({dim: self.step_size * scale})
            elif event.data1 == 9:  # step size slider
                self.set_step_size_idx(2.5 * (64 - event.data2))
                redraw = True

        return redraw
