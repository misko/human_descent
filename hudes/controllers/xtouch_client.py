import logging

import pygame.midi

from hudes.controllers.keyboard_client import KeyboardClient


class XTouchClient(KeyboardClient):
    def init_input(self):
        super().init_input()  # setup keyboard

        self.set_n(8)  # controller uses 8 dimensions

        self.bindings = {
            8: "help",
            16: "quit",
            19: "sgd",
            14: "batch",
            22: "float",
            15: "subspace",
            23: "batchsize",
        }

        self.inverse_bindings = {v: k for k, v in self.bindings.items()}

        self.client_state.set_help_screen_fns(
            [
                "help_screens/hudes_help_start.png",
                "help_screens/hudes_1.png",
                "help_screens/hudes_2.png",
                "help_screens/hudes_3.png",
                "help_screens/hudes_1d_xtouch_controls.png",
                "help_screens/hudes_1d_xtouch_slider_center.png",
            ]
        )

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

            if (
                event.status != 186
                and event.data2 == 127
                and event.data1 == self.inverse_bindings["help"]
            ):
                logging.debug("calling next_help_screen...")
                self.client_state.next_help_screen()
                return True

            if self.client_state.help_screen_idx != -1:
                return False  # we are in help screen mode

            if event.status == 186 and event.data1 >= 1 and event.data1 <= 8:
                dim = event.data1 - 1
                scale = event.data2 - 64
                self.send_dims_and_steps({dim: self.client_state.step_size * scale})
            elif event.data1 == 9:  # step size slider
                self.client_state.set_step_size_idx((100 - event.data2 - 20))
                self.view.update_step_size()
                self.send_config()
                redraw = True
            elif event.data1 in self.bindings and event.data2 == 127:
                action = self.bindings[event.data1]
                if action == "quit":
                    logging.debug("quitting...")
                    self.quit()
                    return False
                    # self.hudes_websocket_client.running = False
                elif action == "sgd":
                    self.get_sgd()
                elif action == "batch":
                    self.get_next_batch()
                elif action == "float":
                    self.client_state.toggle_dtype()
                    self.send_config()
                elif action == "subspace":
                    self.get_next_dims()
                elif action == "batchsize":
                    self.client_state.toggle_batch_size()
                    self.send_config()

        return redraw
