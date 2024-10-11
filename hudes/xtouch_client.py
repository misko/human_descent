import pygame.midi

from hudes.keyboard_client import KeyboardClient


class XTouchClient(KeyboardClient):
    def init_input(self):
        super().init_input()  # setup keyboard

        self.set_n(8)  # controller uses 8 dimensions

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
            print(event)
            if event.data1 >= 1 and event.data1 <= 8:
                dim = event.data1 - 1
                scale = event.data2 - 64
                self.send_dims_and_steps({dim: self.step_size * scale})
            elif event.data1 == 9:  # step size slider
                self.set_step_size_idx(2.5 * (64 - event.data2))
                redraw = True
            elif event.data1 >= 8 and event.data1 <= 15 and event.data2 == 127:
                self.get_next_dims()
            elif event.data1 >= 16 and event.data1 <= 23 and event.data2 == 127:
                self.get_next_batch()

        return redraw
