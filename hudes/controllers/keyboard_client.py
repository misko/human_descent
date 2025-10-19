import pygame as pg

from hudes.hudes_client import HudesClient


class KeyboardClient(HudesClient):
    def init_input(self):
        self.paired_keys = [
            (pg.key.key_code(x), pg.key.key_code(y))
            for x, y in [
                ("w", "s"),
                ("e", "d"),
                ("r", "f"),
                ("u", "j"),
                ("i", "k"),
                ("o", "l"),
            ]
        ]
        self.set_n(len(self.paired_keys))

        for a, b in self.paired_keys:
            self.add_key_to_watch(key=a)
            self.add_key_to_watch(key=b)

        self.key_to_param_and_sign = {}
        for idx in range(self.client_state.n):
            u, d = self.paired_keys[idx]
            self.key_to_param_and_sign[u] = (idx, 1)
            self.key_to_param_and_sign[d] = (idx, -1)

        self.client_state.set_batch_size(512)
        self.client_state.set_dtype("float32")

        self.client_state.set_help_screen_fns(
            [
                "help_screens/hudes_help_start.png",
                "help_screens/hudes_1.png",
                "help_screens/hudes_2.png",
                "help_screens/hudes_3.png",
                "help_screens/hudes_1d_keyboard_controls.png",
            ]
        )

    def usage_str(self) -> str:
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

    def process_key_press(self, event):
        ct = pg.time.get_ticks()
        redraw = self.process_common_keys(event)
        if self.client_state.help_screen_idx != -1:
            return redraw
        if event.type == pg.KEYDOWN:
            if event.key in self.key_to_param_and_sign:
                if self.check_cooldown_time(
                    event=event,
                    key=event.key,
                    ct=ct,
                    cool_down_time=0.2,
                ):
                    dim, sign = self.key_to_param_and_sign[event.key]
                    self.send_dims_and_steps({dim: self.client_state.step_size * sign})

        return redraw
