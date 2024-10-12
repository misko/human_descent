import logging

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

        self.key_to_param_and_sign = {}
        for idx in range(self.client_state.n):
            u, d = self.paired_keys[idx]
            self.key_to_param_and_sign[u] = (idx, 1)
            self.key_to_param_and_sign[d] = (idx, -1)

        self.client_state.set_batch_size(512)
        self.client_state.set_dtype("float32")

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
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_q:
                self.client_state.quit_count += 1
                logging.info("Keep holding to quit!")
                if self.client_state.quit_count > 6:
                    logging.info("Quiting")
                    self.hudes_websocket_client.running = False
                    return False
            else:
                self.client_state.quit_count = 0

            if event.key == pg.K_z:
                self.client_state.help_count += 1
                if self.client_state.help_count > 1:
                    self.view.next_help_screen()
                    self.client_state.help_count = 0
                    return True
            else:
                self.client_state.help_count = 0

            if event.key == pg.K_SLASH:
                self.get_sgd()
                return False
            elif event.key == pg.K_SPACE:
                print("getting new set of vectors")
                self.get_next_dims()
                return False
            elif event.key == pg.K_LEFTBRACKET:
                self.client_state.step_size_increase()
                self.view.update_step_size(
                    self.client_state.log_step_size,
                    self.client_state.max_log_step_size,
                    self.client_state.min_log_step_size,
                )
                self.send_config()
                logging.debug("keyboard_client: increase step size")
                return True
            elif event.key == pg.K_RIGHTBRACKET:
                self.client_state.step_size_decrease()
                self.view.update_step_size(
                    self.client_state.log_step_size,
                    self.client_state.max_log_step_size,
                    self.client_state.min_log_step_size,
                )
                self.send_config()
                logging.debug("keyboard_client: decrease step size")
                return True
            elif event.key == pg.K_QUOTE:
                self.client_state.toggle_dtype()
                self.send_config()
                return False
            elif event.key == pg.K_SEMICOLON:
                self.client_state.toggle_batch_size()
                self.send_config()
                return False
            elif event.key == pg.K_RETURN:
                print("Getting new batch")
                self.get_next_batch()
                return False
            elif event.key in self.key_to_param_and_sign:
                dim, sign = self.key_to_param_and_sign[event.key]
                self.send_dims_and_steps({dim: self.client_state.step_size * sign})
                return False  # we are going to get a response shortly that updates

        return False
