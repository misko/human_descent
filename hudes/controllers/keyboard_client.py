import pygame as pg

from hudes.hudes_client import HudesClient


class KeyboardClient(HudesClient):
    def init_input(self):

        self.paired_keys = [
            ("w", "s"),
            ("e", "d"),
            ("r", "f"),
            ("u", "j"),
            ("i", "k"),
            ("o", "l"),
        ]
        self.set_n(len(self.paired_keys))

        self.key_to_param_and_sign = {}
        for idx in range(self.n):
            u, d = self.paired_keys[idx]
            self.key_to_param_and_sign[u] = (idx, 1)
            self.key_to_param_and_sign[d] = (idx, -1)

        self.set_batch_size(512)
        self.set_dtype("float32")

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
        if event.type == pg.TEXTINPUT:  # event.type == pg.KEYDOWN or
            key = event.text

            if key == "q":
                self.quit_count += 1
                print("Keep holding to quit!")
                if self.quit_count > 4:
                    print("Quiting")
                    self.hudes_websocket_client.running = False
                return False
            self.quit_count = 0

            if key in self.key_to_param_and_sign:
                dim, sign = self.key_to_param_and_sign[key]
                self.send_dims_and_steps({dim: self.step_size * sign})
                return False  # we are going to get a response shortly that updates
            elif key == "[":
                self.step_size_increase()
                return True
            elif key == "]":
                self.step_size_decrease()
                return True
            elif key == " ":
                print("getting new set of vectors")
                self.get_next_dims()
                return False
            elif key == "/":
                self.get_sgd()
                return False
            elif key == "'":
                self.toggle_dtype()
                return False
            elif key == ";":
                self.toggle_batch_size()
                return False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                print("Getting new batch")
                self.get_next_batch()
                return False
        return False
