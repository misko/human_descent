import matplotlib

matplotlib.use("Agg")
import time

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pygame
import torch


# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

    # pygame.init()
    # pygame.display.set_mode(self.window_size)
    return pygame.display.get_surface()


class View:
    def __init__(self):

        self.window_size = (1200, 800)
        self.fig = plt.figure(
            figsize=(12, 8),
        )

        self.fig.subplots_adjust(
            left=0.05, right=0.95, hspace=1, top=0.95, bottom=0.05, wspace=0.5
        )
        self.axd = self.fig.subplot_mosaic(
            #    AAAAAA
            """
                BBCCDD
                BBCCDD
                EEFGHI
                EEJKLI
                """
        )

        self.screen = pygame.display.get_surface()

    def update_examples(self, train_data, val_data):
        self.axd["F"].cla()
        self.axd["F"].imshow(train_data[0])
        self.axd["F"].set_title("Train ex1")

        self.axd["G"].cla()
        self.axd["G"].imshow(train_data[1])
        self.axd["G"].set_title("Train ex2")

        self.axd["H"].cla()
        self.axd["H"].imshow(train_data[2])
        self.axd["H"].set_title("Train ex3")

        # self.axd["I"].cla()
        # self.axd["I"].imshow(train_data[3])

    def update_top(self):
        self.fig.suptitle("Human Descent: MNIST")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_step_size(self, log_step_size, max_log_step_size, min_log_step_size):
        self.axd["I"].cla()
        self.axd["I"].bar([0], log_step_size)
        self.axd["I"].set_ylim(min_log_step_size, max_log_step_size)
        self.axd["I"].set_title("Step")

    def update_example_preds(self, train_preds):
        self.axd["J"].cla()
        self.axd["J"].bar(torch.arange(10), train_preds[0])
        self.axd["J"].set_title("Train ex1")

        self.axd["K"].cla()
        self.axd["K"].bar(torch.arange(10), train_preds[1])
        self.axd["K"].set_title("Train ex2")

        self.axd["L"].cla()
        self.axd["L"].bar(torch.arange(10), train_preds[2])
        self.axd["L"].set_title("Train ex2")

        # self.axd["M"].cla()
        # self.axd["M"].bar(torch.arange(10), train_preds[3])

    def plot_train_and_val(self, train_losses, val_losses):
        n = len(train_losses)
        x = torch.arange(n)
        self.axd["B"].cla()
        self.axd["B"].plot(x, train_losses, label="train")
        self.axd["B"].plot(x, val_losses, label="val")
        self.axd["B"].legend(loc="upper right")
        self.axd["B"].set_title("Loss")
        self.axd["B"].set_xlabel("Step")
        self.axd["B"].set_ylabel("Loss")

        self.axd["C"].cla()
        self.axd["C"].plot(x[n // 2 :], train_losses[n // 2 :], label="train")
        self.axd["C"].set_title("Loss")
        self.axd["C"].set_xlabel("Step")

        self.axd["D"].cla()
        self.axd["D"].plot(x[-8:], train_losses[-8:], label="train")
        self.axd["D"].set_title("Loss")
        self.axd["D"].set_xlabel("Step")

    def draw(self):
        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        surf = pygame.image.frombytes(
            canvas.get_renderer().tostring_rgb(), self.window_size, "RGB"
        )
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()  # draws whole screen vs update that draws a parts
