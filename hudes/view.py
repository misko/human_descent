import math
from typing import List

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
            left=0.07, right=0.95, hspace=0.8, top=0.92, bottom=0.07, wspace=0.5  # 0.5
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
        # self.fig.tight_layout()

        self.screen = pygame.display.get_surface()

    def update_examples(self, train_data: torch.Tensor, val_data: torch.Tensor):
        self.axd["F"].cla()
        self.axd["F"].imshow(train_data[0])
        self.axd["F"].set_title("Ex. 1 img")

        self.axd["G"].cla()
        self.axd["G"].imshow(train_data[1])
        self.axd["G"].set_title("Ex. 2 img")

        self.axd["H"].cla()
        self.axd["H"].imshow(train_data[2])
        self.axd["H"].set_title("Ex. 3 img")

        # self.axd["I"].cla()
        # self.axd["I"].imshow(train_data[3])

    def update_top(self, best_score):
        if best_score is None:
            self.fig.suptitle("Human Descent: MNIST      Top-score: ?")
        else:
            self.fig.suptitle(f"Human Descent: MNIST      Top-score: {best_score:.5e}")
        # self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_step_size(
        self, log_step_size: float, max_log_step_size: float, min_log_step_size: float
    ):
        self.axd["I"].cla()
        self.axd["I"].bar([0], log_step_size)
        self.axd["I"].set_ylim(min_log_step_size, max_log_step_size)
        self.axd["I"].set_title("log(Step size)")
        self.axd["I"].set_xticks([])

    def update_confusion_matrix(self, confusion_matrix: torch.Tensor):
        self.axd["E"].cla()
        self.axd["E"].imshow(confusion_matrix)
        self.axd["E"].set_yticks(range(10))
        self.axd["E"].set_xticks(range(10))
        self.axd["E"].set_ylabel("Ground truth")
        self.axd["E"].set_xlabel("Prediction")
        self.axd["E"].set_title("Confusion matrix")

    def update_example_preds(self, train_preds: List[float]):
        self.axd["J"].cla()
        self.axd["J"].bar(torch.arange(10), train_preds[0])
        self.axd["J"].set_title("Ex. 1 pr(y)")

        self.axd["K"].cla()
        self.axd["K"].bar(torch.arange(10), train_preds[1])
        self.axd["K"].set_title("Ex. 2 pr(y)")

        self.axd["L"].cla()
        self.axd["L"].bar(torch.arange(10), train_preds[2])
        self.axd["L"].set_title("Ex. 3 pr(y)")

        # self.axd["M"].cla()
        # self.axd["M"].bar(torch.arange(10), train_preds[3])

    def plot_train_and_val(
        self,
        train_losses: List[float],
        train_steps: List[int],
        val_losses: List[float],
        val_steps: List[int],
        minimize: bool,
    ):
        if minimize is None:
            self.update_top(None)
        else:
            # TODO THIS LOGIC DOES NOT MAKE SENSE
            if minimize:
                best_score = max(val_losses) if len(val_losses) > 0 else math.inf
            else:
                best_score = min(val_losses) if len(val_losses) > 0 else -math.inf
            self.update_top(best_score)

        n = len(train_losses)
        # x = torch.arange(n)
        self.axd["B"].cla()
        self.axd["B"].plot(train_steps, train_losses, label="train")
        self.axd["B"].plot(val_steps, val_losses, label="val")
        self.axd["B"].legend(loc="upper right")
        self.axd["B"].set_title("Loss")
        self.axd["B"].set_xlabel("Step")
        self.axd["B"].set_ylabel("Loss")

        self.axd["C"].cla()
        self.axd["C"].plot(train_steps[n // 2 :], train_losses[n // 2 :], label="train")
        self.axd["C"].set_title("Loss [half time]")
        self.axd["C"].set_xlabel("Step")
        self.axd["C"].set_yticks([])

        self.axd["D"].cla()
        self.axd["D"].plot(train_steps[-8:], train_losses[-8:], label="train")
        self.axd["D"].set_title("Loss [last 8steps]")
        self.axd["D"].set_yticks([])
        self.axd["D"].set_xlabel("Step")

    def draw(self):
        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        surf = pygame.image.frombytes(
            canvas.get_renderer().tostring_rgb(), self.window_size, "RGB"
        )
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()  # draws whole screen vs update that draws a parts
