import matplotlib

matplotlib.use("module://pygame_matplotlib.backend_pygame")


import pygame
import pygame.display

import matplotlib.pyplot as plt
import numpy as np

screen = pygame.display.set_mode((1200, 800))

# Use the fig as a pygame.Surface
# screen.blit(fig, (0, 0))


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


my_dpi = 96 * 8
fig = plt.figure(
    figsize=(1200 / my_dpi, 800 / my_dpi),
    dpi=my_dpi,  # layout="constrained",
)
fig.subplots_adjust(left=0.5)
axd = fig.subplot_mosaic(
    """
    AAAAAA
    BBCCDD
    BBCCDD
    EEFGHI
    EEFGHI
    """
)
identify_axes(axd)
# fig.savefig("test.png")
fig.subplots_adjust(left=0.1, hspace=0.9)
fig.canvas.draw()
screen.blit(fig)

i = 0
show = True
while show:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Stop showing when quit
            show = False
    # update(i % 40)
    # i += 1

    pygame.display.update()
