import matplotlib

matplotlib.use("Agg")
import time

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pygame

window_size = (1200, 800)


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


def get_fig_and_axd():
    fig = plt.figure(
        figsize=(12, 8),
    )

    fig.subplots_adjust(
        left=0.05, right=0.95, hspace=1, top=0.95, bottom=0.05, wspace=0.5
    )
    axd = fig.subplot_mosaic(
        """
        AAAAAA
        BBCCDD
        BBCCDD
        EEFGHI
        EEJKLM
        """
    )
    return fig, axd


def draw(fig, screen):
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    surf = pygame.image.frombytes(
        canvas.get_renderer().tostring_rgb(), window_size, "RGB"
    )
    screen.blit(surf, (0, 0))


def init_pygame():
    pygame.init()
    pygame.display.set_mode(window_size)
    return pygame.display.get_surface()


fig, axd = get_fig_and_axd()
screen = init_pygame()

while True:
    for event in pygame.event.get():
        pass
    draw(fig, screen)
    pygame.display.flip()  # draws whole screen vs update that draws a parts
