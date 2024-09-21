import matplotlib.pyplot as plt
import numpy as np


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


my_dpi = 96

fig = plt.figure(
    layout="constrained", figsize=(1200 / my_dpi, 800 / my_dpi), dpi=my_dpi
)
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
fig.savefig("test2.png")
breakpoint()
plt.show()
