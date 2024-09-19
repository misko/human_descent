import matplotlib

matplotlib.use("module://pygame_matplotlib.backend_pygame")

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

import pygame
import pygame.display

fig, ax = plt.subplots()
t = np.linspace(0, 3, 40)
g = -9.81
v0 = 12
z = g * t**2 / 2 + v0 * t

v02 = 5
z2 = g * t**2 / 2 + v02 * t

scat = ax.scatter(t[0], z[0], c="b", s=5, label=f"v0 = {v0} m/s")
line2 = ax.plot(t[0], z2[0], label=f"v0 = {v02} m/s")[0]
ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel="Time [s]", ylabel="Z [m]")
ax.legend()

fig.canvas.draw()


screen = pygame.display.set_mode((1200, 800))

# Use the fig as a pygame.Surface
screen.blit(fig, (0, 0))


def update(frame):
    print("update", frame)
    # for each frame, update the data stored on each artist.
    x = t[:frame]
    y = z[:frame]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:
    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])
    fig.canvas.draw()
    screen.blit(fig)
    return (scat, line2)


# ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
# plt.show()


i = 0
show = True
while show:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Stop showing when quit
            show = False
    update(i % 40)
    i += 1

    pygame.display.update()
