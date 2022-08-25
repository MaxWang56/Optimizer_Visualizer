import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from CustomOptimizers import SGD

def rosenbrock(x, y, a, b):

    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def run_optimization(x_0, y_0, optimizer, epochs):
    xy_t = torch.tensor((x_0, y_0), requires_grad=True)
    optimizer = SGD([xy_t], 0.001, 0.01, 10)

    path = np.empty((epochs + 1, 2))
    path[0, :] = (x_0, y_0)

    for i in tqdm(range(1, epochs + 1)):
        optimizer.zero_grad()
        loss = rosenbrock(xy_t[0], xy_t[1], 1, 10)
        loss.backward()
        torch.nn.utils.clip_grad_norm(xy_t, 1.0)
        optimizer.step(i)
        path[i, :] = xy_t.detach().numpy()
    print(path)
    return path

def visualize_path(paths, colors, names,
                   figsize = (12, 12),
                   x_lim = (-2, 2),
                   y_lim = (-1, 3),
                   n_seconds = 5):
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y, 1, 100)

    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    scatters = [ax.scatter(None,
                           None,
                           label=label,
                           c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)

    return anim

n_epochs = 1000
weight_decay = 0.1
path_cosineratedecay = run_optimization(0.3, 0.8, SGD, n_epochs)

freq = 10

paths = [path_cosineratedecay[::freq]]
colors = ['green']
names = ['CRD']

anim = visualize_path(paths,
                            colors,
                            names,
                            figsize=(12, 7),
                            x_lim=(-.1, 1.1),
                            y_lim=(-.1, 1.1),
                            n_seconds=7)
anim.save('result.gif')
