import math

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8")


def Himmelblau(x1, x2):
    return math.pow(x1 * x1 + x2 - 11, 2) + math.pow(x1 + x2 * x2 - 7, 2)


def plotFuction():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    zs = np.array([Himmelblau(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    my_cmap = plt.get_cmap('viridis')
    surf = ax.plot_surface(X, Y, Z, cmap=my_cmap, edgecolor='none')
    fig.colorbar(surf, ax=ax, fraction=0.07, anchor=(1.0, 0.0), )
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Himmelblau function')
    plt.show()


if __name__ == '__main__':
    plotFuction()
