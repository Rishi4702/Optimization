import matplotlib.pyplot as plt
import numpy as np


def ackley(x, y):
    return -20 * np.exp(-0.02 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.e


if __name__ == '__main__':
    x = np.arange(-35, 36, 0.1)
    y = np.arange(-35, 36, 0.1)
    X, Y = np.meshgrid(x, y)

    # calculate the function values for each point in the meshgrid
    Z = ackley(X, Y)
    # print(ackley(0,0))
    print(np.min(Z))
    # create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # set the plot labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Ackley Function')

    # show the plot
    plt.show()
    # r_min, r_max = -32.768, 32.768
    # xaxis = arange(r_min, r_max, 2.0)
    # yaxis = arange(r_min, r_max, 2.0)
    # x, y = meshgrid(xaxis, yaxis)
    # results = ackley(x, y)
    # figure = plt.figure()
    # axis = figure.gca(projection='3d')
    # axis.plot_surface(x, y, results, cmap='jet', shade= "false")
    # plt.show()
    # plt.contour(x,y,results)
    # plt.show()
    # plt.scatter(x, y, results)
    # plt.show()
