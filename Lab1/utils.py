import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
import matplotlib.cm as cm
from Lab1.gradient_descent import gradient_descent
from Lab1.methods import *


def f1(x: np.ndarray):
    f = 2 * x ** 2 + x * 5 - 13
    return f


def f1_grad(x: np.ndarray):
    f = 4 * x + 5
    return f


def f2(x):
    return -2 * x ** 2 - 5 * x + 13


def f2_grad(x):
    return -4 * x - 5


def f3(x: np.ndarray):
    f = (17 * x[0] - 254) ** 2 - 20 * x[0] + (27 * x[1] + 995) ** 2
    return f


def f3_grad(x: np.ndarray):
    f = [-8656 + 578 * x[0],
         54 * (995 + 27 * x[1])]
    return f


class PlotBuilder:
    def __init__(self, a, b, epses):
        self.a = a
        self.b = b
        self.epses = np.array(epses)
        self.data = []

    def add_searcher(self, searcher):
        iterations_axes = []
        function_calls_axes = []
        for eps in self.epses:
            l, r, iterations, function_calls = searcher.search(self.a, self.b, eps)
            iterations_axes.append(iterations)
            function_calls_axes.append(function_calls)
        self.data.append((iterations_axes, function_calls_axes, str(searcher)))

    def show(self):
        x_axes_range = np.array(range(len(self.epses)))
        idx = [0, int(len(x_axes_range) / 2), -1]
        plt.clf()
        for iterations, function_calls, name in self.data:
            plt.plot(x_axes_range, iterations, label=name)
        plt.legend()
        plt.xlabel("eps")
        plt.xticks(x_axes_range[idx],
                   self.epses[idx])
        plt.ylabel("iterations")
        plt.show()
        plt.clf()
        for iterations, function_calls, name in self.data:
            plt.plot(x_axes_range, function_calls, label=name)
        plt.legend()
        plt.xlabel("eps")
        plt.xticks(x_axes_range[idx],
                   self.epses[idx])
        plt.ylabel("function calls")
        plt.show()


def build_plots_for_all_searchers(func, a, b, epses):
    plots = PlotBuilder(a, b, epses)
    plots.add_searcher(LinearSearcher(func))
    plots.add_searcher(BisectionSearcher(func))
    plots.add_searcher(GoldenRatioSearcher(func))
    plots.add_searcher(FibonacciSearcher(func))
    plots.show()


def draw_single_arg_function(func, a, b):
    x = np.linspace(a, b, 100)
    plt.xticks(np.arange(a, b, 0.20))
    plt.plot(x, np.vectorize(func)(x))
    plt.show()


def draw_double_arg_function(func, x1, y1, x2, y2, show=False):
    x_step = 0.1
    y_step = 0.1
    x_s = np.arange(x1, x2, x_step)
    y_s = np.arange(y1, y2, y_step)
    z_s = []
    for y in y_s:
        tmp = []
        for x in x_s:
            tmp.append(func(np.array([x, y])))
        z_s.append(tmp)
    z_s = np.array(z_s)
    plt.clf()
    plt.figure()
    Locator.MAXTICKS = 3000
    cs = plt.imshow(z_s, interpolation='bilinear', cmap=plt.get_cmap("twilight"),
                    origin='lower', extent=[x1, x2, y1, y2],
                    vmax=abs(z_s).max(), vmin=-abs(z_s).max())
    if show:
        plt.show()


def draw_descent_steps(func, grad, start, searcher, eps):
    res, it, path = gradient_descent(func, grad, start, eps, searcher)
    dist = path - res
    print(dist)
    print(len(path), it)
    # draw_double_arg_function(func, res[0] - dist, res[1] - dist, res[0] + dist, res[1] + dist)
    # print(path)
    # for step in path:
    #     plt.scatter(step[0], step[1], c="white", )
    # plt.show()


def run_all_gradients(f, g, start, eps):
    res, it, _ = gradient_descent(f, g, start, eps, None)
    print("LinearStepSearch", (res, it))
    res_, it, _ = gradient_descent(f, g, start, eps, BisectionSearcher)
    print("BisectionSearcher", (res_, it))
    res_, it, _ = gradient_descent(f, g, start, eps, GoldenRatioSearcher)
    print("GoldenRatioSearcher", (res_, it))
    res_, it, _ = gradient_descent(f, g, start, eps, FibonacciSearcher)
    print("FibonacciSearcher", (res_, it))
    return res
