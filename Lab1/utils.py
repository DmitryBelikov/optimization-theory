import numpy as np
import matplotlib.pyplot as plt

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
    f = (x[0] - 2) ** 2 + (x[1] - 1) ** 2
    return f


def test_function_gradient(xs):
    return np.array([[2 * (xs[0][0] - 2),
                      2 * (xs[0][1] - 1)]])


def f3_grad(x: np.ndarray):
    f = [2 * (x[0] - 2),
         2 * (x[1] - 1)]
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
    # plots.add_searcher(LinearSearcher(func))
    plots.add_searcher(BisectionSearcher(func))
    plots.add_searcher(GoldenRatioSearcher(func))
    plots.add_searcher(FibonacciSearcher(func))
    plots.show()


def draw_single_arg_function(func, a, b):
    x = np.linspace(a, b, 100)
    plt.xticks(np.arange(a, b, 0.20))
    plt.plot(x, np.vectorize(func)(x))
    plt.show()


def draw_double_arg_function(func, x1, y1, x2, y2):
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
    plt.figure()
    cs = plt.contour(x_s, y_s, z_s, levels=2000)
    plt.show()


def run_all_gradients(f, g, start):
    # print("LinearStepSearch", gradient_descent(f, g, start, 1e-9, None))
    print("BisectionSearcher", gradient_descent(f, g, start, 1e-9, BisectionSearcher))
    print("GoldenRatioSearcher", gradient_descent(f, g, start, 1e-9, GoldenRatioSearcher))
    print("FibonacciSearcher", gradient_descent(f, g, start, 1e-9, FibonacciSearcher))
