import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
import matplotlib.cm as cm
from Lab1.gradient_descent import gradient_descent, const_gradient_descent
from Lab1.methods import *
from Lab1.newton import newton


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


def norm(matrix):
    return np.linalg.norm(matrix, ord=2)


def get_condition_number(matrix):
    return norm(matrix) * norm(np.linalg.inv(matrix))


def generate_matrix(n, condition_number):
    matrix = np.zeros((n, n))
    matrix[-1, -1] = condition_number
    matrix[0, 0] = 1.0
    for i in range(1, n - 1):
        matrix[i, i] = random.uniform(1.0, condition_number)
    return matrix


def f_by_matrix(matrix: np.ndarray):
    return lambda vec: np.array(vec).dot(matrix).dot(vec)


def f_grad_by_matrix(matrix: np.ndarray):
    def grad(vec):
        vec = np.array(vec)
        return [2.0 * vec[i] * matrix[i, i] for i in range(len(vec))]

    return grad


def f3(x: np.ndarray):
    f = (17 * x[0] - 254) ** 2 - 20 * x[0] + (27 * x[1] + 995) ** 2
    return f


def f3_grad(x: np.ndarray):
    f = [-8656 + 578 * x[0],
         54 * (995 + 27 * x[1])]
    return f


def f4(x: np.ndarray):
    return 3 * (x[0] + 1) ** 2 + (x[1] - 1) ** 2


def f4_grad(x: np.ndarray):
    return [6 * (x[0] + 1), 2 * (x[1] - 1)]


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
        plt.title("Amount of iterations")
        plt.show()
        plt.clf()
        for iterations, function_calls, name in self.data:
            plt.plot(x_axes_range, function_calls, label=name)
        plt.legend()
        plt.xlabel("eps")
        plt.xticks(x_axes_range[idx],
                   self.epses[idx])
        plt.ylabel("function calls")
        plt.title("Amount of function calls")
        plt.show()


def build_plots_for_all_searchers(func, a, b, epses):
    plots = PlotBuilder(a, b, epses)
    plots.add_searcher(FibonacciSearcher(func))
    plots.add_searcher(GoldenRatioSearcher(func))
    plots.add_searcher(BisectionSearcher(func))
    plots.add_searcher(LinearSearcher(func))
    plots.show()


def draw_single_arg_function(func, a, b):
    x = np.linspace(a, b, 100)
    plt.plot(x, np.vectorize(func)(x))
    plt.show()


def draw_double_arg_function(func, xlims, ylims, show=False):
    x_step = 0.1
    y_step = 0.1
    x_step = (xlims[1] - xlims[0]) / 100
    y_step = (ylims[1] - ylims[0]) / 100
    x_s = np.arange(xlims[0], xlims[1], x_step)
    y_s = np.arange(ylims[0], ylims[1], y_step)
    z_s = []
    for y in y_s:
        tmp = []
        for x in x_s:
            tmp.append(func(np.array([x, y])))
        z_s.append(tmp)
    z_s = np.array(z_s)
    plt.clf()
    plt.figure()
    cs = plt.imshow(z_s, interpolation='bilinear', cmap=plt.get_cmap("twilight"),
                    origin='lower', extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                    vmax=abs(z_s).max(), vmin=-abs(z_s).max())
    if show:
        plt.show()


def draw_descent_steps(func, grad, start, searcher, eps, color="white", show=False, name=""):
    res, it, path = gradient_descent(func, grad, start, eps, searcher)
    distance = np.linalg.norm(start - res)
    xlims = (res[0] - distance, res[0] + distance)
    ylims = (res[1] - distance, res[1] + distance)
    if show:
        draw_double_arg_function(func, xlims, ylims)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Iterations: " + str(it) + ", Root:\n%.9f\n%.9f" % (res[0], res[1]))
    prev_step = None
    for step in path:
        if prev_step is not None:
            plt.plot([prev_step[0], step[0]],
                     [prev_step[1], step[1]], linewidth=1, c=color)
            plt.scatter(step[0], step[1], c=color, s=3)
        else:
            plt.scatter(step[0], step[1], c=color, s=3, label=name)
        prev_step = step
    if show:
        plt.legend()
        # plt.savefig("D:/res/" + name)
        plt.show()


def draw_const_descent_steps(func, grad, start, eps, color="white", show=False, name=""):
    res, it, path = const_gradient_descent(func, grad, start, eps, 1e-2)
    distance = np.linalg.norm(start - res)
    xlims = (res[0] - distance, res[0] + distance)
    ylims = (res[1] - distance, res[1] + distance)
    if show:
        draw_double_arg_function(func, xlims, ylims)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Iterations: " + str(it) + ", Root:\n%.9f\n%.9f" % (res[0], res[1]))
    prev_step = None
    for step in path:
        if prev_step is not None:
            plt.plot([prev_step[0], step[0]],
                     [prev_step[1], step[1]], linewidth=1, c=color)
            plt.scatter(step[0], step[1], c=color, s=3)
        else:
            plt.scatter(step[0], step[1], c=color, s=3, label=name)
        prev_step = step
    if show:
        plt.legend()
        # plt.savefig("D:/res/" + name)
        plt.show()


def draw_newton_steps(func, grad, grad2, start, eps, color="white", show=False, name=""):
    res, it, path = newton(func, grad, grad2, start, eps, 1e-2)
    distance = np.linalg.norm(start - res)
    xlims = (res[0] - distance, res[0] + distance)
    ylims = (res[1] - distance, res[1] + distance)
    if show:
        draw_double_arg_function(func, xlims, ylims)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Iterations: " + str(it) + ", Root:\n%.9f\n%.9f" % (res[0], res[1]))
    prev_step = None
    for step in path:
        if prev_step is not None:
            plt.plot([prev_step[0], step[0]],
                     [prev_step[1], step[1]], linewidth=1, c=color)
            plt.scatter(step[0], step[1], c=color, s=3)
        else:
            plt.scatter(step[0], step[1], c=color, s=3, label=name)
        prev_step = step
    if show:
        plt.legend()
        # plt.savefig("D:/res/" + name)
        plt.show()

def draw_all_descent_steps(func, grad, grad2, start, eps, single=True):
    if single:
        start = np.array(start)
        res, it, path = gradient_descent(func, grad, start, eps, None)
        distance = np.linalg.norm(start - res)
        xlims = (res[0] - distance, res[0] + distance)
        ylims = (res[1] - distance, res[1] + distance)
        draw_double_arg_function(func, xlims, ylims)
        delta = 1
        draw_descent_steps(func, grad, start, None, eps, "white", False, "LinearSearcher")
        draw_descent_steps(func, grad, start + [delta, 0], BisectionSearcher, eps, "red", False, "BisectionSearcher")
        draw_descent_steps(func, grad, start + [2 * delta, 0], GoldenRatioSearcher, eps, "green", False,
                           "GoldenRatioSearcher")
        draw_descent_steps(func, grad, start + [3 * delta, 0], FibonacciSearcher, eps, "blue", False,
                           "FibonacciSearcher")
        draw_const_descent_steps(func, grad, start + [4 * delta, 0], eps, "yellow", False,
                                 "Const, step = 1e-2")
        draw_newton_steps(func, grad, grad2, start + [5 * delta, 0], eps, "orange", False, "Newton")
        
        plt.legend()
        plt.show()
    else:
        draw_descent_steps(func, grad, start, None, eps, "white", True, "LinearSearcher")
        draw_descent_steps(func, grad, start, BisectionSearcher, eps, "red", True, "BisectionSearcher")
        draw_descent_steps(func, grad, start, GoldenRatioSearcher, eps, "green", True,
                           "GoldenRatioSearcher")
        draw_descent_steps(func, grad, start, FibonacciSearcher, eps, "blue", True,
                           "FibonacciSearcher")
        draw_const_descent_steps(func, grad, start, eps, "yellow", True, "Const, step = 1e-2")
        draw_newton_steps(func, grad, grad2, start, eps, "orange", True, "Newton")


def run_all_gradients(f, g, start, eps):
    res, it, _ = gradient_descent(f, g, start, eps, None)
    print("LinearStepSearch\n    ans =", list(res), "\n    amount of iterations =", it)
    res_, it, _ = gradient_descent(f, g, start, eps, BisectionSearcher)
    print("BisectionSearcher\n    ans =", list(res_), "\n    amount of iterations =", it)
    res_, it, _ = gradient_descent(f, g, start, eps, GoldenRatioSearcher)
    print("GoldenRatioSearcher\n    ans =", list(res_), "\n    amount of iterations =", it)
    res_, it, _ = gradient_descent(f, g, start, eps, FibonacciSearcher)
    print("FibonacciSearcher\n    ans =", list(res_), "\n    amount of iterations =", it)
    return res


def task1():
    a = -100
    b = 100
    draw_single_arg_function(f1, a, b)
    build_plots_for_all_searchers(f1, a, b, [2 ** x for x in range(-1, -20, -1)])


def task2():
    start = [-23, 86]
    eps = 1e-18
    run_all_gradients(f3, f3_grad, start, eps)


def task6():
    def f1(x: np.ndarray):
        f = ((x[0] - 17) - 3 * x[1]) ** 2 + 4 * x[0] ** 2 - 8 * x[0]
        return f

    def f1_grad(x: np.ndarray):
        f = [10 * x[0] - 6 * (7 + x[1]), -6 * (-17 + x[0] - 3 * x[1])]
        return f

    def f1_grad2(x: np.ndarray):
        f = [
            [10, -6],
            [-6, 18]
        ]
        return f

    def f2(x: np.ndarray):
        f = ((x[0] - 17) - 3 * x[1]) ** 2
        return f

    def f2_grad(x: np.ndarray):
        f = [2 * (x[0] - 17) - 3 * 2 * x[1], 3 * 3 * x[1] - 3 * 2 * (x[0] - 14)]
        return f


    start = [-1, 10]
    eps = 1e-9
    m = generate_matrix(2, 0.001)
    f_1 = f_by_matrix(m)
    f_1_grad = f_grad_by_matrix(m)
    draw_all_descent_steps(f1, f1_grad, f1_grad2, start, eps, False)
    draw_all_descent_steps(f1, f1_grad, f1_grad2, start, eps, True)


def task7():
    n_range = range(3, 20)
    k_range = [2, 3, 4, 5, 10, 20, 25, 50, 100, 150, 200, 250, 500, 1000]
    zs = []
    random.seed(12856)
    for n in n_range:
        tmp = []
        max_its = 0
        for k_ in k_range:
            k = k_
            m = generate_matrix(n, k)
            # print(get_condition_number(m))
            f = f_by_matrix(m)
            f_grad = f_grad_by_matrix(m)
            start = np.ones(n) * 1
            res, it, path = gradient_descent(f, f_grad, start, 1e-9, BisectionSearcher)
            tmp.append(it)
        print(tmp)
        zs.append(tmp)
    zs = np.array(zs)
    cs = plt.imshow(zs, interpolation='bicubic', cmap=plt.get_cmap("hot"),
                    origin='lower')
    plt.show()
