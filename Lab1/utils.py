import numpy as np
import matplotlib.pyplot as plt

from Lab1.methods import Searcher, LinearSearcher


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


class GradientStepSelector:
    def __init__(self, searcher_builder, func, grad, eps):
        self.searcher_builder = searcher_builder
        self.func = func
        self.grad = grad
        self.eps = eps

    def get_step(self, x, d):
        linear = LinearSearcher(lambda a: self.func(x - a * d), lambda a: self.grad(x - a * d))
        _, right_border = linear.search(0, 0, 0)
        searcher = self.searcher_builder(lambda a: self.func(x - a * d), lambda a: self.grad(x - a * d))
        l, r = searcher.search(0, right_border, self.eps)
        return (l + r) / 2
