import numpy as np

from Lab1.methods import BisectionSearcher, LinearSearcher


def stop_criterion(grad, w, w0, eps):
    grad_w = grad(w)
    grad_w0 = grad(w0)
    norm = np.linalg.norm(grad_w, 1) ** 2 < eps * np.linalg.norm(grad_w0, 1) ** 2
    return norm


class GradientStepSelector:
    def __init__(self, searcher_builder, func, eps):
        self.searcher_builder = searcher_builder
        self.func = func
        self.eps = eps

    def get_step(self, x, d):
        linear = LinearSearcher(lambda a: self.func(x - a * d))
        _, right_border, _, _ = linear.search(0, 0, 1e-3)
        searcher = self.searcher_builder(lambda a: self.func(x - a * d))
        l, r, _, _ = searcher.search(0, right_border, self.eps)
        return (l + r) / 2


def gradient_descent(func, grad, w0, searcher=BisectionSearcher, eps=1e-9):
    step_selector = GradientStepSelector(searcher, func, 1e-9)
    w0 = np.array(w0.copy(), np.float64)
    w = np.array(w0.copy(), np.float64)
    while not stop_criterion(grad, w, w0, eps):
        d = grad(w)
        alpha = step_selector.get_step(w, d)
        gradient_value = grad(w)
        delta_w = alpha * gradient_value
        w -= delta_w
    return w
