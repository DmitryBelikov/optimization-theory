import numpy as np

from Lab1.methods import BisectionSearcher
from Lab1.utils import GradientStepSelector


def stop_criterion(grad, w, w0, eps):
    grad_w = grad(w)
    graw_w0 = grad(w0)
    norm = np.linalg.norm(grad_w, 1) ** 2 < eps * np.linalg.norm(graw_w0, 1) ** 2
    return norm


def gradient_descent(func, grad, w0):
    step_selector = GradientStepSelector(BisectionSearcher, func, grad, 1e-10)
    w0 = np.array(w0.copy(), np.float64)
    w = np.array(w0.copy(), np.float64)
    while not stop_criterion(grad, w, w0, 1e-10):
        # print(w)
        d = grad(w0)
        alpha = step_selector.get_step(w, d)
        gradient_value = grad(w)
        delta_w = alpha * gradient_value
        w -= delta_w
    return w
