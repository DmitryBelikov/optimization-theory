import numpy as np

from Lab1.methods import BisectionSearcher, LinearSearcher


def stop_criterion(grad, w, w0, eps):
    grad_w = grad(w)
    grad_w0 = grad(w0)
    norm = np.linalg.norm(grad_w) ** 2 < eps * np.linalg.norm(grad_w0) ** 2
    return norm


class GradientStepSelector:
    def __init__(self, func, grad, eps, searcher_builder=None):
        self.searcher_builder = searcher_builder
        self.func = func
        self.grad = grad
        self.eps = eps
        self.alpha = 1
        self.sigma = 0.05

    def get_step(self, x):
        gradient_value = np.array(self.grad(x))
        if self.searcher_builder is None:
            self.alpha *= 2
            func_value = self.func(x)
            while self.func(x - self.alpha * gradient_value) - func_value > \
                    -self.alpha * self.sigma * np.linalg.norm(gradient_value):
                self.alpha /= 2
            return self.alpha
        linear = LinearSearcher(lambda a: self.func(x - a * gradient_value))
        _, right_border, _, _ = linear.search(0, 0, self.eps)
        searcher = self.searcher_builder(lambda a: self.func(x - a * gradient_value))
        l, r, _, _ = searcher.search(0, right_border, self.eps)
        result = (l + r) / 2
        if result > right_border:
            return right_border
        else:
            return result


def gradient_descent(func, grad, w0, eps=1e-9, searcher=None):
    step_selector = GradientStepSelector(func, grad, 1e-9, searcher)
    w0 = np.array(w0.copy(), np.float64)
    w = np.array(w0.copy(), np.float64)
    iterations = 0
    path = [w0]
    while not stop_criterion(grad, w, w0, eps):
        gradient_value = np.array(grad(w))
        alpha = step_selector.get_step(w)
        delta_w = alpha * gradient_value
        w -= delta_w
        path.append(w.copy())
        iterations += 1
        if alpha < 1e-20:
            # print("Alpha = 0")
            break
    return w, iterations, path
# [-1.5        -0.83966064 -1.25       -1.25      ]
