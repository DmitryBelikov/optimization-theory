import numpy as np

from Lab1.methods import BisectionSearcher, LinearSearcher


def stop_criterion(grad, w, w0, eps):
    grad_w = grad(w)
    grad_w0 = grad(w0)
    norm = np.linalg.norm(grad_w, 1) ** 2 < eps * np.linalg.norm(grad_w0, 1) ** 2
    return norm


class GradientStepSelector:
    def __init__(self, func, grad, eps, searcher_builder=None):
        self.searcher_builder = searcher_builder
        self.func = func
        self.grad = grad
        self.eps = eps
        self.alpha = 1
        self.sigma = 0.05

    def get_step(self, x, d):
        if self.searcher_builder is None:
            self.alpha *= 2
            gradient_value = self.grad(x)
            func_value = self.func(x)
            while all(self.func(x - self.alpha * gradient_value) - func_value >
                      -self.alpha * self.sigma * np.linalg.norm(gradient_value, 2)):
                self.alpha /= 2
            return np.array([self.alpha])
        linear = LinearSearcher(lambda a: self.func(x - a * d))
        _, right_border, _, _ = linear.search(np.zeros(len(x)), np.zeros(len(x)), self.eps)
        # right_border = np.ones(len(x)) * 0.256
        # print("right border =", right_border)
        searcher = self.searcher_builder(lambda a: self.func(x - a * d))
        l, r, _, _ = searcher.search(np.zeros(len(right_border)), right_border, self.eps)
        result = (l + r) / 2
        if all(result > right_border):
            assert False
        else:
            return result


def gradient_descent(func, grad, w0, eps=1e-9, searcher=None):
    step_selector = GradientStepSelector(func, grad, 1e-9, searcher)
    w0 = np.array(w0.copy(), np.float64)
    w = np.array(w0.copy(), np.float64)
    iterations = 0
    while not stop_criterion(grad, w, w0, eps):
        # print("w =", w)
        gradient_value = grad(w)
        alpha = step_selector.get_step(w, grad(w))
        # print("alpha =", alpha)
        # print("gradient =", gradient_value)
        delta_w = alpha * gradient_value
        w -= delta_w
        iterations += 1
        if all(alpha < 1e-20):
            print("Alpha = 0")
            break
        # if iterations > 10:
        #     exit(9)
    return w, iterations
# [-1.5        -0.83966064 -1.25       -1.25      ]
