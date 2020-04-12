from abc import abstractmethod
import matplotlib.pyplot as plt

import numpy as np


class Searcher:
    def __init__(self, func, grad):
        self.func = func
        self.grad = grad

    @abstractmethod
    def search(self, a, b, eps): raise NotImplemented

    def draw_function(self, a, b):
        x = np.linspace(a, b, 100)
        plt.xticks(np.arange(a, b, 0.20))
        plt.plot(x, np.vectorize(self.func)(x))
        plt.show()


class LinearSearcher(Searcher):
    def __init__(self, func, grad):
        super().__init__(func, grad)
        self.step = 2.0
        self.delta = 1e-3

    def search(self, a, b=0, eps=0):
        start_value = self.func(a)
        right_step = self.func(a + self.delta)
        direction = -np.sign(right_step - start_value)
        last_value = start_value
        cur_delta = self.delta
        cur_x = a + self.delta * direction
        while last_value > self.func(cur_x):
            last_value = self.func(cur_x)
            cur_x += cur_delta * direction
            cur_delta *= self.step
        return min(a, cur_x), max(a, cur_x)


class BisectionSearcher(Searcher):
    def __init__(self, func, grad):
        super().__init__(func, grad)

    def search(self, a, b, eps):
        delta = eps / 4
        while abs(a - b) > eps:
            ml = (a + b) / 2 - delta
            mr = (a + b) / 2 + delta
            if self.func(ml) > self.func(mr):
                a = ml
                b = b
            elif self.func(ml) < self.func(mr):
                a = a
                b = mr
            else:
                return a, b
        return a, b


class GoldenRatioSearcher(Searcher):
    def __init__(self, func, grad):
        super().__init__(func, grad)
        self.ratio = 0.618

    def search(self, a, b, eps):
        temp_l = a + (1 - self.ratio) * (b - a)
        temp_r = a + self.ratio * (b - a)
        while abs(b - a) > eps:
            if self.func(temp_l) > self.func(temp_r):
                a = temp_l
                b = b
                temp_l = temp_r
                temp_r = a + self.ratio * (b - a)
            elif self.func(temp_l) < self.func(temp_r):
                a = a
                b = temp_r
                temp_r = temp_l
                temp_l = a + (1 - self.ratio) * (b - a)
            else:
                return a, b
        return a, b
# def fibonacci(a, b, eps):
#     fib = [0.0, 1.0]
#     while fib[-1] <= (b - a) / eps:
#         fib.append(fib[-2] + fib[-1])
#
#     n = len(fib) - 3
#
#     x1 = a + (fib[n] / fib[-1]) * (b - a)
#     x2 = a + (fib[n + 1] / fib[-1]) * (b - a)
#     f1 = f(x1)
#     f2 = f(x2)
#     while n > 0:
#         n -= 1
#         if f(x1) < f(x2):
#             b = x2
#             x2 = x1
#             f2 = f1
#             x1 = a + (fib[n] / fib[-1]) * (b - a)
#             f1 = f(x1)
#         elif f(x1) > f(x2):
#             a = x1
#             x1 = x2
#             f1 = f2
#             x2 = a + (fib[n + 1] / fib[-1]) * (b - a)
#             f2 = f(x2)
#         else:
#             a = x1
#             b = x2
#     return a, b
