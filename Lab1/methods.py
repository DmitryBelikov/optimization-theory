from abc import abstractmethod
import matplotlib.pyplot as plt

import numpy as np


class Searcher:
    def __init__(self, func):
        self.func = func
        self.iterations = -1
        self.function_calls = -1

    @abstractmethod
    def search(self, a, b, eps): raise NotImplemented

    @abstractmethod
    def __str__(self): raise NotImplemented

    def draw_function(self, a, b):
        x = np.linspace(a[0], b[0], 100)
        plt.xticks(np.arange(a[0], b[0], 0.20))
        plt.plot(x, np.vectorize(self.func)(x))
        plt.show()


class LinearSearcher(Searcher):
    def __init__(self, func):
        super().__init__(func)
        self.step = 2.0

    def __str__(self):
        return "linear"

    def search(self, a, b=0, eps=1e-18):
        a = np.array(a)
        b = np.array(b)
        delta = eps
        start_value = self.func(a)
        right_step = self.func(a + delta)
        while start_value == right_step:
            delta *= self.step
            right_step = self.func(a + delta)
        direction = -np.sign(right_step - start_value)
        last_value = start_value
        cur_delta = delta
        cur_x = a + delta * direction
        self.iterations = 0
        self.function_calls = 2
        while True:
            new_last_value = self.func(cur_x)
            self.function_calls += 1
            self.iterations += 1
            if last_value < new_last_value:
                break
            last_value = new_last_value
            cur_x += cur_delta * direction
            cur_delta *= self.step
        if a < cur_x:
            return a, cur_x, self.iterations, self.function_calls
        else:
            return cur_x, a, self.iterations, self.function_calls
    # def search(self, a, b=0, eps=1e-18):
    #     a = np.array(a)
    #     b = np.array(b)
    #     start_value = self.func(a)
    #     right_border = a + eps
    #     cur_delta = eps
    #     while self.func(right_border) <= start_value + eps:
    #         cur_delta *= 2
    #         right_border += cur_delta
    #     return 0, right_border, 0, 0


class BisectionSearcher(Searcher):
    def __init__(self, func):
        super().__init__(func)

    def __str__(self):
        return "bisection"

    def search(self, a, b, eps):
        a = np.array(a)
        b = np.array(b)
        delta = eps / 4
        self.iterations = 0
        self.function_calls = 0
        while np.linalg.norm(a - b) > eps:
            ml = (a + b - 2 * delta) / 2
            mr = (a + b + 2 * delta) / 2
            f_ml = self.func(ml)
            f_mr = self.func(mr)
            self.function_calls += 2
            self.iterations += 1
            if f_ml > f_mr:
                a = ml
                b = b
            elif f_ml < f_mr:
                a = a
                b = mr
            else:
                a = ml
                b = mr
                return a, b, self.iterations, self.function_calls
        return a, b, self.iterations, self.function_calls


class GoldenRatioSearcher(Searcher):
    def __init__(self, func):
        super().__init__(func)
        self.ratio = 0.618

    def __str__(self):
        return "golden ratio"

    def search(self, a, b, eps):
        a = np.array(a)
        b = np.array(b)
        temp_l = a + (1 - self.ratio) * (b - a)
        temp_r = a + self.ratio * (b - a)
        self.iterations = 0
        self.function_calls = 0
        while np.linalg.norm(b - a) > eps:
            f_temp_l = self.func(temp_l)
            f_temp_r = self.func(temp_r)
            self.function_calls += 2
            self.iterations += 1
            if f_temp_l > f_temp_r:
                a = temp_l
                b = b
                temp_l = temp_r
                temp_r = a + self.ratio * (b - a)
            elif f_temp_l < f_temp_r:
                a = a
                b = temp_r
                temp_r = temp_l
                temp_l = a + (1 - self.ratio) * (b - a)
            else:
                return a, b, self.iterations, self.function_calls
        return a, b, self.iterations, self.function_calls


class FibonacciSearcher(Searcher):
    fib = None

    def __init__(self, func):
        super().__init__(func)
        if FibonacciSearcher.fib is None:
            FibonacciSearcher.fib = [1, 1]
            for i in range(2, 1000):
                FibonacciSearcher.fib.append(FibonacciSearcher.fib[-1] + FibonacciSearcher.fib[-2])
        self.gamma = 1e-15

    def __str__(self):
        return "fibonacci"

    def search(self, a, b, eps):
        a = np.array(a)
        b = np.array(b)
        n = 0
        fib = FibonacciSearcher.fib
        # eps = max(1e-3, eps)
        while fib[n] < np.linalg.norm(b - a) / eps:
            n += 1
        temp_l = a + fib[n - 2] / fib[n] * (b - a)
        temp_r = a + fib[n - 1] / fib[n] * (b - a)
        self.iterations = 0
        self.function_calls = 0
        k = -1
        while k < n - 2:
            k += 1
            f_temp_l = self.func(temp_l)
            f_temp_r = self.func(temp_r)
            self.function_calls += 2
            self.iterations += 1
            if f_temp_l > f_temp_r:
                a = temp_l
                b = b
                temp_l = temp_r
                temp_r = a + fib[n - k - 1] / fib[n - k] * (b - a)
            else:
                a = a
                b = temp_r
                temp_r = temp_l
                temp_l = a + fib[n - k - 2] / fib[n - k] * (b - a)
        temp_l = temp_l
        temp_r = temp_l + self.gamma
        self.function_calls += 2
        if self.func(temp_l) == self.func(temp_r):
            a = temp_l
            b = b
        else:
            a = a
            b = temp_r
        return a, b, self.iterations, self.function_calls

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
