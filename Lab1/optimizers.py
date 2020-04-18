from abc import abstractmethod
from Lab1.newton import newton
from Lab1.gradient_descent import gradient_descent, stop_criterion_dx


class Optimizer:
    @abstractmethod
    def minimize(self, x):
        raise NotImplemented


class Newton(Optimizer):
    def __init__(self, func, grad, hess, eps=1e-9):
        self.func = func
        self.grad = grad
        self.hess = hess
        self.eps = eps

    def minimize(self, x):
        return newton(self.func, self.grad, self.hess, x, self.eps, None)[0]


class GradientDescent(Optimizer):
    def __init__(self, func, grad, eps=1e-9, searcher=None, stop_criterion=stop_criterion_dx):
        self.func = func
        self.grad = grad
        self.eps = eps
        self.searcher = searcher
        self.stop_criterion = stop_criterion

    def minimize(self, x):
        return gradient_descent(self.func, self.grad, x, self.eps, self.searcher, self.stop_criterion)[0]
