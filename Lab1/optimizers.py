from abc import abstractmethod
from Lab1.newton import newton
from Lab1.gradient_descent import gradient_descent, stop_criterion_dx


class Optimizer:
    def __init__(self):
        self.iters = 0

    @abstractmethod
    def minimize(self, x):
        raise NotImplemented

    def get_iters(self):
        return self.iters


class Newton(Optimizer):
    def __init__(self, func, grad, hess, eps=1e-9):
        super(Newton, self).__init__()
        self.func = func
        self.grad = grad
        self.hess = hess
        self.eps = eps

    def minimize(self, x):
        result, iters, _ = newton(self.func, self.grad, self.hess, x, self.eps, None)
        self.iters += iters
        return result


class GradientDescent(Optimizer):
    def __init__(self, func, grad, eps=1e-9, searcher=None, stop_criterion=stop_criterion_dx):
        super(GradientDescent, self).__init__()
        self.func = func
        self.grad = grad
        self.eps = eps
        self.searcher = searcher
        self.stop_criterion = stop_criterion

    def minimize(self, x):
        result, iters, _ = gradient_descent(self.func, self.grad, x, self.eps, self.searcher, self.stop_criterion)
        self.iters += iters
        return result
