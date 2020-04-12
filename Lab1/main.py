from Lab1.gradient_descent import gradient_descent
from Lab1.methods import *
from Lab1.utils import *


def main():
    # a = [-10]
    # b = [10]
    # searcher = FibonacciSearcher(f1)
    # searcher.draw_function(a, b)
    # l, r, iterations, function_calls = searcher.search(a, b, 0.001)
    # print(l, r)
    # build_plots_for_all_searchers(f1, a, b, [2 ** x for x in range(-1, -20, -1)])
    print("BisectionSearcher", gradient_descent(f3, f3_grad, [-2, 51, 1, -24], 1e-9, BisectionSearcher))
    print("GoldenRatioSearcher", gradient_descent(f3, f3_grad, [-2, 51, 1, -24], 1e-9, GoldenRatioSearcher))
    print("FibonacciSearcher", gradient_descent(f3, f3_grad, [-2, 51, 1, -24], 1e-9, FibonacciSearcher))
    print("LinearStepSearch", gradient_descent(f3, f3_grad, [-2, 51, 1, -24], 1e-9, None))


if __name__ == '__main__':
    main()
