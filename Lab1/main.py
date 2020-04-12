from Lab1.gradient_descent import gradient_descent
from Lab1.methods import *
from Lab1.utils import *


def main():
    a, b = (-2, 1)
    searcher = BisectionSearcher(f1, f1_grad)
    searcher.draw_function(a, b)
    print(searcher.search(a, b, 0.001))
    print(gradient_descent(f1, f1_grad, [-1], GoldenRatioSearcher))


if __name__ == '__main__':
    main()
