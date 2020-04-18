from Lab1.utils import *
import Lab1.methods


# Single search
# a = [-10]
# b = [10]
# searcher = FibonacciSearcher(f1)
# searcher.draw_function(a, b)
# l, r, iterations, function_calls = searcher.search(a, b, 0.001)
# print(l, r)


def main():
    # draw_descent_steps(f4, f4_grad, start, GoldenRatioSearcher, eps)
    # print("LinearStepSearch", gradient_descent(f3, f3_grad, [-2, 51], 1e-9, None))
    task1()
    task2()
    task6()
    # task7()

def len_change():
    searcher = BisectionSearcher(f1)
    searcher.to_print = True
    print(searcher.search(-5, 5, 1e-9))


if __name__ == '__main__':
    main()
