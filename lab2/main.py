from lab2.utils import *


def main():
    full_test_samples()
    full_test_bnb()
    c, a, b = bad_not_antony()
    my = simplex_method(c, a, b)
    py = python_simplex_method(c, a, b)
    print(my.success, py.success)
    print(my.x, py.x)
    print(my.fun, py.fun)
    c, a, b = bad_frogsrop1()
    my = simplex_method(c, a, b)
    py = python_simplex_method(c, a, b)
    print(my.success, py.success)
    print(my.x, py.x)
    print(my.fun, py.fun)
    c, a, b = bad_frogsrop2()
    my = simplex_method(c, a, b)
    py = python_simplex_method(c, a, b)
    print(my.success, py.success)
    print(my.x, py.x)
    print(my.fun, py.fun)


if __name__ == '__main__':
    main()
