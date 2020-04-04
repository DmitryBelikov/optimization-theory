from lab2.branch_n_bound import *
from lab2.utils import *
from lab2.samples import *
from lab2.vetvei_granis import *


def test(q, a, b):
    myres = bnb_method_my(q, a, b)
    pyres = bnb_method_python(q, a, b)
    nres = vg_method(q, a, b)
    print(pyres.x, myres.x, nres.x)
    print(pyres.fun, myres.fun, nres.fun)
    print(pyres.success, myres.success, nres.success)


def main():
    q, a, b = variants_sample1()
    test(q, a, b)
    q, a, b = variants_sample2()
    test(q, a, b)
    q, a, b = variants_sample3()
    test(q, a, b)
    q, a, b = variants_sample4()
    test(q, a, b)
    q, a, b = variants_sample5()
    test(q, a, b)
    q, a, b = variants_sample6()
    test(q, a, b)
    q, a, b = variants_sample7()
    test(q, a, b)
    q, a, b = variants_sample8()
    test(q, a, b)
    q, a, b = variants_sample9()
    test(q, a, b)
    q, a, b = variants_sample10()
    test(q, a, b)
    q, a, b = variants_sample11()
    test(q, a, b)

    # q, a, b = bad_not_antony()
    # my = simplex_method_ub(q, a, b)
    # py = python_simplex_method_ub(q, a, b)
    # print(my.x, py.x)
    # print(my.fun, py.fun)
    # print(my.success, py.success)
    # q, a, b = bad_frogsrop()
    # my = simplex_method_ub(q, a, b)
    # py = python_simplex_method_ub(q, a, b)
    # print(my.x, py.x)
    # print(my.fun, py.fun)
    # print(my.success, py.success)
    # full_test_simplex()


if __name__ == '__main__':
    main()
