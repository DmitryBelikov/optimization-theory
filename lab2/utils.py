from scipy.optimize import linprog

from lab2.branch_n_bound import branch_n_bound
from lab2.samples import *
from lab2.simplex_method import simplex_method, simplex_method_ub


def bnb_method_my(c, a, b):
    return branch_n_bound(c, a, b, lambda c, a, b: simplex_method_ub(c, a, b))


def bnb_method_python(c, a, b):
    return branch_n_bound(c, a, b, lambda c, a, b: python_simplex_method_ub(c, a, b))


def python_simplex_method(q, a, b):
    ans = linprog(c=q, A_eq=a, b_eq=b, method="simplex")
    return ans


def python_simplex_method_ub(q, a, b):
    ans = linprog(c=q, A_ub=a, b_ub=b, method="simplex")
    return ans


def print_results(method_results, msg=""):
    print(msg)
    if method_results.success:
        print("Result:", method_results.fun)
    else:
        print("Optimization failed.")


def single_test(q, a, b, test_name="Unknown test"):
    python_ans = python_simplex_method(q, a, b)
    my_ans = simplex_method(q, a, b)
    if python_ans.success == my_ans.success:
        if not python_ans.success:
            print(test_name + ": correct", python_ans.success)
            return True
        if (abs(python_ans.fun - my_ans.fun)) < 1e-7:
            print(test_name + ": correct.\n     fun =", python_ans.fun)
            x = my_ans.x
            for i in range(len(x)):
                print("     x_%d = %.3f" % (i, x[i]))
            return True
        else:
            print(test_name + ": wrong")
            print("     My ans =", my_ans.fun)
            print("     Python ans =", python_ans.fun)
            return False
    else:
        print(test_name + ": wrong", python_ans.fun)
        print("     My ans =", my_ans.success)
        print("     Python ans =", python_ans.success)
        return False


def full_test_samples():
    q, a, b = variants_sample1()
    single_test(q, a, b, "sample1")
    q, a, b = variants_sample2()
    single_test(q, a, b, "sample2")
    q, a, b = variants_sample3()
    single_test(q, a, b, "sample3")
    q, a, b = variants_sample4()
    single_test(q, a, b, "sample4")
    q, a, b = variants_sample5()
    single_test(q, a, b, "sample5")
    q, a, b = variants_sample6()
    single_test(q, a, b, "sample6")
    q, a, b = variants_sample7()
    single_test(q, a, b, "sample7")
    q, a, b = variants_sample8()
    single_test(q, a, b, "sample8")
    q, a, b = variants_sample9()
    single_test(q, a, b, "sample9")
    q, a, b = variants_sample10()
    single_test(q, a, b, "sample10")
    q, a, b = variants_sample11()
    single_test(q, a, b, "sample11")
    q, a, b = variants_sample12()
    single_test(q, a, b, "sample12")


def test(q, a, b):
    myres = bnb_method_my(q, a, b)
    pyres = bnb_method_python(q, a, b)
    print(pyres.x, myres.x)
    print(pyres.fun, myres.fun)
    print(pyres.success, myres.success)
    print()


def full_test_bnb():
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
    q, a, b = variants_sample11()
    test(q, a, b)
    q, a, b = variants_sample12()
    test(q, a, b)
