from scipy.optimize import linprog

from samples import *
from simplex_method import simplex_method


def python_simplex_method(q, a, b):
    return linprog(c=q, A_eq=a, b_eq=b, method="simplex")


def print_results(method_results, msg=""):
    print(msg)
    if method_results.success:
        print("Result:", method_results.fun)
    else:
        print("Optimization failed.")


def single_test(q, a, b, test_name):
    python_ans = python_simplex_method(q, a, b)
    my_ans = simplex_method(q, a, b)
    if python_ans.success == my_ans.success:
        if python_ans == "False":
            print(test_name + ": correct")
            return True
        if (python_ans.fun - my_ans.fun) < 1e-15:
            print(test_name + ": correct")
            return True
        else:
            print(test_name + ": wrong")
            print("     My ans =", my_ans.fun)
            print("     Python ans =", python_ans.fun)
            return False
    else:
        print(test_name + ": wrong")
        print("     My ans =", my_ans.success)
        print("     Python ans =", python_ans.success)
        return False


def full_test():
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
