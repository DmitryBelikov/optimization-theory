from lab2.utils import *
from lab2.samples import *
from lab2.vetvei_granis import *


def main():
    q, a, b = bad_not_antony()
    pyres = python_simplex_method_ub(q, a, b)
    myres = simplex_method_ub(q, a, b)
    if pyres.success == myres.success:
        if not pyres.success:
            print("False")
        else:
            print(pyres.x)
            print(myres.x)
    else:
        print(pyres.x)
        print(myres.x)
        print("Py ans != your ans")


# single_test(q, a, b, "test11")
# full_test_simplex()


if __name__ == '__main__':
    main()
