import numpy as np
from scipy.optimize import linprog

from lab2.simplex_method import simplex_method, simplex_method_ub
from lab2.utils import python_simplex_method_ub


class LP:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


def vg_method_my(c, a, b):
    return vg_method(c, a, b, lambda c, a, b: simplex_method_ub(c, a, b))


def vg_method_python(c, a, b):
    return vg_method(c, a, b, lambda c, a, b: python_simplex_method_ub(c, a, b))


def vg_method(c, a, b, f):
    ans = None
    queue = [LP(a, b, c)]
    while queue:
        s = queue.pop(0)
        res = f(s.c, s.a, s.b)
        for i in range(len(res.x)):
            res.x[i] = round(res.x[i], 7)
        if ans is not None and ans.fun <= res.fun:
            continue
        found = False
        for i in range(len(s.a[0])):
            if not res.success:
                return res
            if abs(int(res.x[i]) - res.x[i]) > 1e-7:
                found = True
                tmp = [0] * len(s.a[0])
                tmp[i] = 1
                queue.append(LP(
                    s.a + [tmp],
                    s.b + [int(res.x[i])],
                    s.c)
                )

                tmp2 = [0] * len(s.a[0])
                tmp2[i] = -1
                queue.append(LP(
                    s.a + [tmp2],
                    s.b + [-(int(res.x[i]) + 1)],
                    s.c)
                )

        if not found and (ans is None or ans.fun > res.fun):
            ans = res
    return ans
