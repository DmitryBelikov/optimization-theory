import numpy as np


class simplexMethodResults:
    def __init__(self, success, fun):
        self.success = success
        self.fun = fun


def simplex_method(q, a, b):
    q = np.array(q) * (-1)
    q0 = 0
    a = np.array(a, np.float)
    b = np.array(b)
    # print("q", q)
    # print("a", a)
    # print("b", b)
    while not all(q <= 0):
        max_q = np.amax(q)
        max_column_idx = np.where(q == max_q)[0][0]
        pivot_column = b / a[:, max_column_idx]
        zero_more = [x for x in pivot_column if x >= 0]
        if len(zero_more) == 0:
            return simplexMethodResults(False, q0)
        min_b_div_min_q = np.amin(zero_more)
        min_row_idx = np.where(pivot_column == min_b_div_min_q)[0][0]
        pivot_el = a[min_row_idx][max_column_idx]
        for i in range(0, len(b)):
            if i == min_row_idx:
                continue
            multiplier = -a[i, max_column_idx] / pivot_el
            fix_row = a[min_row_idx] * multiplier
            fix_b = b[min_row_idx] * multiplier
            a[i] = a[i] + fix_row
            b[i] = b[i] + fix_b
        multiplier = -q[max_column_idx] / pivot_el
        q = q + a[min_row_idx] * multiplier
        q0 = q0 + b[min_row_idx] * multiplier
        # print("a = ", a)
        # print("q < 0")
        # print("q = ", q)
        # print("b = ", b)
        # print("q0 = ", q0)
    return simplexMethodResults(True, q0)
