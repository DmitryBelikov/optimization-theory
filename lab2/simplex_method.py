import numpy as np


class SimplexMethodResults:
    def __init__(self, success, fun, x):
        self.success = success
        self.fun = fun
        self.x = x


def check_if_basis(a, j):
    count = 0
    for i in range(a.shape[0]):
        if a[i][j] == 0:
            count += 1
    return count == a.shape[0] - 1


def simplex_method(c, a, b):
    a = np.hstack((np.array(a, np.float), np.eye(len(a))))
    c = np.hstack((np.array(c, np.float) * (-1), np.zeros(a.shape[0], np.float)))
    M = 10000.0
    np.seterr(divide='ignore')
    for i in range(0, len(a)):
        c[-i - 1] = -M
    q0 = 0.0
    for i in range(0, a.shape[0]):
        c = c + a[i] * M
        q0 += b[i] * M
    b = np.array(b, np.float)
    basis = np.arange(a.shape[0]) + a.shape[1] - 2
    eps = 1e-12
    while not all(c <= eps):
        max_c = np.amax(c)
        l = np.where(c == max_c)[0][0]
        pivot_column = b / a[:, l]
        cur_min = np.inf
        r = -1
        for i in range(0, len(pivot_column)):
            if 0 < a[i][l] and pivot_column[i] <= cur_min:
                r = i
                cur_min = pivot_column[i]
        if r == -1:
            return SimplexMethodResults(False, q0, [])
        basis[r] = l
        pivot_el = a[r][l]
        if pivot_el == 0:
            return SimplexMethodResults(False, q0, [])
        for i in range(0, len(b)):
            if i == r:
                continue
            multiplier = -a[i, l] / pivot_el
            fix_row = a[r] * multiplier
            fix_b = b[r] * multiplier
            a[i] = a[i] + fix_row
            b[i] = b[i] + fix_b
        multiplier = -c[l] / pivot_el
        c = c + a[r] * multiplier
        q0 = q0 + b[r] * multiplier
    res = any(basis > a.shape[1] - a.shape[0] - 1)
    if q0 > M:
        res = True
    x = [0] * (a.shape[1] - a.shape[0])
    if not res:
        for i in range(a.shape[0]):
            x[basis[i]] = b[i] / a[i][basis[i]]
    return SimplexMethodResults(not res, q0, x)
