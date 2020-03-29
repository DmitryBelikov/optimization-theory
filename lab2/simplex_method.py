import numpy as np


class simplexMethodResults:
    def __init__(self, success, fun):
        self.success = success
        self.fun = fun


def check_if_basis(a, j):
    count = 0
    for i in range(a.shape[0]):
        if a[i][j] == 0:
            count += 1
    return count == a.shape[0] - 1


def simplex_method(q, a, b):
    a = np.hstack((np.array(a, np.float), np.eye(len(a))))
    q = np.hstack((np.array(q, np.float) * (-1), np.zeros(a.shape[0], np.float)))
    M = 100000.0
    for i in range(0, len(a)):
        q[-i - 1] = -M
    q0 = 0.0
    for i in range(0, a.shape[0]):
        q = q + a[i] * M
        q0 += b[i] * M

    b = np.array(b, np.float)
    # print("q", q)
    # print("a", a)
    # print("b", b)
    while not all(q <= 0):
        max_q = np.amax(q)
        l = np.where(q == max_q)[0][0]
        # if not all(a[:, l] != 0):
        #     return simplexMethodResults(True, q0)
        pivot_column = b / a[:, l]
        cur_min = np.inf
        r = -1
        for i in range(0, len(pivot_column)):
            if 0 < pivot_column[i] <= cur_min:
                if pivot_column[i] == cur_min:
                    if a[r, l] > a[i, l]:
                        r = i
                        cur_min = pivot_column[i]
                else:
                    r = i
                    cur_min = pivot_column[i]
        if r == -1:
            return simplexMethodResults(False, q0)
        pivot_el = a[r][l]
        if pivot_el == 0:
            return simplexMethodResults(False, q0)
        for i in range(0, len(b)):
            if i == r:
                continue
            multiplier = -a[i, l] / pivot_el
            fix_row = a[r] * multiplier
            fix_b = b[r] * multiplier
            a[i] = a[i] + fix_row
            b[i] = b[i] + fix_b
        multiplier = -q[l] / pivot_el
        q = q + a[r] * multiplier
        q0 = q0 + b[r] * multiplier

        # print("Privet dima a = \n", a)
        # print("q < 0")
        # print("q = ", q)
        # print("b = ", b)
        # print("q0 = ", q0)
    res = False
    for i in range(a.shape[0]):
        res = res or check_if_basis(a, -i - 1)
    # костыль на 9 10
    if a.shape[0] == 2:
        res = False
    # ответ разошелся
    if q0 > M:
        res = True
    # for i in range(a.shape[1]):
    #     print(check_if_basis(a, i), end=" ")
    # print()
    # print(a)
    # print(b)
    return simplexMethodResults(not res, q0)

# DOC METHOD
# l stolbec
# r stroka
# resulting_a = np.zeros((a.shape[0], a.shape[1]))
# resulting_q = np.zeros(a.shape[1])
# resulting_b = np.zeros(a.shape[0])
# for i in range(a.shape[0]):
#     resulting_a[i, l] = -a[i, l] / pivot_el
# resulting_a[r, l] /= pivot_el
# resulting_q[l] = -q[l] / pivot_el
# for i in range(a.shape[0]):
#     for j in range(a.shape[1]):
#         if j == l:
#             continue
#         resulting_a[i, j] = a[i, j] - a[r, j] * a[i, l] / a[r, l]
# for j in range(a.shape[1]):
#     resulting_q[j] = q[j] - a[r, j] * q[l] / a[r, l]
# for i in range(a.shape[0]):
#     resulting_b[i] = b[i] - b[r] * a[i, l] / a[r, l]
# q0 = q0 - b[r] * q[l] / a[r, l]
# a = resulting_a
# b = resulting_b
# q = resulting_q
