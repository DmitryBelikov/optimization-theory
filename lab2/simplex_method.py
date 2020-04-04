import numpy as np


class SimplexMethodResults:
    def __init__(self, success, fun, x):
        self.success = success
        self.fun = fun
        self.x = x

    def __str__(self):
        return 'success: {}\n' \
               'fun: {}\n' \
               'x: {}\n'.format(self.success, self.fun, self.x)


def check_if_basis(a, j):
    count = 0
    for i in range(a.shape[0]):
        if a[i][j] == 0:
            count += 1
    return count == a.shape[0] - 1


def simplex_method_ub(c, a, b):
    a = a.copy()
    b = b.copy()
    c = c.copy()
    # добавим единичную матрицу чтобы перевести <= в =
    a = np.hstack((np.array(a, np.float), np.eye(len(a))))
    # добавим нулей для добавочных переменных
    c = np.hstack((np.array(c, np.float), np.zeros(a.shape[0], np.float)))
    # запустим обычное решение
    res = simplex_method(c, a, b)
    # отрежем добавочные переменные
    res.x = res.x[:-len(a)]
    return res


def simplex_method(c, a, b):
    a = a.copy()
    b = b.copy()
    c = c.copy()
    # добавим единичную матрицу для искуственного базиса (M метод)
    a = np.hstack((np.array(a, np.float), np.eye(len(a))))
    # переносим С в левую часть и дополним нулями добавочных переменных(далее заменим на -M)
    c = np.hstack((np.array(c, np.float) * (-1), np.zeros(a.shape[0], np.float)))
    # возьмем большое M
    M = 100000.0
    np.seterr(divide='ignore', invalid='ignore')
    # функцию приравняем к нулю(это наше начальное решение)
    # все х = 0, а добавочные переменные = bi
    q0 = 0.0
    # приведем систему уравнений к виду ax=b, где все b>=0
    for i in range(0, len(a)):
        if b[-i - 1] != 0:
            a[-i - 1] *= np.sign(b[-i - 1])
            b[-i - 1] *= np.sign(b[-i - 1])
    # добавляем штраф на добавочные переменные в размере M. После переноса в левую часть -M
    for i in range(0, a.shape[0]):
        c[-i - 1] = -M
    # избавляемся от коэффициентов M чтобы F(x, x_added) = 0
    for i in range(0, a.shape[0]):
        c = c + a[i] * M
        q0 += b[i] * M
    b = np.array(b, np.float)
    # начальное решение X = 0, X_added_i = bi, F = 0
    # базис при этом X_added
    basis = np.arange(a.shape[0]) + a.shape[0] - 1
    eps = 1e-12
    # пока не все коэффициенты функции меньше равны 0
    while not all(c <= eps):
        max_c = np.amax(c)
        # самый большой коэффициент функции дает наибольший вклад в сумму
        l = np.where(c == max_c)[0][0]
        # выбираем какую переменную из базиса будем заменять
        pivot_column = b / a[:, l]
        cur_min = np.inf
        r = -1
        for i in range(0, len(pivot_column)):
            if 0 < a[i][l] and pivot_column[i] <= cur_min:
                r = i
                cur_min = pivot_column[i]
        if r == -1:
            return SimplexMethodResults(False, q0, [])
        # заменяем
        basis[r] = l
        # разрешаюший элемент
        pivot_el = a[r][l]
        if pivot_el == 0:
            return SimplexMethodResults(False, q0, [])
        # Жорданово исключение
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
    # если значение функции оказалось близко к M. Которое в идеале -> inf. То метод разошелся
    if q0 > 100000 / 10:
        return SimplexMethodResults(False, q0, [])

    x = [0] * (a.shape[1] - a.shape[0])
    for i in range(a.shape[0]):
        # если добавочная переменная вошла в базис, то это плохо
        # (она добавлена искуственно и не несет смысла)
        if basis[i] > a.shape[1] - a.shape[0] - 1:
            # она может зануляться, если занулсяется, то все окей, если нет, то баним решение
            if b[i] / a[i][basis[i]] > 1e-9:
                SimplexMethodResults(False, q0, x)
        else:
            # если переменная не добавочная, то просто расчитываем ее
            x[basis[i]] = b[i] / a[i][basis[i]]
    return SimplexMethodResults(True, q0, x)
