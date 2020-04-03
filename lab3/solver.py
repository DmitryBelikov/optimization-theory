import numpy as np
from scipy.optimize import linprog


def solve_game(a: np.ndarray):
    addition = -a.min(initial=0) + 1
    a += addition

    c = np.array([-1] * a.shape[0])
    b = np.array([-1] * a.shape[1])

    results_x = linprog(c, A_ub=a.T, b_ub=b, method='simplex', bounds=(None, 0))
    v = 1 / results_x.fun

    xs = -results_x.x * v

    b, c = c, b
    b = -b

    results_y = linprog(c, A_ub=a, b_ub=b, method='simplex')
    u = -1 / results_y.fun

    ys = results_y.x * u

    return v - addition, u - addition, xs, ys


def main():
    game = np.array([
        [0, -2/3, -2/3, -2/3, -2/3, -2/3],
        [2/3, 0, -1/3, -1/3, -1/3, -1/3],
        [2/3, 1/3, 0, 0, 0, 0],
        [2/3, 1/3, 0, 0, 1/3, 1/3],
        [2/3, 1/3, 0, -1/3, 0, 2/3],
        [2/3, 1/3, 0, -1/3, -2/3, 0]
    ])

    x_balance, y_balance, xs, ys = solve_game(game)
    print('First player guarantee: {}'.format(x_balance))
    print('Second player guarantee: {}'.format(y_balance))
    print('First player strategy: {}'.format(xs))
    print('Second player strategy: {}'.format(ys))


if __name__ == '__main__':
    main()
