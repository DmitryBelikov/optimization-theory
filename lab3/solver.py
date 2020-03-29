import numpy as np
from scipy.optimize import linprog


def solve_game(a: np.ndarray):
    addition = (-(a.min(initial=0)) + 1)
    a += addition

    a = -a
    c = np.array([1] * a.shape[0])
    b = np.array([-1] * a.shape[1])
    solution_eps = linprog(c=c, A_ub=a, b_ub=b, method='simplex')
    v = 1 / solution_eps.fun
    xs = solution_eps.x * v

    c = np.array([1] * a.shape[1])
    b = np.array([-1] * a.shape[0])
    a = a.T

    solution_nuys = linprog(c=c, A_ub=a, b_ub=b, method='simplex')
    u = 1 / solution_nuys.fun
    ys = solution_nuys.x * u

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
