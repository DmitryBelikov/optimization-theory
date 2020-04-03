import numpy as np
from scipy.optimize import linprog
from lab2.simplex_method import simplex_method_ub


def main():
    game = np.array([
        [1, 1, 1],
        [2, 1, 1],
        [2, 2, 1],
        [1, 2, 2],
        [1, 1, 2],
        [1, 1, 1]
    ], dtype=float)
    n = game.shape[1]
    m = game.shape[0]

    print("Game matrix:")
    print(game)

    # a = np.append(-game, -np.eye(m), axis=1).T
    a = -game.T
    print(a)

    # b = np.append(-np.ones(n), np.zeros(m))
    b = -np.ones(n)
    print(b)

    c = np.ones(m)
    print(c)
    result = linprog(c=c, A_ub=a, b_ub=b)
    
    print(result.success)
    print(simplex_method_ub(c, a, b).x)

    if not result.success:
        print(result.message)
    else:
        print(result.x)


    # print('First player guarantee: {}'.format(x_balance))
    # print('Second player guarantee: {}'.format(y_balance))
    # print('First player strategy: {}'.format(xs))
    # print('Second player strategy: {}'.format(ys))


if __name__ == '__main__':
    main()
