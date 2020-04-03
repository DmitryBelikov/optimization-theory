import numpy as np
from scipy.optimize import linprog
from lab2.simplex_method import simplex_method_ub


def solve_game(a, b, c):
    # result = linprog(c=c, A_ub=a, b_ub=b)
    result = simplex_method_ub(c, a, b)
    if not result.success:
        raise ArithmeticError(getattr(result, "message", "Oops, Something went wrong"))

    return result


def print_results(result):
    print("Guaranteed score: {:.6}".format(1 / np.sum(result)))
    print("Optimal strategy:\n{}".format(result / np.sum(result)))


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

    print("Game matrix:\n{}".format(game))

    try:
        a = -game.T
        b = -np.ones(n)
        c = np.ones(m)
        result = solve_game(a, b, c)
        print("\nPlayer 1 results:")
        print_results(result.x)
        
        a = game
        b = np.ones(m)
        c = -np.ones(n)
        result = solve_game(a, b, c)
        print("\nPlayer 2 results:")
        print_results(result.x)
        
    except ArithmeticError as e:
        print(e)
        return


if __name__ == '__main__':
    main()
