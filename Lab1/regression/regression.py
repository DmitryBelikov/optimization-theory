import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import time
from Lab1.optimizers import GradientDescent, Newton
from Lab1.methods import GoldenRatioSearcher


def regression_function(X, y, w, reg):
    samples = X.shape[0]
    exp_arg = -y * X.dot(w.T)
    return np.ones((1, samples)).dot(np.logaddexp(0, exp_arg)).item() + reg * w[0] * w[0]


def regression_gradient(X, y, w, reg):
    exp_division = scipy.special.expit(-y * X.dot(w.T))
    return X.T.dot(-y * exp_division).T + reg * w


def regression_hessian(X, y, w, reg):
    m, n = X.shape
    result = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            for i in range(m):
                exp_part = np.exp(-y[i] * np.inner(X[i], w))
                special_part = scipy.special.expit(-y[i] * np.inner(X[i], w))
                result[k, j] += y[i] ** 2 * X[i, k] * X[i, j] * special_part / exp_part
    result += reg * np.eye(n)
    return result


def minimization_function(X, y, reg):
    return lambda w: regression_function(X, y, w, reg)


def minimization_gradient(X, y, reg):
    return lambda w: regression_gradient(X, y, w, reg)


def minimization_hessian(X, y, reg):
    return lambda w: regression_hessian(X, y, w, reg)


def process_task(optimizer, w0):
    start_time = time.time()
    weights = optimizer.minimize(w0)
    iterations = optimizer.get_iters()
    end_time = time.time()
    task_time = end_time - start_time
    return {
        'weights': weights,
        'iters': iterations,
        'time': task_time
    }


def logistic_regression(X, y):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    reg = 0.5
    min_func = minimization_function(X, y, reg)
    min_grad = minimization_gradient(X, y, reg)
    min_hess = minimization_hessian(X, y, reg)
    optimizer_descent = GradientDescent(min_func, min_grad, eps=1e-15, searcher=GoldenRatioSearcher)
    optimizer_newton = Newton(min_func, min_grad, min_hess, eps=1e-15)
    w0 = np.array([0] * X.shape[1], dtype=np.float)
    descent_result = process_task(optimizer_descent, w0)
    newton_result = process_task(optimizer_newton, w0)
    return descent_result, newton_result


def get_points(weights):
    xs = [0, 25]
    ys = [-weights[0] / weights[2], (-weights[0] - weights[1] * 25) / weights[2]]
    return xs, ys


def train_regression(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df.drop(['binaryClass'], axis=1).values
    y = df.binaryClass.map(lambda c: -1 if c == 'N' else 1).values
    descent_result, newton_result = logistic_regression(X, y)

    weights_descent = descent_result['weights']
    weights_newton = newton_result['weights']

    xs_descent, ys_descent = get_points(weights_descent)
    xs_newton, ys_newton = get_points(weights_newton)

    plt.plot(xs_descent, ys_descent, color='green')
    plt.plot(xs_newton, ys_newton, color='violet')
    plt.scatter(df.col_1, df.col_2, color=df.binaryClass.map(lambda c: 'red' if c == 'P' else 'blue'))
    plt.show()

    print('Newton took {} iterations and {} seconds to complete'.format(newton_result['iters'], newton_result['time']))
    print('Gradient descent took {} iteration and {} seconds to complete'.format(descent_result['iters'], descent_result['time']))


if __name__ == '__main__':
    train_regression('data/geyser.csv')

