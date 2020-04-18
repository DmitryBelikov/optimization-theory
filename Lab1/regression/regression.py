import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from Lab1.optimizers import GradientDescent
from Lab1.methods import GoldenRatioSearcher


def regression_function(X, y, w, reg):
    samples = X.shape[0]
    exp_arg = -y * X.dot(w.T)
    return np.ones((1, samples)).dot(np.logaddexp(0, exp_arg)).item() + reg * w[0] * w[0]


def regression_gradient(X, y, w, reg):
    exp_division = scipy.special.expit(-y * X.dot(w.T))
    return X.T.dot(-y * exp_division).T + reg * w


def minimization_function(X, y, reg):
    return lambda w: regression_function(X, y, w, reg)


def minimization_gradient(X, y, reg):
    return lambda w: regression_gradient(X, y, w, reg)


def logistic_regression(X, y):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    reg = 0
    min_func = minimization_function(X, y, reg)
    min_grad = minimization_gradient(X, y, reg)
    optimizer = GradientDescent(min_func, min_grad, searcher=GoldenRatioSearcher)
    w0 = np.array([0] * X.shape[1], dtype=np.float)
    weights = optimizer.minimize(w0)
    return weights


def train_regression(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df.drop(['binaryClass'], axis=1).values
    y = df.binaryClass.map(lambda c: -1 if c == 'N' else 1).values
    weights = logistic_regression(X, y)
    print(weights)
    xs = [0, 25]
    ys = [-weights[0] / weights[2], (-weights[0] - weights[1] * 25) / weights[2]]
    plt.plot(xs, ys, color='green')
    plt.scatter(df.col_1, df.col_2, color=df.binaryClass.map(lambda c: 'red' if c == 'P' else 'blue'))
    plt.show()


if __name__ == '__main__':
    train_regression('data/geyser.csv')

