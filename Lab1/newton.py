import numpy as np

from scipy.linalg import cho_factor, cho_solve

def stop_criterion(grad, w, w0, eps):
    grad_w = grad(w)
    grad_w0 = grad(w0)
    norm = np.linalg.norm(grad_w) ** 2 < eps * np.linalg.norm(grad_w0) ** 2
    return norm


def stop_criterion_dx(grad, w, w0, eps):
    norm = np.linalg.norm(w - w0)
    return norm < eps


def get_d(hess, grad: np.ndarray):
    # return np.linalg.solve(hess, -grad)
    # df2 = f_two_prime(x_0)
    # df = f_prime(x_0)
    df2_i = cho_solve(cho_factor(hess), np.eye(len(hess)))
    d = np.matmul(grad, df2_i)
    return -d


def hessian(w, grad):
    epsilon = 1e-12
    N = w.size
    h = np.zeros((N,N))
    df_0 = grad(w)[1]
    for i in range(N):
        xx0 = 1.*w[i]
        w[i] = xx0 + epsilon
        df_1 = grad(w)[1]
        h[i,:] = (df_1 - df_0) / epsilon
        w[i] = xx0
    return h


def newton(func, grad, grad2, w0, eps=1e-9, searcher=None, stop_criterion=stop_criterion_dx):
    w0 = np.array(w0.copy(), np.float64)
    w = np.array(w0.copy(), np.float64)
    iterations = 0
    path = [w0]
    first = True
    while first or not stop_criterion(grad, w, w0, eps):
        first = False
        gradient_value = get_d(grad2(w), np.array(grad(w)))
        alpha = 1.0
        delta_w = alpha * gradient_value
        w0 = w.copy()
        w += delta_w
        path.append(w.copy())
        iterations += 1
        if alpha < 1e-20:
            break
    return w, iterations, path
# [-1.5        -0.83966064 -1.25       -1.25      ]
