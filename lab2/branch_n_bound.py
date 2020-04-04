from scipy.optimize import linprog

from lab2.simplex_method import simplex_method_ub, SimplexMethodResults
import numpy as np


def add_constraint(old_a, old_b, idx, sign, b):
    new_row = np.zeros(old_a.shape[1])
    new_row[idx] = 1
    new_row *= sign
    new_a = np.vstack((old_a, new_row))
    new_constraint = b * sign
    new_b = np.hstack((old_b, new_constraint))
    return new_a, new_b


def get_children(cur_c, cur_a, cur_b, next_step):
    children = []
    for i in range(cur_a.shape[1]):
        if np.ceil(next_step.x[i]) != np.floor(next_step.x[i]):
            upper_a, upper_b = add_constraint(cur_a, cur_b, i, 1, np.floor(next_step.x[i]))
            children.append((cur_c, upper_a, upper_b))
            lower_a, lower_b = add_constraint(cur_a, cur_b, i, -1, np.ceil(next_step.x[i]))
            children.append((cur_c, lower_a, lower_b))
            break
    return children


def branch_n_bound(c, a, b, f):
    order = [(np.array(c, np.float), np.array(a, np.float), np.array(b, np.float))]
    ans = SimplexMethodResults(False, 0, [])
    pos = 0
    while pos < len(order):
        cur_c, cur_a, cur_b = order[pos]
        pos += 1
        next_step = f(cur_c, cur_a, cur_b)
        py_step = linprog(c=cur_c, A_ub=cur_a, b_ub=cur_b, method="simplex")
        for i in range(len(next_step.x)):
            next_step.x[i] = round(next_step.x[i], 7)
        next_step.fun = round(next_step.fun, 7)
        if not next_step.success:
            continue
        if not ans.success or next_step.fun <= ans.fun:
            children = get_children(cur_c, cur_a, cur_b, next_step)
            if len(children) == 0:
                if not ans.success or ans.fun >= next_step.fun:
                    ans = next_step
            else:
                for i in children:
                    order.append(i)
    return ans
