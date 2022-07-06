from itertools import combinations
import numpy as np


def init_x(delta):
    return 0.01 * delta
    # return np.zeros(delta.shape)  # very unfavorite starting condition
    ouders, kinderen = delta.shape
    x = np.zeros(delta.shape)
    for j in range(kinderen):
        f = 1 / delta[:, j].sum()
        x[:, j] = delta[:, j] * f
    return x


def moulin(d, x):
    y = d - x.sum(axis=1)

    def En(z):
        return z * np.ma.log(z) - z
        return z * np.log(z + 1e-12) - z  # +this is faster than masking

    return -En(x).sum() - En(y).sum()


def fix_point_f(d, b, delta, y):
    # Dit is f(y) in het bewijs dat er een unieke oplossing is.
    ouders = range(len(d))
    kinderen = range(len(b))
    y_new = np.zeros_like(y)
    for i in ouders:
        res = 1
        for j in kinderen:
            res += delta[i, j] * b[j] / (y @ delta[:, j])
            # res += delta[i, j] * b[j] / sum(y[k] * delta[k, j] for k in ouders)
        y_new[i] = res
    return d / y_new


def compute_x(b, delta, y):
    ouders, kinderen = delta.shape
    x = np.zeros_like(delta, dtype=float)
    for i in range(ouders):
        for j in range(kinderen):
            x[i, j] = delta[i, j] * y[i] * b[j] / (y @ delta[:, j]).sum()
    return x


def solve_global(d, b, delta, x=None):
    if x is None:
        x = np.zeros(delta.shape)
    M, M_old = moulin(d, x), -np.inf
    y = d - x.sum(axis=1)
    while M - M_old > 1e-16:
        M_old = M
        y = fix_point_f(d, b, delta, y)
        x = compute_x(b, delta, y)
        M = moulin(d, x)
    return M


def make_simple_cases(delta):
    _, n_children = delta.shape
    cases = []
    for child in range(n_children):
        parents = np.where(delta[:, child] == 1)
        for pair in combinations(*parents, 2):
            cases.append([*pair, child])
    return cases


def make_simple_cases_old(delta):
    # very inefficient
    cases = []
    n_parents, n_children = delta.shape
    for i in range(n_parents):
        for j in range(n_children):
            for k in range(i + 1, n_parents):
                if delta[i, j] * delta[k, j]:
                    cases.append([i, k, j])
    return cases


def case(A, B, AB, d, b, x):
    cap_A = d[A] - x[A, :].sum() + x[A, AB]
    cap_B = d[B] - x[B, :].sum() + x[B, AB]
    need_AB = b[AB] - x[:, AB].sum() + x[A, AB] + x[B, AB]
    f = min(need_AB / (cap_A + cap_B), 1)
    return f * cap_A, f * cap_B
