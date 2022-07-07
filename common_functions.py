from collections import defaultdict
from itertools import combinations
import pickle
import numpy as np


def En(z):
    return z * np.log(z + 1e-12) - z
    return z * np.ma.log(z) - z


def moulin(d, x, C):
    res = sum(En(v) for v in x.values())
    for i in C.keys():
        res += En(d[i] - sum(x[i, j] for j in C[i]))
    return -res


def fix_point_f(b, d, y, C, P):
    # Dit is f(y) in het bewijs dat er een unieke oplossing is.
    y_new = np.zeros_like(y)
    B = np.zeros_like(b)
    for j in P.keys():
        B[j] = b[j] / sum(y[k] for k in P[j])
    for i in C.keys():
        y_new[i] = d[i] / (1 + sum(B[j] for j in C[i]))
    return y_new


def compute_x(b, y, C, P):
    x = {}
    B = np.zeros_like(b)
    for j in P.keys():
        B[j] = b[j] / sum(y[k] for k in P[j])
    for i in C.keys():
        for j in C[i]:
            x[i, j] = y[i] * B[j]
    return x


def compute_y(d, x, C):
    y = np.zeros(len(C))
    for i in C.keys():
        y[i] = d[i] - sum(x[i, j] for j in C[i])
    return y


def make_parents_dict(C):
    P = defaultdict(set)
    for i in C.keys():
        for j in C[i]:
            P[j].add(i)
    return P


def make_simple_cases(C):
    P = make_parents_dict(C)
    cases = []
    for child in P.keys():
        for pair in combinations(P[child], 2):
            cases.append((*sorted(pair), child))
    return cases


def case(A, B, AB, b, d, x, C, P):
    cap_A = d[A] + x[A, AB] - sum(x[A, j] for j in C[A])
    cap_B = d[B] + x[B, AB] - sum(x[B, j] for j in C[B])
    need_AB = b[AB] + x[A, AB] + x[B, AB] - sum(x[i, AB] for i in P[AB])
    f = min(need_AB / (cap_A + cap_B), 1)
    # f = need_AB / (cap_A + cap_B)
    return f * cap_A, f * cap_B


def compare_old_to_new(b, d, x, C, P):
    x_old = x.copy()
    cases = make_simple_cases(C)
    for c in cases:
        A, B, AB = c
        x[A, AB], x[B, AB] = case(A, B, AB, b, d, x, C, P)
    return sum(abs(x_old[i, j] - x[i, j]) for i in C for j in C[i])


class Debug:
    def __init__(self):
        self.ids = set()

    def debug(self, *ids):
        self.ids.update(set(ids))

    def undebug(self, *ids):
        self.ids.difference_update(set(ids))

    def output(self, dbg_id, *msg):
        if dbg_id in self.ids:
            print(f"dbg {dbg_id}: {msg}")




def read_network(fname):
    with open(fname, 'rb') as fp:
        b = pickle.load(fp)
        d = pickle.load(fp)
        C = pickle.load(fp)
        M = pickle.load(fp)
    return b, d, C, M
