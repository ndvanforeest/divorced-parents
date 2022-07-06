from itertools import combinations
from collections import defaultdict
import random  #
import numpy as np
from icecream import ic
from pytictoc import TicToc
from priority_queue import priority_dict

import common_functions as common
from common_functions import En

t = TicToc()
thres = 1e-5
random.seed(3)


def fpa_reference(b, d, C):  # fixed point approach
    limit = d.sum() - b.sum()
    P = common.make_parents_dict(C)
    y, n_iters = d.copy(), 0
    while y.sum() > limit + 1e-3:
        y = common.fix_point_f(b, d, y, C, P)
        n_iters += 1
        # print(y.sum() - limit, n_iters)
        if n_iters > 100:
            print("too many iters")
            print(y)
            print(limit, y.sum())
            y_new = common.fix_point_f(b, d, y, C, P)
            print(y_new)
            print(y.sum() - y_new.sum())
            quit()
    x = common.compute_x(b, y, C, P)
    M = common.moulin(d, x, C)
    return M


def fpa(fname):  # fixed point algorithm
    b, d, C, M = common.read_network(fname)
    P = common.make_parents_dict(C)
    y, n_iters, m = d.copy(), 0, -np.inf
    while m < M - thres:
        y = common.fix_point_f(b, d, y, C, P)
        n_iters += 1
        x = common.compute_x(b, y, C, P)
        m = common.moulin(d, x, C)
    return n_iters, 0


def do_cases(fname, shuffle=False, sample=False):
    b, d, C, M = common.read_network(fname)
    P = common.make_parents_dict(C)
    cases = common.make_simple_cases(C)
    init_cases = cases + cases[::-1]  # back and forth
    if shuffle:
        random.shuffle(init_cases)
    x, n_passes, m = defaultdict(float), 0, -np.inf
    while m < M - thres:
        if sample:
            loop_cases = random.choices(init_cases, k=len(init_cases))
        else:
            loop_cases = init_cases
        for case in loop_cases:
            A, B, AB = case
            x[A, AB], x[B, AB] = common.case(A, B, AB, b, d, x, C, P)
        n_passes += 1
        m = common.moulin(d, x, C)
    n_cases = n_passes * len(init_cases)
    return n_passes, n_cases


def wave(fname):  # formerly back and forth
    return do_cases(fname, shuffle=False, sample=False)


def shuffle(fname):  # formerly back and forth
    return do_cases(fname, shuffle=True, sample=False)


def sample(fname):  # formerly called random_cases
    return do_cases(fname, shuffle=False, sample=True)


def convex(fname):
    import cvxpy as cp

    b, d, C, M = common.read_network(fname)
    P = common.make_parents_dict(C)
    delta = np.zeros((len(C), len(P)))
    for i in C.keys():
        for j in C[i]:
            delta[i, j] = 1

    def fairness(d, x):
        y = d - cp.sum(x, axis=1)
        return cp.sum(cp.entr(x)) + cp.sum(x) + cp.sum(cp.entr(y)) + cp.sum(y)

    x = cp.Variable(delta.shape, pos=True)
    objective = cp.Maximize(fairness(d, x))
    constraints = [cp.sum(x, axis=0) == b]  # , y == d - cp.sum(x, axis=1)]

    for index in np.argwhere(delta == 0):
        i, j = index[:]
        constraints += [x[i, j] == 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(reltol=thres, feastol=thres)  # , verbose=True)
    # print('solver status: {}'.format(prob.status))
    # print("solve", prob.solver_stats.solve_time)
    # print("setup", prob.solver_stats.setup_time)
    # print(prob.solver_stats.extra_stats)


# do the largest difference


def prepare(fname):
    b, d, C, M = common.read_network(fname)
    P = common.make_parents_dict(C)
    cases = common.make_simple_cases(C)
    x = defaultdict(float)
    L = common.moulin(d, x, C)
    return b, d, x, C, P, L, M, cases


def moulin_for_virtual_case(b, d, x, y, C, P, case):
    A, B, AB = case
    diff = En(x[A, AB]) + En(x[B, AB]) + En(y[A]) + En(y[B])
    x_1, x_2 = common.case(A, B, AB, b, d, x, C, P)
    y_1 = y[A] - x_1 + x[A, AB]
    y_2 = y[B] - x_2 + x[B, AB]
    diff -= En(x_1) + En(x_2) + En(y_1) + En(y_2)
    return diff


def find_largest_case(b, d, x, C, P, cases):
    diff, best = -np.inf, None
    y = common.compute_y(d, x, C)
    for case in cases:
        m = moulin_for_virtual_case(b, d, x, y, C, P, case)
        if m > diff:
            diff, best = m, case
    return diff, best


def do_largest(fname):
    b, d, x, C, P, L, M, cases = prepare(fname)
    n_real, n_virtual = 0, 0
    while L < M - thres:  # and n_iters < 10:
        n_real += 1
        diff, best = find_largest_case(b, d, x, C, P, cases)
        A, B, AB = best
        x[A, AB], x[B, AB] = common.case(A, B, AB, b, d, x, C, P)
        L += diff
    n_virtual = n_real * len(cases)
    return n_real, n_virtual


# sorted heap algo


def sort_cases(b, d, x, C, P, cases):
    heap = priority_dict()
    y = common.compute_y(d, x, C)
    for case in cases:
        m = moulin_for_virtual_case(b, d, x, y, C, P, case)
        heap[case] = -m
    return heap


def do_sorted(fname):
    b, d, x, C, P, L, M, cases = prepare(fname)
    n_real, n_virtual = 0, 0
    while L < M - thres:  # and n_iters < 10:
        n_real += 1
        heap = sort_cases(b, d, x, C, P, cases)
        while heap:
            diff, case = heap.pop_smallest()
            A, B, AB = case
            x[A, AB], x[B, AB] = common.case(A, B, AB, b, d, x, C, P)
        L = common.moulin(d, x, C)  # we cannot use diff to update L
        n_virtual += len(cases)
    return n_real, n_virtual


# update heap algo


def find_impacted_cases(case, C, P):
    A, B, AB = case
    cases = set()
    for j in C[A]:
        for k in P[j]:
            if k == A:
                continue
            case = (A, k, j) if A < k else (k, A, j)
            cases.add(case)
    for j in C[B]:
        for k in P[j]:
            if k == B:
                continue
            case = (B, k, j) if B < k else (k, B, j)
            cases.add(case)
    for i, k in combinations(P[AB], 2):
        case = (i, k, AB) if i < k else (k, i, AB)
        cases.add(case)
    cases.discard((A, B, AB))
    return cases


def update_heap(b, d, x, C, P, impacted_cases, heap):
    y = common.compute_y(d, x, C)
    for case in impacted_cases:
        m = moulin_for_virtual_case(b, d, x, y, C, P, case)
        heap[case] = -m


def do_update(fname):
    b, d, x, C, P, L, M, cases = prepare(fname)
    n_real, n_virtual = 0, 0
    heap = sort_cases(b, d, x, C, P, cases)
    while L < M - thres:  # and n_iters < 10:
        n_real += 1
        diff, case = heap.pop_smallest()
        L -= diff
        A, B, AB = case
        x[A, AB], x[B, AB] = common.case(A, B, AB, b, d, x, C, P)
        impacted_cases = find_impacted_cases(case, C, P)
        n_virtual += len(impacted_cases)
        update_heap(b, d, x, C, P, impacted_cases, heap)
    return n_real, n_virtual


def time_it(func, fname):
    # print(func.__name__)
    print(func.__name__)
    t.tic()
    func(fname)
    t.toc()


def main():
    fname = "/home/nicky/tmp/networks/linear_10_0.pkl"
    fname = "/home/nicky/tmp/networks/star_50_0.pkl"

    # time_it(fpa, fname)
    # time_it(wave, fname)
    # time_it(shuffle, fname)
    # time_it(sample, fname)
    # time_it(do_largest, fname)
    # time_it(do_sorted, fname)
    # time_it(do_update, fname)
    time_it(convex, fname)


if __name__ == '__main__':
    main()
