from collections import defaultdict
import random
import pickle
import numpy as np


from algorithms import fpa_reference

def check_feasibility(b, d, x, C, P):
    for i in C.keys():
        if not d[i] - sum(x[i, j] for j in C[i]) > 1e-7:
            raise Exception(f"Parent {i} has a shortage")
    for j in P.keys():
        if b[j] - sum(x[i, j] for i in P[j]) > 1e-7:
            print(j, b[j] - sum(x[i, j] for i in P[j]))
            raise Exception(f"Child {j} has a shortage")


def make_star(n_parents, sample):
    # We set the seeds to ensure that random variables are the same
    # when regenerating the networks
    random.seed(sample)
    np.random.seed(sample)
    n_children = n_parents - 1
    C = defaultdict(set)
    for i in range(1, n_parents):
        C[i].add(i - 1)
        C[0].add(i - 1)
    b = np.ones(n_children)
    d = np.random.uniform(0.2, 0.8, n_parents)
    d[0] = 10 * (b.sum() - d[1:].sum())
    write_network(b, d, C, "star", n_parents, sample)


def make_linear(n_parents, sample):
    random.seed(sample)
    np.random.seed(sample)
    n_children = n_parents - 1
    C = defaultdict(set)
    for i in range(n_parents - 1):
        C[i].add(i)
    for i in range(1, n_parents):
        C[i].add(i - 1)
    b = np.ones(n_children)
    d = np.random.uniform(1.1, 2, n_parents)
    d[0] *= 10
    write_network(b, d, C, "linear", n_parents, sample)


def make_random(n_parents, sample):
    random.seed(sample)
    np.random.seed(sample)
    n_children = n_parents - 1
    C = defaultdict(set)
    for i in range(n_parents - 1):
        C[i].add(i)
    for i in range(1, n_parents):
        C[i].add(i - 1)
    for i in range(n_parents):
        for j in random.sample(range(n_children), min(3, n_children)):
            C[i].add(j)
    b = np.ones(n_children)
    d = np.random.uniform(1.1, 5, size=len(C))
    write_network(b, d, C, "random", n_parents, sample)



def write_network(b, d, C, network, n_parents, sample):
    # print(n_parents, sample)
    M = fpa_reference(b, d, C)
    fname = f'/home/nicky/tmp/networks/{network}_{n_parents}_{sample}.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump(b, fp)
        pickle.dump(d, fp)
        pickle.dump(C, fp)
        pickle.dump(M, fp)



def generate_small_networks():
    min_parents, max_parents = 3, 101
    n_samples = 10
    for n_parents in range(min_parents, max_parents):
        for k in range(n_samples):
            make_star(n_parents, k)
            make_linear(n_parents, k)
            make_random(n_parents, k)


def generate_large_networks():
    R = [1e3, 1e4, 1e5, 1e6]
    n_samples = 10
    for n_parents in R:
        for k in range(n_samples):
            make_star(int(n_parents), k)
            make_linear(int(n_parents), k)
            make_random(int(n_parents), k)


def main():
    # generate_small_networks()
    generate_large_networks()


if __name__ == '__main__':
    main()
