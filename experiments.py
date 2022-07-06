from collections import defaultdict
import time
import numpy as np
from tabulate import tabulate

from algorithms import *


def time_it(algo, fname):
    start = time.time()
    algo(fname)
    stop = time.time()
    return stop - start


# table = [["spam",42],["eggs",451],["bacon",0]]
# >>> headers = ["item", "qty"]


def do_experiments(algos, networks, sizes, samples, output):
    # This makes a table with time info for all algorithms for a bunch of networks
    directory = "/home/nicky/tmp/networks/"
    results = []
    for algo in algos:
        for network in networks:
            for size in sizes:
                for sample in samples:
                    print(algo.__name__, network, size, sample)
                    res = [algo.__name__, network, size, sample]
                    fname = directory + f"{network}_{size}_{sample}.pkl"
                    res += [time_it(algo, fname)]
                    results.append(res)
    with open(output, "w") as fp:
        for res in results:
            fp.write(", ".join(str(r) for r in res))
            fp.write("\n")
    quit()


def process_table(algos, networks, sizes, input, factor=1):
    results = defaultdict(list)
    with open(input, "r") as fp:
        for line in fp:
            algo, network, size, sample, T = line.split(",")
            T = float(T)
            key = f"{algo.strip()}-{network.strip()}-{size.strip()}"
            results[key].append(T)
    means = {k: sum(v) / len(v) for k, v in results.items()}
    headers = ["network"] + [algo.__name__.replace("do_", "") for algo in algos]
    table = []
    for size in sizes:
        for network in networks:
            res = [f"{network}-{size}"]
            for algo in algos:
                key = f"{algo.__name__}-{network}-{size}"
                res.append(factor * means[key])
            table.append(res)
    print(tabulate(table, headers=headers, tablefmt="latex", floatfmt=".0f"))
    fname = input.replace("txt", "tex")
    with open(fname, "w") as fp:
        fp.write(
            tabulate(table, headers=headers, tablefmt="latex", floatfmt=".1f")
        )


def make_table_small_networks():
    algos = [
        fpa,
        wave,
        shuffle,
        sample,
        do_sorted,
        do_update,
        do_largest,
        convex,
    ]
    networks = ["star", "linear", "random"]
    sizes = [10, 20, 30, 50]
    samples = list(range(10))
    fname = "table_small_networks.txt"
    # do_experiments(algos, networks, sizes, samples, fname)
    process_table(algos, networks, sizes, fname, factor=1000)


make_table_small_networks()


def make_table_large_networks():
    algos = [fpa, wave, shuffle, sample]
    networks = ["star", "linear", "random"]
    sizes = [100, 1000]  # , 10000]
    samples = list(range(10))
    fname = "table_large_networks.txt"
    # do_experiments(algos, networks, sizes, samples, fname)
    process_table(algos, networks, sizes, fname, factor=1000)


make_table_large_networks()


def make_table_very_large_networks():
    algos = [fpa]
    networks = ["star", "linear", "random"]
    sizes = [10000, 100_000, 1_000_000]
    samples = list(range(10))
    fname = "table_very_large_networks.txt"
    # do_experiments(algos, networks, sizes, samples, fname)
    process_table(algos, networks, sizes, fname, factor=1)


make_table_very_large_networks()
