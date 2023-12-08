import pandas as pd
import barley_break
import numpy as np
from multiprocessing import Pool
import random
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


def gen_barley_break(val):
    h, w = val
    return np.matrix(np.matrix(list(range(1, h * w)) + [0]).reshape(h, w))


def get_data(first):
    barley_break.shuffle(first)
    r, steps, b = barley_break.get_tree(first)
    return len(b.directions) - 1, steps


def selection_branch_func_rand(l):
    return l.pop(random.randrange(0, len(l)))


def selection_branch_func_last(l):
    return l.pop(len(l) - 1)


if __name__ == '__main__':
    pool = Pool(24)
    # barley_breaks = list(pool.map(gen_barley_break, [(4, 4) for i in range(1000)]))
    # with open("barley_breaks.pkl", 'wb') as file:
    #     pickle.dump(barley_breaks, file)
    # print("generation done")
    with open("barley_breaks.pkl", 'rb') as file:
        barley_breaks = pickle.load(file)
    print("load done")

    # data1 = list(pool.map(get_data, barley_breaks))
    # df1 = pd.DataFrame(data1)
    # df1.to_csv("pop0-1000.csv")
    # print("pop0 done")

    # barley_break.selection_branch_func = selection_branch_func_rand
    # data2 = list(pool.map(get_data, barley_breaks))
    # df2 = pd.DataFrame(data2)
    # df2.to_csv("popRandom-1000.csv")
    # print("popRandom done")

    barley_break.selection_branch_func = selection_branch_func_last
    data3 = list(pool.map(get_data, barley_breaks))
    df3 = pd.DataFrame(data3)
    df3.to_csv("popLast-1000.csv")
    print("popLast done")
