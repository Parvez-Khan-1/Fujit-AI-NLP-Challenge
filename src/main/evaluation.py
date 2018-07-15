import os
import json
import numpy as np
from scipy.stats import rankdata


def print_help():
    print("Usage: python evaluation3.py <json>")
    print("Calculate MRR(Mean Reciprocal Rank) about given json lines file.")
    print("The json lines file need to follow the format specified by Fujitsu.")


def calc_mrr(rank):
    rank = list(map(lambda x: 1./x, rank))
    return np.mean(rank)


def main(fname):
    data = []
    with open(fname) as f:
        for line in f.readlines():
            data += [json.loads(line)]
    rank_i = []
    for elem in data:
        cand = elem['candidates']
        results = elem['results']
        cand_ranks = (len(results) - rankdata(results, method='average'))[cand] + 1
        rank_i.append(min(cand_ranks))
    mrr = calc_mrr(rank_i)
    print("MRR: {}".format(mrr))
    with open(".././MRR.txt", "w") as f:
        f.write(str(mrr))


if __name__ == "__main__":
    os.chdir("../../data")
    fname = os.path.abspath(os.curdir) + "/SelQA-ass-result.example.json"
    main(fname)
