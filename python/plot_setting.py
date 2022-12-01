import matplotlib.pylab as plt
import pickle
import copy
import numpy as np
import argparse

setting = [1, 2, 3]
algorithm = ['FW', 'MaxSum', 'MaxAlpha', 'PGA']
algorithm_map = {0: 0, 1: 1, 2: 2, 3: 5}
hatches = ['/', '\\\\', '|', '+', '--', '', '////', 'x', 'o', '.', '\\']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x1, x2, graph_type):
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(10, 3)
    numb_bars = len(algorithm) + 1
    N = len(setting)
    ind = np.arange(0, numb_bars * N, numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax1 = x1[algorithm[i]].values()
        ax[0].bar(ind + i * width, y_ax1, width=width, hatch=hatches[i], label=algorithm[i])
        y_ax2 = x2[algorithm[i]].values()
        ax[1].bar(ind + i * width, y_ax2, width=width, hatch=hatches[i], label=algorithm[i])

    ax[0].tick_params(labelsize=10)
    ax[1].tick_params(labelsize=10)

    ax[0].set_ylabel('Aggregate Utility', fontsize=15)
    ax[1].set_ylabel('Avg. Norm of Est. Error', fontsize=15)

    ax[0].set_xticks(ind + width * 1.5)
    ax[0].set_xticklabels(x1[algorithm[i]].keys(), fontsize=13)

    ax[1].set_xticks(ind + width * 1.5)
    ax[1].set_xticklabels(x2[algorithm[i]].keys(), fontsize=13)

    ax[0].set_ylim(150, 300)
    ax[1].set_ylim(0, 1)
    lgd = fig.legend(labels=algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=len(algorithm), fontsize=15)
    plt.show()
    fig.savefig('Figure/setting_20/{}.pdf'.format(graph_type), bbox_extra_artists=(lgd,), bbox_inches='tight')


parser = argparse.ArgumentParser(description='Plot sensitivity',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                    choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                             'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                             'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                             'servicenetwork', 'ToyExample'])
args = parser.parse_args()

obj1 = {}
obj2 = {}
for alg in algorithm:
    obj1[alg] = {}
    obj2[alg] = {}

fname1 = ['Result_20/Result' + '_old_' + args.graph_type + "_5learners_10sources_5types",
          'Result_20/Result' + '_' + args.graph_type + "_5learners_10sources_5types",
          'Result_20/Result' + '_' + args.graph_type + "_15learners_20sources_10types"]
fname2 = ['Result_20/beta' + '_old_' + args.graph_type + "_5learners_10sources_5types",
          'Result_20/beta' + '_' + args.graph_type + "_5learners_10sources_5types",
          'Result_20/beta' + '_' + args.graph_type + "_15learners_20sources_10types"]

for i in range(len(setting)):
    result1 = readresult(fname1[i])
    result2 = readresult(fname2[i])

    for j in range(len(algorithm)):
        obj1[algorithm[j]][setting[i]] = result1[algorithm_map[j]][1]
        obj2[algorithm[j]][setting[i]] = result2[algorithm_map[j]]

barplot(obj1, obj2, args.graph_type)
