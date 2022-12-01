import matplotlib.pylab as plt
import pickle
import copy
import numpy as np
import argparse

topology_small = ['geant', 'abilene', 'dtelekom']
topology = ['erdos_renyi', 'balanced_tree', 'hypercube', 'star', 'grid_2d', 'small_world']
topology_short = ['ER', 'BT', 'HC', 'star', 'grid', 'SW']
topology_small_short = ['geant', 'abilene', 'dtelekom']
algorithm = ['FW', 'MaxSum', 'MaxAlpha', 'PGA']
algorithm_map = {0:0, 1:1, 2:2, 3:5}
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x, type):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3)
    N = len(topology) + len(topology_small)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
    ax.tick_params(labelsize=12)
    if type == 'Result':
        ylabel = 'Aggregate Utility'
    elif type == 'beta':
        ylabel = 'Avg. Norm of Est. Error'
    ax.set_ylabel(ylabel, fontsize=15)

    ax.set_xticks(ind + width*1.5)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    if type == 'Result':
        plt.ylim(100, 400)

    elif type == 'beta':
        plt.ylim(0, 1)

    lgd = fig.legend(labels = algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=len(algorithm), fontsize=15)
    plt.show()
    if type == 'Result':
        fig.savefig('Figure/topology20_utility_large.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')
    elif type == 'beta':
        fig.savefig('Figure/topology20_beta.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')


parser = argparse.ArgumentParser(description='Plot sensitivity',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--type', type=str, help='Plot est. error or utility', choices=['beta', 'Result'])
args = parser.parse_args()

obj = {}
for alg in algorithm:
    obj[alg] = {}

for i in range(len(topology)):
    fname = 'Result_20/' + args.type + '_old_' + topology[i] + "_5learners_10sources_5types"
    result = readresult(fname)
    if args.type == 'Result':
        for j in range(len(algorithm)):
            obj[algorithm[j]][topology_short[i]] = result[algorithm_map[j]][1]
    elif args.type == 'beta':
        for j in range(len(algorithm)):
            obj[algorithm[j]][topology_short[i]] = result[algorithm_map[j]]

for i in range(len(topology_small)):
    fname = 'Result_20/' + args.type + '_old_' + topology_small[i] + "_3learners_3sources_3types"
    result = readresult(fname)
    if args.type == 'Result':
        for j in range(len(algorithm)):
            obj[algorithm[j]][topology_small_short[i]] = result[algorithm_map[j]][1]
    elif args.type == 'beta':
        for j in range(len(algorithm)):
            obj[algorithm[j]][topology_small_short[i]] = result[algorithm_map[j]]

barplot(obj, args.type)

