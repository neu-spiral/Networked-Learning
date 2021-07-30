import matplotlib.pylab as plt
import pickle
import copy
import numpy as np

topology = ['erdos_renyi', 'balanced_tree', 'hypercube', 'star', 'geant', 'abilene', 'dtelekom', 'grid_2d', 'small_world']
topology_short = ['ER', 'BT', 'HC', 'star', 'geant', 'abilene', 'dtelekom', 'grid', 'SW']
algorithm = ['FW', 'MaxSum', 'MaxAlpha', 'PGA']
algorithm_map = {0:0, 1:1, 2:2, 3:5}
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']

def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result

def barplot(x):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3)
    N = len(topology)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Normalized utility', fontsize=15)
    ax.set_xticks(ind + width*1.5)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    plt.ylim(0.8, 1.05)
    lgd = fig.legend(labels = algorithm, loc='upper center', ncol=len(algorithm), fontsize=15)
    plt.show()
    fig.savefig('Figure/topology_rate.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')

obj = {}
for alg in algorithm:
    obj[alg] = {}
    for top in topology_short:
        obj[alg][top] = 0

for i in range(len(topology)):
    fname = 'Result_smallrate/Result_' + topology[i]
    result = readresult(fname)
    for j in range(len(algorithm)):
        obj[algorithm[j]][topology_short[i]] = result[algorithm_map[j]][1]/ result[0][1]


barplot(obj)

