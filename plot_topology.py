import matplotlib.pylab as plt
import pickle
import copy
import numpy as np

topology = ['erdos_renyi', 'balanced_tree', 'hypercube', 'star', 'geant', 'abilene', 'dtelekom']
topology_short = ['ER', 'BT', 'HC', 'star', 'geant', 'abilene', 'dtelekom']
algorithm = ['FW', 'Max1', 'U1', 'Max2', 'U2', 'PA']
hatches = ['/', '\\\\', '+', '--', '', '////', '\\']

def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result

def barplot(x):
    fig, ax = plt.subplots()
    N = len(topology)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Utility', fontsize=10)
    ax.set_xticks(ind + width*2.5)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=10)
    lgd = fig.legend(labels = algorithm, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(algorithm), fontsize=10)
    plt.show()
    fig.savefig('Figure/topology_rate.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')

obj = {}
for alg in algorithm:
    obj[alg] = {}
    for top in topology_short:
        obj[alg][top] = 0

for i in range(len(topology)):
    fname = 'Result_rate/Result_' + topology[i]
    result = readresult(fname)
    for j in range(len(algorithm)):
        obj[algorithm[j]][topology_short[i]] = result[j][1]

barplot(obj)

