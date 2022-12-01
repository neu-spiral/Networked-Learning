import matplotlib.pylab as plt
import pickle
import copy
import numpy as np

algorithm = ['FW', 'MaxSum', 'MaxAlpha', 'PGA']
algorithm_map = {0:0, 1:1, 2:2, 3:5}
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x1, x2):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(6, 3)
    N = 1
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x1[algorithm[i]]
        ax[0].bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
        ax[0].tick_params(labelsize=12)
        ax[0].set_ylabel('Aggregate Utility', fontsize=15)
        ax[0].set_ylim(6, 12)

    for i in range(len(algorithm)):
        y_ax = x2[algorithm[i]]
        ax[1].bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
        ax[1].tick_params(labelsize=12)
        ax[1].set_ylabel('Avg. Norm of Est. Error', fontsize=15)
        ax[1].set_ylim(0, 0.3)

    lgd = fig.legend(labels = algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(algorithm), fontsize=15)
    plt.tight_layout()
    plt.show()
    fig.savefig('Figure/ToyExample2.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')

obj1 = {}
obj2 = {}

fname1 = 'Result_5/Result_ToyExample2'
fname2 = 'Result_5/beta_ToyExample2'
result1 = readresult(fname1)
result2 = readresult(fname2)

for j in range(len(algorithm)):
    obj1[algorithm[j]] = result1[algorithm_map[j]][1]
    obj2[algorithm[j]] = result2[algorithm_map[j]]

barplot(obj1, obj2)

