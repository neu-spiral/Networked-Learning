import matplotlib.pylab as plt
import pickle
from ProbGenerate import Problem


algorithm = ['FW', 'MaxSum', 'MaxAlpha', 'PGA']
algorithm_map = {0:0, 1:1, 2:2, 3:5}


def barplot2(plotname, Ys, P):
    fig, ax = plt.subplots(nrows=len(Ys), ncols=1)
    fig.set_size_inches(23, 8)
    for j in range(len(Ys)):
        title = algorithm[j]
        ax[j].set_title(title, fontsize=20)
        Y = Ys[j]
        x_ax = []
        y_ax = []
        for l in P.learners:
            for i in P.catalog:
                x_ax.append(str((l, i)))
                y_ax.append(Y[l][i])
        for e in P.G.edges():
            for i in P.catalog:
                for t in set(P.types.values()):
                    x_ax.append(str((e, i, t)))
                    y_ax.append(Y[e][i][t])
        ax[j].bar(x_ax, y_ax)
        # ax[j].tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    fig.savefig(plotname)


fname = 'Problem_5/Problem_ToyExample2'
with open(fname, 'rb') as f:
    P = pickle.load(f)

fname = 'Result_5/Result_ToyExample2'
plotname = 'Figure/ToyExample2_distribution.pdf'
with open(fname, 'rb') as f:
    result = pickle.load(f)

Ys = []
for j in range(len(algorithm)):
    Ys.append(result[algorithm_map[j]][0])

barplot2(plotname, Ys, P)
