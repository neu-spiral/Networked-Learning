import matplotlib.pyplot as plt
import logging, argparse
import pickle
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


coefficients = {'learner': [5, 10, 15, 20], 'source': [10, 20, 30, 40]}

algorithm = ['FW', 'MaxSum', 'MaxAlpha', 'PGA']
algorithm_map = {0:0, 1:1, 2:2, 3:5}

# colors = ['r', 'sandybrown', 'gold', 'darkseagreen', 'c', 'dodgerblue', 'm']
line_styles = ['s-', '*:', 'd--', '^-', 'v-', '.:']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def plotSensitivity(x1, x2, change, graph, min_bandwidth):
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(8, 3.5)
    for i in range(len(algorithm)):
        alg = algorithm[i]
        ax[0].plot(coefficients[change], x1[alg], line_styles[i], markersize=10, label=alg, linewidth=3)
        ax[1].plot(coefficients[change], x2[alg], line_styles[i], markersize=10, label=alg, linewidth=3)

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)

    ax[0].set_ylabel('Aggregate Utility', fontsize=15)
    ax[1].set_ylabel('Avg. Norm of Est. Error', fontsize=15)

    if change == 'source':
        xlabel = '|S|'
    elif change == 'learner':
        xlabel = '|L|'
    ax[0].set_xlabel(xlabel, fontsize=15)
    ax[1].set_xlabel(xlabel, fontsize=15)

    lgd = fig.legend(labels = algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(algorithm), fontsize=15)
    plt.tight_layout()
    plt.show()
    fig.savefig('Figure/sens_{}/{}_{}.pdf'.format(change, graph, min_bandwidth),  bbox_extra_artists=(lgd,), bbox_inches='tight')
    logging.info('saved in Figure/sens_{}/{}_{}.pdf'.format(change, graph, min_bandwidth))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot sensitivity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'ToyExample'])
    parser.add_argument('--learners', default=3, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--min_bandwidth', default=20, type=float, help='Minimum bandwidth of each edge')

    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--change', type=str, help='changed variable', choices=['learner', 'source'])


    args = parser.parse_args()
    np.random.seed(args.random_seed + 2021)

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj1 = {}
    obj2 = {}
    if args.change == 'learner':
        for j in range(len(algorithm)):
            obj1[algorithm[j]] = []
            for i in range(len(coefficients[args.change])):
                fname = "Result_{}/Result_{}_{}learners_20sources_10types".format(int(args.min_bandwidth),
                        args.graph_type, coefficients[args.change][i])
                result = readresult(fname)
                obj1[algorithm[j]].append(result[algorithm_map[j]][1])
            obj2[algorithm[j]] = []
            for i in range(len(coefficients[args.change])):
                fname = "Result_{}/beta_{}_{}learners_20sources_10types".format(int(args.min_bandwidth),
                        args.graph_type, coefficients[args.change][i])
                result = readresult(fname)
                obj2[algorithm[j]].append(result[algorithm_map[j]])
    elif args.change == 'source':
        for j in range(len(algorithm)):
            obj1[algorithm[j]] = []
            for i in range(len(coefficients[args.change])):
                fname = "Result_{}/Result_{}_10learners_{}sources_10types".format(int(args.min_bandwidth),
                        args.graph_type, coefficients[args.change][i])
                result = readresult(fname)
                obj1[algorithm[j]].append(result[algorithm_map[j]][1])
            obj2[algorithm[j]] = []
            for i in range(len(coefficients[args.change])):
                fname = "Result_{}/beta_{}_10learners_{}sources_10types".format(int(args.min_bandwidth),
                        args.graph_type, coefficients[args.change][i])
                result = readresult(fname)
                obj2[algorithm[j]].append(result[algorithm_map[j]])
    plotSensitivity(obj1, obj2, args.change, args.graph_type, int(args.min_bandwidth))