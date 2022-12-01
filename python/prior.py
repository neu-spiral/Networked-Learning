import torchvision
import torch
from torchvision import datasets, transforms
import logging, argparse
import topologies
from networkx import DiGraph
import pickle
import numpy as np
import statistics as stat


class Problem:
    def __init__(self, sourceRates, learners, sources, bandwidth, G, paths, prior, T, sourceParameters):
        self.sourceRates = sourceRates
        self.learners = learners
        self.sources = sources
        self.bandwidth = bandwidth
        self.G = G
        self.paths = paths
        self.prior = prior
        self.T = T
        self.sourceParameters = sourceParameters

def main():


    parser = argparse.ArgumentParser(description='Simulate Toy Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--T', default=1, type=float, help="Duration of experiment")

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)

    logging.basicConfig(level=args.debug_level)
    np.random.seed(args.random_seed + 2021)

    data_test = datasets.MNIST(root="./data/",
                               transform=transforms.ToTensor(),
                               train=False,
                               download=True)

    images, labels = zip(*data_test)

    noice = 0.5

    dimension = len(images[0].numpy().reshape(-1, 1))
    images_1, labels_1 = [], []
    images_2, labels_2 = [], []

    for i in range(data_test):
        if labels[i] in [0, 1, 2, 3, 4]:
            images_1.append(images[i].numpy().reshape(-1, 1))
            labels_1.append(labels[i] + np.random.normal(0, noice, size=(100,)))
        elif labels[i] in [5, 6, 7, 8, 9]:
            images_2.append(images[i].numpy().reshape(-1, 1))
            labels_2.append(labels[i] + np.random.normal(0, noice, size=(100,)))

    Cov_0 = np.diag(np.ones(dimension))

    temp = np.zeros((dimension, dimension))
    for i in range(len(images_1)):
        temp += np.dot(images_1[i], images_1[i].transpose())
    Cov_1 = np.dot(Cov_0, temp)
    temp = temp + noice * np.linalg.inv(Cov_0)
    temp = np.linalg.inv(temp)
    Cov_1 = np.dot(Cov_1, temp)

    # Cov_1 = Cov_1 + Cov_0
    # inv_1 = np.linalg.det(np.linalg.inv(Cov_1))

    temp = np.zeros((dimension, dimension))
    for i in range(len(images_2)):
        temp += np.dot(images_2[i], images_2[i].transpose())
    Cov_2 = np.dot(Cov_0, temp)
    temp = temp + noice * np.linalg.inv(Cov_0)
    temp = np.linalg.inv(temp)
    Cov_2 = np.dot(Cov_2, temp)

    # Cov_2 = Cov_2 + Cov_0
    # inv_2 = np.linalg.det(np.linalg.inv(Cov_2))


    logging.info('Generating MNIST')
    temp_graph = topologies.MNIST()
    logging.debug('nodes: ' + str(temp_graph.nodes()))  # list
    logging.debug('edges: ' + str(temp_graph.edges()))  # list of node pair
    G = DiGraph()  # generate a DiGraph

    number_map = dict(zip(temp_graph.nodes(), range(len(temp_graph.nodes()))))
    G.add_nodes_from(number_map.values())  # add node from temp_graph to G
    for (x, y) in temp_graph.edges():  # add edge from temp_graph to G
        xx = number_map[x]
        yy = number_map[y]
        G.add_edge(xx, yy)

    graph_size = G.number_of_nodes()
    edge_size = G.number_of_edges()
    logging.info('...done. Created graph with %d nodes and %d edges' % (graph_size, edge_size))
    logging.debug('G is:' + str(G.nodes()) + str(G.edges()))

    logging.info('Generating bandwidth')
    source_rates, learners_nodes, bandwidths = topologies.MNIST_parameters()
    bandwidths = dict([((number_map[u], number_map[v]), bandwidths[(u, v)]) for (u, v) in bandwidths])

    logging.info('Generating learners')
    learners = [number_map[node] for node in learners_nodes]

    logging.info('Generating sources')
    sources = [number_map[s] for s in source_rates]
    sourceRates = {}
    for s in source_rates:
        sourceRates[number_map[s]] = source_rates[s]

    logging.info('Generating prior')
    prior = {}
    prior['noice'] = {}
    prior['cov'] = {}

    prior['cov'][learners[0]] = Cov_1
    prior['noice'][sources[0]] = noice
    prior['cov'][learners[1]] = Cov_2
    prior['noice'][sources[1]] = noice

    logging.info('Generating paths and parameters of data')
    sourceParameters = {}
    sourceParameters['mean'] = {}
    sourceParameters['cov'] = {}
    sourceParameters['data'] = {}
    paths = {}

    s1 = sources[0]
    s2 = sources[1]
    images_1_np = np.squeeze(images_1)
    sourceParameters['mean'][s1] = np.mean(images_1_np, 0)
    sourceParameters['cov'][s1] = np.cov(images_1_np, rowvar=False)
    sourceParameters['data'][s1] = images_1_np
    images_2_np = np.squeeze(images_2)
    sourceParameters['mean'][s2] = np.mean(images_2_np, 0)
    sourceParameters['cov'][s2] = np.cov(images_2_np, rowvar=False)
    sourceParameters['data'][s2] = images_2_np

    for s in sources:
        paths[s] = []
        for l in learners:
            paths[s].append(l)

    P = Problem(sourceRates, learners, sources, bandwidths, G, paths, prior, args.T, sourceParameters)
    fname = 'Problem_5/Problem_MNIST2'
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(P, f)


if __name__ == '__main__':
    main()
