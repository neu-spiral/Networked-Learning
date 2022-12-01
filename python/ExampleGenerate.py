from networkx import DiGraph, shortest_path
import networkx
import pickle
import topologies
import numpy as np
import logging, argparse
import random


class Problem:
    def __init__(self, sources, learners, catalog, bandwidth, G, paths, features, prior, T, types):
        self.sources = sources
        self.learners = learners
        self.catalog = catalog
        self.bandwidth = bandwidth
        self.G = G
        self.paths = paths
        self.features = features
        self.prior = prior
        self.T = T
        self.types = types


def main():
    parser = argparse.ArgumentParser(description='Simulate Toy Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--catalog_size', default=2, type=int, help='Catalog size')
    parser.add_argument('--dimension', default=10, type=int, help='Feature dimension')

    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--T', default=1, type=float, help="Duration of experiment")

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)

    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 2021)

    logging.info('Generating Toy Example')
    temp_graph = topologies.ToyExample()
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

    logging.info('Generating catalog')
    catalog = list(range(args.catalog_size))

    logging.info('Generating bandwidth')
    source_rates, learner_types, bandwidths = topologies.ToyExample_parameters()
    bandwidths = dict([((number_map[u], number_map[v]), bandwidths[(u, v)]) for (u, v) in bandwidths])

    logging.info('Generating learners')
    learners = [number_map[node] for node in learner_types.keys()]

    logging.info('Generating features')
    features = {}
    for i in catalog:
        features[i] = np.random.uniform(0, 0.01, (args.dimension,1))
        upper = int(np.floor(args.dimension / len(learners)))
        j = (i % len(learners)) * upper
        features[i][j:j + upper] = np.random.uniform(1, 2, (upper, 1))

    type_set = set(learner_types.values())
    type_map = dict(zip(type_set, range(len(type_set))))

    logging.info('Generating types')
    types = dict([(number_map[l], type_map[learner_types[l]]) for l in learner_types])

    logging.info('Generating prior')
    prior = {}
    prior['noice'] = {}
    prior['cov'] = {}
    prior['beta'] = {}
    i = 0
    noice = np.random.uniform(0.5, 1, len(type_set))
    for l in learners:

        diag = np.random.uniform(0, 0.01, args.dimension)
        upper = np.floor(args.dimension / len(learners))
        j = int(i + upper)
        diag[i:j] = np.random.uniform(100, 200, j - i)
        prior['cov'][l] = np.diag(diag)
        prior['noice'][l] = noice[types[l]]
        prior['beta'][l] = np.zeros((args.dimension, 1))
        prior['beta'][l][i:j] = np.ones((j - i, 1))
        i = j

    logging.info('Generating sources')
    sources = {}
    for s in source_rates:
        sources[number_map[s]] = {}
        for i in catalog:
            sources[number_map[s]][i] = {}
            for t in source_rates[s]:
                sources[number_map[s]][i][type_map[t]] = source_rates[s][t]

    P = Problem(sources, learners, catalog, bandwidths, G, [], features, prior, args.T, types)
    fname = 'Problem_5/Problem_ToyExample2'
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(P, f)


if __name__ == '__main__':
    main()
