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
    parser = argparse.ArgumentParser(description='Simulate a Network',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--min_bandwidth', default=10, type=float, help='Minimum bandwidth of each edge')
    parser.add_argument('--max_bandwidth', default=20, type=float, help="Maximum bandwidth of each edge")

    parser.add_argument('--min_datarate', default=100, type=float, help='Minimum data rate of each item at each sources')
    parser.add_argument('--max_datarate', default=200, type=float, help="Maximum bandwidth of each edge")

    parser.add_argument('--catalog_size', default=20, type=int, help='Catalog size')
    parser.add_argument('--types', default=3, type=int, help='Number of types')
    parser.add_argument('--learners', default=3, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--dimension', default=20, type=int, help='Feature dimension')

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz','regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom','servicenetwork'])
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--graph_degree', default=4, type=int, help='Degree. Used by balanced_tree, regular, barabasi_albert, watts_strogatz')
    parser.add_argument('--graph_p', default=0.10, type=int, help='Probability, used in erdos_renyi, watts_strogatz')
    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--T', default=10, type=float, help="Duration of experiment")

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)

    def graphGenerator():
        if args.graph_type == "erdos_renyi":
            return networkx.erdos_renyi_graph(args.graph_size, args.graph_p)
        if args.graph_type == "balanced_tree":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(args.graph_degree)))
            return networkx.balanced_tree(args.graph_degree, ndim)
        if args.graph_type == "cicular_ladder":
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.circular_ladder_graph(ndim)
        if args.graph_type == "cycle":
            return networkx.cycle_graph(args.graph_size)
        if args.graph_type == 'grid_2d':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.grid_2d_graph(ndim, ndim)
        if args.graph_type == 'lollipop':
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.lollipop_graph(ndim, ndim)
        if args.graph_type == 'expander':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.margulis_gabber_galil_graph(ndim)
        if args.graph_type == "hypercube":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(2.0)))
            return networkx.hypercube_graph(ndim)
        if args.graph_type == "star":
            ndim = args.graph_size - 1
            return networkx.star_graph(ndim)
        if args.graph_type == 'barabasi_albert':
            return networkx.barabasi_albert_graph(args.graph_size, args.graph_degree)
        if args.graph_type == 'watts_strogatz':
            return networkx.connected_watts_strogatz_graph(args.graph_size, args.graph_degree, args.graph_p)
        if args.graph_type == 'regular':
            return networkx.random_regular_graph(args.graph_degree, args.graph_size)
        if args.graph_type == 'powerlaw_tree':
            return networkx.random_powerlaw_tree(args.graph_size)
        if args.graph_type == 'small_world':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.navigable_small_world_graph(ndim)
        if args.graph_type == 'geant':
            return topologies.GEANT()
        if args.graph_type == 'dtelekom':
            return topologies.Dtelekom()
        if args.graph_type == 'abilene':
            return topologies.Abilene()
        if args.graph_type == 'servicenetwork':
            return topologies.ServiceNetwork()

    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 2021)

    logging.info('Generating graph')
    temp_graph = graphGenerator()  # use networkx to generate a graph
    # networkx.draw(temp_graph)
    # plt.draw()
    V = len(temp_graph.nodes())
    E = len(temp_graph.edges())
    logging.debug('nodes: ' + str(temp_graph.nodes()))  # list
    logging.debug('edges: ' + str(temp_graph.edges()))  # list of node pair
    G = DiGraph()  # generate a DiGraph

    number_map = dict(zip(temp_graph.nodes(), range(len(temp_graph.nodes()))))
    G.add_nodes_from(number_map.values())  # add node from temp_graph to G
    for (x, y) in temp_graph.edges():  # add edge from temp_graph to G
        xx = number_map[x]
        yy = number_map[y]
        G.add_edges_from(((xx, yy), (yy, xx)))
    graph_size = G.number_of_nodes()
    edge_size = G.number_of_edges()
    logging.info('...done. Created graph with %d nodes and %d edges' % (graph_size, edge_size))
    logging.debug('G is:' + str(G.nodes()) + str(G.edges()))

    logging.info('Generating catalog')
    catalog = list(range(args.catalog_size))

    logging.info('Generating learners')
    learners = np.random.choice(range(graph_size), args.learners)

    logging.info('Generating bandwidth')
    bandwidth = {}
    for e in G.edges():
        bandwidth[e] = random.uniform(args.min_bandwidth, args.max_bandwidth)

    logging.info('Generating features')
    features = {}
    for i in catalog:
        features[i] = np.random.uniform(0, 0.1, (args.dimension,1))
        upper = int(np.floor(args.dimension / len(learners)))
        j = np.random.randint(args.dimension-upper)
        features[i][j:j+upper] = np.random.uniform(1000, 2000, (upper,1))
        # features[i] = np.random.rand(args.dimension,1)

    logging.info('Generating types')
    types = dict(zip(learners, range(args.types)))

    logging.info('Generating prior')
    prior = {}
    prior['noice'] = {}
    prior['cov'] = {}
    i = 0
    noice = np.random.rand(args.types)
    for l in learners:
        # covariance is PSD

        # A = np.random.rand(dimension, dimension)
        # B = np.dot(A, A.transpose())
        # C = B + B.T  # makesure symmetric
        #
        # prior['cov'][l] = C

        # matrix = np.random.rand(dimension, 1)
        # prior['cov'][l] = np.dot(matrix, matrix.transpose())

        diag = np.random.uniform(0, 1, args.dimension)
        upper = np.floor(args.dimension/len(learners))
        j = int(i+upper)
        diag[i:j] = np.random.uniform(1000, 2000, j-i)
        prior['cov'][l] = np.diag(diag)
        prior['noice'][l] = noice[types[l]]
        i = j

    logging.info('Generating sources')
    sources_set = np.random.choice(range(graph_size), args.sources)
    sources = {}
    for s in sources_set:
        sources[s] = {}
        for i in catalog:
            sources[s][i] = {}
            for t in set(types.values()):
                sources[s][i][t] = np.random.uniform(args.min_datarate, args.max_datarate, 1)

    P = Problem(sources, learners, catalog, bandwidth, G, [], features, prior, args.T, types)
    fname = 'Problem_/Problem_' + args.graph_type
    with open(fname, 'wb') as f:
        pickle.dump(P, f)

if __name__ == '__main__':
    main()


