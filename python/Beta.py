import logging, argparse
import pickle
import numpy as np
from ProbGenerate import Problem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate model through MAP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type', default="dtelekom", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'ToyExample2'])
    parser.add_argument('--types', default=3, type=int, help='Number of types')
    parser.add_argument('--learners', default=3, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--min_bandwidth', default=20, type=float, help='Minimum bandwidth of each edge')

    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()
    np.random.seed(args.random_seed + 2020)
    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    fname = "Result_{}/Result_old_{}_{}learners_{}sources_{}types".format(int(args.min_bandwidth),
        args.graph_type, args.learners, args.sources, args.types)
    # fname = "Result_{}/Result_ToyExample2".format(int(args.min_bandwidth))
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    fname = "Problem_{}/Problem_old_{}_{}learners_{}sources_{}types".format(int(args.min_bandwidth),
        args.graph_type, args.learners, args.sources, args.types)
    # fname = "Problem_{}/Problem_ToyExample2".format(int(args.min_bandwidth))
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)
    X = P.features
    beta = P.prior['beta']
    covariance = P.prior['cov']
    noice = P.prior['noice']
    learners = P.learners
    catalog = P.catalog
    T = P.T

    distance = []
    samples = 1000
    for r in range(len(results)):
        result = results[r][0]
        dist = 0
        if result == 0:
            distance.append(dist)
            continue
        for l in learners:
            norm = 0
            for j in range(samples):
                a = 0
                b = 0
                for i in catalog:
                    n = np.random.poisson(result[l][i] * T)
                    a += n * np.dot(X[i], X[i].transpose())
                    for k in range(n):
                        y = np.dot(X[i].transpose(), beta[l]) + np.random.normal(0, noice[l])
                        if y < 0:
                            pass
                        b += X[i] * y
                temp1 = a + noice[l] * np.linalg.inv(covariance[l])
                temp1 = np.linalg.inv(temp1)
                temp2 = np.dot(np.linalg.inv(covariance[l]), beta[l])
                map_l = np.dot(temp1, b) + np.dot(temp1, temp2)*noice[l]

                # map_l = np.dot(temp1, b)
                norm += np.linalg.norm(map_l - beta[l])
            norm /= samples
            dist += norm
        distance.append(dist / args.learners)
    print(distance)
    fname = "Result_{}/beta_old_{}_{}learners_{}sources_{}types".format(int(args.min_bandwidth),
        args.graph_type, args.learners, args.sources, args.types)
    # fname = "Result_{}/beta_ToyExample2".format(int(args.min_bandwidth))

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(distance, f)
