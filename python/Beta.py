import logging, argparse
import pickle
import numpy as np
from ProbGenerate import Problem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate model through MAP',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type', default="dtelekom", type=str, help='Graph type',choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz','regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom','servicenetwork'])
    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()
    np.random.seed(args.random_seed + 2020)
    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    fname = 'Result_smallrate/Result_more_' + args.graph_type
    logging.info('Read data from '+fname)
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    fname = 'Problem_smallrate/Problem_' + args.graph_type
    logging.info('Read data from '+fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)
    X = P.features
    beta = P.prior['beta']
    # beta0 = P.prior['beta']
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
        for l in learners:
            norm = 0
            for j in range(samples):
                a = 0
                b = 0
                for i in catalog:
                    n = np.random.poisson(result[l][i] * T)
                    a += n * np.dot(X[i], X[i].transpose())
                    for k in range(n):
                        y = np.dot(X[i].transpose(), beta[l]) + np.random.normal(0,noice[l])
                        if y < 0:
                            pass
                        b += X[i] * y
                temp1 = a + noice[l]*np.linalg.inv(covariance[l])
                temp1 = np.linalg.inv(temp1)
                # temp2 = np.dot(np.linalg.inv(covariance[l]), beta0[l])
                # map_l = np.dot(temp1, b) + np.dot(temp1, temp2)*noice[l]

                map_l = np.dot(temp1, b)
                norm += np.linalg.norm(map_l-beta[l])
            norm /= samples
            dist += norm
        distance.append(dist)
    print(distance)
    fname = 'Result_smallrate/beta_more_' + args.graph_type
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(distance, f)

