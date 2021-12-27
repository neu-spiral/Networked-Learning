import logging, argparse
import pickle
from FrankWolf import FrankWolf
from ProbGenerate import Problem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz','regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom','servicenetwork'])
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    fname = 'Problem_smallrate/Problem_' + args.graph_type
    logging.info('Read data from '+fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)
    alg1 = FrankWolf(P)


    fname = 'Result_smallrate/Result_more_' + args.graph_type
    logging.info('Read data from '+fname)
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    results_large = []
    for r in range(len(results)):
        result = results[r][0]
        print(results[r][1])
        results_large.append(alg1.objU(Y=result, samples=1000))
        print(results_large[r])

    fname = 'Result_smallrate/Result2_more_' + args.graph_type
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(results_large, f)


