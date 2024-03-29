import cvxpy as cp
import pickle
import numpy as np
from ProbGenerate import Problem
import copy
import logging, argparse
import time


class Gradient:

    def __init__(self, P):
        self.sources = P.sources
        self.learners = P.learners
        self.catalog = P.catalog
        # self.problem_size = (len(self.learners), len(self.catalog))
        self.bandwidth = P.bandwidth
        self.G = P.G
        self.paths = P.paths
        self.features = P.features
        self.prior = P.prior
        self.T = P.T
        self.types = P.types

    def objG(self, n, l):
        temp = 0
        for i in self.catalog:
            temp += n[i] * np.dot(self.features[i], self.features[i].transpose())
        # temp1 = np.linalg.det(temp / self.prior['noice'][l])
        # temp2 = np.linalg.det(np.linalg.inv(self.prior['cov'][l]))
        # temp1 = np.log(temp1)
        # temp2 = np.log(temp2)
        temp = temp / self.prior['noice'][l] + np.linalg.inv(self.prior['cov'][l])
        temp = np.linalg.det(temp)
        obj = np.log(temp)
        return obj

    def objU(self, Y, samples):
        obj = 0
        zero = {}
        for i in self.catalog:
            zero[i] = 0
        for l in self.learners:
            N = self.generate_sample(Y, samples, l)
            for j in range(samples):
                obj += self.objG(N[j], l) - self.objG(zero, l)
        obj /= samples
        return obj

    def generate_sample(self, Y, samples, l):
        N = []
        for j in range(samples):
            n = {}
            for i in self.catalog:
                n[i] = np.random.poisson(Y[l][i] * self.T)
            N.append(n)
        return N

    def Estimate_Gradient(self, Y, head, samples):

        # L, I = self.problem_size
        # Z = np.zeros(self.problem_size)
        Z = {}
        for l in self.learners:
            Z[l] = {}
            for i in self.catalog:
                Z[l][i] = cp.Variable()

        for l in self.learners:
            N = self.generate_sample(Y, samples, l)
            for i in self.catalog:
                t1 = time.time()
                gradient = 0
                for t in range(head):
                    obj1 = 0
                    obj2 = 0
                    for j in range(samples):
                        # n1 = copy.deepcopy(N[j])
                        n1 = N[j].copy()
                        n1[i] = t + 1
                        obj1 += self.objG(n1, l)
                        # n2 = copy.deepcopy(N[j])
                        n2 = N[j].copy()
                        n2[i] = t
                        obj2 += self.objG(n2, l)
                    obj1 /= samples
                    obj2 /= samples
                    gradient += (obj1 - obj2) * Y[l][i] ** t * self.T ** (t + 1) / np.math.factorial(t) * np.exp(
                        -Y[l][i] * self.T)
                Z[l][i] = gradient
                t2 = time.time()
                print(t2-t1)

        return Z

    def adapt(self, Y, D, gamma):
        for l in self.learners:
            for i in self.catalog:
                Y[l][i] += gamma * D[l][i]

    def alg(self, iterations, head, samples, routing='hop'):
        pass


class FrankWolf(Gradient):
    def find_max(self, Z, routing='hop'):
        """
        Solve a linear programing given gradient Z
        Z: a dictionary
        routing: 'hop'
        return: a dictionary
        """
        D = {}
        for l in self.learners:
            D[l] = {}
            for i in self.catalog:
                D[l][i] = 0

        constr = []
        if routing == 'hop':

            for e in self.G.edges():
                D[e] = {}
                for i in self.catalog:
                    D[e][i] = {}
                    for t in set(self.types.values()):
                        D[e][i][t] = cp.Variable()
                        constr.append(D[e][i][t] >= 0)

            flow = {}
            for e in self.G.edges():
                flow[e] = 0
                for i in self.catalog:
                    for t in set(self.types.values()):
                        flow[e] += D[e][i][t]
                constr.append(flow[e] <= self.bandwidth[e])

            in_flow = {}
            out_flow = {}
            for v in self.G.nodes():
                in_edges = self.G.in_edges(v)
                out_edges = self.G.out_edges(v)
                in_flow[v] = {}
                out_flow[v] = {}

                for i in self.catalog:
                    in_flow[v][i] = {}
                    out_flow[v][i] = {}

                    for t in set(self.types.values()):
                        in_flow[v][i][t] = 0
                        out_flow[v][i][t] = 0

                        if v in self.learners:
                            if t == self.types[v]:
                                for e in in_edges:
                                    in_flow[v][i][t] += D[e][i][t]
                                D[v][i] = in_flow[v][i][t]
                            for e in out_edges:
                                out_flow[v][i][t] += D[e][i][t]
                            constr.append(out_flow[v][i][t] == 0)

                        elif v in self.sources:
                            if t in self.sources[v][i]:
                                in_flow[v][i][t] = self.sources[v][i][t]
                            else:
                                in_flow[v][i][t] = 0
                            for e in out_edges:
                                out_flow[v][i][t] += D[e][i][t]
                            constr.append(out_flow[v][i][t] <= in_flow[v][i][t])

                        else:
                            for e in in_edges:
                                in_flow[v][i][t] += D[e][i][t]
                            for e in out_edges:
                                out_flow[v][i][t] += D[e][i][t]
                            constr.append(out_flow[v][i][t] <= in_flow[v][i][t])

        obj = 0
        for l in self.learners:
            for i in self.catalog:
                obj += D[l][i] * Z[l][i]

        self.problem = cp.Problem(cp.Maximize(obj), constr)
        self.problem.solve(solver=cp.MOSEK)
        print("status:", self.problem.status)
        # print('solve_time', self.problem.solution.attr['solve_time'])

        if self.problem.status == 'optimal':
            for l in self.learners:
                for i in self.catalog:
                    D[l][i] = D[l][i].value if D[l][i].value >= 0 else 0.
        else:
            for l in self.learners:
                for i in self.catalog:
                    D[l][i] = 0
        return D

    def alg(self, iterations, head, samples, routing='hop'):

        # Y = np.zeros(self.problem_size)
        Y = {}
        for l in self.learners:
            Y[l] = {}
            for i in self.catalog:
                Y[l][i] = 0
        gamma = 1. / iterations
        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, samples)
            D = self.find_max(Z, routing)
            self.adapt(Y, D, gamma)
            print(t)

        return Y


class ProjectAscent(Gradient):

    def project(self, Y, routing='hop'):
        """
        Solve a project given Y: argmin_D (D-Y)^2
        Y: a dictionary
        routing: 'hop'
        return: a dictionary
        """
        D = {}
        for l in self.learners:
            D[l] = {}
            for i in self.catalog:
                D[l][i] = 0

        constr = []
        if routing == 'hop':

            for e in self.G.edges():
                D[e] = {}
                for i in self.catalog:
                    D[e][i] = {}
                    for t in set(self.types.values()):
                        D[e][i][t] = cp.Variable()
                        constr.append(D[e][i][t] >= 0)

            flow = {}
            for e in self.G.edges():
                flow[e] = 0
                for i in self.catalog:
                    for t in set(self.types.values()):
                        flow[e] += D[e][i][t]
                constr.append(flow[e] <= self.bandwidth[e])

            in_flow = {}
            out_flow = {}
            for v in self.G.nodes():
                in_edges = self.G.in_edges(v)
                out_edges = self.G.out_edges(v)
                in_flow[v] = {}
                out_flow[v] = {}

                for i in self.catalog:
                    in_flow[v][i] = {}
                    out_flow[v][i] = {}

                    for t in set(self.types.values()):
                        in_flow[v][i][t] = 0
                        out_flow[v][i][t] = 0

                        if v in self.learners:
                            if t == self.types[v]:
                                for e in in_edges:
                                    in_flow[v][i][t] += D[e][i][t]
                                D[v][i] = in_flow[v][i][t]
                            for e in out_edges:
                                out_flow[v][i][t] += D[e][i][t]
                            constr.append(out_flow[v][i][t] == 0)

                        elif v in self.sources:
                            if t in self.sources[v][i]:
                                in_flow[v][i][t] = self.sources[v][i][t]
                            else:
                                in_flow[v][i][t] = 0
                            for e in out_edges:
                                out_flow[v][i][t] += D[e][i][t]
                            constr.append(out_flow[v][i][t] <= in_flow[v][i][t])

                        else:
                            for e in in_edges:
                                in_flow[v][i][t] += D[e][i][t]
                            for e in out_edges:
                                out_flow[v][i][t] += D[e][i][t]
                            constr.append(out_flow[v][i][t] <= in_flow[v][i][t])

        obj = 0
        for l in self.learners:
            for i in self.catalog:
                obj += (D[l][i] - Y[l][i]) ** 2

        self.problem = cp.Problem(cp.Minimize(obj), constr)
        self.problem.solve(solver = cp.MOSEK)
        # self.problem.solve()
        print("status:", self.problem.status)
        # print('solve_time', self.problem.solution.attr['solve_time'])

        D_value = {}
        if self.problem.status == 'optimal':
            for l in self.learners:
                D_value[l] = {}
                for i in self.catalog:
                    D_value[l][i] = D[l][i].value if D[l][i].value >= 0 else 0.
        else:
            for l in self.learners:
                D_value[l] = {}
                for i in self.catalog:
                    D_value[l][i] = 0
        return D_value

    def alg(self, iterations, head, samples, routing='hop'):
        Y = {}
        for l in self.learners:
            Y[l] = {}
            for i in self.catalog:
                Y[l][i] = 0
        # gamma = 1. / iterations
        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, samples)
            self.adapt(Y, Z, 1. / (t + 1))
            Y = self.project(Y, routing)
            print(t)

        return Y


class UtilityMax:
    def __init__(self, P, routing='hop'):

        self.sources = P.sources
        self.learners = P.learners
        self.catalog = P.catalog
        self.bandwidth = P.bandwidth
        self.G = P.G
        self.paths = P.paths
        self.features = P.features
        self.prior = P.prior
        self.types = P.types

        self.D = {}
        for l in self.learners:
            self.D[l] = {}
            for i in self.catalog:
                self.D[l][i] = 0

        self.constr = []
        if routing == 'hop':

            for e in self.G.edges():
                self.D[e] = {}
                for i in self.catalog:
                    self.D[e][i] = {}
                    for t in set(self.types.values()):
                        self.D[e][i][t] = cp.Variable()
                        self.constr.append(self.D[e][i][t] >= 0)

            flow = {}
            for e in self.G.edges():
                flow[e] = 0
                for i in self.catalog:
                    for t in set(self.types.values()):
                        flow[e] += self.D[e][i][t]
                self.constr.append(flow[e] <= self.bandwidth[e])

            in_flow = {}
            out_flow = {}
            for v in self.G.nodes():
                in_edges = self.G.in_edges(v)
                out_edges = self.G.out_edges(v)
                in_flow[v] = {}
                out_flow[v] = {}

                for i in self.catalog:
                    in_flow[v][i] = {}
                    out_flow[v][i] = {}

                    for t in set(self.types.values()):
                        in_flow[v][i][t] = 0
                        out_flow[v][i][t] = 0

                        if v in self.learners:
                            if t == self.types[v]:
                                for e in in_edges:
                                    in_flow[v][i][t] += self.D[e][i][t]
                                self.D[v][i] = in_flow[v][i][t]
                            for e in out_edges:
                                out_flow[v][i][t] += self.D[e][i][t]
                            self.constr.append(out_flow[v][i][t] == 0)

                        elif v in self.sources:
                            if t in self.sources[v][i]:
                                in_flow[v][i][t] = self.sources[v][i][t]
                            else:
                                in_flow[v][i][t] = 0
                            for e in out_edges:
                                out_flow[v][i][t] += self.D[e][i][t]
                            self.constr.append(out_flow[v][i][t] <= in_flow[v][i][t])

                        else:
                            for e in in_edges:
                                in_flow[v][i][t] += self.D[e][i][t]
                            for e in out_edges:
                                out_flow[v][i][t] += self.D[e][i][t]
                            self.constr.append(out_flow[v][i][t] <= in_flow[v][i][t])

    def objective(self, alpha):
        pass

    def solve(self, alpha):
        obj = self.objective(alpha)
        self.problem = cp.Problem(cp.Maximize(obj), self.constr)
        self.problem.solve(solver=cp.MOSEK)
        print("status:", self.problem.status)

        D_value = {}
        if self.problem.status == 'optimal':
            for l in self.learners:
                D_value[l] = {}
                for i in self.catalog:
                    D_value[l][i] = self.D[l][i].value if self.D[l][i].value >= 0 else 0.
        else:
            for l in self.learners:
                D_value[l] = {}
                for i in self.catalog:
                    D_value[l][i] = 0
        return D_value


class NeededRate(UtilityMax):
    def objective(self, alpha):
        def utility(x, alpha):
            if alpha == 1.0:
                return cp.log(x)
            else:
                return x ** (1 - alpha) / (1 - alpha)

        obj = 0
        for l in self.learners:
            obj_l = 0
            for i in self.catalog:
                obj_l += self.D[l][i]
            obj += utility(obj_l, alpha)
        return obj


class AllRate(UtilityMax):
    def objective(self, alpha):
        def utility(x, alpha):
            if alpha == 1.0:
                return cp.log(x)
            else:
                return x ** (1 - alpha) / (1 - alpha)

        obj = 0

        for l in self.learners:
            obj_l = 0
            for e in self.G.in_edges(l):
                for i in self.catalog:
                    for t in set(self.types.values()):
                        obj_l += self.D[e][i][t]
            obj += utility(obj_l, alpha)
        return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'ToyExample'])
    parser.add_argument('--types', default=3, type=int, help='Number of types')
    parser.add_argument('--learners', default=3, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--min_bandwidth', default=20, type=int, help='Minimum bandwidth of each edge')

    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()
    np.random.seed(args.random_seed + 2021)

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    fname = "Problem_{}/Problem_old_{}_{}learners_{}sources_{}types".format(int(args.min_bandwidth),
        args.graph_type, args.learners, args.sources, args.types)
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)
    alg1 = FrankWolf(P)

    Y1 = alg1.alg(iterations=50, head=50, samples=20)
    obj1 = alg1.objU(Y=Y1, samples=100)
    print(Y1)

    alg2 = NeededRate(P)
    Y2 = alg2.solve(0)
    obj2 = alg1.objU(Y=Y2, samples=100)
    print(Y2)

    alg2 = NeededRate(P)
    Y3 = alg2.solve(5.0)
    obj3 = alg1.objU(Y=Y3, samples=100)
    print(Y3)

    # alg3 = AllRate(P)
    # Y4 = alg3.solve(0)
    # obj4 = alg1.objU(Y=Y4, samples=100)
    # print(Y4)
    #
    # alg3 = AllRate(P)
    # Y5 = alg3.solve(5.0)
    # obj5 = alg1.objU(Y=Y5, samples=100)
    # print(Y5)

    alg4 = ProjectAscent(P)
    Y6 = alg4.alg(iterations=50, head=50, samples=20)
    obj6 = alg1.objU(Y=Y6, samples=100)
    print(Y6)

    print(obj1, obj2, obj3, obj6)

    fname = "Result_{}/Result_old_{}_{}learners_{}sources_{}types".format(int(args.min_bandwidth),
        args.graph_type, args.learners, args.sources, args.types)
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump([(Y1, obj1), (Y2, obj2), (Y3, obj3), (0, 0), (0, 0), (Y6, obj6)], f)
