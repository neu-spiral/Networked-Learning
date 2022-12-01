import cvxpy as cp
import pickle
import numpy as np
from prior import Problem
import copy
import logging, argparse
import time


class Gradient:

    def __init__(self, P):
        self.sourceRates = P.sourceRates
        self.learners = P.learners
        self.sources = P.sources
        self.bandwidth = P.bandwidth
        self.G = P.G
        self.paths = P.paths
        self.prior = P.prior
        self.T = P.T
        self.sourceParameters = P.sourceParameters

    def objG(self, features, n, noices, cov):
        temp = 0
        for s in self.sources:
            for i in range(n[s]):
                temp += np.dot(features[s][i], features[s][i].transpose()) / noices[s]
        temp = temp + np.linalg.inv(cov)
        temp = np.linalg.det(temp)
        obj = np.log(temp)
        return obj

    def objU(self, Y, N1, N2):
        obj = 0
        zeros = {}
        for s in self.sources:
            zeros[s] = 0
        for l in self.learners:
            for i in range(N1):
                n = self.generate_sample1(Y, l)
                for j in range(N2):
                    features = self.generate_sample2(n)
                    noices = self.prior['noice']
                    cov = self.prior['cov'][l]
                    obj += self.objG(features, n, noices, cov) - self.objG(zeros, zeros, noices, cov)
        obj = obj / N1 / N2

        return obj

    def generate_sample1(self, Y, l):
        n = {}
        for s in self.sources:
            rate = Y[s][l]
            n[s] = np.random.poisson(rate * self.T)

        return n

    def generate_sample2(self, n):
        features = {}
        for s in self.sources:
            # mean = self.sourceParameters['mean'][s]
            # cov = self.sourceParameters['cov'][s]
            size = n[s]
            # features[s] = np.random.multivariate_normal(mean, cov, size)
            index = np.random.choice(len(self.sourceParameters['data'][s]), size)
            features[s] = self.sourceParameters['data'][s][index]

        return features

    def Estimate_Gradient(self, Y, head, N1, N2):

        Z = {}
        for s in self.paths:
            Z[s] = {}
            for l in self.paths[s]:
                Z[s][l] = 0

        for l in self.learners:
            noices = self.prior['noice']
            cov = self.prior['cov'][l]

            for i in range(N1):
                n = self.generate_sample1(Y, l)
                n_h = {}  # each arrival is greater than head
                for s in n:
                    n_h[s] = max(head, n[s])
                t1 = time.time()
                for j in range(N2):
                    features = self.generate_sample2(n_h)
                    for s in self.sources:
                        n_copy = n.copy()
                        rate = Y[s][l]
                        delta = 0
                        for h in range(head):
                            n_copy[s] = h + 1
                            obj1 = self.objG(features, n_copy, noices, cov)
                            n_copy[s] = h
                            obj2 = self.objG(features, n_copy, noices, cov)

                            delta += (obj1 - obj2) * rate ** h * self.T ** (h + 1) / np.math.factorial(h) * \
                                     np.exp(-rate * self.T)
                        Z[s][l] += delta / N1 / N2
                t2 = time.time()
                print(t2-t1)

        return Z

    def adapt(self, Y, D, gamma):
        for s in self.paths:
            for l in self.paths[s]:
                Y[s][l] += gamma * D[s][l]

    def alg(self, iterations, head, N1, N2):
        pass


class FrankWolf(Gradient):
    def find_max(self, Z):
        constr = []

        D = {}
        for s in self.paths:
            D[s] = {}
            for l in self.paths[s]:
                D[s][l] = cp.Variable()
                constr.append(D[s][l] >= 0)

        for (u, v) in self.G.edges():
            constr.append(D[u][v] <= self.bandwidth[(u, v)])

        for s in self.paths:
            temp = 0
            for l in self.paths[s]:
                temp += D[s][l]
            constr.append(temp <= self.sourceRates[s])

        obj = 0
        for s in self.paths:
            for l in self.paths[s]:
                obj += D[s][l] * Z[s][l]

        problem = cp.Problem(cp.Maximize(obj), constr)
        problem.solve()
        print("status: ", problem.status)

        for s in self.paths:
            for l in self.paths[s]:
                D[s][l] = D[s][l].value

        return D

    def alg(self, iterations, head, N1, N2):

        Y = {}
        for s in self.paths:
            Y[s] = {}
            for l in self.paths[s]:
                Y[s][l] = 0

        gamma = 1. / iterations
        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, N1, N2)
            D = self.find_max(Z)
            self.adapt(Y, D, gamma)
            print(t, Y)

        return Y


class ProjectAscent(Gradient):

    def project(self, Y, routing='hop'):
        constr = []

        D = {}
        for s in self.paths:
            D[s] = {}
            for l in self.paths[s]:
                D[s][l] = cp.Variable()
                constr.append(D[s][l] >= 0)

        for (u, v) in self.G.edges():
            constr.append(D[u][v] <= self.bandwidth[(u, v)])

        for s in self.paths:
            temp = 0
            for l in self.paths[s]:
                temp += D[s][l]
            constr.append(temp <= self.sourceRates[s])

        obj = 0
        for s in self.paths:
            for l in self.paths[s]:
                obj += (D[s][l] - Y[s][l]) ** 2

        problem = cp.Problem(cp.Minimize(obj), constr)
        problem.solve()
        print("status: ", problem.status)

        for s in self.paths:
            for l in self.paths[s]:
                D[s][l] = D[s][l].value

        return D

    def alg(self, iterations, head, N1, N2):
        Y = {}
        for s in self.paths:
            Y[s] = {}
            for l in self.paths[s]:
                Y[s][l] = 0

        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, N1, N2)
            self.adapt(Y, Z, 1. / (t + 1))
            Y = self.project(Y)
            print(t, Y)

        return Y


class UtilityMax:
    def __init__(self, P):

        self.sourceRates = P.sourceRates
        self.learners = P.learners
        self.sources = P.sources
        self.bandwidth = P.bandwidth
        self.G = P.G
        self.paths = P.paths
        self.prior = P.prior
        self.T = P.T

        self.D = {}
        for l in self.learners:
            self.D[l] = {}
            for i in self.catalog:
                self.D[l][i] = 0

        self.constr = []

        D = {}
        for s in self.paths:
            D[s] = {}
            for l in self.paths[s]:
                D[s][l] = cp.Variable()
                self.constr.append(D[s][l] >= 0)

        for (u, v) in self.G.edges():
            self.constr.append(D[u][v] <= self.bandwidth[(u, v)])

        for s in self.paths:
            temp = 0
            for l in self.paths[s]:
                temp += D[s][l]
            self.constr.append(temp <= self.sourceRates[s])

    def objective(self, alpha):
        def utility(x, alpha):
            if alpha == 1.0:
                return cp.log(x)
            else:
                return x ** (1 - alpha) / (1 - alpha)

        obj = 0
        for l in self.learners:
            obj_l = 0
            for s in self.sources:
                obj_l += self.D[s][l]
            obj += utility(obj_l, alpha)
        return obj

    def solve(self, alpha):
        obj = self.objective(alpha)
        self.problem = cp.Problem(cp.Maximize(obj), self.constr)
        self.problem.solve(solver=cp.MOSEK)
        print("status:", self.problem.status)

        for s in self.paths:
            for l in self.paths[s]:
                self.D[s][l] = self.D[s][l].value
        return self.D




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MNIST FW algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()
    np.random.seed(args.random_seed + 2021)

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    fname = 'Problem_5/Problem_MNIST2'
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)
    alg1 = FrankWolf(P)

    Y1 = alg1.alg(iterations=20, head=20, N1=20, N2=20)
    obj1 = alg1.objU(Y=Y1, samples=100)
    print(Y1)

    alg2 = UtilityMax(P)
    Y2 = alg2.solve(0)
    obj2 = alg1.objU(Y=Y2, samples=100)
    print(Y2)

    alg2 = UtilityMax(P)  # returned Y will cover the original variable
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
    Y6 = alg4.alg(iterations=20, head=20, N1=20, N2=20)
    obj6 = alg1.objU(Y=Y6, samples=100)
    print(Y6)

    print(obj1, obj2, obj3, obj6)

    fname = 'Result_5/Result_MNIST2'
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump([(Y1, obj1), (Y2, obj2), (Y3, obj3), (0, 0), (0, 0), (Y6, obj6)], f)
