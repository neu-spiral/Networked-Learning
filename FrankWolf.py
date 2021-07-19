import cvxpy as cp
import pickle
import numpy as np
from ProbGenerate import Problem
import copy


class FrankWolf:

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

    def find_max(self, Z, routing='hop'):
        """
        Solve a linear programing given gradient Z
        Z: a dictionary
        routing: 'hop' or 'source'
        return: a dictionary
        """
        D = {}
        for l in self.learners:
            D[l] = {}
            for i in self.catalog:
                D[l][i] = cp.Variable()

        constr = []
        if routing == 'hop':

            for e in self.G.edges():
                D[e] = {}
                for i in self.catalog:
                    D[e][i] = cp.Variable()
                    constr.append(D[e][i] >= 0)

            flow = {}
            for e in self.G.edges():
                flow[e] = 0
                for i in self.catalog:
                    flow[e] += D[e][i]
                constr.append(flow[e] <= self.bandwidth[e])

            in_flow = {}
            out_flow = {}
            for v in self.G.nodes():
                in_edges = self.G.in_edges(v)
                out_edges = self.G.out_edges(v)
                in_flow[v] = {}
                out_flow[v] = {}

                for i in self.catalog:
                    in_flow[v][i] = 0
                    out_flow[v][i] = 0

                    if v in self.learners:
                        for e in in_edges:
                            in_flow[v][i] += D[e][i]
                        constr.append(in_flow[v][i] == D[v][i])
                        for e in out_edges:
                            out_flow[v][i] += D[e][i]
                        constr.append(out_flow[v][i] == 0)

                    elif v in self.sources:
                        for e in out_edges:
                            out_flow[v][i] += D[e][i]
                        constr.append(out_flow[v][i] <= self.sources[v][i])
                        for e in in_edges:
                            in_flow[v][i] += D[e][i]
                        constr.append(in_flow[v][i] == 0)

                    else:
                        for e in in_edges:
                            in_flow[v][i] += D[e][i]
                        for e in out_edges:
                            out_flow[v][i] += D[e][i]
                        constr.append(out_flow[v][i] <= in_flow[v][i])

        else:
            for l in self.learners:
                D[l] = {}
                for s in self.sources:
                    D[l][s] = {}
                    for i in self.catalog:
                        D[l][s][i] = {}
                        for h in self.paths[l][s]:
                            D[l][s][i][h] = cp.Variable()
                            constr.append(D[l][s][i][h] >= 0)

            in_flow = {}
            for l in self.learners:
                in_flow[l] = {}
                for i in self.catalog:
                    in_flow[l][i] = 0
                    for s in self.sources:
                        for h in self.paths[l][s]:
                            in_flow[l][i] += D[l][s][i][h]
                    constr.append(in_flow[l][i] == D[l][i])

            out_flow = {}
            for s in self.sources:
                out_flow[s] = {}
                for i in self.catalog:
                    out_flow[s][i] = 0
                    for l in self.learners:
                        for h in self.paths[l][s]:
                            out_flow[s][i] += D[l][s][i][h]
                    constr.append(out_flow[s][i] <= self.sources[s][i])

            flow = {}
            for e in self.G.edges():
                flow[e] = 0
                for l in self.learners:
                    for s in self.sources:
                        for i in self.catalog:
                            for h in self.paths[l][s]:
                                if e in h:
                                    flow[e] += D[l][s][i][h]
                constr.append(flow[e] <= self.bandwidth[e])

        obj = 0
        for l in self.learners:
            for i in self.catalog:
                obj += D[l][i] * Z[l][i]

        self.problem = cp.Problem(cp.Maximize(obj), constr)
        self.problem.solve()
        print("status:", self.problem.status)

        for l in self.learners:
            for i in self.catalog:
                D[l][i] = D[l][i].value if D[l][i].value>=0 else 0.
        for e in self.G.edges():
            for i in self.catalog:
                D[e][i] = D[e][i].value if D[e][i].value>=0 else 0.
        return D

    def objG(self, n, l):
        temp = 0
        for i in self.catalog:
            temp += n[i] * np.dot(self.features[i], self.features[i].transpose())
        temp = np.linalg.det(temp + self.prior['noice'] ** 2 * self.prior['cov'][l])
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
                n[i] = np.random.poisson(Y[l][i] * self.T, 1)
            N.append(n)
        return N

    def Estimate_Gradient(self, Y, head, samples):

        # L, I = self.problem_size
        # Z = np.zeros(self.problem_size)
        Z = {}
        for l in self.learners:
            Z[l] = {}
            for i in self.catalog:
                Z[l][i] = 0

        for l in self.learners:
            N = self.generate_sample(Y, samples, l)
            for i in self.catalog:
                gradient = 0
                for t in range(head):
                    obj1 = 0
                    obj2 = 0
                    for j in range(samples):
                        n1 = copy.deepcopy(N[j])
                        n1[i] = t + 1
                        obj1 += self.objG(n1, l)
                        n2 = copy.deepcopy(N[j])
                        n2[i] = t
                        obj2 += self.objG(n2, l)
                    obj1 /= samples
                    obj2 /= samples
                    gradient += (obj1 - obj2) * Y[l][i] ** t * self.T ** (t + 1) / np.math.factorial(t) * np.exp(
                        -Y[l][i] * self.T)
                Z[l][i] = gradient
        return Z

    def adapt(self, Y, D, gamma):
        for l in self.learners:
            for i in self.catalog:
                Y[l][i] += gamma * D[l][i]

    def FW(self, iterations, head, samples, routing='hop'):

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
            print(t, Y)

        return Y

class UtilityMax:
    def __init__(self, P, routing = 'hop'):

        self.sources = P.sources
        self.learners = P.learners
        self.catalog = P.catalog
        self.bandwidth = P.bandwidth
        self.G = P.G
        self.paths = P.paths
        self.features = P.features
        self.prior = P.prior

        self.D = {}
        for l in self.learners:
            self.D[l] = {}
            for i in self.catalog:
                self.D[l][i] = cp.Variable()

        self.constr = []
        if routing == 'hop':

            for e in self.G.edges():
                self.D[e] = {}
                for i in self.catalog:
                    self.D[e][i] = cp.Variable()
                    self.constr.append(self.D[e][i] >= 0)

            flow = {}
            for e in self.G.edges():
                flow[e] = 0
                for i in self.catalog:
                    flow[e] += self.D[e][i]
                self.constr.append(flow[e] <= self.bandwidth[e])

            in_flow = {}
            out_flow = {}
            for v in self.G.nodes():
                in_edges = self.G.in_edges(v)
                out_edges = self.G.out_edges(v)
                in_flow[v] = {}
                out_flow[v] = {}

                for i in self.catalog:
                    in_flow[v][i] = 0
                    out_flow[v][i] = 0

                    if v in self.learners:
                        for e in in_edges:
                            in_flow[v][i] += self.D[e][i]
                        self.constr.append(in_flow[v][i] == self.D[v][i])
                        for e in out_edges:
                            out_flow[v][i] += self.D[e][i]
                        self.constr.append(out_flow[v][i] == 0)

                    elif v in self.sources:
                        for e in out_edges:
                            out_flow[v][i] += self.D[e][i]
                        self.constr.append(out_flow[v][i] <= self.sources[v][i])
                        for e in in_edges:
                            in_flow[v][i] += self.D[e][i]
                        self.constr.append(in_flow[v][i] == 0)

                    else:
                        for e in in_edges:
                            in_flow[v][i] += self.D[e][i]
                        for e in out_edges:
                            out_flow[v][i] += self.D[e][i]
                        self.constr.append(out_flow[v][i] <= in_flow[v][i])

        else:
            for l in self.learners:
                self.D[l] = {}
                for s in self.sources:
                    self.D[l][s] = {}
                    for i in self.catalog:
                        self.D[l][s][i] = {}
                        for h in self.paths[l][s]:
                            self.D[l][s][i][h] = cp.Variable()
                            self.constr.append(self.D[l][s][i][h] >= 0)

            in_flow = {}
            for l in self.learners:
                in_flow[l] = {}
                for i in self.catalog:
                    in_flow[l][i] = 0
                    for s in self.sources:
                        for h in self.paths[l][s]:
                            in_flow[l][i] += self.D[l][s][i][h]
                    self.constr.append(in_flow[l][i] == self.D[l][i])

            out_flow = {}
            for s in self.sources:
                out_flow[s] = {}
                for i in self.catalog:
                    out_flow[s][i] = 0
                    for l in self.learners:
                        for h in self.paths[l][s]:
                            out_flow[s][i] += self.D[l][s][i][h]
                    self.constr.append(out_flow[s][i] <= self.sources[s][i])

            flow = {}
            for e in self.G.edges():
                flow[e] = 0
                for l in self.learners:
                    for s in self.sources:
                        for i in self.catalog:
                            for h in self.paths[l][s]:
                                if e in h:
                                    flow[e] += self.D[l][s][i][h]
                self.constr.append(flow[e] <= self.bandwidth[e])

    def objective(self, alpha):
        def utility(x, alpha):
            if alpha == 1.0:
                return cp.log(x)
            else:
                return x ** (1-alpha) / (1-alpha)
        obj = 0
        for l in self.learners:
            obj_l = 0
            for i in self.catalog:
                obj_l += self.D[l][i]
            obj += utility(obj_l, alpha)
        return obj

    def solve(self, alpha):
        obj = self.objective(alpha)
        self.problem = cp.Problem(cp.Maximize(obj), self.constr)
        self.problem.solve()
        print("status:", self.problem.status)

        for l in self.learners:
            for i in self.catalog:
                self.D[l][i] = self.D[l][i].value if self.D[l][i].value>=0 else 0.
        for e in self.G.edges():
            for i in self.catalog:
                self.D[e][i] = self.D[e][i].value if self.D[e][i].value>=0 else 0.
        return self.D

if __name__ == '__main__':
    fname = 'Problem'
    with open(fname, 'rb') as f:
        P = pickle.load(f)
    alg1 = FrankWolf(P)
    Y1 = alg1.FW(iterations=2, head=10, samples=10)
    obj1 = alg1.objU(Y=Y1, samples=1000)

    alg2 = UtilityMax(P)
    Y2 = alg2.solve(0)
    obj2 = alg1.objU(Y=Y2, samples=1000)


    alg2 = UtilityMax(P)
    Y4 = alg2.solve(2.0)
    obj4 = alg1.objU(Y=Y4, samples=1000)
    print(obj1, obj2, obj4)




