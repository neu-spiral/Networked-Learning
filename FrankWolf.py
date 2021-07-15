import cvxpy as cp
import pickle
import numpy as np

class FrankWolf:

    def __init__(self, P):
        self.sources = P.sources
        self.learners = P.learners
        self.catalog = P.catalog
        self.problem_size = (len(self.learners), len(self.catalog))
        self.bandwidth = P.bandwidth
        self.G = P.G
        self.paths = P.paths
        self.features = P.features
        self.prior = P.prior


    def find_max(self, Z, routing):
        """
        Solve a linear programing given gradient Z
        Z: a dictionary
        routing: 'hop' or 'source'
        return: a dictionary
        """
        D ={}
        for l in self.learners:
            D[l] = {}
            for i in self.catalog:
                D[l][i]= cp.Variable()

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

                    elif v in self.sources:
                        for e in out_edges:
                            out_flow[v][i] += D[e][i]
                        constr.append(out_flow[v][i] <= self.sources[v][i])

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
                obj += D[l][i]*Z[l][i]


        self.problem = cp.Problem(cp.Maximize(obj),constr)
        self.problem.solve()

        for l in self.learners:
            for i in self.catalog:
                D[l][i]= D[l][i].value
        return D

    def objective(self, n):
        obj = 0
        for l in self.learners:
            g = 0
            for i in self.catalog:
                g += n[l][i]*np.dot(self.features[i], np.linalg.inv(self.features[i]))
            obj += np.log(np.linalg.det(g + self.prior['delta']**2 * self.prior['sigma']))

    def generate_sample(self, Y, samples, T):
        N = []
        for j in range(samples):
            n = {}
            for l in self.learners:
                n[l] = {}
                for i in self.catalog:
                    n[l][i] = np.random.poisson(Y[l][i] * T, 1)
            N.append(n)
        return N

    def Estimate_Gradient(self, Y, head, samples, T):

        L, I = self.problem_size
        Z = np.matlib.zeros(self.problem_size)
        N = self.generate_sample(Y, samples, T)
        for l in range(L):
            for i in range(I):
                gradient = 0
                for t in range(head):
                    obj1 = 0
                    obj2 = 0
                    for j in range(samples):
                        n1 = N[t]
                        n1[l][i] = t+1
                        obj1 += self.objective(n1)
                        n2 = N[t]
                        n2[l][i] = t
                        obj2 += self.objective(n2)
                    obj1 /= samples
                    obj2 /= samples
                    gradient += (obj1 - obj2) * Y[l][i]**t * T**(t+1) / np.math.factorial(t) * np.exp(-Y[l][i]*T)
                Z[l][i] = gradient
        return Z


    def FW(self, iterations, head, samples, T, routing):
        def adapt(Y, D, gamma):
            for l in D:
                for i in D[l]:
                    Y[l][i] += gamma

        Y = np.matlib.zeros(self.problem_size)
        gamma = 1. / iterations
        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, samples, T)
            D = self.find_max(Z, routing)
            adapt(Y, D, gamma)

        return Y


if __name__ == '__main__':
    fname = ''
    with open(fname, 'rb') as f:
        P = pickle.load(f)
    FrankWolf(P)