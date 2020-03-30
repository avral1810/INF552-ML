import copy
import numpy as np
import matplotlib.pyplot as plt


class Pocket():
    
    def __init__(self, weights=[], alpha=0.01, maxIter=1000):
        self.weights = weights
        self.alpha = alpha
        self.maxIter = maxIter
        self.errorCounts = []
        self.bestWeights = []
        self.bestErrorCount = float('inf')
        self.section = maxIter/14
    
    def train(self, X, Y):
        d = X.shape[1]
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.random(d+1) # make space for W0
        iter = 0
        while iter < self.maxIter:
            self.errorCounts.append(0)
            for x, y in zip(X, Y):
                prod = np.dot(x, self.weights)
                if prod > 0 and y < 0:
                    self.weights -= self.alpha * x
                elif prod < 0 and y > 0:
                    self.weights += self.alpha * x

            tmp = np.sign(np.dot(X, self.weights))
            self.errorCounts[-1] = X.shape[0] - np.where(tmp==Y)[0].shape[0]
            if self.bestErrorCount > self.errorCounts[-1]:
                self.bestErrorCount = self.errorCounts[-1]
                self.bestWeights = copy.deepcopy(self.weights)
                self.bestIterationNo = iter
            iter += 1
            if self.errorCounts[-1] == 0:
                break
            if iter % self.section == 0:
                print("#",end = "")

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.sign(np.dot(X, self.weights))

        return iter
