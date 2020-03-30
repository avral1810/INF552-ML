from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, weights=[], alpha=0.001, maxIter=7000):
        self.weights = weights
        self.alpha = alpha
        self.maxIter = maxIter
        self.errorCounts = []
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
            iter += 1
            if self.errorCounts[-1] == 0:
                break
            if iter % self.section == 0:
                print("#",end = "")

        return iter

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.sign(np.dot(X, self.weights))