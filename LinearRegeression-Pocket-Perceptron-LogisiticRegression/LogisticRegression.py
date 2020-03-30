import numpy as np
import math

class LogisticRegression():
    @staticmethod
    def findGradient(weights, x, y):
        arg = y * np.dot(weights, x)
        tmp = 1 + np.exp(arg)
        res = (x * y) / tmp
        return res
    
    @staticmethod
    def findProb(weights, x, y):
        arg = y * np.dot(weights, x)
        tmp = np.exp(arg)
        return tmp / (1 + tmp)


    def __init__(self, weights=[], alpha=0.001, maxIter=1000):
        self.weights = weights
        self.alpha = alpha
        self.maxIter = maxIter
        self.section = maxIter/14
    
    def train(self, X, Y):
        N, d = X.shape
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.random(d+1)
        iter = 0
        while iter < self.maxIter:
            gradient = np.zeros(d+1)
            for x, y in zip(X, Y):
                gradient = np.add(gradient, LogisticRegression.findGradient(self.weights, x, y))
            gradient /= N
            self.weights += self.alpha * gradient
            iter += 1
            if iter % self.section == 0:
                print("#",end = "")
        return iter

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        Y = [None for _ in range(X.shape[0])]
        for idx, x in enumerate(X):
            prob_1 = LogisticRegression.findProb(self.weights, x, 1)
            if prob_1 > 0.5:
                Y[idx] = 1
            else:
                Y[idx] = -1

        return np.asarray(Y)


