import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


class LinearRegression:
    def __init__(self,data):
        self.data = data


    def linearregression(self,X, y):
        return inv(X.T.dot(X)).dot(X.T).dot(y)

    def run(self):
        y=[]
        for item in self.data:
            y=self.data[2]
            x=list(zip(self.data[0],self.data[1]))
        X= np.matrix(x)
        X = np.column_stack((np.ones(len(X)), X))
        p=self.linearregression(X,y)
        print('Final Weights [W0, W1, W2,...]: {0}'.format(p))
