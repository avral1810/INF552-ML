from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numpy.linalg import inv

from Perceptron import Perceptron
from Pocket import Pocket
from LogisticRegression import LogisticRegression
from LinearRegression import LinearRegression

maxIter = 7000


#standard Lib

def SKPerceptron(X,Y):
    from numpy import loadtxt, where
    from sklearn.linear_model import Perceptron

    alpha = float(0.001)
    print("SciKit-Learn Perceptron with alpha ", alpha, "and " ,maxIter," iterations")
    model = Perceptron(alpha=alpha, verbose=False)
    model.n_iter = 7000
    model.fit(X, Y)

    YPred = model.predict(X)

    nCorrect = where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = (nCorrect / nTotal)*100


    print('Final Weights: {0}'.format(model.coef_))
    print('\n\t\t'.join(['Accuracy on the train dataset: {0}%', 'Predicted Correctly: {1}', 'Total Samples: {2}']).format(accuracy, nCorrect, nTotal))

def perceptronMain():
    alpha = float(0.001)
    print("Perceptron with alpha ", alpha, "and " ,maxIter," iterations")
    data = np.loadtxt("classification.txt", delimiter=',', dtype='float',usecols=(0,1,2,3))
    X = data[:, :-1]
    Y = data[:, -1]

    model = Perceptron(alpha=alpha, maxIter=maxIter)
    nIterations = model.train(X, Y)

    YPred = model.predict(X)
    nCorrect = np.where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = (nCorrect / nTotal) * 100

    print('Final Weights [W0, W1, W2,...]: {0}'.format(model.weights))
    print('\n\t\t'.join(['Accuracy on the train dataset: {0}%', 'Predicted Correctly: {1}','Total Samples: {2}']).format(accuracy, nCorrect, nTotal))
    print("\n\nSTANDARD LIB PERCEPTRON\n\n")
    SKPerceptron(X,Y)



def pocketMain():
    alpha = 0.01
    print("Pocket with alpha ", alpha, "and " ,maxIter," iterations")
    data = np.loadtxt("classification.txt", delimiter=',',dtype='float',usecols=(0,1,2,4))
    X = data[:, :-1]
    Y = data[:, -1]

    model = Pocket(alpha=alpha, maxIter=maxIter)
    nIterations = model.train(X, Y)

    YPred = model.predict(X)
    nCorrect = np.where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = (nCorrect / nTotal) * 100

    print('Best Weights: {0}'.format(model.bestWeights))
    print('Best/Least No. of Violations: {0}'.format(model.bestErrorCount))
    print('\n\t\t'.join(['Accuracy on the train dataset: {0}%', 'Predicted Correctly: {1}', 'Total Samples: {2}']).format(accuracy, nCorrect, nTotal))

    plt.ylabel('No. of violations')
    plt.xlabel('iteration')
    plt.plot(model.errorCounts)
    plt.show()



def SKLogisticRegression(X,Y):
    from numpy import loadtxt, where
    from sklearn.linear_model import LogisticRegression
    print("SciKit-Learn Logistic regression ")
    model = LogisticRegression()
    model.fit(X, Y)

    YPred = model.predict(X)
    nCorrect = where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = (nCorrect / nTotal) * 100

    print('Final Weights: {0}'.format(model.coef_))
    print('\n\t\t'.join(['Accuracy on the train dataset: {0}%', 'Predicted Correctly: {1}', 'Total Samples: {2}']).format(accuracy, nCorrect, nTotal))

def logisticRegressionMain():
    alpha = 0.05
    print("Logistic regression with alpha ", alpha, "and " ,maxIter," iterations")
    data = np.loadtxt('classification.txt',delimiter=',',dtype='float',usecols=(0,1,2,4))

    X = data[:, :-1]
    Y = data[:, -1]

    model = LogisticRegression(alpha=alpha, maxIter=maxIter)
    nIterations = model.train(X, Y)

    YPred = model.predict(X)
    nCorrect = np.where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = (nCorrect / nTotal) * 100

    print('No. of Iterations: {0}'.format(nIterations))
    print('Final Weights [W0, W1, W2,...]: {0}'.format(model.weights))
    print('\n\t\t'.join(['Accuracy on the train dataset: {0}%', 'Predicted Correctly: {1}', 'Total Samples: {2}']).format(accuracy, nCorrect, nTotal))

    print("\n\nSTANDARD LIB LOGISTIC REGRESSION\n\n")
    SKLogisticRegression(X,Y)



def SKLinearRegression():
    print("SciKit-Learn Linear regression ")
    from numpy import loadtxt
    from sklearn.linear_model import LinearRegression
    data = np.loadtxt("linear-regression.txt", delimiter=',', dtype='float',usecols=(0,1,2))
    X = data[:, :-1]
    Y = data[:, -1]
    model = LinearRegression()
    model.fit(X, Y)

    print('Final Weights: {0}'.format(model.coef_))

def linearRegressionMain():
    print("Linear regression ")
    data=pd.read_csv("linear-regression.txt",header=None)
    model = LinearRegression(data)
    model.run()
    print("\n\nSTANDARD LIB LINEAR REGRESSION\n\n")
    SKLinearRegression()


def main():
    print("\n\n\n")
    perceptronMain()
    print("\n\n\n")
    logisticRegressionMain()
    print("\n\n\n")
    linearRegressionMain()
    print("\n\n\n")
    pocketMain()



main()


##Standard Library








