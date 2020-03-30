"""Aviral Upadhyay
Vandit Maheshwari"""



import numpy as np
import matplotlib.pyplot as plt
from FastMap import FastMap
from PCA import PCA






if __name__ == '__main__':
    pca = PCA()    
    fm = FastMap()
    dimension = 2
    datapoints = pca.getInputData('pca-data.txt')   
    object_pair, distance = fm.getInputData('fastmap-data.txt')
    label_set = fm.getObjectName('fastmap-wordlist.txt')
    
    z_k = pca.execute(datapoints, dimension)
    # print ('pca.covar=', pca.covar)
    # print('pca.eigenvalue.T=', pca.eigenvalue.T)
    print('\npca.sorted k eigenvector  \n', pca.sorted_k_eigenvector.T) #Transform the format to original format

    print('Dimensions reduce to (', dimension, '), and results as below:\n', z_k) 

    
     
    
    k_d_distance = fm.execute(object_pair, distance, dimension)
    
    print ('The ', dimension, ' dimensions result = \n', k_d_distance)
    
    fm.plot(label_set)


"""STANDARD LIB"""
from sklearn.decomposition import PCA as stanPCA
data = np.genfromtxt('pca-data.txt',delimiter = '\t')
stanpca = stanPCA(n_components = 2)
projection = stanpca.fit_transform(data)
print("Standard Lib o/p\n",projection)