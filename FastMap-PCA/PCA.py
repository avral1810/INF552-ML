"""Aviral Upadhyay
Vandit Maheshwari"""


import numpy as np
import matplotlib.pyplot as plt


class PCA ():

    def __init__(self, datapoints=None, dimensions=0):   
        self.x = np.array(datapoints)
        self.k = dimensions
        self.covar = np.array
        self.mn_x = np.array  
        self.eigenvalue = np.array
        self.eigenvector = np.array
        self.sorted_eigenvalue = np.array
        self.sorted_eigenvector = np.array
        self.sorted_k_eigenvector = np.array
        self.z_k_T = np.array
        
    def mean_normalization(self, x):

        _mean = np.mean(x, axis=0)  
        self.mn_x = np.array(x - _mean)

        return self.mn_x
    
    def covariance_matrix(self, x):
        covar = np.array
        _x = np.array(x)

        covar = np.cov(_x.T)  
        self.covar = covar

        return self.covar
    
    def get_sorted_eigenvector_nk (self, covarience, k):
       
        eigenvalue = np.array
        eigenvector = np.array  
        _v = np.array     
        sorted_value_idx = np.array

        eigenvector, eigenvalue, _v = np.linalg.svd(covarience)

        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
        
        sorted_value_idx = np.argsort(-eigenvalue)  

        eigenvector = eigenvector[:, sorted_value_idx]  
        self.sorted_eigenvector = eigenvector
        
        eigenvalue = eigenvalue[sorted_value_idx]  
        self.sorted_eigenvalue = eigenvalue
        
        self.sorted_k_eigenvector = self.sorted_eigenvector[:, :k]  
        
        return self.sorted_k_eigenvector
    
    def k_dimension_projection(self, v, x):
        
        z_k = np.array
        
        z_k = v.T.dot(x.T)  
        
        return z_k
    
    def execute(self, datapoints, dimensions):
        covar = self.covar
        z_k = np.array
        nx = np.array  
        self.x = datapoints
        self.k = dimensions
        v_k = np.array 
        
        if(datapoints is None):
            return None
        else :        
            nx = self.mean_normalization(self.x)
            covar = self.covariance_matrix(nx)
            v_k = self.get_sorted_eigenvector_nk(covar, self.k)               
            z_k = self.k_dimension_projection(v_k, nx)
            self.z_k_T = z_k.T
            return self.z_k_T
            
    
    def getInputData(self,filename):
        data = np.genfromtxt(filename, delimiter='\t')
        return data    