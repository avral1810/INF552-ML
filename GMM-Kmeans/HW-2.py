# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:11:44 2020

@author: Aviral Upadhyay
        Vandit Maheshwari
"""

import numpy as np
from collections import defaultdict
from numpy import genfromtxt
import random
import matplotlib.pyplot as plt


class kmeans:
    
    def __init__(self,total_clusters=3,centroids=None,total_iter=500,threshold=0.01):
        self.total_clusters=total_clusters
        self.centroids=centroids
        self.total_iter=total_iter
        self.threshold=threshold
        
    def fit(self,data=None):
        i=1
        centroids = np.array(random.sample(list(data),3))
        while True:
            new_centroids = np.array(self.compute_centroids(data,centroids))
            if np.array_equal(new_centroids,centroids):
                break
            if (np.abs(new_centroids,centroids) < self.threshold).all():
                break
            if i>self.total_iter:
                break
            i+=1
            centroids=new_centroids
        return new_centroids
    
    def distance(self,x,y):
        return np.sqrt(np.sum((x-y)**2))
    
    def compute_centroids(self,data,centroids):
        dict_clusters=defaultdict(list)
        dict_clusters_indices = defaultdict(list)
        for i in range(0,len(data)):
            val, min_index = min((val,idx) for (idx,val) in enumerate([self.distance(data[i],x) for x in centroids]))
            dict_clusters_indices[min_index].append(i)
            dict_clusters[min_index].append(data[i].tolist())
        new_centroids = []
        for j in dict_clusters:
            new_centroids.append(np.array(dict_clusters[j]).mean(axis=0))
        
        return new_centroids
    


class GMM:
    def __init__(self,num_of_clusters=3,max_num_of_iter=200,means=[],cov_vars=[],amplitude=[],threshold=0.01):
        self.num_of_clusters=num_of_clusters
        self.max_num_of_iter=max_num_of_iter
        self.means=means
        self.cov_vars=cov_vars
        self.amplitude=amplitude
        self.threshold=threshold
    
    def fit(self,data=None):
        iter=1
        membership_weights=None
        while True:
            self.mstep(data,membership_weights)
            new_membership_weights=self.estep(data)
            if membership_weights is not None and new_membership_weights is not None:
                if (np.abs(new_membership_weights-membership_weights) < self.threshold).all():
                    break
            if iter >= self.max_num_of_iter:
                break
            iter+=1
            membership_weights=new_membership_weights
        return
    def kmeans(self,data):
        cluster_centroids=np.array(random.sample(list(data), 3))
        dict_of_clusters=defaultdict(list)
        for i in range(0,len(data)):
            val, min_index = min((val, idx) for (idx, val) in enumerate([self.euclidean_dist(data[i],x) for x in cluster_centroids]))
            dict_of_clusters[min_index].append(data[i].tolist())
        dict_of_clusters=[np.array(dict_of_clusters[i]) for i in dict_of_clusters]
        return dict_of_clusters
    
    def euclidean_dist(self,x,y):
        return np.sqrt(np.sum((x-y)**2))
    
    def mstep(self,data,membership_weights=None):
        self.means=[]
        self.cov_vars=[]
        self.amplitude = []
        num_of_var=len(data[0])
        num_of_data_points=len(data)
        sum=np.sum
        if membership_weights is not None:
            self.amplitude = sum(membership_weights, axis=0) / num_of_data_points
            for i in range(0,self.num_of_clusters):
                self.means.append(sum(np.multiply(data,membership_weights[:,i].reshape(len(data),1)),axis=0) / sum(membership_weights[:,i]))
                cov_temp_sum=np.zeros((num_of_var,num_of_var))
                for j in range(0,num_of_data_points):
                    temp=data[j]-self.means[i]
                    temp=np.dot(temp.T.reshape(num_of_var,1),temp.reshape(1,num_of_var))
                    temp=temp*membership_weights[j][i]
                    cov_temp_sum=np.add(cov_temp_sum,temp)
                cov_temp_sum=cov_temp_sum / sum(membership_weights[:,i])
                self.cov_vars.append(cov_temp_sum)
        else:
            clusters=self.kmeans(data)
            self.amplitude=np.ones(self.num_of_clusters) / self.num_of_clusters
            self.amplitude=self.amplitude.tolist()
            for i in range(0,self.num_of_clusters):
                self.means.append(np.mean(clusters[i], axis=0))
                self.cov_vars.append(np.cov(clusters[i].T))
        return 
    
    def estep(self,data):
        num_of_data_points=len(data)
        pdfs = np.empty([num_of_data_points,self.num_of_clusters])
        for i in range(0,self.num_of_clusters):
            m=self.means[i]
            cov=self.cov_vars[i]
            invcov=np.linalg.inv(cov)
            norm_factor = 1 / np.sqrt((2*np.pi)**2 * np.linalg.det(cov))
            for row in range(0,num_of_data_points):
                temp = data[row,:] - m
                temp = temp.T
                temp = np.dot(-0.5*temp, invcov)
                temp = np.dot(temp, (data[row,:] - m))
                pdfs[row][i] = norm_factor*np.exp(temp)
        membership_weights = np.empty([num_of_data_points,self.num_of_clusters])
        for i in range(0,num_of_data_points):
            denominator=np.sum(self.amplitude*pdfs[i])
            for j in range(0,self.num_of_clusters):
                membership_weights[i][j]=self.amplitude[j]*pdfs[i][j] / denominator
        return membership_weights
        


filename='clusters.txt'
X = genfromtxt(filename, delimiter=',')

def main():
        
        clf=kmeans()
        centroids=clf.fit(X)
        print("Kmeans\n\n")
        print(centroids)
        print("\n\n")
        g=GMM(num_of_clusters=3)
        g.fit(X)
        print("Amplitudes:")
        print(g.amplitude)
        print("means:")
        print(np.array(g.means))
        print("Covariance Matrix:")
        print(np.array(g.cov_vars))        

def standardLibKmeans():
    colors = 2*["r.", "g.", "c.", "b.", "y."]
    clusters = 3
    trials = 10
    from sklearn.cluster import KMeans
    print ("X.shape="+str(X.shape))
    print ('===================')
    print ('     K-Means')
    print ('===================')
    km = KMeans(n_clusters=clusters, n_init=trials, algorithm ='full')
    km.fit(X)

    centroids = km.cluster_centers_
    labels = km.labels_

    plt.xlabel("x-axis") 
    plt.ylabel("y-axis") 
    # Max 10 clusters can be marked by 5 different colors
    colors = 2*["r.", "g.", "c.", "b.", "y."] 

    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=20, linewidths=10)
    plt.show()

    print ('centroids=', centroids, '\n\n\n')


def standardLibGMM():
    colors = 2*["r.", "g.", "c.", "b.", "y."]
    clusters = 3
    trials = 10
    from sklearn import mixture
    print ('===================')
    print ('     EM-GMM')
    print ('===================')
    gmm = mixture.GaussianMixture(n_components=clusters, n_init=trials, covariance_type="full")
    gmm.fit(X)
    labels = gmm.predict(X)
    weights = gmm.weights_
    means = gmm.means_
    n_cov = gmm.covariances_

    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

    plt.scatter(means[:,0], means[:,1], marker='x', s=20, linewidths=10)
    plt.show()

    print ('GMM weights:', weights)
    print ('GMM means:', means)
    print ('GMM covars: components=', n_cov)


main()
standardLibKmeans()
standardLibGMM()