
# Code from Chapter 14 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import math

class kmeans:
    """ The k-Means algorithm"""
    def __init__(self,k,data):
        self.nData = np.shape(data)[0]
        self.nDim = np.shape(data)[1]
        self.k = k
    
    def k_plus(self,data,maxIterations=10):
        minima = data.min(axis=0)
        maxima = data.max(axis=0)
        # randomly select first center
        first_centre = np.random.rand(1,self.nDim)*(maxima-minima)+minima
        # pick the fartherst sample from the current center
        newCentres = []
        newCentres.append(first_centre)
        newCentres = np.asarray(newCentres)[0][0]
        numCentre = 1
        while numCentre < self.k:
            distance = []
            for item in data:
                tempD = []
                for c in newCentres:
                    tempD.append(np.sqrt(sum(np.power(item-c,2))))
                distance.append(sum(tempD))
            newCentres = np.vstack((newCentres,data[distance.index(max(distance))]))
            numCentre += 1

        # same as kmeans
        oldCentres = np.random.rand(2,self.nDim)*(maxima-minima)+minima
        centres = self.kmeansAlg(data,newCentres,oldCentres,2,maxIterations)
        return centres


    def biKmeans(self,data,maxIterations=10):
        centres = np.asarray([])
        return self.biKmeansRec(data,centres,self.k,maxIterations)
    
    def biKmeansRec(self,data,centres,k,maxIterations):
        # Find the minimum and maximum values for each feature
        minima = data.min(axis=0)
        maxima = data.max(axis=0)
        
        nDim = np.shape(data)[1]
        newCentres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
        oldCentres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
        # Array of centres
        print("nC:",newCentres)
        print("oC:",oldCentres)
        tempCentres = self.kmeansAlg(data,newCentres,oldCentres,self.k,maxIterations)
        print("nnC:",tempCentres)
        '''
        clusters = self.kmeansfwd(data,newCentres)
        print(clusters)
        
        if k >= 1:
            # For each centre, calculate and find the farthurest Euclid Distance
            tempD = 0-math.inf
            tempI = 0
            for i in range(2):
                if tempD < totalEuclid(newCentres[i],clusters[i]):
                    tempI = i
            return biKmeansRec(clusters[i],newCentres[i],k-1,maxIterations)
    
        else:
            return np.vstack((centres,newCentres))
        '''
    def kmeanstrain(self,data,maxIterations=10):
        nData = np.shape(data)[0]
        nDim = np.shape(data)[1]
        # Find the minimum and maximum values for each feature
        minima = data.min(axis=0)
        maxima = data.max(axis=0)
        
        newCentres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
        oldCentres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
        centres = self.kmeansAlg(data,newCentres,oldCentres,self.k,maxIterations)
        return centres

    def kmeansAlg(self,nData,nDim,data,newCentres,oldCentres,k,maxIterations):
        count = 0
        #print centres
        while np.sum(np.sum(oldCentres-newCentres))!= 0 and count<maxIterations:
            oldCentres = newCentres.copy()
            count += 1

            # Compute distances
            distances = np.ones((1,self.nData))*np.sum((data-newCentres[0,:])**2,axis=1)
            for j in range(self.k-1):
                distances = np.append(distances,np.ones((1,self.nData))*np.sum((data-newCentres[j+1,:])**2,axis=1),axis=0)
	
            # Identify the closest cluster
            cluster = distances.argmin(axis=0)
            cluster = np.transpose(cluster*np.ones((1,self.nData)))
	
            # Update the cluster centres
            for j in range(self.k):
                thisCluster = np.where(cluster==j,1,0)
                if sum(thisCluster)>0:
                    newCentres[j,:] = np.sum(data*thisCluster,axis=0)/np.sum(thisCluster)
        return newCentres
	
    def kmeansfwd(self,data,centres):
        nData = np.shape(data)[0]
        # Compute distances
        distances = np.ones((1,nData))*np.sum((data-centres[0,:])**2,axis=1)
        for j in range(self.k-1):
            distances = np.append(distances,np.ones((1,nData))*np.sum((data-centres[j+1,:])**2,axis=1),axis=0)
                
        # Identify the closest cluster
        cluster = distances.argmin(axis=0)
        cluster = np.transpose(cluster*np.ones((1,nData)))

        return cluster


