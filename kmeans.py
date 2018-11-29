import numpy as np
import math

def kmeansAlg(nData,nDim,data,newCentres,oldCentres,k,maxIterations):
    count = 0
    #print centres
    while np.sum(np.sum(oldCentres-newCentres))!= 0 and count<maxIterations:
        oldCentres = newCentres.copy()
        count += 1
            
        # Compute distances
        distances = np.ones((1,nData))*np.sum((data-newCentres[0,:])**2,axis=1)
        for j in range(k-1):
            distances = np.append(distances,np.ones((1,nData))*np.sum((data-newCentres[j+1,:])**2,axis=1),axis=0)
        
        # Identify the closest cluster
        cluster = distances.argmin(axis=0)
        cluster = np.transpose(cluster*np.ones((1,nData)))
            
        # Update the cluster centres
        for j in range(k):
            thisCluster = np.where(cluster==j,1,0)
            if sum(thisCluster)>0:
                newCentres[j,:] = np.sum(data*thisCluster,axis=0)/np.sum(thisCluster)
    return newCentres

def kmeansfwd(data,k,nData,centres):
    # Compute distances
    distances = np.ones((1,nData))*np.sum((data-centres[0,:])**2,axis=1)
    for j in range(k-1):
        distances = np.append(distances,np.ones((1,nData))*np.sum((data-centres[j+1,:])**2,axis=1),axis=0)
        
    # Identify the closest cluster
    cluster = distances.argmin(axis=0)
    cluster = np.transpose(cluster*np.ones((1,nData)))
    return cluster

def kmeanstrain(data,k,maxIterations=10):
    nData = np.shape(data)[0]
    nDim = np.shape(data)[1]
    # Find the minimum and maximum values for each feature
    minima = data.min(axis=0)
    maxima = data.max(axis=0)
    
    newCentres = np.random.rand(k,nDim)*(maxima-minima)+minima
    oldCentres = np.random.rand(k,nDim)*(maxima-minima)+minima
    centres = kmeansAlg(nData,nDim,data,newCentres,oldCentres,k,maxIterations)
    return centres

def k_plus(data,k,maxIterations=10):
    nData = np.shape(data)[0]
    nDim = np.shape(data)[1]
    # Find the minimum and maximum values for each feature
    minima = data.min(axis=0)
    maxima = data.max(axis=0)
    # randomly select first center
    first_centre = np.random.rand(1,nDim)*(maxima-minima)+minima
    # pick the fartherst sample from the current center
    newCentres = []
    newCentres.append(first_centre)
    newCentres = np.asarray(newCentres)[0][0]
    numCentre = 1
    while numCentre < k:
        distance = []
        for item in data:
            tempD = []
            for c in newCentres:
                tempD.append(np.sqrt(sum(np.power(item-c,2))))
            distance.append(sum(tempD))
        newCentres = np.vstack((newCentres,data[distance.index(max(distance))]))
        numCentre += 1

    # same as kmeans
    oldCentres = np.random.rand(k,nDim)*(maxima-minima)+minima
    centres = kmeansAlg(nData,nDim,data,newCentres,oldCentres,k,maxIterations)
    return centres

def biKmeans(data,k,maxIterations=10):
    centres = np.asarray([])
    return biKmeansRec(data,centres,k,maxIterations)
    
def biKmeansRec(data,centres,k,maxIterations):
    # Find the minimum and maximum values for each feature
    minima = data.min(axis=0)
    maxima = data.max(axis=0)
   
    nData = np.shape(data)[0]
    nDim = np.shape(data)[1]
    newCentres = np.random.rand(2,nDim)*(maxima-minima)+minima
    oldCentres = np.random.rand(2,nDim)*(maxima-minima)+minima
    # Array of centres
    tempCentres = kmeansAlg(nData,nDim,data,newCentres,oldCentres,2,maxIterations)

    clusters = kmeansfwd(data,2,nData,newCentres)
    c = [[],[]]
    for i in range(len(clusters)):
        if clusters[i][0] == 0:
            c[0].append(i)
        else:
            c[1].append(i)

    if k > 1:
        # For each centre, calculate and find the farthurest Euclid Distance
        tempD = 0-math.inf
        tempI = 0
        for i in range(2):
            if tempD < totalEuclid(newCentres[i],c[i],data):
                tempI = i
        return np.vstack((newCentres[1-i], biKmeansRec(newCluster(data,c[i]), newCentres[i], k-1, maxIterations)))
    else:
        return newCentres

def totalEuclid(centre,cluster,data):
    total = 0
    for i in range(len(cluster)):
        total += np.sqrt(sum(np.power(data[cluster[i]]-centre,2)))
    return total


def newCluster(data,cluster):
    # return the new data from given cluster
    newData = data[cluster[0]]
    for i in range(1,len(cluster)):
        newData = np.vstack((newData,data[cluster[i]]))
    return newData


