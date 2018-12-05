import numpy as np
import math

'''
Original K-means algorithm
'''
def kmeansAlg(nData,nDim,data,newCentres,oldCentres,k,maxIterations):
    count = 0
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

'''
K-means algorithm with initialization
'''
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

'''
K-means algorithm with bisection
'''
def biKmeans(data,k,maxIterations=10):
    #print(data)
    if k < 2:
        return (data, np.zeros((len(data),1)))
    else:
        minima = data.min(axis=0)
        maxima = data.max(axis=0)
        
        nData = np.shape(data)[0]
        nDim = np.shape(data)[1]
        newCentres = np.random.rand(2,nDim)*(maxima-minima)+minima
        oldCentres = np.random.rand(2,nDim)*(maxima-minima)+minima
    
        centres = kmeansAlg(nData,nDim,data,newCentres,oldCentres,2,maxIterations)

        oldCluster = []
        for i in range(len(data)):
            oldCluster.append(i)
        # @newData = [cluster1Data, cluster2Data, etc.]
        clusterData = getCluster(oldCluster,kmeansfwd(data,2,nData,centres))

        #print(data)
        #print(clusterData)
        #print(centres)
        if k == 2:
            return (centres,clusterData)
        else:
            return biKmeansRec(data,clusterData,centres,k-1,maxIterations)

#@clusterData list of index in data [[1,2,3][4,5,6]...], @centers 3*n ndarray
def biKmeansRec(data,clusterData,centres,k,maxIterations):
    #print("clusterData:",clusterData)
    #print("centres:",centres)
    dist = 0-math.inf
    index = 0
    #for i in range(len(newData)):
    for i in range(len(clusterData)):
        # data = 3*n ndarray, clusterData = [[1,2,3][4,5,6]...]
        # newData = 3*n ndarray of cluster[i]
        # need to have at least 2 objects in the cluster to split
        if len(clusterData[i]) > 1:
            newData = getData(data,clusterData[i]) #newData = list of 3*n ndarray
        #print("newD:",newData)
            tempDist = totalEuclid(centres[i],np.asarray(newData))
        #print("cur")
            if tempDist > dist:
                dist = tempDist
                index = i

    # get data in the choosen cluster
    # @newData = 3*n ndarray of cluster[index]
    newData = getData(data,clusterData[index])
    oldCluster = clusterData[index]
    
    #delete oldCluster
    del clusterData[index]

    #delete old center
    centres = np.delete(centres,(index),axis=0)

    #perform kmeans with k=2 for new data
    minima = newData.min(axis=0)
    maxima = newData.max(axis=0)
    
    nData = np.shape(newData)[0]
    nDim = np.shape(newData)[1]
    newCentres = np.random.rand(2,nDim)*(maxima-minima)+minima
    oldCentres = np.random.rand(2,nDim)*(maxima-minima)+minima

    newCentres = kmeansAlg(nData,nDim,newData,newCentres,oldCentres,2,maxIterations)
    #add new centre
    centres = np.vstack((centres,newCentres))
    
    #get new cluster with in cluster
    newC = kmeansfwd(newData,2,nData,newCentres)
    #print("cluster:",newC)
    #newClusters = getCluster(newData,kmeansfwd(newData,2,nData,newCentres))
    
    #get index of objects in newCluster
    newClusters = getCluster(oldCluster,newC)
    clusterData.append(newClusters[0])
    clusterData.append(newClusters[1])

    if k > 2:
        return biKmeansRec(data,clusterData,centres,k-1,maxIterations)
    else:
        return (centres,clusterData)

def totalEuclid(centre,data):
    total = 0
    for i in range(len(data)):
        total += np.sqrt(sum(np.power(data[i]-centre,2)))
    return total

def getCluster(clusterData,cluster):
    c0 = []
    c1 = []
    for i in range(len(cluster)):
        if cluster[i][0] == 0:
            c0.append(clusterData[i])
        else:
            c1.append(clusterData[i])
    return [c0,c1]

def getData(data,cluster):
    newData = data[cluster[0]]
    for i in range(1, len(cluster)):
        newData = np.vstack((newData,data[cluster[i]]))
    return newData




