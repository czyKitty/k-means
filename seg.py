import sys
import math
import time
import argparse

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from cImage import *
from kmeans import *
import kmedoids

def MSE(data,centres,cluster):
    MSE = 0
    for i in range(len(data)):
        MSE += np.sqrt(sum(np.power(data[i]-centres[int(cluster[i])],2)))
    return MSE

def bikmeans_MSE(data,centres,cluster):
    MSE = 0
    for i in range(len(centres)):
        for j in range(len(cluster[i])):
            MSE += np.sqrt(sum(np.power(data[cluster[i][j]]-centres[i],2)))
    return MSE

def kmedoid_MSE(data,centres,cluster):
    MSE = 0
    for i in range(len(centres)):
        for j in range(len(cluster[i])):
            MSE += np.sqrt(sum(np.power(data[cluster[i][j]]-data[centres[i]],2)))
    return MSE

def main():
    parser = argparse.ArgumentParser(description='Test kmeans algorithm.')
    parser.add_argument('imageFile', type=str, default='Earth.gif', help='Name of input graph')
    parser.add_argument('k', type=int, help='Number of clusters')
    parser.add_argument('-l', '--logFile', type=str, default='none', help='Name of log file (default:none)')
    parser.add_argument('-d', '--display', type=str, default='yes', help='Display image (default:yes)')
    
    args = parser.parse_args()
    # get sys input
    image = FileImage(args.imageFile)
    k = args.k
    
    width = image.getWidth()
    height = image.getHeight()
    
    # get data
    r = []
    g = []
    b = []
    for row in range(height):
        for col in range(width):
            pixel = image.getPixel(col,row)
            r.append([pixel[0]])
            g.append([pixel[1]])
            b.append([pixel[2]])
    pixels = np.hstack((r,g,b))

    nData = np.shape(pixels)[0]

    # perform k-mean
    print("--- Kmeans ---")
    start_time = time.time()
    c1 = kmeanstrain(pixels,k)
    cluster1 = kmeansfwd(pixels,k,nData,c1)
    t1 = time.time() - start_time
    MSE1 = MSE(pixels,c1,cluster1)/len(pixels)
    print("Time:",t1,"second")
    print("MSE:",MSE1)

    # perform kmeans-plus
    print("--- Kmeans-plus ---")
    start_time = time.time()
    c2 = k_plus(pixels,k)
    cluster2 = kmeansfwd(pixels,k,nData,c2)
    t2 = time.time() - start_time
    MSE2 = MSE(pixels,c2,cluster2)/len(pixels)
    print("Time:",t2,"second")
    print("MSE:",MSE2)

    # perform bi-kmeans
    print("--- bi-Kmeans ---")
    start_time = time.time()
    result = biKmeans(pixels,k)
    c3 = result[0]
    cluster3 = result[1]
    t3 = time.time() - start_time
    MSE3 = bikmeans_MSE(pixels,c3,cluster3)/len(pixels)
    print("Time:",t3,"second")
    print("MSE:",MSE3)

    t4 = "None"
    MSE4 = "None"
    if(len(pixels) < 900):
        print("--- Kmedoid ---")
        start_time = time.time()
        M,C = kmedoids.kMedoids(pairwise_distances(pixels, metric='euclidean'), k)
        t4 = time.time() - start_time
        MSE4 = kmedoid_MSE(pixels,M,C)/len(pixels)
        print("Time:",t4,"second")
        print("MSE:",MSE4)

    if args.logFile != 'none':
        fout = open(args.logFile,'w')
        fout.write("time\n")
        fout.write(str(t1)+"\n"+str(t2)+"\n"+str(t3)+"\n"+str(t4)+"\n")
        fout.write("MSE\n")
        fout.write(str(MSE1)+"\n"+str(MSE2)+"\n"+str(MSE3)+"\n"+str(MSE4))

    # create images
    image1 = EmptyImage(width,height)
    image2 = EmptyImage(width,height)
    for row in range(height):
        for col in range(width):
            p1 = c1[int(cluster1[row*width+col][0])]
            p2 = c2[int(cluster2[row*width+col][0])]
            image1.setPixel(col,row,Pixel(int(round(p1[0])),int(round(p1[1])),int(round(p1[2]))))
            image2.setPixel(col,row,Pixel(int(round(p2[0])),int(round(p2[1])),int(round(p2[2]))))

    image3 = EmptyImage(width,height)
    tempArr = pixels.tolist()
    for i in range(len(cluster3)):
        for j in range(len(cluster3[i])):
            p3 = c3[i]
            index = cluster3[i][j]
            image3.setPixel(index%width,index//width,Pixel(int(round(p3[0])),int(round(p3[1])),int(round(p3[2]))))

    if args.display == 'draw':
        # draw original image
        myimagewindow = ImageWin("newImage",4*width,height)
        
        image.setPosition(0,0)
        image1.setPosition(width,0)
        image2.setPosition(2*width,0)
        image3.setPosition(3*width,0)

        image.draw(myimagewindow)
        image1.draw(myimagewindow)
        image2.draw(myimagewindow)
        image3.draw(myimagewindow)
        myimagewindow.exitOnClick()

main()

