import sys
import math

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from cImage import *
from kmeans import *
import kmedoids

'''
def totalEuclid(centre,data):
    total = 0
    for i in range(len(centre)):
        for j in range(len(data[i])):
            array = np.asarray(data[i][j])
            print("array:")
            total += np.sqrt(sum(np.power(array-centre[i],2)))
    return total
'''
def kmedoid_MSE(data,centres,cluster):
    MSE = 0
    for i in range(len(centres)):
        for j in range(len(cluster[i])):
            MSE += np.sqrt(sum(np.power(data[cluster[i][j]]-data[centres[i]],2)))
    return MSE

def main():
    # get sys input
    image = FileImage(sys.argv[1])
    k = int(sys.argv[2])
    
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

    #nData = np.shape(pixels)[0]
    # perform k-mean
    #c1 = kmeanstrain(pixels,k)
    #cluster1 = kmeansfwd(pixels,k,nData,c1)
    M,C = kmedoids.kMedoids(pairwise_distances(pixels, metric='euclidean'), k)
    #print(M)
    #print(C)
    print(kmedoid_MSE(pixels,M,C))

    '''
    # draw original image
    myimagewindow = ImageWin("newImage",2*width,height)
    # draw new image

    image1 = EmptyImage(width,height)
    for i in range(len(M)):
        #print(i)
        for j in range(len(C[i])):
            index = C[i][j]
            p = pixels[M[i]]
            #print(index%height,index//width,p[0])
            image1.setPixel(index%height,index//width,Pixel(p[0],p[1],p[2]))
    #print("END")
    image.setPosition(0,0)
    image.draw(myimagewindow)
    image1.setPosition(width,0)
    image1.draw(myimagewindow)
    myimagewindow.exitOnClick()
    
    for row in range(height):
        for col in range(width):
            #p1 = c1[int(cluster1[row*width+col][0])]
            image1.setPixel(col,row,Pixel(int(round(p1[0])),int(round(p1[1])),int(round(p1[2]))))
            image2.setPixel(col,row,Pixel(int(round(p2[0])),int(round(p2[1])),int(round(p2[2]))))
            image3.setPixel(col,row,Pixel(int(round(p3[0])),int(round(p3[1])),int(round(p3[2]))))

    image.setPosition(0,0)
    image.draw(myimagewindow)
    image1.setPosition(width,0)
    image1.draw(myimagewindow)
    image2.setPosition(2*width,0)
    image2.draw(myimagewindow)
    image3.setPosition(3*width,0)
    image3.draw(myimagewindow)
    myimagewindow.exitOnClick()
    '''
main()

