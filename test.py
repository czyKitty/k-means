import sys
import math

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from cImage import *
from kmeans import *
import kmedoids


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
    print(M)
    print(C)
    '''
    # draw original image
    myimagewindow = ImageWin("newImage",2*width,height)
    # draw new image

    image1 = EmptyImage(width,height)
    for row in range(height):
        for col in range(width):
            p1 = c1[int(cluster1[row*width+col][0])]
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

