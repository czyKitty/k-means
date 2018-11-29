import sys
import math

import numpy as np
from cImage import *
from kmeans import *

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
    #print(pixels[0][0])

    nData = np.shape(pixels)[0]
    # perform k-mean
    c1 = kmeanstrain(pixels,k)
    cluster1 = kmeansfwd(pixels,k,nData,c1)
    #
    c2 = k_plus(pixels,k)
    cluster2 = kmeansfwd(pixels,k,nData,c2)

    #
    c3 = biKmeans(pixels,k)
    cluster3 = kmeansfwd(pixels,k,nData,c3)

    # draw original image
    myimagewindow = ImageWin("newImage",4*width,height)
    # draw new image

    image1 = EmptyImage(width,height)
    image2 = EmptyImage(width,height)
    image3 = EmptyImage(width,height)
    for row in range(height):
        for col in range(width):
            p1 = c1[int(cluster1[row*width+col][0])]
            p2 = c2[int(cluster2[row*width+col][0])]
            p3 = c3[int(cluster3[row*width+col][0])]
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

main()

