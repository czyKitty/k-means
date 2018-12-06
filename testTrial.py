import os
import sys

def loop(img,k):
    time = [0,0,0,0]
    MSE = [0,0,0,0]
    testFile = open("k="+k+".txt",'w')
    for i in range(20):
        os.system("python3 seg.py "+img+" "+k+" -l test.txt"+" -d no")
        dataFile = open("test.txt",'r')
        line = dataFile.readline()
        while line != '':
            if line == "time\n":
                for j in range(4):
                    line = dataFile.readline()
                    time[j] += float(line[:-1])
            line = dataFile.readline()
            if line == "MSE\n":
                for j in range(4):
                    line = dataFile.readline()
                    MSE[j] += float(line[:-1])

    testFile.write("time\n")
    for i in range(4):
        testFile.write(str(time[i]/20)+"\n")
    testFile.write("MSE\n")
    for i in range(4):
        testFile.write(str(MSE[i]/20)+"\n")
    testFile.close()

def main():
    for i in range(2,20):
        loop(sys.argv[1],str(i))
main()
