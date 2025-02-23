import numpy as npy
from sklearn.preprocessing import MinMaxScaler

def squashingSigmoidFunc(xParameter):
    sigmoidResult = 1 / (1 + npy.exp(-xParameter))
    return sigmoidResult

def squashingSigmoidDiff(yParameter):
    sigmoidDiffResult = yParameter * (1 - yParameter)
    return sigmoidDiffResult

def tangentFunc(x):
    tangentVal = npy.tanh(x)
    return tangentVal

def tangentFuncDiff(y):
    tangentDiffVal = 1 - y * y
    return tangentDiffVal

def getArray(parameter):
    arrayRes = npy.array(parameter)
    return arrayRes


def getSequences(df, winSize):
    X = []
    Y = []
    len = df.shape[0]
    itr =1;
    while(itr <= len):
        if itr + winSize + 1 > df.index[-1]:
            break
        locX = df.iloc[itr:itr+winSize]
        locY = df.iloc[itr+winSize+1]
        X.append(locX)
        Y.append(locY)
        arrayX = getArray(X)
        arrayY = getArray(Y)
        itr +=1
    return arrayX, arrayY

def split(X, Y, r):
    len = X.shape[0]
    splitData = int(len*r)
    X1 = X[:splitData]
    X2 = X[splitData:]
    Y1 = Y[:splitData]
    Y2 = Y[splitData:]
    return X1, X2, Y1, Y2

def getRandomValue(frame, sdWeight, var,d):
    frameX = frame[0]
    frameY = frame[1]
    if(d == 30):
      val = npy.random.rand(frameX, frameY) * sdWeight + var
    return val

def getNumpyZerosLike(frame):
    arr = npy.zeros_like(frame)
    return arr
  
def getNumpyZeros(layr, val):
    arr = npy.zeros((layr, val))
    return arr

def stackArray(vecA,vecB):
    stack = npy.vstack((vecA, vecB))
    return stack

def getIpData(df):
    IpData = df.iloc[:, [1]]
    return IpData

def readCSV(url):
    file = pds.read_csv(url)
    return file

sclr = MinMaxScaler()
def fitScalerData(IpData):
    sclr.fit(IpData);
def transformData(IpData):
    IpData = sclr.transform(IpData)
    return IpData
def downloadReadings(Readings):
    Readings.to_csv('Readings.csv', index=False)
    return True