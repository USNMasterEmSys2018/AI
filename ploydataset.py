# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 14:38:57 2018

@author: liubo
"""


from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D

            
class ploydataset:
    def __init__(self, m: int = 1):
        self.__n = 200*m+1
        self.a = numpy.linspace(-m,m,self.__n)

    def generatedata(self):
        x = self.a
        self.feature = numpy.random.rand(2)*2-1
        #self.feature = [0.5,0.5]
        print(self.feature)
        b = numpy.random.rand(self.__n)*2-1 #noise
        var=[[1]*self.__n,x]
        y = numpy.matmul(self.feature,var) + b
        self.var = var
        self.y = y

def fetchdesentdataset(var: list, dataset):
    X = numpy.linspace(-3,3,101)
    X = (numpy.meshgrid(X, X))
    X =  numpy.transpose(X)
    
    H = [[[]*len(var)]*len(X[0])]*len(X[0])
    for i in range(len(X)):
        for j in range(len(X[i])):
            H[i][j] = list(numpy.matmul(X[i][j],var))
            
    J = numpy.array([numpy.array([0]*len(X[0]))]*len(X[0]))
    for i in range(len(X)):
        for j in range(len(X[i])):
            J[i][j] = numpy.matmul([1]*(len(dataset)),numpy.transpose((abs(numpy.matmul(X[i][j],var)-dataset)**2)/2))
            
    X = numpy.transpose(X)     
    return X, J

def f(feature: list, vari: list, dataset: list):
    h = numpy.matmul(feature,vari)
    dj = numpy.matmul(vari,(h-dataset))#(h-data.z)*vari
    #print(dj)
    j = numpy.matmul([1]*len(h),(h-dataset)**2/2)
    return j, dj,h

def plotfigure(feature: list, var: list, dataset):
    plt.figure(2)
    plt.plot(var[1], dataset,"b*")
    y = numpy.transpose(numpy.matmul(feature, var))
    #print(numpy.array(feature).shape)
    plt.plot(var[1], y, "r")
    plt.show()
    
def plot3d(var: list, feature: list, dataset, X,J):
    #plotfigure(feature, var, dataset)
    plt.figure(1)
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    j = []*len(feature)
    #print()
    H = numpy.matmul(feature,var)
    j = numpy.matmul([1]*len(var[0]),numpy.transpose(abs(H-[dataset])**2/2))
    feature = numpy.transpose(feature)
    surf = ax.plot_surface( X[0], X[1], J, cmap=cm.coolwarm)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.scatter(feature[0],feature[1],j, c="r")
    plt.show() 