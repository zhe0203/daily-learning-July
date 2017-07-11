# -*- coding: utf-8 -*-
'''
优点：泛化错误率低，计算开销不大，结果容易理解
缺点：对参数调节和核函数的选择敏感，原始分类器不加修改近使用于处理二分类
适用数据类型：数值型和标称型数据（离散数据）
'''

# SMO算法中的辅助函数
import numpy as np
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 其中i是第一个alpha的下标，m是所有alpha的数目。只要函数值不等于输入值i，函数就会进行随机选择。
def selectJrand(i,m):
    j = i
    while (j == i):
        i = int(np.random.uniform(0,m))
    return j

# 用于调整大于H或小于L的alpha的值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 加载数据
dataArr,labelArr = loadDataSet('testSet.txt')
# print(labelArr)
# print(dataArr)

# SMO算法
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(o,alphas[j] - alphas[i])
                    H = min(C,C+alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[i] + alphas[i])
                if L == H:
                    print('L==H')
                    continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei-Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*\
                             (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold) * \
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold) * \
                    dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i] and (C > alphas[i])):
                    b = b1
                elif (0 < alphas[j] and (C > alphas[j])):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print("iter:%d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number:%d' % iter)
    return b,alphas

b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
print(b)
print(alphas[alphas>0])



# 绘制支持向量
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import  Circle

xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers = []
colors = []
fr = open('testSet.txt')
for line in fr.readlines():
    lineSplit = line.strip().split('\t')
    xPt = float(lineSplit[0])
    yPt = float(lineSplit[1])
    label = int(lineSplit[2])
    if label == -1:
        xcord0.append(xPt)
        ycord0.append(yPt)
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)
fr.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0,ycord0,marker='s',s=90)
ax.scatter(xcord1,ycord1,marker='o',s=50,c='red')
plt.title('Support Vectors Circled')
# 圈出支持向量的点
circle = Circle((4.6581910000000004, 3.507396), 0.3, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((3.4570959999999999, -0.082215999999999997), 0.3, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((6.0805730000000002, 0.41888599999999998), 0.3, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
b = -3.75567
w0 = 0.8065
w1 = -0.2761
x = arange(-2.0,12.0,0.1)
y = (-w0*x-b)/w1
ax.plot(x,y)
ax.axis([-2,12,-8,6])
plt.show()

# 完整版的Platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

# 对数据进行分类
def calcWs(alphas,dataArr,classLabels):
    x = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(x)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],x[i,:].T)
    return w
# 对于每一个数据点
ws = calcWs(alphas,dataArr,labelArr)
datMat = mat(dataArr)
print(datMat[0]*mat(ws)+b) # 其值大于0属于1类，小于0属于-1类
