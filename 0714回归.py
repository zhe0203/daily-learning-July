# -*- coding: utf-8 -*-
'''
优点：结果易于理解，计算上不复杂
缺点：对非线性的数据拟合不好
适用数据类型：数值型和标称型  对于标称型数据将被转成二值型数据
'''
# 标准回归函数和数据导入函数
from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  # 计算特征的个数
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print('this matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 加载数据
xArr,yArr = loadDataSet('ex0.txt')
print(xArr[0:5],yArr[0:5])
ws = standRegres(xArr,yArr)
print(ws)

# 绘图
xMat = mat(xArr)
yMat = mat(yArr)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten(),yMat.T[:,0].flatten().A[0],alpha=0.5)

# 绘制拟合曲线
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:,1],yHat,'r--',linewidth=2)
plt.show()

# 计算拟合值和真实值的相关系数
# print(yHat)  # 需要对其进行转置
# print(yMat)
print(corrcoef(yHat.T,yMat))

#################### 局部加权回归 ################
'''
给定数据空间中的任意一点testPonit计算出对应的预测值yHat,创建对角权重矩阵，阶数等于样本点的个数
'''
# 对于其中的每一个样本点都进行如下的计算
def lwlr(testPoint,xArr,yArr,k=1.0):   # 参数k为控制衰减速度
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))    #　创建对角矩阵，用来给每个样本点赋予权重

    # 遍历每个样本点
    for j in range(m):
        # 随着样本点与待预测点之间距离的递增，权重以指数级递减
        diffMat = testPoint - xMat[j,:]    # 计算样本点与预测点之间的距离
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k **2))  # 高斯核
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('this matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))   # 计算估计系数OLS
    return testPoint * ws   # 计算待测点的预测值

# 对于所有的样本点使用局部加权回归进行预测
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):    # 用于为每一个数据点调用lwlr函数
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

xArr,yArr = loadDataSet('ex0.txt')
# 为了得到数据集里所有点的估计，可以调用lwlrTest函数
yHat = lwlrTest(xArr,xArr,yArr,0.01)

# 绘图
fig = plt.figure()
i = 0
for k in [1,0.01,0.003]:
    yHat = lwlrTest(xArr,xArr,yArr,k)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)   # 返回数据排序大小的位置
    xSort = xMat[srtInd][:,0,:]     # 返回排序后的数据

    ax = fig.add_subplot(311 + i)   # 绘制子图
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    ax.axis([0,1,3,5])
    i += 1

plt.show()

############## 示例
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()
abX,abY = loadDataSet('abalone.txt')

yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
print(rssError(abY[0:99],yHat01.T))
print(rssError(abY[0:99],yHat1.T))
print(rssError(abY[0:99],yHat10.T))

################### 预测 #################
'''使用已经建立好的模型，在新的数据集上进行预测'''
yHat01 = lwlrTest(abX[100:103],abX[0:99],abY[0:99],0.1)
print('k=0.1:',rssError(abY[100:103],yHat01.T))
yHat1 = lwlrTest(abX[100:103],abX[0:99],abY[0:99],1)
print('k=1:',rssError(abY[100:103],yHat1.T))
yHat10 = lwlrTest(abX[100:103],abX[0:99],abY[0:99],10)
print('k=10:',rssError(abY[100:103],yHat10.T))

############### 岭回归 ##############
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('this matrix is singular,cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    # 数据标准化
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat= (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat


ridgeWeights = ridgeTest(abX,abY)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()

########## lasso ###########
# 前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))   # 返回权重矩阵
    ws = zeros((n,1))     # 初始化权重为0
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):   # 进行迭代次数
        print(ws.T)  # 打印权重矩阵
        lowestError = inf   # 设置误差
        for j in range(n):  # 对于每个特征
            for sign in [-1,1]:  # 增大或者减少
                wsTest = ws.copy()
                wsTest[j] += eps*sign  # 得到新的权重
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)  # 计算误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

xArr,yArr = loadDataSet('abalone.txt')
a = stageWise(xArr,yArr,0.01,200)
print(a)

# 交叉验证测试岭回归
def crossValidation(xArr,yArr,numVal=10):
    m = len(xArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)  # 对数据进行打乱
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
            wMat = ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))
