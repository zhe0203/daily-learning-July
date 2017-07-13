# -*- coding: utf-8 -*-
'''
优点：泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整
缺点：对离群点敏感
适用数据类型：数值型和标称型数据

bagging:基于数据随机重抽样的分类器构建
    从原数据集中选择S次后得到S个新数据集，然后建立S个分类器，选择分类器投票结果最多的类别作为最后的分类结果。随机森林
    分类器的权重相等
boosting:每个新分类器都是根据以训练出的分类器的性能来进行训练
    分类器的权重不等，每个权重代表的是其对应分类器上一轮迭代中的成功度
'''

# 基于单层决策树构建弱分类器
from numpy import *
import matplotlib.pyplot as plt
def loadSimpData():
    datMat = matrix([[1,2.1],[2,1.1],[1.3,1],[1,1],[2,1]])
    classLabels = [1,1,-1,-1,1]
    return datMat,classLabels

datMat,classLabels = loadSimpData()


xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
for i in range(len(classLabels)):
    if classLabels[i] == 1:
        xcord0.append(datMat[i][0,0])
        ycord0.append(datMat[i][0,1])
    else:
        xcord1.append(datMat[i][0,0])
        ycord1.append(datMat[i][0,1])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0,ycord0,marker='o',s=90)
ax.scatter(xcord1,ycord1,marker='s',s=90)
plt.show()

# 单层决策树生成函数
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10   # 用于在特征的所有可能值进行遍历
    bestStump = {}  # 用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    minError = inf  # 错误率出初始化为无穷大，之后用于寻找可能的最小错误率
    for i in range(n):    # 对于数据中的每一个特征进行循环
        rangeMin = dataMatrix[:,i].min()  # 通过计算最大、最小值来了解应该需要多大的步长
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1,int(numSteps)+1):   # 对于每个步长
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr          # 计算加权错误率
                print('split:dim %d,thresh %.2f,thresh ineqal: %s,the weighted error is %.3f'\
                      % (i,threshVal,inequal,weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

D = mat(ones((5,1)) / 5) # 初始化权重
print(D)
buildStump(datMat,classLabels,D)

############# 完整AdaBoost算法的实现过程  ##############
def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1)) / m)  # 初始化样本权重向量
    aggClassEst = mat(zeros((m,1)))   # 记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print('D:',D.T)   # 打印权重向量
        alphas = float(0.5 * log((1.0-error) / max(error,1e-16)))   #　计算这个弱分类器的权重
        bestStump['alphas'] = alphas
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)

        # 为下一次的迭代计算D
        expon = multiply(-1*alphas*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D / D.sum()

        # 错误率累加计算
        aggClassEst += alphas * classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))  #　计算错误率
        errorRate = aggErrors.sum() / m
        print('total error:',errorRate,'\n')
        if errorRate == 0.0:
            break
    return weakClassArr

########### 测试
classifierArray = adaBoostTrainDS(datMat,classLabels,10)
# print(classifierArray)

###### AdaBoost分类函数
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArray)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alphas'] * classEst
        # print(aggClassEst)
    return sign(aggClassEst)
datArr,labelArr = loadSimpData()
classifierArr = adaBoostTrainDS(datArr,labelArr,30)
# print(adaClassify([0,0],classifierArray))

# 自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

datArr,labelArr = loadDataSet('horseColicTraining.txt')
classifierArray = adaBoostTrainDS(datArr,labelArr,10)
print(classifierArray)

testArr,testLabelArr = loadDataSet('horseColicTest.txt')
prediction10 = adaClassify(testArr,classifierArray)
errArr = mat(ones((67,1)))
print(errArr[prediction10 != mat(testLabelArr).T].sum())


################### 非均衡问题 ##########################
# ROC曲线的绘制及AUC计算函数
def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1)) / m)  # 初始化样本权重向量
    aggClassEst = mat(zeros((m,1)))   # 记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print('D:',D.T)   # 打印权重向量
        alphas = float(0.5 * log((1.0-error) / max(error,1e-16)))   #　计算这个弱分类器的权重
        bestStump['alphas'] = alphas
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)

        # 为下一次的迭代计算D
        expon = multiply(-1*alphas*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D / D.sum()

        # 错误率累加计算
        aggClassEst += alphas * classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))  #　计算错误率
        errorRate = aggErrors.sum() / m
        print('total error:',errorRate,'\n')
        if errorRate == 0.0:
            break
    return aggClassEst

'''
先从排名最低的样例开始，所有排名更低的样例都被判为反例，而所有排名更高的样例都被判为正例
然后，将其移到排名低的样例中去，如果该样例属于正例，那么对真阳率进行修改，即y轴的方向下降一个步长，x轴不变
否则，如果样例属于反例，那么对假阴率进行修改，即x轴的方向下降一个步长，y轴不变
'''

def plotROC(predStrengths,classLabels):
    cur = (1,1)
    ySum = 0                  # 用计算AUC
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:    # y轴下降方向
            delX = 0
            delY = yStep
        else:                            # x轴下降方向
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)           # 更新最新的点
    ax.plot([0,1],[0,1],'b--')
    ax.axis([0,1,0,1])
    plt.show()
    print(ySum * xStep)  # 简化为步长乘以每一个高度来计算AUC

aggClassEst = adaBoostTrainDS(datArr,labelArr,50)
plotROC(aggClassEst.T,labelArr)
