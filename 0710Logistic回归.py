# -*- coding: utf-8 -*-
'''
优点：计算代价不高，易于理解和实现
缺点：容易欠拟合，分类精度可能不高
使用数据类型：数值型和标称型数据（离散数据）
'''
import numpy as np
# loadDataSet主要是打开文件并逐行读取
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# 定义sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# 梯度上升算法
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)      # 将自变量转换为矩阵的形式
    labelMat = np.mat(classLabels).transpose()   # 转换为矩阵的形式并转置
    m,n = np.shape(dataMatrix)                   # 计算维度
    alpha = 0.001                          # 向目标移动的步长
    maxCycles = 500                        # 迭代次数
    weights = np.ones((n,1))              # 将权重初始化为1
    for k in range(maxCycles):           # 进行迭代
        h = sigmoid(dataMatrix*weights)             # 使用更新的weights计算拟合值，其值为列向量
        error = (labelMat - h)                      # 计算拟合值与真实值之间的误差
        weights = weights + alpha * dataMatrix.transpose() * error   # 按照差值的方向调整回归系数
    return weights

#################################  示例运算   #######################################
# 读取数据
dataArr,labelMat  = loadDataSet()    #　读取文本文件testSet.txt，并返回给相应的数据框
# 进行梯度上升算法求解最优的拟合系数
weights = gradAscent(dataArr,labelMat)
print(weights)
# [[ 4.12414349]
#  [ 0.48007329]
#  [-0.6168482 ]]

################################ 分析数据：画出决策边界  ##############################
# 画出分割线，确定不同类别数据之间的分割线
import matplotlib.pyplot as plt
def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()    # 加载数据集
    dataArr = np.array(dataMat)         # 转换为数组的形式
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:           # 对于标签为1的数据分为一类
            xcord1.append(dataArr[i,1])     # 设置x轴位置
            ycord1.append(dataArr[i,2])     # 设置y轴位置
        else:                               # 对于标签为0的数据分为另一类
            xcord2.append(dataArr[i,1])     # 设置x轴位置
            ycord2.append(dataArr[i,2])     # 设置y轴位置
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')    # 绘制标签为1的数据
    ax.scatter(xcord2,ycord2,s=30,c='green')             # 绘制标签为0的数据
    x = np.arange(-3,3,0.1)                              # 设置分割线的横坐标值
    # 此处设置了sigmoid值为0，设定0=w0x0+w1x2+w2x2，可以求解出x2和x1的关系式，如下所示
    y = (-weights[0]-weights[1]*x)/weights[2]            # 将曲线转换为x2 =  x1的形式
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 进行绘图
'''
weights返回的是矩阵的形式，在这里需要的是数组的形式，因此可以使用numpy中的getA()函数
其主要是将矩阵matrix形式的数据转换为array数组的形式
'''
# plotBestFit(weights.getA())

############################ 训练算法：随机梯度上升  ############################
"""
梯度上升算法在每次更新回归系数时都需要遍历整个数据集，该方法在处理100个样本的数据时尚可，但是如果有数十亿的特征，那么
该方法的计算复杂度太高。一种改进的方法是一次使用一个样本点来更新回归系数，该方法成为随机梯度上升算法。
"""
# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.1
    weights = np.ones(n)     # 初始化拟合系数全部为1
    for i in range(m):       # 对于数据中每个样本进行如下的操作
        h = sigmoid(sum(dataMatrix[i] * weights))   # 计算第i个样本的拟合值
        error = classLabels[i] - h                  # 计算第i个样本的误差
        weights = weights + alpha * error * dataMatrix[i]   # 使用第i个样本进行权重值的更新
    return weights    # 返回最终的回归系数

weights = stocGradAscent0(np.array(dataArr),labelMat)
# plotBestFit(weights)

############################### 改进的随机梯度上升算法 ##########################
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):    # 更新迭代的次数
        dataIndex = list(range(m))
        for i in range(m):
            aplha = 4/(1+j+i)+0.00001     #　更新步长每次迭代都会调整
            # uniform返回的是含有小数的数值，使用int可以将其取为整数
            randIndex = int(np.random.uniform(0,len(dataIndex)))  # 随机选择样本来更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + aplha*error*dataMatrix[randIndex]
            del dataIndex[randIndex]   # 从列表中删掉该值
        return weights

weights = stocGradAscent1(np.array(dataArr),labelMat,numIter=1000)
# plotBestFit(weights)

##########################  绘制回归系数变化的曲线图  #####################################
def plot_weights(dataMatrix,classLabels,numIter=500):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    w0 = [];w1 = [];w2 = []
    for j in range(numIter):  # 更新迭代的次数
        dataIndex = list(range(m))
        for i in range(m):
            aplha = 4 / (1 + j + i) + 0.1  # 更新步长每次迭代都会调整
            # uniform返回的是含有小数的数值，使用int可以将其取为整数
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # 随机选择样本来更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + aplha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]  # 从列表中删掉该值
        w0.append(weights[0])
        w1.append(weights[1])
        w2.append(weights[2])
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(list(range(numIter)), w0,'-')  # 绘制标签为1的数据
    plt.ylabel('X0')

    ax = fig.add_subplot(312)
    ax.plot(list(range(numIter)), w1, '-')  # 绘制标签为1的数据
    plt.ylabel('X1')

    ax = fig.add_subplot(313)
    ax.plot(list(range(numIter)), w2, '-')  # 绘制标签为1的数据
    plt.ylabel('X2')

    plt.show()

# plot_weights(np.array(dataArr),labelMat,numIter=1000)

################## 实战 ###################################
# 数据准备：处理数据中的缺失值
'''
1、使用可用特征的均值来填补缺失值
2、使用特殊值来填补缺失值，如-1
3、忽略有缺失值的样本
4、使用相似样本的均值来填补缺失值
5、使用另外的机器学习算法来预测缺失值
'''
# 定义函数，以回归系数和特征向量来计算对应的sigmoid值，如果sigmoid值大于0.5则返回1否则返回0
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# 定义打开测试集、训练集的数据，并对数据进行格式化处理的函数
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []   # 测试集的特征
    trainingLabels = []  # 测试集的标签
    # 将数据的训练集分割
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    # 计算回归系数
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,5000)

    # 计算拟合的模型在测试集上的错误率
    errorCount = 0                   # 初始化错误的个数为0
    numTestVec = 0.0                 # 初始化样本的总数
    for line in frTest.readlines():
        # 对于每一个测试集的样本执行下面的操作
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))       # 对于每一个样本进行提取特征
        # 计算每一个样本的拟合值是否与真实值一样
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = ((float(errorCount)) / numTestVec)   # 计算错误率
    print('错误率为:%f' % errorRate)
    return errorRate

# 定义函数执行colicTest10次，求平均值
def multiTest():
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print('10次循环后的平均值为：%f'  % (errorSum / float(numTests)))

multiTest()
