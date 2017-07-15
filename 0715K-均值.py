# -*- coding: utf-8 -*-
'''
优点：容易实现
缺点：可能收敛到局部最小值，在大规模数据集上收敛较慢
适用数据类型：数值型
'''
from numpy import *
# K-均值聚类支持函数
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

#　计算距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]   # 特征数
    centroids = mat(zeros((k,n)))   #　构建k个质心的初始化值为0
    for j in range(n):             # 对于每一个特征
        minJ = min(dataSet[:,j])   # 寻找最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)  # 计算每个特征的最大最小值之差
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)  # 保证随机质心在数据集的边界内
    return centroids

datMat = mat(loadDataSet('testSet.txt'))

centroids = randCent(datMat,5)
# print(centroids)
# print(distEclud(datMat[0],datMat[1]))

# K-均值聚类算法
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]  # 样本数
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)  # 创建质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):   # 遍历每一个样本点
            minDist = inf
            minIndex = -1
            for j in range(k):   # 遍历每一个质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])  # 寻找最近质心
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist ** 2
        # print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

# 测试
myCentroids,clustAssing = kMeans(datMat,4)
# print(myCentroids)
# print(clustAssing)

# 绘图
import matplotlib.pyplot as plt
category = clustAssing[:,0].flatten().A[0]
fig = plt.figure()
ax = fig.add_subplot(111)
for i,j in zip([0,1,2,3],['s','o','x','v']):
    df = datMat[category == i,:]
    ax.scatter(df[:,0],df[:,1],marker=j,s=30)

# 绘制质心
for i in range(4):
    ax.scatter(myCentroids[i,0],myCentroids[i,1],c='red',s=100,marker = '+')

plt.show()

# 评价聚类的质量
'''
合并最近的质心；合并两个是的SSE增幅最小的质心
'''
# 二分K-均值算法
'''
首先将所有点作为一个簇，然后将该簇一分为二，之后选择其中一个簇进行划分，
选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE值
另一种做法是选择SSE最大的簇进行划分，直到簇数目达到用户指定的数目为止
将所有点看出一个簇
当簇数目小于k时
    对于每一个簇
        计算总误差
        在给定的簇上面进行K-均值聚类（k=2)
        计算将该簇一分为二之后的总误差
    选择是的误差最小的那个簇进行划分操作
'''
def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]  # 样本数
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0]   # 将其视为一个簇进行计算质心
    centList = [centroid0]
    for j in range(m):    # 对于每一个样本点计算距离
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = sum(splitClustAss[:,1])  # 计算误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            print('sseSplit,notSplit:',sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestcenttosplit is ',bestCentToSplit)
        print('the len of bestclustAss is:',len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return centList,clusterAssment

# datMat3 = mat(loadDataSet('testSet2.txt'))
# centList,myNewAssments = biKmeans(datMat3,3)
# print(array(centList)[:,0][:,0])
datMat3 = mat(random.rand(100,2))
centList,myNewAssments = biKmeans(datMat3,3)

# 绘图
category = myNewAssments[:,0].flatten().A[0]
fig = plt.figure()
ax = fig.add_subplot(111)
for i,j in zip([0,1,2],['s','o','x']):
    df = datMat3[category == i,:]
    ax.scatter(df[:,0],df[:,1],marker=j,s=30)

# 绘制质心
for i in range(3):
    ax.scatter(array(centList)[:,0][:,0],array(centList)[:,0][:,1],c='red',s=100,marker = '+')

plt.show()


# 获取地址
import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  #JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress,city)
    url_params = urllib.urlencode(params)  # 将创建的字典转换为可以通过URL进行传递的字符串格式
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt','w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1],lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()

# 球面距离计算及簇绘图函数
def distSLC(vecA,vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy
import matplotlib
def clusterClubs(numClust=5):
    datList = []   # 经纬度
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])])
        datMat = mat(datList)
        myCentroids,clustAssing = biKmeans(datMat,numClust,distMeas=distSLC)
        fig = plt.figure()
        rect = [0.1,0.1,0.8,0.8]
        scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
        axprops = dict(xticks=[],yticks=[])
        ax0 = fig.add_axes(rect,label='ax0',**axprops)
        imgP = plt.imread('Portland.png')
        ax0.imshow(imgP)
        ax1  = fig.add_axes(rect,label='ax1',frameon=False)
        for i in range(numClust):
            ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0],:]
            markerStyle = scatterMarkers[i % len(scatterMarkers)]
            ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],
                        ptsInCurrCluster[:,1].flatten().A[0],
                        marker=markerStyle,s=90)
        ax1.scatter(myCentroids[:,0].flatten().A[0],
                    myCentroids[:,1].flatten().A[0],
                    marker='+',s=300)
        plt.show()
clusterClubs(5)
