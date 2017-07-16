# -*- coding: utf-8 -*-
# Apriori算法中的辅助函数
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

# 返回物品集中的所有单一物品集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))  # 对于每一个数据都做frozenset变换

def scanD(D,CK,minSupport):
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):    # can返回的是集合型数据，可直接用于判断是否数据子集
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    print(ssCnt)
    numItems = float(len(D))  # 计算样本数
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        # print(support >= 0.5)
        if support >= minSupport:
            retList.insert(0,key)
            supportData[key] = support
    return retList,supportData

# 测试
dataSet = loadDataSet()
C1 = createC1(dataSet)
D = list(map(set,dataSet))
# print(D)
L1,suppData0 = scanD(D,C1,0.5)
print(L1)
print(suppData0)

# Apriori算法
def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

L,suppData = apriori(dataSet)
print(L)
print(suppData)

# 关联规则生成函数
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
