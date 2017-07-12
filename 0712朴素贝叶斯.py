# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
'''
优点：在数据较少的情况下仍然有效，可以处理多类别问题
缺点：对于输入数据的准备方式较为敏感
适用数据类型：标称型数据

建模步骤：
考虑出现在所有文档中的单词，再决定将哪些单词纳入词汇表或者说所要的词汇集合，然后必须要将每一篇文档转换为词汇表上的向量
'''
# 词表到向量的转换函数
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]  # 将上述的list打上标签。1表示侮辱性文字，0表示无
    return postingList,classVec

# 创建一个包含在所有文档中出现的不重复的列表
def createVocabList(dataSet):
    vocabSet = set([])   # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)   # 创建两个集合的并集
    return list(vocabSet)

# 使用下面函数将输入参数的词汇表及某个文档输出向量，向量的每一个元素为1或0，分别表示词汇表中的单词再输入文档中是否出现
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)   # 创建一个其中所含元素都为0 的向量
    for word in inputSet:
        if word in vocabList:          # 检验是否在向量词汇中
            returnVec[vocabList.index(word)] = 1
        else:
            print('这个 %s 单词不存在向量中' % word)
    return returnVec

################ 执行程序 ################
listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)

# 将文档内容转换为向量的形式
for i in range(len(listOPosts)):
    print(setOfWords2Vec(myVocabList,listOPosts[i]))

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                        # 计算文章总数
    numWords = len(trainMatrix[0])                         # 计算词汇量的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)    # 计算每个类别的概率
    p0Num = zeros(numWords)                               # 初始化概率
    p1Num = zeros(numWords)
    p0Denom,p1Denom = 0.0,0.0
    for i in range(numTrainDocs):            # 对于每篇训练文章
        if trainCategory[i] == 1:            # 对于第一个类别
            p1Num += trainMatrix[i]           # 增加每一个词的计数值
            p1Denom += sum(trainMatrix[i])   #　计算对于这一类词的总频数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    '''使用每一类每个词出现的频数除以这一类的总频数即可得到条件概率向量p(w0|x) p(w1|x)......'''
    p1Vect = p1Num / p1Denom         # 将该词条的数目除以总词条数目得到条件概率
    p0Vect = p0Num / p0Denom
    return p0Vect,p1Vect,pAbusive

############## 测试  ##########################
myVocabList = createVocabList(listOPosts)
trainMat = []                                    # 构建矩阵表
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

p0V,p1V,pAb = trainNB0(trainMat,listClasses)
print(p0V)
print(p1V)
print(pAb)

# 修改分类器
'''
让了防止某个词在某些类别的概率为0，即某个p(wi|1)为0，使得分子为0，初始化所有词的出现的数初始化为1，并将分母初始化为2
'''
# 进行如下的修改
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                        # 计算文章总数
    numWords = len(trainMatrix[0])                         # 计算词汇量的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)    # 计算每个类别的概率
    p0Num = ones(numWords)                                  # 初始化概率为1
    p1Num = ones(numWords)                                  # 初始化概率为1
    p0Denom,p1Denom = 2.0,2.0
    for i in range(numTrainDocs):            # 对于每篇训练文章
        if trainCategory[i] == 1:            # 对于第一个类别
            p1Num += trainMatrix[i]           # 增加每一个词的计数值
            p1Denom += sum(trainMatrix[i])   #　计算对于这一类词的总频数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    '''使用每一类每个词出现的频数除以这一类的总频数即可得到条件概率向量p(w0|x) p(w1|x)......'''
    '''另外，遇到的问题是下溢出问题，可以对乘积取自然对数'''
    p1Vect = log(p1Num / p1Denom)         # 将该词条的数目除以总词条数目得到条件概率
    p0Vect = log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    # 由于去了对数，因此可以使用如下的计算
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  #　对于每一个样本计算分子
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1) # 计算属于另外一个类别的分子
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()          #　对测试集进行计算分类

# 文档词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

######### 使用朴素被夜色过滤垃圾邮件 ##############
# 数据准备：切分文本
'''对于一个字符串，可以使用string.split()方法将其切分'''
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon'
print(mySent.split())

# 可以使用正则来切分句子，其中分隔符是除单词、数字外的任意字符串
import re
regEx = re.compile('\\W*')
listOfTokens = regEx.split(mySent)
print(listOfTokens)

# 去掉空字符串，并转换为小写
print([tok.lower() for tok in listOfTokens if len(tok) > 0])

######## 完整的文件解析和垃圾邮件测试函数

def textParse(bigString):
    import re
    # listOfTokens = re.split(r'\W*',bigString)
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        f = open('email/spam/%d.txt' % i)
        wordList = textParse(f.read()) #decode("utf-8"))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        f = open('email/ham/%d.txt' % i)
        wordList = textParse(f.read()) #decode("utf-8"))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('错误率为：',float(errorCount) / len(testSet))
spamTest()
