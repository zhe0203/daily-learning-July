# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# sign函数
"""
符号函数：-1 if x < 0, 0 if x==0, 1 if x > 0
"""
print(np.sign(-1))  # 0
print(np.sign(2))   # 1
print(np.sign(0))   # 0
print(np.sign([-1,2,3.5]))  # [-1.  1.  1.]

# np.multiply函数
"""
计算两个数组之间的乘积，对应的元素相乘，支持brodcast
"""
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
print(x1,x2)
print(np.multiply(x1, x2))
print(x1 * x2)   # 等价于

# del 函数元素的删除
a = [1,2,3]
del a[2]  # 删除a中的第二个元素3
print(a)

# transpose与T区别
df = np.arange(12).reshape(3,-1)
print(df)
print(df.transpose())
print(df.T)

# 使用异常来捕获错误的文件
def textParse(bigString):
    import re
    # listOfTokens = re.split(r'\W*',bigString)
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

for i in range(1,26):
    try:
        f = open('C:/Users/jk/Desktop/Python学习/每日一学习/7月/email/ham/%d.txt' % i)
        print(textParse(f.read()))
    except:
        print(i)

# tolist函数
# 将Numpy数组转换为列表进行迭代
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.tolist())

# numpy 中 nonzero()函数
'''用于得到数组array中非零元素的位置（数组索引）的函数'''
a = np.array([[0,0,3],[0,0,0],[0,0,9]])
b = np.nonzero(a)
print(b)   # 返回数组中非零元素的位置
a = [0,1,2]
print(np.nonzero(a)[0])  # array([1,2])  非零元素位置

# frozenset()函数与set()函数区别
'''
set无序排序且不重复，是可变的，有add（），remove（）等方法.基本功能包括关系测试和消除重复元素.
集合对象还支持union(联合), intersection(交集), difference(差集)和sysmmetric difference(对称差集)等数学运算.
sets 支持 x in set, len(set),和 for x in set。作为一个无序的集合，sets不记录元素位置或者插入点。
因此，sets不支持 indexing, 或其它类序列的操作。

frozenset是冻结的集合，它是不可变的，存在哈希值，好处是它可以作为字典的key，也可以作为其它集合的元素。
缺点是一旦创建便不能更改，没有add，remove方法。
'''
s=set('cheeseshop')
print(s,type(s))
t=frozenset('bookshop')
print(t,type(t))

# issubset()函数比较一个集合是否是另外一个集合的子集
a = [1,2,3]
b = [1,2,3,4,5]
# 首先需要转换成集合的形式
print(set(a).issubset(set(b)))

a_set = set(a)
a_frozenset = frozenset(a)
for i in a_set:
    print(i,type(i))  # int
for j in a_frozenset:
    print(j,type(j))  # int
