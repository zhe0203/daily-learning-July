'''
给出数据1,2,3,5,6,7,8,10,12,13,14,15
要求输出：
    1,2,3
    5,6,7,8
    10
    12,13,14,15
'''
import numpy as np
list = [1,2,3,5,6,7,8,10,12,13,14,15]
# 首先进行差分
list_diff = np.diff(list)
a = []
for i,j in enumerate(list_diff):
    if j > 1:
        a.append(i)
# print(a)

# 进行数据的分割 split
list = np.array(list)
a = np.array(a)
result = np.split(list,a+1)
for i in result:
    print(i)

def data_split(num):
    a = [i+1 for i,j in enumerate(np.diff(num)) if j > 1]
    num = np.array(num)
    print(np.split(num,a))
data_split([1,2,3,5,6,7,8,10,12,13,14,15])
