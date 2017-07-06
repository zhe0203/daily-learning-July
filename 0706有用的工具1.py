# -*- coding: utf-8 -*-

# pprint模块:打印Python对象
'''
pprint 是pretty printer的缩写，用来打印python的数据结构，与print相比
它打印出来的结构更加整齐、便于阅读
'''
import pprint
# 生成一个python对象
data = (
    'this is a string',
    [1,2,3,4],
    ('more tuples',1,2.3,4.5),
    'this is yet another string.'
)
# 普通打印
print(data)
## ('this is a string', [1, 2, 3, 4], ('more tuples', 1, 2.3, 4.5), 'this is yet another string.')

# 使用pprint打印
pprint.pprint(data)  # 可以看到打印出来的公式更加美观
# ('this is a string',
#  [1, 2, 3, 4],
#  ('more tuples', 1, 2.3, 4.5),
#  'this is yet another string.')

# pickle,cPickle模块：序列化Python对象
'''
pickle模块实现了一种算法，可以将任意一个python对象转化为一系列的字节
也可以将这些字节重构为一个有相同特征的新对象
由于字节可以被传输或者存储，因此pickle事实上实现了传递或保存python对象的功能
cPickle 使用 C 而不是 Python 实现了相同的算法，因此速度上要比 pickle 快一些。
但是它不允许用户从 pickle 派生子类。如果子类对你的使用来说无关紧要，那么 cPickle 是个更好的选择
'''
try:
    import cPickle as pickle
except:
    import pickle

# 编码和解码
'''
使用pickle.dumps()可以将一个对象转换为字符串
'''
data = [{'a':'A','b':2,'c':3.0}]
data_string = pickle.dumps(data)
print(data)
print(data_string)

'''
虽然pickle编码的字符串并不一定可读，但是我们可以用pickle.loads()
来从这个字符串中恢复原对象中的内容
'''

data_from_string = pickle.loads(data_string)
print(data_from_string)

'''
dumps 可以接受一个可省略的 protocol 参数（默认为 0），目前有 3 种编码方式：
    0：原始的 ASCII 编码格式
    1：二进制编码格式
    2：更有效的二进制编码格式
'''
data_string_1 = pickle.dumps(data,1)
print(data_string_1)
data_string_2 = pickle.dumps(data,2)
print(data_string_2)
print(pickle.loads(data_string_2))

# 存储和读取pickle文件
'''
除了将对象转换为字符串这种方式，pickle还支持将对象写入一个文件中
通常我们将这个文件命名为xxx.pkl,以表示他是一个pickle文件
存储和读取的函数分别为：
pickle.dump(obj,file,protocol=0)将对象序列化并存入file文件中
pickle.load(file)从file文件中的内容恢复对象
'''
# 将对象存入文件
# with open('data.pkl','wb') as f:
#     pickle.dump(data,f)
# 从文件中读取
# with open('data.pkl') as f:
#     data_from_file = pickle.load(f)
# print(data_from_file)

# 清理生成的文件
# import os
# os.remove('data.pkl')


# json模块:处理JSON数据
'''
JSON对象与字典对象的区别在于，JSON对象首尾包含"""   """
'''

import json
from pprint import pprint
info_string = """
{
    "name": "echo",
    "age": 24,
    "coding skills": ["python", "matlab", "java", "c", "c++", "ruby", "scala"],
    "ages for school": {
        "primary school": 6,
        "middle school": 9,
        "high school": 15,
        "university": 18
    },
    "hobby": ["sports", "reading"],
    "married": false
}
"""

# 使用json.loads()方法从字符串中读取JSON数据
info = json.loads(info_string)   # 将JSON数据变成了一个python对象
pprint(info)

# 使用json.dumps()将一个python对象转换为JSON对象
info_json = json.dumps(info)
print(info_json)

# 生成和读取JSON文件
'''
json.dump(obj,file)将对象保存为JSON格式文件
json.load(file)从JSON文件中读取数据
'''
with open('info.json','w') as f:
    json.dump(info,f)
# 查看info.json的内容
with open('info.json') as f:
    print(f.read())

# 从文件中读取数据
with open('info.json') as f:
    info_from_file = json.load(f)
pprint(info_from_file)
