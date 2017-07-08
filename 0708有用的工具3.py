# -*- coding: utf-8 -*-
# string模块：字符串处理
import string
# 标点符号
print(string.punctuation)   # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# 字母表
print(string.ascii_letters)
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
print(string.ascii_uppercase)
# ABCDEFGHIJKLMNOPQRSTUVWXYZ
print(string.ascii_lowercase)
# abcdefghijklmnopqrstuvwxyz

# 数字
print(string.digits)  # 0123456789

# 16进制
print(string.hexdigits)
# 0123456789abcdefABCDEF

# 将每个单词的首字符大写
print(string.capwords('this is a big world'))
# This Is A Big World

# 将指定的单词放到中央
# print(string.center(20,'test'))

'''
string.capitalize()      把字符串的第一个字符大写
string.center()      返回一个原字符串集中，并使用空格填充长度至width
string.count(str)   返回str在string里面出现的次数
string.endswith()    检查字符串是否以obj结束
string.expandtabs()   把字符串string中的tab符号转为空格
string.find()  检测str是否包含在string中，如果 beg 和 end 指定范围，则检查是否包含在指定范围内，如果是返回开始的索引值，否则返回-1
string.format()   格式化字符串
string.index(str)  跟find一样
string.isalnum()  如果 string 至少有一个字符并且所有字符都是字母或数字则返回 True,否则返回 False
string.isalpha() 如果 string 至少有一个字符并且所有字符都是字母则返回 True,否则返回 False
string.isdecimal() 如果 string 只包含十进制数字则返回 True 否则返回 False.
string.isdigit() 如果 string 只包含数字则返回 True 否则返回 False.
string.islower() 如果 string 中包含至少一个区分大小写的字符，并且所有这些(区分大小写的)字符都是小写，则返回 True，否则返回 False
string.isnumeric() 如果 string 中只包含数字字符，则返回 True，否则返回 False
string.isspace()如果 string 中只包含空格，则返回 True，否则返回 False.
string.istitle()如果 string 是标题化的(见 title())则返回 True，否则返回 False
string.isupper()如果 string 中包含至少一个区分大小写的字符，并且所有这些(区分大小写的)字符都是大写，则返回 True，否则返回 False
string.join(seq)以 string 作为分隔符，将 seq 中所有的元素(的字符串表示)合并为一个新的字符串
string.ljust(width)返回一个原字符串左对齐,并使用空格填充至长度 width 的新字符串
string.lower()转换 string 中所有大写字符为小写.
string.lstrip()截掉 string 左边的空格
string.maketrans(intab, outtab])maketrans() 方法用于创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。
max(str)返回字符串 str 中最大的字母。
min(str)返回字符串 str 中最小的字母。
string.partition(str)有点像 find()和 split()的结合体,从 str 出现的第一个位置起,把 字 符 串 string 分 成 一 个 3 元 素 的 元 组 (string_pre_str,str,string_post_str),如果 string 中不包含str 则 string_pre_str == string.
string.replace(str1, str2,  num=string.count(str1))把 string 中的 str1 替换成 str2,如果 num 指定，则替换不超过 num 次.
string.rfind(str, beg=0,end=len(string) )类似于 find()函数，不过是从右边开始查找.
string.rindex( str, beg=0,end=len(string))类似于 index()，不过是从右边开始.
string.rjust(width)返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串
string.rpartition(str)类似于 partition()函数,不过是从右边开始查找.
string.rstrip()删除 string 字符串末尾的空格.
string.split(str="", num=string.count(str))以 str 为分隔符切片 string，如果 num有指定值，则仅分隔 num 个子字符串
string.splitlines([keepends])按照行('\r', '\r\n', \n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为 False，不包含换行符，如果为 True，则保留换行符。
string.startswith(obj, beg=0,end=len(string))检查字符串是否是以 obj 开头，是则返回 True，否则返回 False。如果beg 和 end 指定值，则在指定范围内检查.
string.strip([obj])在 string 上执行 lstrip()和 rstrip()
string.swapcase()翻转 string 中的大小写
string.title()返回"标题化"的 string,就是说所有单词都是以大写开始，其余字母均为小写(见 istitle())
string.translate(str, del="")根据 str 给出的表(包含 256 个字符)转换 string 的字符,要过滤掉的字符放到 del 参数中
string.upper()转换 string 中的小写字母为大写
string.zfill(width)返回长度为 width 的字符串，原字符串 string 右对齐，前面填充0
string.isdecimal() isdecimal()方法检查字符串是否只包含十进制字符。这种方法只存在于unicode对象。
'''

#　collections模块：更多数据结构
import collections

# 计数器
# 可以使用Counter(seq)对序列中出现的元素个数进行统计
from string import punctuation
sentence = "One, two, three, one, two, tree, I come from China."

# string.translate(str, del="")
# 根据 str 给出的表(包含 256 个字符)转换 string 的字符,要过滤掉的字符放到 del 参数中

words_count = collections.Counter(sentence.translate(punctuation).lower().split())
print(words_count)
print(sentence)
print(sentence.translate(punctuation))

# 双端队列  支持从对头对尾出入对
dq = collections.deque()
for i in range(10):
    dq.append(i)
print(dq)

for i in range(10):
    print(dq.pop())   #　先删除对尾

for i in range(10):
    dq.appendleft(i)   # 注意appendleft的使用
print(dq)

for i in range(10):
    print(dq.popleft())  # 注意popleft的使用，可以查看list的用法函数

# 与列表相比，双端队列在对头的操作更快
import time
lst = []
dq = collections.deque()

lst.insert(0,10)
dq.appendleft(10)

# 有序字典
items = (
    ('A',1),('B',2),('C',3)
)
regular_dict = dict(items)
ordered_dict = collections.OrderedDict(items)
print('Regular Dict:')
for k,v in regular_dict.items():
    print(k,v)

print('Ordered Dict:')
for k,v in ordered_dict.items():
    print(k,v)

# 带默认值的字典
'''
对于 Python 自带的词典 d，当 key 不存在的时候，调用 d[key] 会报错，
但是 defaultdict 可以为这样的 key 提供一个指定的默认值，
我们只需要在定义时提供默认值的类型即可，如果 key 不存在返回指定类型的默认值：
'''
dd = collections.defaultdict(list)
print(dd['foo'])   # []
dd = collections.defaultdict(int)  # 默认值为整数
print(dd['foo'])    #　0
dd = collections.defaultdict(float)  #　默认值为浮点型数据
print(dd['foo'])  # 0.0

# requests模块
import requests

# 传入 URL 参数
# 假如我们想访问 httpbin.org/get?key=val，我们可以使用 params 传入这些参数：

payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.get("http://httpbin.org/get", params=payload)

#　查看url
print(r.url)

# 读取响应内容
r = requests.get('https://github.com/timeline.json')
print(r.text)

# 查看文字的编码
print(r.encoding)

# Requests 中也有一个内置的 JSON 解码器处理 JSON 数据：
print(r.json())

# 响应状态码
r = requests.get('http://httpbin.org/get')
print(r.status_code)  # 打印状态码
# 响应头
print(r.headers['Content-Type'])
