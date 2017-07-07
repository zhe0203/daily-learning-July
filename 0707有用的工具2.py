# -*- coding: utf-8 -*-

# glob模块
import glob
'''
glob模块提供了方便的文件模式匹配方法
'''
# 找出所有于.py结尾的文件
print(glob.glob('*.py'))

'''
glob函数支持三种格式的语法：
    *   匹配单个或多个字符
    ？  匹配任意单个字符
    []  匹配指定范围内的字符，如[0-9]匹配数字
'''
print(glob.glob('070[1-5]*.py'))
print(glob.glob(r'../*.py'))    # 返回上级目录中所有的py文件

'''
#获取指定目录下的所有图片
glob.glob(r"E:/Picture/*/*.jpg")

#获取上级目录的所有.py文件
glob.glob(r'../*.py') #相对路径
'''

# glob.iglob
'''
获取一个可遍历对象，使用它可以逐个获取匹配的文件路径名
glob.glob同时获取所有的匹配路径
glob.iglob一次只获取一个匹配路径
'''
for py in glob.iglob('*.py'):
    print(py)

# shutil 模块：高级文件操作
import shutil
import os

# 复制文件
with open('test.file','w') as f:
    pass
print('test.file' in os.listdir(os.curdir))

## 使用shutil.copy(src,dst)将源文件复制到目标地址
shutil.copy('test.file','test.copy.file')

print("test.file" in os.listdir(os.curdir))
print("test.copy.file" in os.listdir(os.curdir))

## 如果目标地址中间的文件夹不存在则会报错
## 另外的一个函数shutil.copyfile(src,dst)与shutil.copy使用方法一致
## 不过只是简单复制文件的内容，并不会赋值文件本身的读写可执行权限
# os.renames("test.file", "test_dir/test.file")
# os.renames("test.copy.file", "test_dir/test.copy.file")
#
# shutil.copytree('test_dir/','test_dir_copy/')
# print('test_dir_copy' in os.listdir(os.curdir))

# 删除非空文件夹
# os.removedirs()不能删除非空文件夹
try:
    os.removedirs('test_dir_copy')
except Exception as msg:
    print(msg)

# 使用shutil.rmtree来删除非空文件夹
# shutil.rmtree('test_dir_copy')

# 产生压缩文件
# 查看支持的压缩文件格式：

# shutil.get_archive_formats()
# shutil.make_archive(basename, format, root_dir)
# shutil.make_archive("test_archive", "zip", "test_dir/")
# os.remove("test_archive.zip")
# shutil.rmtree("test_dir/")

# gzip zipfile tarfile模块：处理压缩文件
import zlib,gzip,bz2,zipfile,tarfile

# zlib模块 提供了对字符串进行压缩和解压的功能
orginal = b'this is a test string'
compressed = zlib.compress(orginal)
print(compressed)
print(zlib.decompress(compressed))

# 同时提供了两种校验和计算方法
print(zlib.adler32(orginal))
print(zlib.crc32(orginal))

# gzip模块 可以产生.gz格式的文件。其压缩方式由zlib模块提供
# 我们可以通过gzip.open方法来读写.gz格式的文件
content = b'lots of content here'
with gzip.open('file.txt.gz','wb') as f:
    f.write(content)

# 读
with gzip.open('file.txt.gz','rb') as f:
    file_content = f.read()
print(file_content)

# 将压缩文件内容解压出来
with gzip.open('file.txt.gz','rb') as f_in,open('file.txt','wb') as f_out:
    shutil.copyfileobj(f_in,f_out)

# bz2 模块
# bz2 模块提供了另一种压缩文件的方法：
orginal = b"this is a test string"
compressed = bz2.compress(orginal)
print(compressed)
print(bz2.decompress(compressed))

# zipfile模块   产生一些file.txt的复制
for i in range(10):
    shutil.copy('file.txt','file.txt.'+str(i))
# 将这些赋值全部压缩到一个zip文件中
f = zipfile.ZipFile('files.zip','w')
for name in glob.glob('*.txt.[0-9]'):
    f.write(name)
    os.remove(name)
f.close()

# 解压这个文件用namelist方法查看压缩文件中的子文件名
f = zipfile.ZipFile('files.zip','r')
print(f.namelist())

# 用f.read()方法来读取name文件中的内容
for name in f.namelist():
    print(name,'content:',f.read(name))
f.close()

# 可以用 extract(name) 或者 extractall() 解压单个或者全部文件

# tarfile模块
## 支持tar格式文件的读写
f = tarfile.open('file.txt.tar','w')
f.add('file.txt')
f.close()

# 清理生成的文件
os.remove('file.txt')
os.remove('file.txt.tar')
os.remove('file.zip')
