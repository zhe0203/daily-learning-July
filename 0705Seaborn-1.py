# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# 定义种子
np.random.seed(sum(map(ord,'aesthetics')))

# 定义一个含便移的正弦图像，来比较传统的matplotlib和seaborn的不同
def sinplot(flip=1):
    x = np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*0.5)*(7-i)*flip)
    plt.show()
sinplot()

import seaborn as sns
sinplot()

# 样式控制：axes_style()和set_style()
'''
darkgrid 黑色网格（默认）
whitegrid  白色网格
dark 黑色背景
white 白色背景
ticks 应该是四周都有刻度线的白色背景
'''
# sns.set_style('whitegrid')
data = np.random.normal(size=(20,6)) + np.arange(6) / 2
# print(data)
# sns.boxplot(data=data)


sns.set_style('white')
sns.set_style('ticks',{'xtick.major.size':8,'ytick.major.size':8})
sns.despine()
sinplot()
plt.show()

f,ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10,trim=True)  #　offset两坐标轴离开的距离
plt.show()

# 通过往despine()中添加参数去控制边框
sns.set_style('whitegrid')
sns.boxplot(data=data,palette='deep')
sns.despine(left=True)  # 删除左边边框
st = sns.axes_style('darkgrid')
plt.show()

# 参数设置
'''
despine(fig=None,ax=None,top=True,right=True,left=False,
        bottom=False,offset=None,trim=False)
'''

'''
虽然来回切换非常容易，但sns也允许用with语句中套用axes_style()达到
临时设置参数的效果（仅对with块内的绘图函数起作用）。这也允许创建不同风格的坐标轴。
'''
with sns.axes_style('darkgrid'):
    plt.subplot(211)
    sinplot()
    plt.subplot(212)
    sinplot(-1)

# 查看具体的参数
'''
sns.axes_style()

{'axes.axisbelow': True,
 'axes.edgecolor': 'white',
 'axes.facecolor': '#EAEAF2',
 'axes.grid': True,
 'axes.labelcolor': '.15',
 'axes.linewidth': 0.0,
 'figure.facecolor': 'white',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
  'Liberation Sans',
  'Bitstream Vera Sans',
  'sans-serif'],
 'grid.color': 'white',
 'grid.linestyle': '-',
 'image.cmap': 'Greys',
 'legend.frameon': False,
 'legend.numpoints': 1,
 'legend.scatterpoints': 1,
 'lines.solid_capstyle': 'round',
 'text.color': '.15',
 'xtick.color': '.15',
 'xtick.direction': 'out',
 'xtick.major.size': 0.0,
 'xtick.minor.size': 0.0,
 'ytick.color': '.15',
 'ytick.direction': 'out',
 'ytick.major.size': 0.0,
 'ytick.minor.size': 0.0}
'''

# 另一组参数控制绘图元素的规模，这应该让您使用相同的代码来制作适合在较大或较小的情节适当的场景中使用的情节
# 首先，可以通过sns.set()重置参数

# 四种预设，按相对尺寸的顺序(线条越来越粗)，分别是paper，notebook, talk, and poster。notebook的样式是默认的，上面的绘图都是使用默认的notebook预设。

# default 默认设置
sns.set_context("notebook")
plt.figure(figsize=(8,6))
sinplot()

sns.set_context('poster')
plt.figure(figsize=(8,6))
sinplot()

sns.set_context('notebook',font_scale=1.5,
                rc={'lines.linewidth':2.5})
sinplot()
