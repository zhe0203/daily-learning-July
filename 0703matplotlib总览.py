# _*_ coding: utf-8 _*_

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simple_plot():
    '''
    simple plot
    '''
    # 生成测试数据
    x = np.linspace(-np.pi,np.pi,256,endpoint=True)
    y_cos,y_sin = np.cos(x),np.sin(x)

    # 生成画布
    plt.figure(figsize=(8,6),dpi=80)  # 生成画布大小
    plt.title('plot title')     # 生成图标题
    plt.grid(True)     # 网格线是否显示

    # 设置x轴
    plt.xlabel('x lable')
    plt.xlim(-4,4)
    plt.xticks(np.linspace(-4,4,9,endpoint=True))   # 设置x轴显示v标签的名称

    # 设置y轴
    plt.ylabel('y label')
    plt.ylim(-1,1)
    plt.yticks(np.linspace(-1,1,9,endpoint=True))

    # 画两条曲线
    plt.plot(x,y_cos,'b--',linewidth=2,label='cos')
    plt.plot(x,y_sin,'g-',linewidth=2,label='sin')

    # 设置图列位置，loc可以为upper lower left right center
    plt.legend(loc='upper left',shadow=True)

    # 图形显示
    plt.show()
    return
# simple_plot()

def simple_advanced_plot():

    '''
    simple advanced plot
    '''
    # 生成测试数据
    x = np.linspace(-np.pi,np.pi,256,endpoint=True)
    y_cos,y_sin = np.cos(x),np.sin(x)

    # 生成画布
    plt.figure(figsize=(8,6),dpi=80)
    plt.title('plot title')
    plt.grid(True)

    # 画图的另一种方式
    ax_1 = plt.subplot(111)
    ax_1.plot(x,y_cos,color='blue',linewidth=2,linestyle='--',label='cos in left')
    ax_1.legend(loc='upper left',shadow=False)

    # 设置y轴(左边)
    ax_1.set_ylabel('y label for cos in left')
    ax_1.set_ylim(-1,1)
    ax_1.set_yticks(np.linspace(-1,1,9,endpoint=True))

    # 画图的另外一种方式
    ax_2 = ax_1.twinx()
    ax_2.plot(x,y_sin,color='green',linewidth=2,linestyle="-", label="sin in right")
    ax_2.legend(loc='upper right',shadow=True)

    # 设置y轴(右边）
    ax_2.set_ylabel('y label for sin in right')
    ax_2.set_ylim(-2,2)
    ax_2.set_yticks(np.linspace(-2,2,9,endpoint=True))

    # 设置x轴（共同）
    ax_2.set_xlabel('x label')
    ax_2.set_xlim(-4,4)
    ax_2.set_xticks(np.linspace(-4,4,9,endpoint=True))

    # 图像显示
    plt.show()
    return
# simple_advanced_plot()

def subplot_plot():
    """
    subplot plot
    """
    # 子图的style列表
    style_list = ["g+-", "r*-", "b.-", "yo-"]

    # 依次画图
    for num in range(4):
        # 生成测试数据
        x = np.linspace(0.0, 2+num, num=10*(num+1))
        y = np.sin((5-num) * np.pi * x)

        # 子图的生成方式
        plt.subplot(2, 2, num+1)
        plt.plot(x, y, style_list[num])
        plt.xlabel('x label')

    # 图形显示
    plt.grid(True)
    plt.show()
    return
# subplot_plot()

def barh_plot():

    # 生成测试数据
    means_men = (20,35,30,35,27)
    means_women = (25,32,34,20,25)

    # 设置相关参数
    index = np.arange(len(means_men))
    bar_height = 0.35

    # 画柱状图（水平方向）
    plt.barh(index,means_men,height=bar_height,alpha=0.2,color='b',label='men')
    plt.barh(index+bar_height,means_women,height=bar_height,alpha=0.8,color='r',label='women')
    plt.legend(loc='upper right',shadow=True)

    # 设置柱状图标示
    for x, y in zip(index, means_men):
        plt.text(y + 0.3, x + (bar_height / 2), y, ha="left", va="center")
    for x, y in zip(index, means_women):
        plt.text(y + 0.3, x + bar_height + (bar_height / 2), y, ha="left", va="center")

    # 设置刻度范围/坐标轴名称等
    plt.xlim(0,45)
    plt.xlabel('scores')
    plt.ylabel('group')
    plt.yticks(index+bar_height,("A组", "B组", "C组", "D组", "E组"))

    # 图像显示
    plt.show()
    return

barh_plot
def barh_plot():
    """
    barh plot
    """
    # 生成测试数据
    means_men = (20, 35, 30, 35, 27)
    means_women = (25, 32, 34, 20, 25)

    # 设置相关参数
    index = np.arange(len(means_men))
    bar_height = 0.35

    # 画柱状图(水平方向)
    plt.barh(index, means_men, height=bar_height, alpha=0.2, color="b", label="Men")
    plt.barh(index+bar_height, means_women, height=bar_height, alpha=0.8, color="r", label="Women")
    plt.legend(loc="upper right", shadow=True)

    # 设置柱状图标示
    for x, y in zip(index, means_men):
        plt.text(y+0.3, x+(bar_height/2), y, ha="left", va="center")
    for x, y in zip(index, means_women):
        plt.text(y+0.3, x+bar_height+(bar_height/2), y, ha="left", va="center")
    # 这里根据x,y的定位进行添加数据的标签，y+0.3相当于x的位置，x+bar_height相当于y的位置,y表示要添加的数据

    # 设置刻度范围/坐标轴名称等
    plt.xlim(0, 45)
    plt.xlabel("Scores")
    plt.ylabel("Group")
    plt.yticks(index+bar_height, ("A组", "B组", "C组", "D组", "E组"))

    # 图形显示
    plt.show()
    return
# barh_plot()
