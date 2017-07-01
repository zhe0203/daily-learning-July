# -*- coding: utf-8 -*-
'''
输入一个正整数N，输出以N为边长的螺旋矩阵
1  2  3   4
12 13 14  5
11 16 15  6
10 9  8   7
'''
import numpy as np
def result(N):
    num = range(1, N * N + 1)
    mat = np.zeros((N, N))
    k = 0
    for i in range(1,int(np.ceil(N / 2)) + 1):
        row,col = [],[]
        row.extend([i-1]*(N-i))
        row.extend(list(range(i-1,N-i)))
        row.extend([N-i]*(N-i))
        row.extend(list(range(N-i,i-1,-1)))
        col.extend(list(range(i-1, N - i)))
        col.extend([N - i] * (N-i))
        col.extend(list(range(N - i, i-1, -1)))
        col.extend([i - 1] * (N-i))

        a = list(zip(row, col))
        b = []

        for i in range(len(a)):
            if a[i] not in b:
                b.append(a[i])

        for l, m in b:
            mat[l,m] = num[k]
            k += 1
    print(mat)

result(6)
