# -*- coding: utf-8 -*-

import sys
import operator
from numpy import *
from PIL import Image



def K_means(dataSet, k, e):
    # 目前 dataSet 为 二维数组
    n, m = dataSet.shape

    maxNum = dataSet.max()
    minNum = dataSet.min()

    # 随机创建中心点
    center = zeros((k, m))
    for i in range(k):
        for j in range(m):
            center[i][j] = random.randint(minNum, maxNum)

    # 临时矩阵
    center_tmp = zeros((k, m))
    # 误差矩阵
    eMat = ones((k, m)) * e

    diff = center - center_tmp
    while (diff > eMat).any() :
        center_tmp = center.copy()
        center = move_center_by(center, dataSet)

        diff = abs(center - center_tmp)

    return center


def move_center_by(center, dataSet):
    n, m = dataSet.shape

    # 用于标记，每个点归哪个中心点
    labels = zeros((n, 1))
    for i in range(n):
        labels[i] = min_distances(center, dataSet[i])


    numOfCenter = center.shape[0]
    # 移动中心点
    for i in range(numOfCenter):
        numOfLabel = 1
        sumOfAxis = ones((1, m))*128
        for j in range(n):
            if labels[j] == i:
                numOfLabel += 1
                sumOfAxis += dataSet[j]

        center[i] = sumOfAxis / numOfLabel
        center = center.astype(int)

    return center


def min_distances(center, data):
    # 计算距离， 选择距离最短的中心点，并标记

    k = center.shape[0]

    # 该点到个点的每个坐标的距离
    diffMat = tile(data, (k, 1)) - center
    # 平方
    sqDiffMat = diffMat**2
    # 求和
    sqDistances = sqDiffMat.sum(axis = 1)
    # 开根号
    distances = sqDistances**0.5

    minDis = distances[0]
    min_index = 0
    for i in range(1, k):
        if distances[i] < minDis:
            minDis = distances[i]
            min_index = i

    return min_index


def getPixel(filename):
    pixel = []
    with Image.open(filename) as im:
        xsize, ysize = im.size
        for i in range(xsize):
            for j in range(ysize):
                r, g, b = im.getpixel((i,j))
                pixel.append((r, g, b))

    return pixel


def colorSort(color):
    return color[argsort(color.T)[0]]


def drawColorBlock(color, blockSize, filename):
    n, m = color.shape

    Size = blockSize.copy()
    Size[0] *= n

    im = Image.new('RGB', Size)
    for i in range(n):
        for j in range(Size[1] * i, Size[1] * (i + 1)):
            for k in range(Size[1]):
                im.putpixel((j, k), tuple(color[i]))

    im.save(filename)


def extractColor(filename, k=5, size=60):
    filename = filename.split('\\')[-1]
    pixel = getPixel(filename)

    e = 5                       # 误差
    color = K_means(array(pixel), k, e)
    color = colorSort(color)

    fname, ftype = filename.split('.')
    output = fname + '_color.' + ftype
    blockSize = [size, size]
    drawColorBlock(color, blockSize, output)

    return color


def deColor(filename, k=10):
    filename = filename.split('\\')[-1]
    with Image.open(filename) as im:
        xsize, ysize = im.size
        newIm = Image.new('RGB', (xsize, ysize))

    pixel = getPixel(filename)
    color = K_means(array(pixel), k, 5)

    for i in range(xsize):
        for j in range(ysize):
            min_index = min_distances(color, pixel[i * ysize + j])
            newIm.putpixel((i,j), tuple(color[min_index]))


    fname, ftype = filename.split('.')
    output = fname + '_de.' + ftype
    newIm.save(output)


def main():
    if len(sys.argv) == 1:
        print('Not found pictures.')
    else:
        color = extractColor(sys.argv[1])
        print(color)


if __name__ == "__main__":
    main()
