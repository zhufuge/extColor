# -*- coding: utf-8 -*-

import sys
import operator
from numpy import *
from PIL import Image



def K_means(dataSet, k, e):
    """
    Descript:
     - 用 K-means 聚类算法 聚类

    Args:
     - dataSet: 用于聚类的数据集，目前只处理二维矩阵
     - k: 簇的数量
     - e: 数据处理值时，所使用的误差

    Return:
     - center: 包含每个簇的中心值的矩阵
    """
    n, m = dataSet.shape        # 取行列数值

    # 随机创建中心点
    center = zeros((k, m))
    for i in range(k):
        for j in range(m):
            center[i][j] = random.randint(dataSet.min(), dataSet.max())

    # 临时矩阵，用于计算误差
    center_tmp = zeros((k, m))
    # 误差矩阵，用于误差比较
    eMat = ones((k, m)) * e

    # any() boolean矩阵 求 &， 有一个及以上的 false 就为 false
    while (abs(center - center_tmp) > eMat).any():
        center_tmp = center.copy()
        center = move_center_by(center, dataSet)

    return center


def move_center_by(center, dataSet):
    """
    Descript:
     - 通过用 dataSet 中的数据来 移动 初始center，并返回新的center的值
     - 具体：标记 dataSet 中的每个数据点 的最短距离的中心点，并打上标签。同标签
    为同一组，旧中心点移动到组的中心，形成新的中心点

    Args:
     - center: 中心点的矩阵
     - dataSet: 用于移动中心点的数据集矩阵

    Return:
     - center: 移动后的中心点矩阵
    """
    n, m = dataSet.shape

    # 标签，记录每个数据点的最近中心点的下标
    labels = zeros((n, 1))
    for i in range(n):
        labels[i] = min_distance_index(center, dataSet[i])

    # 移动中心点
    # 通过记录的下标分组
    # 使用算术平均数，计算组的中心

    # 每个组的数量，初始化为 1，避免除0 的情况
    numOfLabel = ones((center.shape[0], 1))
    # center 置零，计算组的值的总和
    center = zeros(center.shape)
    for i in range(n):
        center[int(labels[i])] += dataSet[i]
        numOfLabel[int(labels[i])] += 1

    center = center / numOfLabel
    center = center.astype(int) # 取整

    return center


def min_distance_index(center, data):
    """
    Descript:
     - 计算并选择距离最短的中心点的下标

    Args:
     - center: 中心点
     - data: 一个数据点

    Return:
     - min_index: 最近点的下标
    """

    k = center.shape[0]

    # 该点到个点的每个坐标的距离
    # 欧拉距离公式
    diffMat = tile(data, (k, 1)) - center
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5

    # 找最近中心点的下标
    minDis = distances[0]
    min_index = 0
    for i in range(1, k):
        if distances[i] < minDis:
            minDis = distances[i]
            min_index = i

    return min_index


def getPixel(filename):
    """
    Descript:
     - 返回图片的像素点的rgb值的矩阵

    Args:
     - filename: 图片文件名

    Return:
     - pixel: 像素点rgb矩阵
    """
    pixel = []
    with Image.open(filename) as im:
        xsize, ysize = im.size
        for i in range(xsize):
            for j in range(ysize):
                r, g, b = im.getpixel((i,j))
                pixel.append((r, g, b))

    return pixel


def colorSort(color):
    """
    Descript:
     - 通过 r值 大小排序

    Args:
     - color: 像素点rgb矩阵

    Return:
     - color[argsort(color.T)[0]]: 排序后的矩阵

    先对 color 转置，使各像素点的 r值 在矩阵的第一行
    通过对第一行排序的下标，改变 color 中各像素点的顺序

    TODO 寻找综合排序，不再只通过 r值 排序
    """
    return color[argsort(color.T[0])]


def drawColorBlock(color, blockSize, filename):
    """
    Descript:
     - 通过像素点矩阵，绘制颜色方块图片，并保存。每个色块为正方形，大小为 blockSize

    Args:
     - color: 像素点rgb矩阵
     - blockSize: 每块色块的大小
     - filename: 保存的文件名
    """

    numOfColor = color.shape[0]

    im = Image.new('RGB', (blockSize * numOfColor, blockSize))
    # 将矩阵的每个点对应的图片的每个像素
    for i in range(numOfColor):
        for j in range(blockSize * i, blockSize * (i + 1)):
            for k in range(blockSize):
                im.putpixel((j, k), tuple(color[i]))

    im.save(filename)


def extractColor(filename, numOfColor=5, size=60):
    """
    Descript:
     - 提取图片中的主要颜色，并生成色块图片, 返回颜色矩阵。色块数量为 k，大小为size

    Args:
     - filename: 需提取的文件名
     - numOfColor: 颜色数量。默认为 5
     - size: 色块大小。默认为 60*60 px

    Return:
     - color: 提取的颜色矩阵
    """
    # 将文件路径去掉，使用当前文件夹路径
    filename = filename.split('\\')[-1]

    # 获得主要颜色的矩阵
    pixel = getPixel(filename)
    color = K_means(array(pixel), numOfColor, 5)
    color = colorSort(color)

    fname, ftype = filename.split('.')
    output = fname + '_color.' + ftype
    drawColorBlock(color, size, output)

    return color


def deColor(filename, numOfColor=10):
    """
    Descript:
     - 降低图片的颜色数量，用 numOfColor 种颜色来覆盖与其相近的颜色

    Args:
     - filename: 图片文件名
     - numOfColor: 颜色数量。默认为 10
    """

    # 将文件路径去掉，使用当前文件夹路径
    filename = filename.split('\\')[-1]

    with Image.open(filename) as im:
        newIm = Image.new('RGB', im.size)

    # 聚类后的颜色矩阵
    pixel = getPixel(filename)
    color = K_means(array(pixel), numOfColor, 5)

    # 颜色矩阵覆盖
    xsize, ysize = newIm.size
    for i in range(xsize):
        for j in range(ysize):
            min_index = min_distance_index(color, pixel[i * ysize + j])
            newIm.putpixel((i,j), tuple(color[min_index]))

    fname, ftype = filename.split('.')
    output = fname + '_de.' + ftype
    newIm.save(output)


def main():
    # 外部传参
    if len(sys.argv) == 1:
        print('Not found pictures.')
    else:
        color = extractColor(sys.argv[1])
        print(color)


if __name__ == "__main__":
    main()
