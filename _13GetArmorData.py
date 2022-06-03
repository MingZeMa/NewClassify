# 装甲板数据分析
import copy
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xml.dom.minidom
from _0Utility import Traverse_Folder

# xml标签读取
def xml_reader(XmlLabel):
    """
    :param XmlLabel: xml标签
    :param Target: 目标装甲板列表
    :return: 标签中对应的四点列表
    """
    print("Start Process xml Label")
    # 存储单组点的列表
    Point = []
    # 存储单张图中所有的装甲板四点信息
    Points = []

    # 读取xml的相关信息
    Xml = xml.dom.minidom.parse(XmlLabel)
    root = Xml.documentElement
    # 获取标签对应的图像的名字
    Img_Name = root.getElementsByTagName('filename')[0].childNodes[0].data
    name = root.getElementsByTagName('name')
    bndbox = root.getElementsByTagName('bndbox')

    for i in range(len(list(bndbox))):
        # 存储装甲板角点的坐标与ID
        for child in bndbox[i].childNodes:
            if (child.nodeName != '#text'):
                Point.append(float(child.childNodes[0].data))
        Points.append(Point)
        Point = []

    return Points


# txt标签
def txt_reader(TxtLabel):
    """
    :param TxtLabel: txt标签
    :param Target: 目标装甲板
    :return: 标签中对应的四点列表
    """
    print("Start Process txt Label")
    # 存储单组点的列表
    Point = []
    # 存储单张图中所有的装甲板四点信息
    Points = []


    # 读取txt的相关信息
    txt = pd.read_table(TxtLabel, sep=' ', header=None)
    for i in range(txt.shape[0]):
        # 开始存储角点信息
        for j in range(txt.shape[1] - 1):
            # 读取四个角点，左上，左下，右下，右上
            Point.append(txt[j + 1][i])

        Points.append(Point)
        Point = []

    return Points

# 标签矫正，处理一些神奇的标签
def correct_label(Points):
    """
    :param Points: 某个装甲板对应的四个角点的坐标
    :return: None
    """
    if Points[0] > Points[4]:
        Points = Points[::-1]
    return Points

# 获取初始数据，将八个点的坐标存起来
def get_data():
    # 八个点的标签
    PointDict = {"LeftTopX": 0, "LeftTopY": 0, "LeftBottomX": 0, "LeftBottomY": 0,
                 "RightBottomX": 0, "RightBottomY": 0, "RightTopX": 0, "RightTopY": 0}
    PointList = []
    # 获取所有装甲板的标签路径
    DataPath = "Dataset/DatasetSource"
    DataList = []
    ArmorData = Traverse_Folder(os.listdir(DataPath), DataPath, DataList, [".txt", ".xml"])

    # 数据读取
    for num, data in enumerate(ArmorData):
        # 读取标签的后缀
        LabelSuffix = data.split("\\")[-1].split(".")[-1]
        if LabelSuffix == "txt":
            Points = txt_reader(data)
        elif LabelSuffix == "xml":
            Points = xml_reader(data)
        else:
            print("Suffix Error!!!")

        for num, point in enumerate(Points):
            # 矫正标反的数据
            Point = correct_label(point)

            for num, key in enumerate(PointDict.keys()):
                PointDict[key] = Point[num]
                NewDict = copy.deepcopy(PointDict)

            PointList.append(NewDict)

    PointDataFrame = pd.DataFrame(PointList)
    PointDataFrame.to_excel("PointData.xlsx")
    print(PointDataFrame.head())

# 数据处理
def processdata():
    ArmorDataFrame = pd.read_excel("PointData.xlsx")





if __name__ == "__main__":
    get_data()
    ArmorData = pd.read_excel("PointData.xlsx")
    # 计算装甲板数据
    ArmorData["ArmorWideth"] = " "
    ArmorData["ArmorWideth"] = (ArmorData)