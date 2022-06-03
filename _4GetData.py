"""
@Function: 数据获取(支持txt格式与xml格式)

@Time:2022-3-15

@Author:马铭泽
"""
import cv2 as cv
from _0Utility import *
import xml.dom.minidom
import random
import time
import pandas as pd
import numpy as np

# 装甲板索引列表
ArmorList = ["armor_sentry_blue", "armor_hero_blue", "armor_engineer_blue", "armor_infantry_3_blue",
             "armor_infantry_4_blue", "armor_infantry_5_blue", "armor_outpost_blue", "armor_bs_blue",
             "armor_base_blue", "armor_sentry_red", "armor_hero_red", "armor_engineer_red",
             "armor_infantry_3_red", "armor_infantry_4_red", "armor_infantry_5_red", "armor_outpost_red",
             "armor_bs_red", "armor_base_red", "armor_sentry_none", "armor_hero_none",
             "armor_engineer_none", "armor_infantry_3_none", "armor_infantry_4_none", "armor_infantry_5_none",
             "armor_outpost_none", "armor_bs_none", "armor_base_none", "armor_sentry_purple",
             "armor_hero_purple", "armor_engineer_purple", "armor_infantry_3_purple", "armor_infantry_4_purple",
             "armor_infantry_5_purple", "armor_outpost_purple", "armor_bs_purple", "armor_base_purple"]

# 标签矫正，处理一些神奇的标签
def correct_label(Points):
    """
    :param Points: 某个装甲板对应的四个角点的坐标
    :return: None
    """
    if Points[0] > Points[4]:
        Points = Points[::-1]
    return Points


# 等比例裁剪
def resize_equal(src, size, equal = True):
    """
    :param src: 原图
    :param size: 目标大小
    :param equal: 是否等比例裁剪
    :return: 裁剪过后的图
    """
    print("Start to resize Img")
    if equal:
        # 对ROI区域进行等比例缩放
        OldSize = src.shape[0:2]
        Ratio = min(float(size[i]) / (OldSize[i]) for i in range(len(OldSize)))
        NewSize = tuple([int(i * Ratio) for i in OldSize])
        src = cv.resize(src, (NewSize[1], NewSize[0]))
        pad_w = size[1] - NewSize[1]
        pad_h = size[0] - NewSize[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        # 对边界进行填充
        NewImg = cv.copyMakeBorder(src, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
    else:
        NewImg = cv.resize(src, (size[1], size[0]))

    return NewImg

# 防止最小外接矩形超出图像范围
def preventExceed(minx, miny, temp_width, temp_height, img_size):
    """
    :param minx: x方向上最小的点
    :param miny: y方向上最小的点
    :param temp_width: 宽度上的差值
    :param temp_height: 高度上的差值
    :param img_size: 原图
    :return:
    """


# 图片保存
def saveimg(img, ratio, train_cnt, val_cnt, id, index, img_name = None, mode="TwoFolders"):
    """
    :param img: 要保存的图片
    :param ratio: 测试集和训练集的比率
    :param train_cnt: 训练集数量统计
    :param val_cnt: 测试集数量统计
    :param id: 图片对应的标签
    :param index: 图片的索引值
    :param mode: 模式,支持训练集+测试集,若选择"OneFolders"则都会保存到测试集中
    :return:
    """
    print("Start to save Img")
    if (ratio > 1) | (ratio < 0):
        print("Out of range!!!")
        return 0

    #通过图片通道数判断是否改变id
    if len(img.shape) == 2:
        id //= 3
        # 为了负样本要让id+1
        id += 1

    if mode == "TwoFolders":
        if random.random() < ratio:
            cv.imwrite("Dataset/Train/Img{}_{}.bmp".format(index, id), img)
            train_cnt[id] += 1
        else:
            cv.imwrite("Dataset/Val/Img{}_{}.bmp".format(index, id), img)
            val_cnt[id] += 1

    if mode == "OneFolder":
        cv.imwrite("Dataset/Val/Img{}_{}.bmp".format(index, id), img)
        val_cnt[id] += 1

    if mode == "DeBug":
        ImgName = img_name.split(".")[0].split("\\")[-2:]
        img = cv.resize(img, ImgSize)
        cv.imwrite("Dataset/Debug/{}_{}.bmp".format(ImgName, id), img)


# xml标签读取
def xml_reader(XmlLabel, Target):
    """
    :param XmlLabel: xml标签
    :param Target: 目标装甲板列表
    :return: 标签中对应的ID值与对应的四点列表
    """
    print("Start Process xml Label")
    # 存储单组点的列表
    Point = []
    # 存储单张图中所有的装甲板四点信息
    Points = []
    # 存储每个四点信息对应的类别
    ArmorClasses = []

    # 读取xml的相关信息
    Xml = xml.dom.minidom.parse(XmlLabel)
    root = Xml.documentElement
    # 获取标签对应的图像的名字
    Img_Name = root.getElementsByTagName('filename')[0].childNodes[0].data
    name = root.getElementsByTagName('name')
    bndbox = root.getElementsByTagName('bndbox')

    for i in range(len(list(bndbox))):
        # 存储装甲板角点的坐标与ID
        Class = name[i].childNodes[0].data
        if Class not in Target:
            continue
        else:
            ID = Target.index(Class)
        ArmorClasses.append(ID)
        for child in bndbox[i].childNodes:
            if (child.nodeName != '#text'):
                Point.append(float(child.childNodes[0].data))
        Points.append(Point)
        Point = []

    return ArmorClasses, Points


# txt标签
def txt_reader(TxtLabel, Target):
    """
    :param TxtLabel: txt标签
    :param Target: 目标装甲板
    :return: 标签中对应的ID值与对应的四点列表
    """
    print("Start Process txt Label")
    # 存储单组点的列表
    Point = []
    # 存储单张图中所有的装甲板四点信息
    Points = []
    # 存储每个四点信息对应的类别
    ArmorClasses = []

    # 读取txt的相关信息
    txt = pd.read_table(TxtLabel, sep=' ', header=None)
    for i in range(txt.shape[0]):
        # 获取装甲板类型
        ID = txt[0][i]
        if ID not in [ArmorList.index(i) for i in Target]:
            continue
        else:
            # ID转换，根据目标类别更改类别
            ID = [ArmorList.index(i) for i in Target].index(ID)
        ArmorClasses.append(ID)

        # 开始存储角点信息
        for j in range(txt.shape[1] - 1):
            # 读取四个角点，左上，左下，右下，右上
            Point.append(txt[j + 1][i])

        Points.append(Point)
        Point = []

    return ArmorClasses, Points


# 图像预处理
def imgprocess(Img, Point, Mode):
    """
    :param Img: 原图
    :param Point: 装甲板点集
    :param Mode: 模式："WarpAndGray"-透射变换+灰度, "Rectangle":最小外接矩形, "RectangleAndGray":最小外接矩形+灰度, "Mask":掩码提取(无背景外接矩形), "MaskAndGray":掩码加灰度
    :return: 处理后的图像
    """
    print("Start to process Img, Mode is:", Mode)
    # 投射变换加转换灰度模式
    if Mode == "WarpAndGray":
        for i in range(len(Point)):
            if Point[i] < 0:
                Point[i] = 0
        Height = ((Point[2] * ImgSize[0] - Point[0] * ImgSize[0]) ** 2 +
                  (Point[3] * ImgSize[1] - Point[1] * ImgSize[1]) ** 2) ** 0.5
        Width = ((Point[6] * ImgSize[0] - Point[0] * ImgSize[0]) ** 2 +
                 (Point[7] * ImgSize[1] - Point[1] * ImgSize[1]) ** 2) ** 0.5
        p1 = np.float32([[Point[0] * ImgSize[0], Point[1] * ImgSize[1]],
                         [Point[2] * ImgSize[0], Point[3] * ImgSize[1]],
                         [Point[4] * ImgSize[0], Point[5] * ImgSize[1]],
                         [Point[6] * ImgSize[0], Point[7] * ImgSize[1]]])
        p2 = np.float32([[0, 0],
                         [0, Height - 1],
                         [Width - 1, Height - 1],
                         [Width - 1, 0], ])
        M = cv.getPerspectiveTransform(p1, p2)
        WarpImg = cv.warpPerspective(Img, M, (int(Width), int(Height)))
        GrayImg = cv.cvtColor(WarpImg, cv.COLOR_BGR2GRAY)

        return GrayImg


    # 最小外接矩形模式
    if (Mode == "Rectangle") | (Mode == "RectangleAndGray"):
        Height = ((Point[2] * ImgSize[0] - Point[0] * ImgSize[0]) ** 2 +
                  (Point[3] * ImgSize[1] - Point[1] * ImgSize[1]) ** 2) ** 0.5

        MinX = int(min(Point[0] * ImgSize[0], Point[2] * ImgSize[0]))
        if MinX < 0:
            MinX = 0

        MaxX = int(max(Point[4] * ImgSize[0], Point[6] * ImgSize[0]))
        if MaxX > ImgSize[0]:
            MaxX = ImgSize[0]

        MinY = int(min(Point[1] * ImgSize[1], Point[7] * ImgSize[1]) - Height//2)
        if MinY < 0:
            MinY = 0

        MaxY = int(max(Point[3] * ImgSize[1], Point[5] * ImgSize[1]) + Height//2)
        if MaxY > ImgSize[1]:
            MaxY = ImgSize[1]

        print(int(MinY),int(MaxY), int(MinX),int(MaxX))
        ROI = Img[int(MinY):int(MaxY), int(MinX):int(MaxX), :]
        if Mode == "RectangleAndGray":
            ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)

        return ROI

    if (Mode == "Mask") | (Mode == "MaskAndGray"):
        # 读取装甲板的点集信息
        Points = np.array([[Point[0] * ImgSize[0], Point[1] * ImgSize[1]],
                           [Point[2] * ImgSize[0], Point[3] * ImgSize[1]],
                           [Point[4] * ImgSize[0], Point[5] * ImgSize[1]],
                           [Point[6] * ImgSize[0], Point[7] * ImgSize[1]]],dtype=np.int32)

        # 生成掩码(注意必须是单通道的！！！)
        Mask = np.zeros(Img.shape[:2],dtype=np.uint8)
        # 绘制装甲板的多边形
        cv.polylines(Mask, np.int32([Points]), isClosed=True, color=(255,255,255), thickness = 1)
        # 填充
        cv.fillPoly(Mask, np.int32([Points]),(255, 255, 255))
        # 筛选出每张图片中的装甲板，两种方式都可以实现遮罩
        # MaskedImg = cv.add(np.array(Img, dtype=np.uint8), np.zeros_like(Img,dtype=np.uint8), mask=np.array(Mask, dtype=np.uint8))
        MaskedImg = cv.bitwise_and(Img, Img, mask = Mask)


        #截取ROI
        MinX = np.min(Points[:,0], axis=0)
        if MinX < 0: MinX = 0
        MaxX = np.max(Points[:,0], axis=0)
        MinY = np.min(Points[:,1], axis=0)
        if MinY < 0: MinY = 0
        MaxY = np.max(Points[:,1], axis=0)

        ROI = MaskedImg[MinY:MaxY, MinX:MaxX, :]
        if Mode == "MaskAndGray":
            ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
        #
        # cv.imshow("Mask",MaskedImg)
        # cv.imshow("ROI",cv.resize(ROI, (480,360)))
        # if cv.waitKey(500) == ord('q'):
        #     time.sleep(5)
        # cv.waitKey(500)

        return ROI

    # 二值化模式
    if (Mode == "Thresholed"):
        Height = ((Point[2] * ImgSize[0] - Point[0] * ImgSize[0]) ** 2 +
                  (Point[3] * ImgSize[1] - Point[1] * ImgSize[1]) ** 2) ** 0.5
        Width = ((Point[6] * ImgSize[0] - Point[0] * ImgSize[0]) ** 2 +
                 (Point[7] * ImgSize[1] - Point[1] * ImgSize[1]) ** 2) ** 0.5

        MinX = int(min(Point[0] * ImgSize[0], Point[2] * ImgSize[0]) + Width // 5)
        if MinX < 0:
            MinX = 0

        MaxX = int(max(Point[4] * ImgSize[0], Point[6] * ImgSize[0]) - Width // 5)
        if MaxX > ImgSize[0]:
            MaxX = ImgSize[0]

        MinY = int(min(Point[1] * ImgSize[1], Point[7] * ImgSize[1]) - Height // 3)
        if MinY < 0:
            MinY = 0

        MaxY = int(max(Point[3] * ImgSize[1], Point[5] * ImgSize[1]) + Height // 3)
        if MaxY > ImgSize[1]:
            MaxY = ImgSize[1]

        Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
        ret, Otsu = cv.threshold(Img, 30, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        ROI = Otsu[int(MinY):int(MaxY), int(MinX):int(MaxX)]


        return ROI


if __name__ == "__main__":
    # 图像以及标签目标路径
    # 2021内录
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\Dataset2021"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\Dataset2021"
    # sj内录
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\WinterData"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\WinterData"
    #训练使用
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\UsingTrain"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\UsingTrain"
    # 使用全部数据集
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource"
    # 西南交大
    ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\GKD\data"
    LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\GKD\data"

    ImgList = []
    LabelList = []
    # 目标装甲板类别
    TargetArmor = [
               "armor_hero_red", "armor_hero_blue","armor_hero_none",
               "armor_engineer_red", "armor_engineer_blue", "armor_engineer_none",
               "armor_infantry_3_red", "armor_infantry_3_blue", "armor_infantry_3_none",
               "armor_infantry_4_red", "armor_infantry_4_blue", "armor_infantry_4_none",
               "armor_infantry_5_red", "armor_infantry_5_blue", "armor_infantry_5_none",
               "armor_sentry_red", "armor_sentry_blue", "armor_sentry_none",
               "armor_outpost_none", "armor_outpost_blue", "armor_outpost_none",
               "armor_base_red", "armor_base_blue", "armor_base_none"
                ]

    # 训练集和测试集样本数量统计
    TrainCount = [0 for i in range(len(TargetArmor))]
    ValCount = [0 for j in range(len(TargetArmor))]
    #原图裁剪大小
    ImgSize = (640,480)
    # 最终得到的图片大小
    TargetSize = (36,48)
    # 各种索引值
    CorrectIndex = 0
    ImgIndex = 0




    # 读取数据并排序
    FatherImg_Folder = os.listdir(ImgPath)
    Imgs = sorted(Traverse_Folder(FatherImg_Folder, ImgPath, ImgList, [".jpg", ".png"]))
    FatherLabel_Folder = os.listdir(LabelPath)
    Labels = sorted(Traverse_Folder(FatherLabel_Folder, LabelPath, LabelList, [".txt", ".xml"]))
    print("Img", len(Imgs))
    print("label", len(Labels))

    # 遍历标签标签对位
    for num, label in enumerate(Labels):
        # 读取标签的后缀
        LabelSuffix = label.split("\\")[-1].split(".")[-1]
        # 图片与标签的对位(图片比标签多)
        try:
            while (label.split("\\")[-1].split(".")[0] !=
                   Imgs[CorrectIndex].split("\\")[-1].split(".")[0]):
                CorrectIndex = CorrectIndex + 1
        except:
            print("Error! Can't match!")
            print("label", label.split("\\")[-1].split(".")[0].split("-")[-1])

        # 读取图片
        print(Imgs[CorrectIndex])
        Img = cv.imread(Imgs[CorrectIndex])
        Img = cv.resize(Img, ImgSize) # 这一步放缩非常重要！！！
        # 读取标签和点
        if LabelSuffix == "txt":
            ArmorClasses, Points = txt_reader(label, TargetArmor)
        elif LabelSuffix == "xml":
            ArmorClasses, Points = xml_reader(label, TargetArmor)
        else:
            print("Suffix Error!!!")

        # 开始处理图片
        for num, point in enumerate(Points):
            # 标签检查,防止一些奇怪的从右上开始标签
            point = correct_label(point)
            print(point)
            # 获取预处理之后截取的装甲板
            ArmorImg = imgprocess(Img, point, Mode = "RectangleAndGray") #"MaskAndGray""RectangleAndGray"
            # 图像裁剪
            CropArmorImg = resize_equal(ArmorImg,TargetSize, equal=False)
            # 图像保存
            saveimg(CropArmorImg, 0.8, TrainCount, ValCount,
                    ArmorClasses[num], ImgIndex,img_name=Imgs[CorrectIndex], mode="OneFolder")
            print("Index:",ImgIndex)
            ImgIndex += 1

    print("TrainCount:", TrainCount, "ValCount:", ValCount)

