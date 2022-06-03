import numpy as np
import pandas as pd
import matplotlib as plt
import cv2 as cv
import xml.dom.minidom

from _0Utility import *

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

def box_visualization(img, points, img_size, img_name, mode = "New"):
    """
    :param img: 输入图像
    :param points: 图中所有的点
    :param img_size: 图像大小
    :param img_name: 要显示的图像的名字
    :param mode: “New”是直接显示处旋转完的，“Old”则显示没旋转前的(坐标还是相对坐标)
    :return:
    """
    if mode == "Old":
        for point in points:
            Point = np.array([[point[0] * img_size[0], point[1] * img_size[1]],
                               [point[2] * img_size[0], point[3] * img_size[1]],
                               [point[4] * img_size[0], point[5] * img_size[1]],
                               [point[6] * img_size[0], point[7] * img_size[1]]], dtype=np.int32)
            cv.polylines(img, np.int32([Point]), isClosed=True, color=(0,0,255), thickness = 1)
        cv.imshow(img_name,img)
        cv.waitKey(1000)

    if mode == "New":
        for point in points:
            cv.polylines(img, np.int32([point]), isClosed=True, color=(0,0,255), thickness = 1)
        cv.imshow(img_name,img)
        cv.waitKey(1000)

def rotate_img(img, points, angle):
    Width = img.shape[1]
    Height = img.shape[0]

    M = cv.getRotationMatrix2D((Width / 2, Height / 2), angle, 1.0)

    Cos1 = abs(M[0,0])
    Sin1 = abs(M[0,1])

    NewWidth = int(Cos1 * Width + Sin1 * Height)
    NewHeight = int(Sin1 * Width + Cos1 * Height)

    M[0,2] += (NewWidth / 2 - Width / 2)
    M[1,2] += (NewHeight / 2 - Height / 2)
    RotatedImg = cv.warpAffine(img, M, (NewWidth, NewHeight), cv.INTER_LINEAR, 0)

    # 对标签点进行透射变换
    for num,point in enumerate(points):
        points[num] = np.array([[point[0] * Width, point[1] * Height],
                           [point[2] * Width, point[3] * Height],
                           [point[4] * Width, point[5] * Height],
                           [point[6] * Width, point[7] * Height]], dtype=np.int32)

        print(M[:2, :2],"\ndot\n",np.dot(points[num], M[:2, :2]))
        points[num] = np.dot(M[:2, :2],points[num].T).T + M[:, -1]


    # cv.imshow("RotatedImg", RotatedImg)
    # cv.waitKey(1000)
    return RotatedImg, points

if __name__ == "__main__":
    # 图像以及标签目标路径
    # 2021内录
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\Dataset2021"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\Dataset2021"
    # sj内录
    ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\WinterData"
    LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\WinterData"
    # 训练使用
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\UsingTrain"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\UsingTrain"
    # 使用全部数据集
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource"
    # 西南交大
    # ImgPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\GKD\data"
    # LabelPath = r"E:\718\Code\AI\NewClassify\Dataset\DatasetSource\GKD\data"

    ImgList = []
    LabelList = []
    # 目标装甲板类别
    TargetArmor = [
        "armor_hero_red", "armor_hero_blue", "armor_hero_none",
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
    # 原图裁剪大小
    ImgSize = (640, 480)
    # 最终得到的图片大小
    TargetSize = (36, 48)
    # 各种索引值
    CorrectIndex = 0
    ImgIndex = 0

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
        Img = cv.resize(Img, ImgSize)  # 这一步放缩非常重要！！！
        # 读取标签和点
        if LabelSuffix == "txt":
            ArmorClasses, Points = txt_reader(label, TargetArmor)
        elif LabelSuffix == "xml":
            ArmorClasses, Points = xml_reader(label, TargetArmor)
        else:
            print("Suffix Error!!!")


        # box_visualization(Img, Points, ImgSize, "InitImg", mode="Old")
        RotatedImg, NewPoints = rotate_img(Img, Points, -30)
        box_visualization(RotatedImg, NewPoints, ImgSize, "RotatedImg", mode="New")
    #     # 开始处理图片
    #     for num, point in enumerate(Points):
    #         # 标签检查,防止一些奇怪的从右上开始标签
    #         point = correct_label(point)
    #         print(point)
    #        
    #
    #         ArmorImg = imgprocess(Img, point, Mode="RectangleAndGray")  # "MaskAndGray""RectangleAndGray"
    #         # 图像裁剪
    #         CropArmorImg = resize_equal(ArmorImg, TargetSize, equal=False)
    #         # 图像保存
    #         saveimg(CropArmorImg, 0.8, TrainCount, ValCount,
    #                 ArmorClasses[num], ImgIndex, img_name=Imgs[CorrectIndex], mode="OneFolder")
    #         print("Index:", ImgIndex)
    #         ImgIndex += 1
    #
    # print("TrainCount:", TrainCount, "ValCount:", ValCount)