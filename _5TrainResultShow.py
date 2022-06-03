"""
@Function: 根据日志绘制训练集和测试集的Loss还有Accuracy曲线

@Time:2022-3-15

@Author:马铭泽
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#numpy数据控制
np.set_printoptions(suppress=True, precision=8)

#获取loss和accuracy的各项初始数据
def getdata(DatainLines):
    """
    :param DatainLines: 日志中每一行的数据
    :return: 测试集和验证集的Accuracy与Loss
    """
    # 创建存储的列表
    # 创建存储损失的列表
    TrainLosses = []
    ValidLosses = []
    # 创建存储准确率的列表
    TrainAccuracys = []
    ValAccuracys = []
    for i, line in enumerate(DatainLines):
        # 切分训练集日志，.strip()的作用时清除两侧空格
        if "Train" in line:
            Epoch = float(line.strip().split("\t")[2].split(":")[1])
            Lr = float(line.strip().split("\t")[3].split(":")[1])
            Loss = float(line.strip().split("\t")[4].split(":")[1])

            # 获取准确度
            Accuracy = float(line.strip().split("\t")[5].split(":")[1])
            TrainAccuracys.append(np.array([Epoch, Lr, Accuracy]))

            if Loss < 1:
                TrainLosses.append(np.array([Epoch, Lr, Loss]))

        # 切分验证集日志,有几个空格就是第几个
        if "Valid" in line:
            Epoch = float(line.strip().split("\t")[16].split(":")[1])
            Lr = float(line.strip().split("\t")[17].split(":")[1])
            Loss = float(line.strip().split("\t")[18].split(":")[1])

            # 获取准确度
            Accuracy = float(line.strip().split("\t")[19].split(":")[1])
            ValAccuracys.append(np.array([Epoch, Lr, Accuracy]))
            # if Loss<1:
            ValidLosses.append(np.array([Epoch, Lr, Loss]))

    # 对数据按行拼接
    TrainLosses = np.vstack(TrainLosses)
    ValidLosses = np.vstack(ValidLosses)
    TrainAccuracys = np.vstack(TrainAccuracys)
    ValAccuracys = np.vstack(ValAccuracys)

    return TrainAccuracys, TrainLosses, ValAccuracys, ValidLosses


# 处理需要根据学习率分割的数据
def dataprocess(Datas):
    """
    :param Datas: 需要根据学习率进行分割的数据
    :return: 分割后统一存放的列表
    """
    # 将训练集的损失按照学习率的不同分隔开
    DifferentLrs = np.unique(Datas[..., 1])
    # 分割数据
    NewData = []
    for Lr in DifferentLrs:
        Index = np.where(Datas[:, 1] == Lr)
        Data = Datas[Index, :][0]
        NewData.append(Data)

    return NewData

def get_cmap(n, name='rainbow'):
    """
    :param n: 数
    :param name: 色彩空间名字
    :return: 任意一个颜色
    """
    return plt.cm.get_cmap(name, n)



def Visualization(TrainDataList, ValDataList, XLabel, YLabel, Title, LabelPosition, LogNames, splitwithLr=True ):
    """
    :param TrainDatas: 训练集的数据集合
    :param ValDatas: 测试集的数据集合
    :param XLabel: x轴的标签
    :param YLabel: y轴的标签
    :param Title: 图片的标题
    :param LabelPosition: 标签的位置
    :param LogNames: 每个日志的版本号
    :param splitwithLr: 时候按照学习率分割训练集的数据
    :return: 无
    """
    # 获取颜色
    cmap = get_cmap(len(ValDataList))
    for num, (TrainDatas, ValDatas, LogName) in enumerate(zip(TrainDataList, ValDataList, LogNames)):
        if splitwithLr:
            for TrainData in TrainDatas:
                plt.plot(TrainData[...,0],TrainData[...,2],label = None)
        else:
            plt.plot(TrainDatas[..., 0], TrainDatas[..., 2], label=None)

        plt.plot(ValDatas[...,0],ValDatas[...,2],label = "Val" + LogName, color=cmap(num))

    #设置x，y轴标签
    plt.ylabel(XLabel)
    plt.xlabel(YLabel)
    plt.title(Title)
    #显示标签框
    plt.legend(loc=LabelPosition,fontsize=7)
    plt.show()


if __name__ == "__main__":
    # 判断是否根据学习率去进行分割
    SplitWithLr = False

    TrainLossesList = []
    ValLossesList = []
    TrainAccuracyList = []
    ValAccuracyList = []
    LogNames = []
    LogFolder = "Output/log"
    for logfile in os.listdir(LogFolder):
        LogNames.append(logfile.split("_")[0])
        with open(os.path.join(LogFolder, logfile), "r") as log:
            lines = log.readlines()
            TrainAccuracys, TrainLosses, ValAccuracys, ValLosses = getdata(lines)
            if SplitWithLr:
                # 根据学习率对数据进行划分
                NewTrainLosses = dataprocess(TrainLosses)
                NewTrainAccuracys = dataprocess(TrainAccuracys)
                TrainLossesList.append(NewTrainLosses)
                TrainAccuracyList.append(NewTrainAccuracys)
            else:
                # 若不用学习率进行分割则直接不处理放入列表中
                TrainLossesList.append(TrainLosses)
                TrainAccuracyList.append(TrainAccuracys)
            # 单个数据存入列表
            ValLossesList.append(ValLosses)
            ValAccuracyList.append(ValAccuracys)


    #绘制相关曲线
    Visualization(TrainLossesList, ValLossesList, "Loss", "Epoch", "Loss-Curve", "upper right",
                  LogNames, False)
    Visualization(TrainAccuracyList, ValAccuracyList, "Accuracy", "Epoch", "Accuracy-Curve",
                  'lower right', LogNames, False)
