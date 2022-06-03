"""
@Function: 通用的各类函数与类

@Time:2022-3-14

@Author:马铭泽
"""
import logging
import os
from logging import handlers
import matplotlib.pyplot as plt
import itertools
import numpy as np

#日志初始化类
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',mode='w',fmt='%(asctime)s- %(levelname)s：%(message)s '):
        #设置日志名称
        self.filename = os.path.join("Output/log",filename + ".log")
        self.logger = logging.getLogger(filename)
        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))
        # 设置日志格式
        format_str = logging.Formatter(fmt,datefmt='%Y-%m-%d-%H:%M:%S')
        # 往文件里写入
        FileHandler = logging.FileHandler(filename=filename,mode=mode)
        # 设置文件里写入的格式
        FileHandler.setFormatter(format_str)
        #把对象加到logger里
        self.logger.addHandler(FileHandler)



# 递归遍历文件夹
def Traverse_Folder(Father_Folder,Init_Path,Lists,Target_Suffix) -> list:
    """
    :param Father_Folder: 要寻找的目标路径
    :param Init_Path: 最初始的顶层路径
    :param Lists: 用于存储最终找到的所有内容
    :param Target_Suffix: 目标类型文件
    :return:
    """
    # print(Father_Folder)
    for child1 in Father_Folder:
        Child1_Path = os.path.join(Init_Path, child1)
        Suffix = os.path.splitext(Child1_Path)[1]
        if(os.path.isdir(Child1_Path)):
            Child1_Path_Folders = os.listdir(Child1_Path)
            Traverse_Folder(Child1_Path_Folders,Child1_Path,Lists,Target_Suffix)
        else:
            if(Suffix in Target_Suffix):
                # for Num,item in enumerate(Father_Folder):
                #     Father_Folder[Num] = os.path.join(Init_Path,item)
                # print(Child1_Path)
                Lists.append(Child1_Path)
    return Lists


#绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    :param cm: 计算出的混淆矩阵的值
    :param classes: 混淆矩阵中每一行每一列对应的列
    :param normalize: True:显示百分比, False:显示个数
    :param title: 图像标题
    :param cmap: 颜色条
    :return: 变换过的Matrix(如果计算准确度的话返回它才有意义)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.figure(figsize=(8, 8))#设置画布大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return cm


if __name__ == "__main__":
    path = r"E:\718\Code\AI\Armor_Classify\NewDataset\WinterData\sj1.9"
    list = []
    Father_Folder = os.listdir(path)
    list = Traverse_Folder(Father_Folder, path, list, ".txt")
