"""
@Function: 相关评估指标的计算与绘制(现支持PRC ROC)

@Time:2022-3-14

@Author:马铭泽
"""
import numpy as np
import numpy.random as r
import sklearn.metrics as m
import matplotlib.pyplot as plt
import sys

# numpy数据输出设置
np.set_printoptions(suppress=True, precision=4)


# Roc曲线绘制
def ROC_AUC(Labels, Outputs, ShowROC=False):
    """
    :param Labels: 标签集合
    :param Outputs: 输出集合
    :param ShowROC: 是否展示
    :return: 各项指标
    """
    fprs = []
    tprs = []
    AUCs = []
    for num,(LabelData,OutputData) in enumerate(zip(Labels,Outputs)):
        fpr, tpr, thresholds = m.roc_curve(LabelData, OutputData)
        AUC = m.auc(fpr, tpr)  # AUC其实就是ROC曲线下边的面积
        fprs.append(fpr)
        tprs.append(tpr)
        AUCs.append(AUC)
        if ShowROC:
            plt.figure('ROC curve')
            plt.plot(fpr, tpr, label=num)

    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc="lower right", fontsize=7)
    plt.show()

    return fprs, tprs, AUCs


def PRC(Labels, Outputs, ShowPRC=False):
    """
    :param Labels: 标签集合
    :param Outputs: 输出集合
    :param ShowPRC: 是否展示
    :return: 各项指标
    """
    recalls = []
    precisions = []
    MFs = []
    APs = []
    for num, (LabelData,OutputData) in enumerate(zip(Labels,Outputs)):
        # 返回查准率和召回率
        precision, recall, thresholds = m.precision_recall_curve(LabelData, OutputData)
        # 计算F1Score,它是用来均衡查准率和召回率的一个数据
        F1Score = 2 * (precision * recall) / ((precision + recall) + sys.float_info.min)# TODO:不知道为什么分母上要加float
        # 找到F1Score对应的最大值的索引值
        MF = F1Score[np.argmax(F1Score)]
        # 计算平均精度
        AP = m.average_precision_score(LabelData, OutputData)
        recalls.append(recall)
        precisions.append(precision)
        MFs.append(MF)
        APs.append(AP)
        # 输出PRC图象
        if ShowPRC:
            plt.figure('Precision recall curve')
            plt.plot(recall, precision, label=num)

    plt.ylim([0.0, 1.0])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc="lower left",fontsize=7)
    plt.show()

    return recalls, precisions, MFs, APs





