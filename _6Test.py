"""
@Function: 对模型的详细测试

@Time:2022-3-15

@Author:马铭泽
"""
import os.path
import time

import matplotlib.pyplot as plt

from _3ClassifyNet import *
from _2Dataset_Load import *
from _7Evaluate_Model import *
from _8ParametersReader import *
from _0Utility import plot_confusion_matrix
from _1TrainMain import UsingNet

# 展示预测的图片以及预测结果
def resultVisualization(InputImg, Label, Output, ShowTime = None, Display = False):
    """
    :param InputImg: 输入的图像
    :param Label: 标签
    :param Output: 输出
    :param ShowTime: 展示的时间
    :param Display: 是否展示
    :return: 可以进行可视化的图片
    """
    # 处理图片格式
    ShowImg = np.transpose(InputImg.cpu().numpy()[0], (1, 2, 0))
    ShowImg = cv.cvtColor(ShowImg, cv.COLOR_RGB2BGR)
    ShowImg = (Normalization(ShowImg) * 255).astype(np.uint8)
    ShowImg = cv.resize(ShowImg, (480, 360))
    # 显示预测值和实际值
    # 显示标签
    cv.putText(ShowImg, 'Label:' + str(Label.numpy()[0]), (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    # 显示预测值
    cv.putText(ShowImg, 'Predict:' + str(list(Output).index(max(Output))), (120, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 0, 255), 1)
    if Display:
        cv.imshow("InitImg",ShowImg)
        cv.waitKey(ShowTime)

    return ShowImg


# 测试图片
def test(InputImg, Label, Matrix):
    """
    :param InputImg: 待测试的输入图片
    :param Label: 真实的标签
    :param Matrix: 混淆举证
    :return: 模型的输出(特征向量)
    """
    # OutputImg = Classify(InputImg)
    OutputImg = Net(InputImg)
    softmax = torch.softmax(OutputImg, 1, torch.float)
    Output = softmax.cpu().numpy()[0]
    Matrix[Label.numpy()[0], list(Output).index(max(Output))] += 1

    return Output


# 画ROC PRC 曲线
def drawPRCAndROC(Outputs, Labels):
    """
    :param Outputs: 模型的输出合集
    :param Labels: 标签合集
    :return: 无
    """
    # 绘制ROC曲线
    fprs, tprs, AUCs = ROC_AUC(Labels, Outputs, ShowROC=True)
    print('AUCs:', AUCs)

    # 绘制PRC曲线
    recalls, precisions, MFs, APs = PRC(Labels, Outputs, ShowPRC=True)
    print('MF:', MFs)
    print('AP:', APs)


# 画准确度随着模型变化的变化
def drawAccuracyChange(AccuracyList, ModelNum):
    """
    :param AccuracyList: 准确度集合
    :return: 无
    """
    Nums = len(AccuracyList[0])
    x = [str(i) for i in range(ModelNum + 1)]
    fig = plt.figure()
    for i in range(Nums):
        Accuracy = np.array(AccuracyList)[:, i]
        plt.plot(x, Accuracy, label = str(i))
        plt.plot(x, Accuracy, "o")

    plt.ylabel("Accuracy")
    plt.xlabel('Model')
    plt.title("AccuracyChange")
    # 显示标签框
    plt.legend(loc='lower right', fontsize=7)
    plt.show()


if __name__ == "__main__":
    # 要分类的类别数
    ClassNum = 9
    # 初始化网络结构
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读取数据集
    Data_Folder = "Dataset"
    # 导入训练集与测试集
    TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(Data_Folder, 1, TrainTransform=TrainImg_Transform, ValTransform=ValImg_Transform)

    # 读取模型文件
    ModelPath = r"Output\model"
    Models = os.listdir(ModelPath)

    # 存储所有的曲线绘制数据
    OutputList = []
    LabelList = []

    # 存储每一次混淆矩阵计算完各类的准确度
    AccuracyList = []
    # 记录待测模型个数
    ModleIndex = 0

    # 遍历每一个指标综合评估
    for num, model in enumerate(Models):
        FullModelPath = os.path.join(ModelPath,model)
        # 初始化模型
        # TODO:目前遍历测试与评估只支持同一网络结构
        if UsingNet == "Net1":
            Net = ClassifyNet(In_Channels=1, Out_Channels=ClassNum, device=device, file_path=None, Features=6,
                              LastLayerWithAvgPool=False).to(device)
            # 导入模型
            Net.load_state_dict(torch.load(FullModelPath, map_location=device))
            # 开启测试模式，取消梯度更新
            Net.eval()

        if UsingNet == "Net2":
            Net = HitNet(In_Channels=1, Out_Channels=7,Features= 7).to(device)
            # 导入模型
            Net.load_state_dict(torch.load(FullModelPath, map_location=device))
            # 开启测试模式，取消梯度更新
            Net.eval()

        torch.set_grad_enabled(False)

        # 初始化混淆矩阵
        Matrix = np.zeros(ClassNum * ClassNum).reshape(ClassNum, ClassNum)
        # 初始化混淆矩阵底部标签
        Classes = [str(i) for i in range(ClassNum)]

        # 创一个文件夹保存每个模型输出的测试结果
        OutputFolderName = "Output_" + str(num)
        OutputFolderPath = os.path.join("Output/testImg",OutputFolderName)
        os.makedirs(OutputFolderPath,exist_ok=True)
        #记录正确与错误的序号
        TrueNum = 0
        FalseNum = 0

        # 模型评估的相关初始化
        # 存储计算单个ROC AUC的列表
        Outputs = []
        Labels = []


        # 创建保存各类的文件夹
        for i in range(ClassNum):
            ClassPath = os.path.join(OutputFolderPath,str(i))
            os.makedirs(ClassPath, exist_ok=True)
            # 创建正类与负类
            os.makedirs(os.path.join(ClassPath,"true"), exist_ok=True)
            os.makedirs(os.path.join(ClassPath,"false"), exist_ok=True)


        #开始测试
        for Num, (InputImg, Label, SampleName) in enumerate(ValDataLoader):
            InputImg = InputImg.float().to(device)
            Output = test(InputImg, Label, Matrix)
            OutputID = list(Output).index(max(Output))
            ShowImg = resultVisualization(InputImg, Label, Output, Display=False)
            LabelID = Label.numpy()[0]
            # if LabelID == OutputID:
            #     cv.imwrite(OutputFolderPath + r"\{0}\true\{1}_{2}.jpg".format(LabelID, LabelID, TrueNum), cv.resize(ShowImg, (48, 36)))
            #     TrueNum += 1
            # else:
            #     cv.imwrite(OutputFolderPath + r"\{0}\false\{1}_{2}_{3}_{4}.jpg".format(LabelID, LabelID, FalseNum, OutputID, max(Output)),cv.resize(ShowImg, (48, 36)))
            #     FalseNum += 1

            # 存储评估的数据
            Outputs.append(Output)
            LabelVector = [0 for j in range(len(list(Output)))]
            LabelVector[Label] = 1
            Labels.extend(LabelVector)

            # 绘制分类效果
            if Num < 12:
                ax = plt.subplot(3, 4, Num + 1)
                ax.set_yticks([])
                ax.set_xticks([])
                plt.subplots_adjust(hspace=0.4, wspace=0.4)
                plt.imshow(ShowImg)
                plt.title(str(OutputID))

        plt.savefig("result")
        plt.show()

        # 拼接并拉平
        Outputs = np.vstack(Outputs).ravel()
        Labels = np.array(Labels).ravel()

        # 将每个模型的输出和标签数据存储起来
        OutputList.append(Outputs)
        LabelList.append(Labels)

        # 绘制单个模型的混淆矩阵
        AccuracyMatrix = plot_confusion_matrix(Matrix, classes=Classes, normalize=True, title='confusion matrix' + str(num))
        Accuracys = [AccuracyMatrix[i][i] for i in range(ClassNum)]
        AccuracyList.append(Accuracys)
        ModleIndex = num


    # 绘制ROC PRC
    drawPRCAndROC(OutputList, LabelList)
    # 画出整体的变化趋势
    drawAccuracyChange(AccuracyList,ModleIndex)