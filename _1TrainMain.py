"""
@Function:主训练程序

@Time:2022-3-14

@Author:马铭泽
"""
import os, logging
import torch
from _2Dataset_Load import *
from _3ClassifyNet import *
from _0Utility import *

# 选取要使用的网络
UsingNet = "Net1"  # Net1是全连接神经网络、Net2是带残差层的网络

def train(Epoch,Threshold = False,Ratio = 0.6):
    # 开启训练模式，防止Batch Normalization造成的误差
    # Classify.train()
    Net.train()
    # 用来储存损失
    Train_Loss = 0
    # torch.cuda.empty_cache()
    # 显示当前的代数和学习率
    print("Epoch:", Epoch, "Lr=", Lr_Update.get_last_lr()[0], flush=True)

    # 记录计算正确的个数
    TrainCorrect = 0
    Drop_num_Train = 0

    for Num, (InputImg, Label, SampleName) in enumerate(TrainDataLoader):
        # print("第",Num,"次训练",flush=True)
        # 将图像与标签导入GPU
        InputImg = InputImg.float().to(ParaDic.get("Device"))
        Label = Label.to(ParaDic.get("Device"))
        # 权重清零
        Optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            torch.cuda.empty_cache()
            # OutputImg = Classify(InputImg)
            OutputImg = Net(InputImg)
            # 判断是否更新
            # print(Classify.Conv1.Conv1conv1.weight.data.requires_grad)
            BatchLoss = Criterion(OutputImg, Label)
            # 反向传播
            BatchLoss.backward()
            # 权重更新
            Optimizer.step()
            # 损失的叠加,一定要写item
            Train_Loss += BatchLoss.item()


            # 计算Accuracy
            # 首先找到最大的那个输出，然后和label比较
            Output = torch.softmax(OutputImg, 1, torch.float)

            # 筛选出准确度高于0.7的样本进行计算
            if Threshold:
                Label = Label[Output.max(dim=1).values > Ratio]
                Drop_num_Train += (Output.max(dim=1).values <= Ratio).sum().item()
                Output = Output[Output.max(dim=1).values > Ratio]

            Predict_ID = Output.argmax(dim=1)
            TrainCorrect += (Predict_ID == Label).sum().item()

    # 计算准确率
    if not Threshold:
        Train_Accuracy = TrainCorrect / (TrainDataset.__len__())
    else:
        Train_Accuracy =TrainCorrect / (TrainDataset.__len__() - Drop_num_Train)

    # 计算平均损失
    Average_Train_Loss = Train_Loss / TrainDataset.__len__() * ParaDic.get("Batchsize")
    print("Loss is", Average_Train_Loss, "Accuracy is", Train_Accuracy, flush=True)
    Log.logger.warning('\tTrain\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}\tAccuracy:{3:08f}'.format(
        Epoch,
        Lr_Update.get_last_lr()[0],
        Average_Train_Loss,
        Train_Accuracy))


def test(Epoch,Threshold = False,Ratio = 0.6):
    # Classify.eval()
    Net.eval()
    # 用来储存损失
    Val_Loss = 0
    # 用来存储正确的个数
    ValCorrect = 0
    # 存储不到概率的值
    Drop_num_Val = 0
    # 显示当前的代数和学习率
    print("Epoch:", Epoch, "Lr=", Lr_Update.get_last_lr()[0], flush=True)
    for Num, (InputImg, Label, SampleName) in enumerate(ValDataLoader):
        # 将图像与标签导入GPU
        InputImg = InputImg.float().to(ParaDic.get("Device"))
        Label = Label.to(ParaDic.get("Device"))
        with torch.set_grad_enabled(False):
            # OutputImg = Classify(InputImg)
            OutputImg = Net(InputImg)
            BatchLoss = Criterion(OutputImg, Label)
            # 损失的叠加,一定要写item
            Val_Loss += BatchLoss.item()

            # 判断预测是否正确
            Output = torch.softmax(OutputImg, 1, torch.float)

            # 筛选出准确度高于0.7的样本进行计算
            if Threshold:
                Label = Label[Output.max(dim=1).values > Ratio]
                Drop_num_Val += (Output.max(dim=1).values <= Ratio).sum().item()
                Output = Output[Output.max(dim=1).values > Ratio]

            Predict_ID = Output.argmax(dim=1)
            ValCorrect += (Predict_ID == Label).sum().item()


    # 计算准确率
    if not Threshold:
        Val_Accuracy = ValCorrect / (ValDataset.__len__())
    else:
        Val_Accuracy = ValCorrect / (ValDataset.__len__() - Drop_num_Val)
    # 计算平均损失
    Average_Val_Loss = Val_Loss / ValDataset.__len__()
    print("Loss is", Average_Val_Loss, "Accuracy is", Val_Accuracy, flush=True)
    Log.logger.warning(
        '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tValid\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}\tAccuracy:{3:08f}'.format(
        Epoch,
        Lr_Update.get_last_lr()[0],
        Average_Val_Loss,
        Val_Accuracy))




if __name__ == "__main__":
    # 记录版本号
    Version = 1.0
    # 存储网络的相关参数
    ParaDic = {"Epoch": 150,
               "Lr": 0.0007,# 0.0009
               "Batchsize": 50,
               "LrUpdate_Ratio": 0.2,# 0.2
               "LrUpdate_Epoch": 60,
               "TestEpoch": 10,
               "SaveEpoch": 10,
               "Device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
               }

    # 路径读取
    # 数据集路径
    DatasetPath = "Dataset"
    # 模型输出路径
    OutputPath = "Output/model"
    # 日志输出路径
    LogOutputPath = "Output/log"
    os.makedirs(OutputPath, exist_ok=True)
    # 读取数据集
    TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(DatasetPath, ParaDic["Batchsize"], TrainTransform=TrainImg_Transform, ValTransform=ValImg_Transform)

    #从此处开始可迭代
    for feature in [6]:
        # 初始化日志系统
        Log = Logger(os.path.join(LogOutputPath, str(Version) + "_" + str(ParaDic.get("Epoch")) + ".log"))


        # SJ网络
        # 实例化对象
        if UsingNet == "Net1":
            Net = ClassifyNet(In_Channels=1, Out_Channels=9, file_path=None, device=ParaDic.get("Device"),
                                   Features=feature, LastLayerWithAvgPool=False)
            # Classify.Para_Init()
            Net.to(ParaDic.get("Device"))

        # 本部网络
        if UsingNet == "Net2":
            Net = HitNet(In_Channels=1, Out_Channels=9,Features=feature)
            Net.to(ParaDic.get("Device"))


        # 定义代价函数和优化器，并且将他们转移到GPU中
        Criterion = nn.CrossEntropyLoss().to(ParaDic.get("Device"))
        # 初始化优化器
        # Optimizer = torch.optim.Adam(Classify.parameters(), lr=ParaDic.get("Lr"))
        Optimizer = torch.optim.Adam(Net.parameters(), lr=ParaDic.get("Lr"))
        # 新的优化器
        # Optimizer = torch.optim.AdamW(Net.parameters(), lr=ParaDic.get("Lr"))
        # Optimizer = torch.optim.Adam([{'params':Classify.Conv3.parameters(),'lr':Lr / 10},
        #                               {'params':Classify.classifier.parameters(),'lr':Lr}
        #                              ])  # 第一个参数是可用于迭代优化的参数或者定义参数组的dicts
        # 配置学习率的更新
        # Lr_Update = torch.optim.lr_scheduler.ReduceLROnPlateau(Optimizer, 'max', factor=0.5, patience=10 )
        # 按照比例更新学习率
        Lr_Update = torch.optim.lr_scheduler.StepLR(Optimizer, ParaDic.get("LrUpdate_Epoch"), gamma=ParaDic.get("LrUpdate_Ratio"))
        # 一种新的学习率更新策略
        # Lr_Update = torch.optim.lr_scheduler.CyclicLR(Optimizer, base_lr=0.001, max_lr=0.0011, step_size_up=75,
        #                                               step_size_down=75, mode='triangular', cycle_momentum=False)

        for epoch in range(1, ParaDic.get("Epoch") + 1):
            train(epoch)
            if epoch % ParaDic.get("TestEpoch") == 0 or epoch == 1:
                test(epoch)

            if epoch % ParaDic.get("SaveEpoch") == 0:
                # torch.save(Classify.state_dict(), os.path.join(OutputPath, '{0}_{1:04d}.pt'.format(Version,epoch)))
                torch.save(Net.state_dict(), os.path.join(OutputPath, '{0}_{1:04d}.pt'.format(Version,epoch)))

            # 学习率的更新
            Lr_Update.step()

        Version += 0.1
        # 保留两位有效数字，防止浮点型精度失真问题
        Version = round(Version, 2)
