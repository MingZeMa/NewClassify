"""
@Function:主体网络的搭建

@Time:2022-3-14

@Author:马铭泽
"""
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np



class ClassifyNet(nn.Module):
    def __init__(self, In_Channels, Out_Channels,device, file_path, Features=64, fcNodes = 60, LastLayerWithAvgPool = False):
        super(ClassifyNet, self).__init__()
        self.file_path = file_path
        self.device = device
        self.LastLayerWithAvgPool = LastLayerWithAvgPool
        # 特征提取过程
        # 第一层卷积
        self.Conv1 = ClassifyNet.block(In_Channels, Features, "Conv1",Kernel_Size = 5,Padding=2)
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#上交采用的是平均池化，但是平均池化保留的是背景，最大保留的是纹理，我不理解为什么用平均
        # 第二层卷积
        self.Conv2 = ClassifyNet.block(Features, Features * 2 , "Conv2",Kernel_Size = 3,Padding=1)
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三层卷积
        self.Conv3 = ClassifyNet.block(Features * 2, Features * 4 , "Conv3",Kernel_Size = 3,Padding=1)
        # 拉平
        self.Flatten = nn.Flatten()
        # 全局平均池化
        self.AdaptiveAvgPool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        # 第一层全连接
        self.FC1 = ClassifyNet.fc_block(fcNodes,"fc1")
        # 最后一层分类
        self.FC_End = nn.LazyLinear(Out_Channels)


    # 封装两次卷积过程
    @staticmethod
    def block(In_Channels, Features, name,Kernel_Size,Padding = 0):
        """
        :param In_Channels:输入通道数
        :param Features: 第一层卷积后图像的通道数
        :param name: 卷积层的名称
        :param Kernel_Size: 卷积核的大小
        :param Padding: 填充的个数
        :return: 卷积层块(包含卷积层、BN层、Relu)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                name + "conv",
                nn.Conv2d(In_Channels, Features, kernel_size = Kernel_Size, padding=Padding, bias=True)#这里注意可以改是否要加b
            ),
            # batch Normal处理数据分布
            (name + "norm", nn.BatchNorm2d(Features)),
            # Relu激活
            (name + "relu", nn.ReLU(inplace=True)),
            # Gelu激活
            # (name + "Gelu", nn.GELU()),
        ]))

    @staticmethod
    def fc_block(NodesNum,Name,DropRatio = 0.5):
        """
        :param NodesNum: 全连接测层的神经元个数
        :param Name: 网络层名称
        :param DropRatio: Dropout的比率
        :return: 全连接层块(包含线性层、Relu、Dropout)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                Name + "fc",
                nn.LazyLinear(NodesNum)
            ),
            # Relu激活
            (Name + "relu", nn.ReLU(inplace=True)),
            # Gelu激活
            # (Name + "Gelu", nn.GELU()),
            #Dropout
            (Name + "dropout", nn.Dropout(DropRatio)),
        ]))


    @staticmethod
    def flatten(Input,InputShape):
        """
        :param Input: 需要拉成一维向量的输入
        :return: 处理完的一维向量
        """
        OutputShape = InputShape[1] * InputShape[2] * InputShape[3]
        return Input.view(InputShape[0], OutputShape)


    # 正向传播过程
    def forward(self, x):
        """
        :param x: 网络的输入
        :return: 网络的输出
        """
        # 下采样
        self.Conv1.parameters()
        Conv1 = self.Conv1(x)
        Conv2 = self.Conv2(self.Pool1(Conv1))
        Conv3 = self.Conv3(self.Pool2(Conv2))
        # 判断是否使用全局平均池化
        if self.LastLayerWithAvgPool:
            AdaptiveAvgPool = self.AdaptiveAvgPool(Conv3)
            FC1 = self.FC1(AdaptiveAvgPool)
            Output = self.FC_End(FC1)
        else:
            Flatten = self.Flatten(Conv3)
            FC1 = self.FC1(Flatten)
            Output = self.FC_End(FC1)

        return Output

    def get_w(self,fp):
        """
        :param fp: 文件句柄
        :return: 卷积层的权重
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            list[i] = float(list[i].rstrip('\n'))
        weight = torch.Tensor(np.zeros([int(list[1]), int(list[0]), int(list[2]), int(list[3])]))
        index = 0
        for i in range(weight.shape[1]):
            for j in range(weight.shape[0]):
                for k in range(weight.shape[2]):
                    for l in range(weight.shape[3]):
                        weight[j, i, k, l] = list[index + 4]
                        # print(list[index + 4])
                        index += 1

        return weight

    def get_b(self, fp):
        """
        :param fp: 文件句柄
        :return: 卷积层的偏差
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            list[i] = float(list[i].rstrip('\n'))
        bias = torch.Tensor(np.zeros([int(list[0])]))
        for i in range(bias.shape[0]):
            bias[i] = list[i + 1]


        return bias

    def get_fc_w(self,fp):
        """
        :param fp:文件句柄
        :return: 全连接层的权重
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            list[i] = float(list[i].rstrip('\n'))
        weight = torch.Tensor(np.zeros([int(list[1]), int(list[0])])) # (60,560)，nn.Linear乘上的是权重的转置
        index = 0
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                weight[i, j] = list[index + 2]
                # print(list[index + 4])
                index += 1

        return weight

    def get_fc_b(self, fp):
        """
        :param fp: 文件句柄
        :return: 全连接层的偏差
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            #去除最后的换行符
            list[i] = float(list[i].rstrip('\n'))
        bias = torch.Tensor(np.zeros([int(list[0])]))
        # print("bia shape")
        for i in range(bias.shape[0]):
            bias[i] = list[i + 1]

        return bias


    #权重初始化
    def Para_Init(self):
        """
        :return: None
        """
        for name, parameters in self.named_parameters():
            # 读取卷积层参数
            # conv1
            # print(name.split("."))
            moedl_dict = self.state_dict()
            # print(moedl_dict)
            if (name.split(".")[1] == "Conv1conv1"):#这里读取参数只能通过对中间段进行切片进行，因为中间还有BN层
                if(name.split(".")[2] == "weight"):
                    Full_Path = os.path.join(self.file_path,'conv1_w')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_w(fp))
                        parameters.requires_grad = False

                if (name.split(".")[2] == "bias"):
                    Full_Path = os.path.join(self.file_path, 'conv1_b')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_b(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
                        # print(list(self.Conv1[0].parameters())[0])

            #conv2
            if (name.split(".")[1] == "Conv2conv1"):
                if (name.split(".")[2] == "weight"):
                    Full_Path = os.path.join(self.file_path, 'conv2_w')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_w(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
                if (name.split(".")[2] == "bias"):
                    Full_Path = os.path.join(self.file_path, 'conv2_b')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_b(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
            #conv3
            if (name.split(".")[1] == "Conv3conv1"):
                if (name.split(".")[2] == "weight"):
                    Full_Path = os.path.join(self.file_path, 'conv3_w')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_w(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
                if (name.split(".")[2] == "bias"):
                    Full_Path = os.path.join(self.file_path, 'conv3_b')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_b(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)

            # fc1
            # if (name.split(".")[1] == "0"):
            #     if (name.split(".")[2] == "weight"):
            #         Full_Path = os.path.join(self.file_path, 'fc1_w')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_w(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)
            #     if (name.split(".")[2] == "bias"):
            #         Full_Path = os.path.join(self.file_path, 'fc1_b')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_b(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)
            #
            # # fc2
            # if (name.split(".")[1] == "3"):
            #     if (name.split(".")[2] == "weight"):
            #         Full_Path = os.path.join(self.file_path, 'fc2_w')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_w(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)
            #     if (name.split(".")[2] == "bias"):
            #         Full_Path = os.path.join(self.file_path, 'fc2_b')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_b(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)

# 残差块类
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


class HitNet(nn.Module):
    def __init__(self, In_Channels, Out_Channels, Features=64):
        super(HitNet, self).__init__()
        # 特征提取过程
        #第一层卷积
        self.Conv1 = HitNet.block(In_Channels, Features, "Conv1",Kernel_Size = 5,Padding=2)
        #第二层卷积
        self.Conv2 = HitNet.block(Features, Features * 2 , "Conv2",Kernel_Size = 3,Padding=1)
        #残差块
        self.Residual = HitNet.residual(Features * 2, Features * 4, "residual")
        #获取残差块另一边的数据
        self.Conv1x1 = nn.Conv2d(Features * 2, Features * 4, stride=2, kernel_size=1)
        #最后一步的处理
        self.LastBlock = nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(Features *4), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        #最后一层分类
        self.FC_End = nn.LazyLinear(Out_Channels)


    # 封装两次卷积过程
    @staticmethod
    def block(In_Channels, Features, name,Kernel_Size,Padding = 0):
        """
        :param In_Channels:输入通道数
        :param Features: 第一层卷积后图像的通道数
        :param name: 卷积层的名称
        :param Kernel_Size: 卷积核的大小
        :param Padding: 填充的个数
        :return: 卷积层块(包含卷积层、BN层、Relu)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                name + "conv",
                nn.Conv2d(In_Channels, Features, kernel_size = Kernel_Size, padding=Padding, bias=True)#这里注意可以改是否要加b
            ),
            # batch Normal处理数据分布
            (name + "norm", nn.BatchNorm2d(Features)),
            # Relu激活
            (name + "relu", nn.ReLU(inplace=True)),
        ]))

    @staticmethod
    def fc_block(NodesNum,Name,DropRatio = 0.5):
        """
        :param NodesNum: 全连接测层的神经元个数
        :param Name: 网络层名称
        :param DropRatio: Dropout的比率
        :return: 全连接层块(包含线性层、Relu、Dropout)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                Name + "fc",
                nn.LazyLinear(NodesNum)
            ),
            # Relu激活
            (Name + "relu", nn.ReLU(inplace=True)),
            #Dropout
            (Name + "dropout", nn.Dropout(DropRatio)),
        ]))


    @staticmethod
    def flatten(Input,InputShape):
        """
        :param Input: 需要拉成一维向量的输入
        :return: 处理完的一维向量
        """
        OutputShape = InputShape[1] * InputShape[2] * InputShape[3]
        return Input.view(InputShape[0], OutputShape)

    @staticmethod
    def residual(input_channels, num_channels, name):
        return nn.Sequential(OrderedDict([
            # batch Normal处理数据分布
            (name + "norm", nn.BatchNorm2d(input_channels)),
            # Relu激活
            (name + "relu", nn.ReLU(inplace=True)),
            (name + "relu", nn.MaxPool2d(kernel_size=2, stride=2)),
            # 卷积
            (
                name + "conv",
                nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding=1, bias=True)#这里注意可以改是否要加b
            ),
        ]))


    # 正向传播过程
    def forward(self, x):
        """
        :param x: 网络的输入
        :return: 网络的输出
        """
        # 下采样
        Conv1 = self.Conv1(x)
        Conv2 = self.Conv2(Conv1)
        Residual = self.Residual(Conv2)
        Conv1x1 = self.Conv1x1(Conv2)
        Input = Residual + Conv1x1
        FC1 = self.LastBlock(Input)
        Output = self.FC_End(FC1)

        return Output