"""
@Function:数据集的预处理与导入

@Time:2022-3-13

@Author:马铭泽
"""
import torch, os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 用os处理报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

Input_Img_Size = (36, 48)
# 训练数据集原图片的转换
TrainImg_Transform = transforms.Compose([
    transforms.Resize(Input_Img_Size),
    # 将图片转换为一个张量
    transforms.ToTensor(),
    # 标准化(每个通道的均值,每个通道的标准差)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #单通道
    transforms.Normalize(mean=[0.444], std=[0.225]),
])

# 测试数据集原图片的转换
ValImg_Transform = transforms.Compose([
    transforms.Resize(Input_Img_Size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #单通道
    transforms.Normalize(mean=[0.444], std=[0.225]),
])

# 数据导入的类的创建
class PipeDataset(Dataset):
    def __init__(self, DatasetFolderPath, ImgTransform,  ShowSample=False):
        self.DatasetFolderPath = DatasetFolderPath
        self.ImgTransform = ImgTransform
        self.ShowSample = ShowSample
        self.SampleFolders = os.listdir(self.DatasetFolderPath)

    def __len__(self):
        return len(self.SampleFolders)

    def __getitem__(self, item):
        # 样本文件夹路径
        SampleFolderPath = os.path.join(self.DatasetFolderPath, self.SampleFolders[item])
        #读取图片
        FusionImg = Image.open(SampleFolderPath)
        #数据预处理
        FusionImg = self.ImgTransform(FusionImg)
        #读取图片的标签
        Label = int(SampleFolderPath.split("\\")[2].split(".")[0].split("_")[1])

        # %% 显示Sample
        if self.ShowSample:
            plt.figure(self.SampleFolders[item])
            # 转化成数组然后取第一层
            Img = FusionImg.numpy()[0]
            # 进行数据的标准化，统一规格
            Img = (Normalization(Img) * 255).astype(np.uint8)
            plt.figure()
            plt.imshow(Img)
            plt.show()

        return FusionImg, Label, self.SampleFolders[item]


# 数据集的导入以及处理的函数
def PipeDatasetLoader(FolderPath, BatchSize=1, ShowSample=False, TrainTransform = None, ValTransform = None):
    """
    :param FolderPath: 放数据集的主路径
    :param BatchSize: 每次取的图片数量
    :param ShowSample: 是否展示图片
    :param TrainTransform: 训练集的预处理
    :param ValTransform: 测试集的预处理
    :return: 训练集图片、测试集图片
    """
    # 训练集文件路径的合成
    TrainFolderPath = os.path.join(FolderPath, 'Train')
    # 获得经过转换后的训练数据集
    TrainDataset = PipeDataset(TrainFolderPath, TrainTransform,  ShowSample)
    # 将得到的训练数据集进行数据导入(数据集,每一次batch的数量,是否打乱顺序,当数据不够最后一次batch是否变小,线程数,设置为锁页内存可以让速度更快)
    TrainDataLoader = DataLoader(TrainDataset, batch_size=BatchSize, shuffle=True, drop_last=False, num_workers=0,
                                 pin_memory=True)

    # 对标签集进行处理,与训练集处理相同
    ValFolderPath = os.path.join(FolderPath, 'Val')
    ValDataset = PipeDataset(ValFolderPath, ValTransform,ShowSample)
    ValDataLoader = DataLoader(ValDataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    return TrainDataset, TrainDataLoader, ValDataset, ValDataLoader


# 数组归一化到0~1,便于对图像的归一化处理
def Normalization(Array):
    """
    :param Array: 需要归一化的矩阵
    :return: 归一化后的矩阵
    """
    min = np.min(Array)
    max = np.max(Array)
    if max - min == 0:
        return Array
    else:
        return (Array - min) / (max - min)