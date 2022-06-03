"""
@Function: 利用torchvision进行数据增强

@Time:2022-3-14

@Author:马铭泽
"""
import random

import torch,os,glob
import cv2 as cv
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
# import matplotlib.pyplot as plt

#定义相关路径
Folder_Path = ""
Save_Path = ""


Img_Transform = transforms.Compose([
    #进行随机仿射变换(旋转度数范围,在一定范围内进行水平和垂直位移,在一定范围内随机缩放,度数范围)
    # transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.5, 2.), shear=10),
    #随机概率水平翻转图片,不填的话默认是0.5
    # transforms.RandomHorizontalFlip(),
    #裁剪为随机大小和宽高比(预期大小,原始尺寸的比例,选取的插值方法(线性))
    # transforms.RandomResizedCrop(Input_Img_Size, scale=(1., 1.), interpolation=Image.BILINEAR),
	# 竖直方向翻转
	# transforms.RandomVerticalFlip(p=0.5),
	# transforms.RandomHorizontalFlip(p=0.5),
	#随机旋转
	transforms.RandomRotation(degrees= (0, 30)),# 逆时针为正方向
	#修改亮度饱和度(亮度范围0.8~1.2)
	# transforms.ColorJitter(hue=0.04),
	transforms.ColorJitter(brightness=((0.4, 0.7))),#(8, 12)超亮
	# transforms.ColorJitter(saturation=0.1),
    #将图片转换为一个张量
    #transforms.ToTensor(),
    #标准化(每个通道的均值,每个通道的标准差)
	#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 按照所需的数量随机获取图片
def get_random_data(imgs_path, expect_num, output_path, All = False):
	Imgs = os.listdir(imgs_path)
	Index = 0

	if All:
		for num, img in enumerate(Imgs):
			Img = cv.imread(os.path.join(imgs_path, img), cv.IMREAD_GRAYSCALE)
			# print(output_path + "BadImg{0}_6.bmp".format(Index))
			cv.imwrite(output_path + "/BadImg{0}_0.bmp".format(Index), Img)
			Index += 1
	else:
		for num, img in enumerate(Imgs):
			if Index < expect_num:
				if random.random() < 0.6:
					# opencv读取默认是会变成3通道，要设置一下
					Img = cv.imread(os.path.join(imgs_path, img), cv.IMREAD_GRAYSCALE)
					# print(output_path + "BadImg{0}_6.bmp".format(Index))
					cv.imwrite(output_path + "/BadImg{0}_4.bmp".format(Index), Img)
					Index += 1

	print("Process Done!")
	print("Process Num:", Index)


#生成随机的图像
def create_img(index):
	# 产生随机数组`
	# randomByteArray = bytearray(os.urandom(1728))
	# randomArray = np.random.randn(1728) * 255
	# flatNumpyArray = np.array(randomByteArray)

	# 数组转换为一个300*400的灰度图像
	# grayImage = randomArray.reshape(36, 48)

	#随机正态分布
	# grayImage = np.random.normal(np.random.random(), np.random.random(), (36,48)) * 255
	# 随机泊松分布
	grayImage = np.random.poisson(lam=np.abs(np.random.randint(low=1, high=7, size=1)), size=(36,48)) * 255.0


	# cv.imshow("img",grayImage)
	# cv.waitKey(1000)
	cv.imwrite('Dataset\Train\PoissonImg{0}_6.bmp'.format(index), grayImage)



# 最大池化
def max_pooling(img, G=2, mode = "Max"):
	"""
	:param img: 要处理的图像
	:param G: 池化的步长(也不完全算作步长，它同时也是卷积核的大小)
	:param mode: Mean是平均池化、Max是最大池化
	:return: 处理后的图片
	"""
	out = img.copy()

	# 单通道
	if len(img.shape) == 2:
		H, W = img.shape

		Nh = int(H / G)
		Nw = int(W / G)

		for y in range(Nh):
			for x in range(Nw):
				if mode == "Mean":
					out[G * y:G * (y + 1), G * x:G * (x + 1)] = np.mean(out[G * y:G * (y + 1), G * x:G * (x + 1)])
				if mode == "Max":
					out[G * y:G * (y + 1), G * x:G * (x + 1)] = np.max(out[G * y:G * (y + 1), G * x:G * (x + 1)])

	# 多通道
	else:
		H, W, C = img.shape
		Nh = int(H / G)
		Nw = int(W / G)

		for y in range(Nh):
			for x in range(Nw):
				for c in range(C):
					if mode == "Mean":
						out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c])
					if mode == "Max":
						out[G*y:G*(y+1), G*x:G*(x+1), c] = np.max(out[G*y:G*(y+1), G*x:G*(x+1), c])


	return out

# 自适应二值化测试
def AdaptiveThreshold(img_path):
	Imgs = os.listdir(img_path)
	for img in Imgs:
		if img[:6] == "BadImg":
			continue

		Suffix = os.path.splitext(img)[1]
		if Suffix == ".txt":
			continue


		ImgFullPath = os.path.join(img_path, img)

		kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小


		# Img = cv.imread(ImgFullPath, cv.IMREAD_GRAYSCALE)
		Img = cv.imread(ImgFullPath)
		Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
		Img = cv.resize(Img, (640, 480))
		ThresholdImg = cv.adaptiveThreshold(Img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 1)
		ThresholdImgGauss = cv.adaptiveThreshold(Img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 1)
		dst = cv.erode(ThresholdImg, kernel)  # 膨胀操作
		cv.imshow("Img",Img)
		cv.imshow("Threshold",ThresholdImg)
		cv.imshow("ThresholdGauss",ThresholdImgGauss)
		cv.imshow("dilate",dst)
		cv.waitKey(300)

def Otsu(img_path):
	Imgs = os.listdir(img_path)
	for img in Imgs:
		if img[:6] == "BadImg":
			continue

		Suffix = os.path.splitext(img)[1]
		if Suffix == ".txt":
			continue


		ImgFullPath = os.path.join(img_path, img)


		Img = cv.imread(ImgFullPath, cv.IMREAD_GRAYSCALE)
		# Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
		Img = cv.resize(Img, (640, 480))
		# Img = cv.resize(Img, (640, 480))[:, 100:540]
		ret, Otsu = cv.threshold(Img, 30, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
		cv.imshow("Img",Img)
		cv.imshow("Threshold",Otsu)
		# cv.imshow("ThresholdGauss",ThresholdImgGauss)
		# cv.imshow("dilate",dst)
		print(ret)
		cv.waitKey(300)



#数据增强的类的创建
class Dataset_Strengthen(object):
	def __init__(self, Folder_Path,Save_Path,ImgTransform):
		self.Folder_Path = Folder_Path
		self.Save_Path = Save_Path
		self.ImgTransform = ImgTransform


	def Read_Transform(self):
		Imgs = os.listdir(self.Folder_Path)

		for Img in Imgs:
			Suffix = os.path.splitext(Img)[1]
			if Suffix == "":
				continue
			#设置图像读取路径
			Img_Path = os.path.join(self.Folder_Path,Img)
			#以PIL格式读取图片(RGB)
			# Initial_Img = Image.open(Img_Path).convert('RGB')
			#以PIL格式读取图片(GRAY)
			Initial_Img = Image.open(Img_Path)
			#图像转换
			Transformed_Img = self.ImgTransform(Initial_Img)
			# 尝试补集
			# Initial_Img = np.array(Initial_Img, dtype=np.float32)
			# Transformed_Img = 255 - Initial_Img
			Transformed_Img = np.array(Transformed_Img)

			# print(Transformed_Img.shape)
			# 将RGB图像转为BGR,只对三通道
			# Transformed_Img = cv.cvtColor(Transformed_Img,cv.COLOR_RGB2BGR)

			# 最大池化
			Transformed_Img = max_pooling(Transformed_Img, G=2, mode="Mean")

			Strengthen_Path = self.Save_Path + "\\"+ "Streng3" +str(Img)
			# print(Strengthen_Path)
			# cv.imshow("Img",Transformed_Img)
			cv.imwrite(Strengthen_Path,Transformed_Img)
			# cv.waitKey(100)

if __name__ == '__main__':
	# Strengthen = Dataset_Strengthen(r"Dataset\StrengthSource",r"Dataset\StrengthSource\StrengthenOutput",Img_Transform)
	# Strengthen.Read_Transform()
	# 存储数据
	# get_random_data(r"Dataset\StrengthSource\StrengthenOutput", expect_num=10000, output_path="Dataset\Val", All=True)

	# 创建随机图片
	# for index in range(3000):
	# 	create_img(index)

	# 测试自适应二值化
	ImgPath = "Dataset/Train"
	# ImgPath = r"Dataset\DatasetSource\UsingTrain\no use\pics"
	# cv.createTrackbar("value", "Threshold", 0, 10, AdaptiveThreshold)
	# AdaptiveThreshold(ImgPath)
	Otsu(ImgPath)

