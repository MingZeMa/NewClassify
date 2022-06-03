"""
@Function: 读取文件中的参数(已弃用)

@Time:2022-3-14

@Author:马铭泽
"""
#模块导入
import numpy
import torch


#读取卷积weight
def get_conv_w(name,parameters):
    print("Start upload Conv1 parameters")
    with open(name + ".txt", mode="w+") as fp:
        #存储参数的大小
        for shape in range(len(parameters.shape)):
            print(parameters.shape[shape], file=fp)
        #存储具体的参数
        for in_channel in range(parameters.shape[1]):
            for out_channel in range(parameters.shape[0]):
                for row in range(parameters.shape[2]):
                    for col in range(parameters.shape[3]):
                        print(round(float(parameters[out_channel][in_channel][row][col]), 8), file=fp)




#读取卷积bias
def get_conv_b(name, parameters):
    print("Start upload Conv1 bias")
    with open(name + ".txt", mode="w+") as fp:
        # 存储参数的大小
        for shape in range(len(parameters.shape)):
            print(parameters.shape[shape], file=fp)
        # 存储具体的参数
        for channel in range(parameters.shape[0]):
            print(round(float(parameters[channel]), 8), file=fp)




#读取全连接wight
def get_linear_w(name, parameters):
    print("Start upload Conv1 parameters")
    with open(name + ".txt", mode="w+") as fp:
        # 存储参数的大小
        for shape in range(len(parameters.shape)):
            print(parameters.shape[shape], file=fp)
        # 存储具体的参数
        for in_channel in range(parameters.shape[1]):
            for out_channel in range(parameters.shape[0]):
                print(round(float(parameters[out_channel][in_channel]), 8), file=fp)


#读取全连接bias
def get_linear_b(name, parameters):
    print("Start upload Conv1 bias")
    with open(name + ".txt", mode="w+") as fp:
        # 存储参数的大小
        for shape in range(len(parameters.shape)):
            print(parameters.shape[shape], file=fp)
        # 存储具体的参数
        for channel in range(parameters.shape[0]):
            print(round(float(parameters[channel]), 8), file=fp)




#写入全部参数