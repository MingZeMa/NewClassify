"""
@Function: 数据降维可视化(支持二维和三维)

@Time:2022-3-14

@Author:马铭泽
"""

import numpy as np
import os
import cv2 as cv
from time import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 进行3D图像绘制


def get_data(file_fold):
    Imgs = os.listdir(file_fold)
    Data = []
    Label = []
    for num,img in enumerate(Imgs):
        img = os.path.join(file_fold,img)
        Img = cv.imread(img)
        Img = np.ravel(Img)
        Data.append(Img)
        Label.append(img.split('.')[0].split('_')[-1])
    return Data, Label

# def get_data():
#     digits = datasets.load_digits(n_class=6)
#     data = digits.data
#     label = digits.target
#     n_samples, n_features = data.shape
#     return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)


    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(int(label[i]) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def plot_embedding3D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1],data[i,2], str(label[i]),
                 color=plt.cm.Set1(int(label[i]) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    File_Folder = r'E:\718\Code\AI\armor_classify_fineturn\Dataset\Val'
    data, label = get_data(File_Folder)
    # data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()


if __name__ == '__main__':
    main()