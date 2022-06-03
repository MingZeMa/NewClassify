"""
@Function: 单通道颜色数据增强

@Time:2022-3-14

@Author:马铭泽
"""

import os
import numpy as np
import cv2 as cv


def find_pics(path):
    files = os.listdir(path=path)
    pics = []
    for file in files:
        pics.append(os.path.join(path, file))
    return pics


def main():
    index = 0
    pic_files = find_pics("Dataset/Color_Change")
    cv.namedWindow("adjuster")
    for channel in "RGB":
        cv.createTrackbar(channel, "adjuster", 255, 510, lambda x: 0)

    for pic in pic_files:
        rgb = np.array([214, 250, 332], dtype=np.int)
        print(pic[-3:])
        if pic[-3:] == "new":
            continue
        img = cv.imread(pic)
        # img = cv.resize(img, (480, 360))
        cv.imshow("raw img", img)
        while cv.waitKey(100) != ord('q'):
            for idx, channel in enumerate("RGB"):
                rgb[idx] = cv.getTrackbarPos(channel, "adjuster")
            new_img = img.copy().astype(np.int)
            mask = new_img.sum(-1) > 0
            for i in range(3):
                new_img[mask, 2 - i] = np.clip(new_img[mask, 2 - i] + rgb[i] - 255, 0, 255)
            cv.imshow("new img", cv.resize(new_img.astype(np.uint8), (480, 360)))
        # new_img = cv.resize(new_img,(48,36))
        cv.imwrite("Dataset/Color_Change/new/B5new" + str(index) + "_{0}.bmp".format(pic.split('_')[-1].split('.')[0]),new_img)
        index += 1



if __name__ == "__main__":
    main()