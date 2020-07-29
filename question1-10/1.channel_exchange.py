# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def main():
    # img.shape(270,377,3)，颜色通道为BGR
    img = cv.imread('../assets/sample_2.jpg')
    # img2转换成RGB通道
    img2 = img[:, :, (2, 1, 0)].copy()
    cv.imshow('', img2)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
