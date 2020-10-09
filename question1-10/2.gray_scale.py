# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def main():
    img = cv.imread('../assets/sample_2.jpg')
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # cv.imshow('use one of RGB as gray(b as example)', b)
    # cv.waitKey(0)
    # out=np.zeros([img.shape[0],img.shape[1],1])
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         out[i,j]=max(b[i,j],g[i,j],r[i,j])
    # cv.imshow('use max of RGB as gray', out)
    # cv.waitKey(0)
    # 会偏白，甚至几乎全白
    # out = (b + g + r) // 3
    # cv.imshow('use average of RGB as gray', out)
    # cv.waitKey(0)
    out = 0.0722 * b + 0.7152 * g + 0.2126 * r
    out = out.astype(np.uint8)
    cv.imshow('weighted average, better in showing the picture', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
