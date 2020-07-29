# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def main():
    img = cv.imread('../assets/sample_2.jpg')
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    out = 0.0722 * b + 0.7152 * g + 0.2126 * r
    out = out.astype(np.uint8)
    out[out < 128] = 0
    out[out >= 128] = 255
    cv.imshow('', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
