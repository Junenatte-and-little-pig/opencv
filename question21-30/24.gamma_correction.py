# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def gamma_correction(img, c=1, g=2.2):
    out = img.copy()
    # 使用运算赋值会报错
    out = np.divide(out, 255.)
    out = (1 / c * out) ** (1 / g)
    out = np.multiply(out, 255.)
    return out.astype(np.uint8)


def main():
    img = cv.imread('../assets/sample_2.jpeg')
    out = gamma_correction(img)
    cv.imshow('', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
