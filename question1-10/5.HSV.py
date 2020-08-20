# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def RGB2HSV(_img):
    # 0<=H<=180, 0<=S<=255, 0<=V<=255
    img = _img.copy() / 255.
    c_max = np.max(img, axis=2)
    c_min = np.min(img, axis=2)
    c_argmax = np.argmax(img, axis=2)
    delta = c_max - c_min
    hsv = np.zeros_like(img, dtype=np.float32)
    hsv[..., 0][np.where(c_max == c_min)] = 0
    ind = np.where(c_argmax == 0)
    hsv[..., 0][ind] = 60 * np.divide((img[..., 2][ind] - img[..., 1][ind]),
                                      (delta[ind])) + 240
    ind = np.where(c_argmax == 1)
    hsv[..., 0][ind] = 60 * np.divide((img[..., 0][ind] - img[..., 2][ind]),
                                      (delta[ind])) + 120
    ind = np.where(c_argmax == 2)
    hsv[..., 0][ind] = 60 * np.divide((img[..., 1][ind] - img[..., 0][ind]),
                                      (delta[ind]))
    hsv[..., 0] = hsv[..., 0] / 2.

    hsv[..., 1][np.where(c_max == 0)] = 0
    hsv[..., 1][np.where(c_max != 0)] = delta[np.where(c_max != 0)] / c_max[
        np.where(c_max != 0)]
    hsv[..., 1] = hsv[..., 1] * 255.

    hsv[..., 2] = c_max.copy() * 255.
    return hsv


def main():
    img = cv.imread('../assets/sample_2.jpg')
    hsv = RGB2HSV(img).astype(np.uint8)
    hsv[..., 0] = hsv[..., 0] * 255. / 180
    # cv.cvtColor(img,cv.COLOR_BGR2HSV)
    cv.imshow('', hsv)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
