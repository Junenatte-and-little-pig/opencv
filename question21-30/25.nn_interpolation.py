# -*- encoding: utf-8 -*-
import numpy as np
import cv2 as cv


def nn_interpolation(img, ax=1., ay=1.):
    H, W, _ = img.shape

    aH = int(ay * H)
    aW = int(ax * W)

    y = np.arange(aH).repeat(aW).reshape(aH, -1)
    x = np.tile(np.arange(aW), (aH, 1))
    y = np.round(y / ay).astype(np.int)
    x = np.round(x / ax).astype(np.int)

    out = img[y, x]

    out = out.astype(np.uint8)

    return out


def main():
    img = cv.imread('../assets/sample_2.jpg')
    out = nn_interpolation(img,ax=1.5,ay=1.5)
    cv.imshow('', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
