# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def otsu(img):
    max_sigma = 0
    max_t = 0
    H, W = img.shape
    for t in range(1, 255):
        v0 = img[np.where(img < t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = img[np.where(img >= t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = t
    print("threshold:{}".format(max_t))
    img[img < max_t] = 0
    img[img >= max_t] = 255
    return img


def main():
    img = cv.imread('../assets/sample_2.jpg')
    b=img[:,:,0].copy()
    g=img[:,:,1].copy()
    r=img[:,:,2].copy()
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out=otsu(out)
    cv.imshow('',out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
