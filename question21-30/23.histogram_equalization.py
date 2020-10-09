# -*- encoding: utf-8 -*-
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def histogram_equalization(img, z_max=255):
    H, W, C = img.shape
    S = H * W * C * 1.

    out = img.copy()
    sum_h = 0.
    for i in range(255):
        idx = np.where(img == i)
        sum_h += len(img[idx])
        z_prime = z_max / S * sum_h
        out[idx] = z_prime
    out = out.astype(np.uint8)
    return out


def main():
    img = cv.imread('../assets/sample_2_gray.jpg')
    out = histogram_equalization(img)
    plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()


if __name__ == '__main__':
    main()
