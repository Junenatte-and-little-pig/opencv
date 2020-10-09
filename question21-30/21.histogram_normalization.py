# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def histogram_normalization(img,a=0,b=255):
    c=img.min()
    d=img.max()

    out=img.copy()
    out=(b-a)/(d-c)*(out-c)+a
    out[out<a]=a
    out[out>b]=b
    out=out.astype(np.uint8)
    return out


def main():
    img = cv.imread('../assets/sample_2_gray.jpg')
    out=histogram_normalization(img)
    plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()


if __name__ == '__main__':
    main()
