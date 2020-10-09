# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def histogram_operation(img,m_0=128,s_0=52):
    m=img.mean()
    s=img.std()

    out=img.copy()
    out=s_0/s*(out-m)+m_0
    out=out.astype(np.uint8)
    return out


def main():
    img = cv.imread('../assets/sample_2_gray.jpg')
    out=histogram_operation(img)
    plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()


if __name__ == '__main__':
    main()
