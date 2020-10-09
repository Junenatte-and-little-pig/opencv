# -*- encoding: utf-8 -*-
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    img = cv.imread('../assets/sample_2_gray.jpg')
    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()


if __name__ == '__main__':
    main()
