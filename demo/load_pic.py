# -*- encoding: utf-8 -*-
import cv2 as cv


def main():
    img = cv.imread('../assets/sample_2.jpg')
    print("shape of image:{}".format(img.shape))
    print("type of image:{}".format(img.dtype))
    cv.imshow('', img)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
