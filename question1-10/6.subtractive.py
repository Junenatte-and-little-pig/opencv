# -*- encoding: utf-8 -*-
import cv2 as cv


def subtractive(_img):
    img = _img.copy()
    img = img // 64 * 64 + 32
    return img


def main():
    img = cv.imread('../assets/sample_2.jpg')
    out = subtractive(img)
    cv.imshow('', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
