# -*- encoding: utf-8 -*-
import cv2 as cv


def main():
    img = cv.imread('../assets/sample_2.jpg')
    img2 = img.copy()
    img2[:50, :50] = 0
    cv.imshow('', img2)
    cv.waitKey(0)
    cv.imwrite('../assets/sample_2_out.jpg', img2)


if __name__ == '__main__':
    main()
