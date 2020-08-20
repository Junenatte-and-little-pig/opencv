# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def max_pooling(img,G=4):
    out=img.copy()
    H,W,C=img.shape
    for i in range(H//G):
        for j in range(W//G):
            for c in range(C):
                out[i*G:(i+1)*G,j*G:(j+1)*G,c]=np.max(out[i*G:(i+1)*G,j*G:(j+1)*G,c]).astype(int)
    return out


def main():
    img=cv.imread('../assets/sample_2.jpg')
    out=max_pooling(img)
    cv.imshow('',out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
