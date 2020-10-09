# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)
    return out


def differential_vertical_filter(img):
    H, W, _ = img.shape
    padding = 1
    gray = BGR2GRAY(img)
    out = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float)
    out[padding:padding + H, padding:padding + W] = gray.copy().astype(np.float)
    tmp = out.copy()

    k = [[0., -1., 0.], [0., 1., 0.], [0., 0., 0.]]
    for x in range(H):
        for y in range(W):
            out[padding + x, padding + y] = np.sum(k * tmp[x:x + 3, y:y + 3])
    out = np.clip(out, 0, 255)
    out = out[padding:padding + H, padding:padding + W].astype(np.uint8)
    return out


def differential_horizontal_filter(img):
    H, W, _ = img.shape
    padding = 1
    gray = BGR2GRAY(img)
    out = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float)
    out[padding:padding + H, padding:padding + W] = gray.copy().astype(np.float)
    tmp = out.copy()

    k = [[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]]
    for x in range(H):
        for y in range(W):
            out[padding + x, padding + y] = np.sum(k * tmp[x:x + 3, y:y + 3])
    out = np.clip(out, 0, 255)
    out = out[padding:padding + H, padding:padding + W].astype(np.uint8)
    return out


def main():
    img = cv.imread('../assets/sample_2.jpg')
    out_v = differential_vertical_filter(img)
    cv.imshow('', out_v)
    cv.waitKey(0)
    out_h = differential_horizontal_filter(img)
    cv.imshow('', out_h)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
