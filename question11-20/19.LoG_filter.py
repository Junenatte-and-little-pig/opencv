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


def LoG_filter(img, K_size=5, sigma=3):
    H, W, _ = img.shape
    padding = K_size // 2
    gray = BGR2GRAY(img)
    out = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float)
    out[padding:padding + H, padding:padding + W] = gray.copy().astype(np.float)
    tmp = out.copy()

    k = np.zeros((K_size, K_size))
    for x in range(-padding, -padding + K_size):
        for y in range(-padding, -padding + K_size):
            k[padding + x, padding + y] = (
                                                      x ** 2 + y ** 2 - sigma ** 2) * np.exp(
                -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    k /= 2 * np.pi * (sigma ** 6)
    k /= k.sum()

    for x in range(H):
        for y in range(W):
            out[padding + x, padding + y] = np.sum(
                k * tmp[x:x + K_size, y:y + K_size])
    out = np.clip(out, 0, 255)
    out = out[padding:padding + H, padding:padding + W].astype(np.uint8)
    return out


def main():
    img = cv.imread('../assets/sample_2.jpg')
    out_v = LoG_filter(img)
    cv.imshow('', out_v)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
