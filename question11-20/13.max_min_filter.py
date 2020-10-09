# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def BGR2GRAY(image):
    b = image[:, :, 0].copy()
    g = image[:, :, 1].copy()
    r = image[:, :, 2].copy()

    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)
    return out


def max_min(image, K_size=3):
    H, W, _ = image.shape
    padding = K_size // 2
    out = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float)
    gray = BGR2GRAY(image)
    out[padding:padding + H, padding:padding + W] = gray.copy().astype(np.float)
    tmp = out.copy()

    for x in range(H):
        for y in range(W):
            out[padding + x, padding + y] = np.max(
                tmp[x:x + K_size, y:y + K_size]) - np.min(
                tmp[x:x + K_size, y:y + K_size])

    out = out[padding:padding + H, padding:padding + W].astype(np.uint8)
    return out


def main():
    img = cv.imread('../assets/sample_2.jpg')
    out = max_min(img)
    cv.imshow('', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
