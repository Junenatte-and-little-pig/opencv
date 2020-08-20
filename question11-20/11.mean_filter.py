# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def mean_filter(img, K_size=3):
    H, W, C = img.shape
    padding = K_size // 2
    out = np.zeros([H + padding * 2, W + padding * 2, C], dtype=np.float32)
    out[padding:padding + H, padding:padding + W] = img.copy()

    tmp = out.copy()
    for x in range(H):
        for y in range(W):
            for c in range(C):
                out[padding + x, padding + y, c] = np.mean(tmp[x:x + K_size, y:y + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[padding:padding + H, padding:padding + W].astype(np.uint8)
    return out


def main():
    img = cv.imread('../assets/sample_2.jpg')
    out = mean_filter(img)
    cv.imshow('', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
