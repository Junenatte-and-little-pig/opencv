# -*- encoding: utf-8 -*-
import cv2 as cv
import numpy as np


def bl_interpolation(img,ax=1.,ay=1.):
    H, W, _ = img.shape

    aH = int(ay * H)
    aW = int(ax * W)

    y = np.arange(aH).repeat(aW).reshape(aH, -1)
    x = np.tile(np.arange(aW), (aH, 1))
    y=y/ay
    x=x/ax

    iy=np.floor(y).astype(np.int8)
    ix=np.floor(x).astype(np.int8)

    iy=np.minimum(iy,H-2)
    ix=np.minimum(ix,W-2)

    dy=y-iy
    dx=x-ix

    dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
    dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

    out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (
                1 - dx) * dy * img[iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


def main():
    img = cv.imread('../assets/sample_2.jpg')
    out = bl_interpolation(img,ax=1.5,ay=1.5)
    cv.imshow('', out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()