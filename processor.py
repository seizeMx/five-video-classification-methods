"""
Process an image that we can pass to our networks.
"""
import random

import cv2
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


def rotate(img, angle, center, w, h):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (nW, nH))

    return rotated

def process_image(image, target_shape, chanel_3d=False, type_0=0):
    """Given an image, process it and return the array."""

    # Load the image.
    h, w, _ = target_shape
    image = load_img(image[0], target_size=(h, w))
    img_arr = img_to_array(image)

    center = (h / 2, w / 2)
    a = 5

    if type_0 == 1:
        img_arr = cv2.flip(img_arr, 1)
    elif type_0 == 2:
        img_arr = rotate(img_arr, random.uniform(-a, a), center, w, h)[:32, :32, :]
    # Turn it into numpy, normalize and return.
    if chanel_3d:
        # TODO YUV
        x = (img_arr[:, :, 1] / 255.).astype(np.float32)
    else:
        x = (img_arr / 255.).astype(np.float32)

    return x


if __name__ == '__main__':
    process_image(r'd:\datasetConvert\300VW_Dataset_2015_12_14\114\0\001788.jpg', (32, 32, 3))
