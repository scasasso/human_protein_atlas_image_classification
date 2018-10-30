import numpy as np
import cv2
from imgaug import augmenters as iaa
from PIL import Image


def load_image(path):
    R = Image.open(path + '_red.png')
    G = Image.open(path + '_green.png')
    B = Image.open(path + '_blue.png')
    Y = Image.open(path + '_yellow.png')

    im = np.stack((
        np.array(R),
        np.array(G),
        np.array(B),
        np.array(Y)), -1)

    return im

