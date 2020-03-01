from __future__ import division
import cv2
import numpy as np
import math

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.
    Args:
        img (CV2 Image): CV2 Image to be rotated.
        angle ({float, int}): In degrees degrees counter clockwise order.
        resample ({CV.Image.NEAREST, CV.Image.BILINEAR, CV.Image.BICUBIC}, optional):
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    if not _is_numpy_image(img):
        raise TypeError('img should be nparray Image. Got {}'.format(type(img)))

    if center == None:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
    if expand == True:
        ratio = math.sin(2 * math.pi * angle / 360) * w / h + math.cos(2 * math.pi * angle / 360)
    else:
        ratio = 1
    M = cv2.getRotationMatrix2D(center, angle, ratio)

    return cv2.warpAffine(img, M, (w, h))
