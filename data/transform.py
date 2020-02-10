import cv2
import random
import numpy as np
import torch
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ReadImage(object):
    """read image"""
    def __call__(self, sample):
        if 'image' not in sample:
            sample['image'] = cv2.imread(sample['image_path'])
        return sample

class ResizeImage(object):
    """Resize image."""
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, sample):
        image = sample['image']
        new_image = cv2.resize(image, (self.width, self.height))
        sample['image'] = new_image
        if 'boxes' in sample:
            h, w, _ = image.shape
            boxes = sample['boxes']
            scale_x = self.width / w
            scale_y = self.height / h
            new_boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
            sample['boxes'] = new_boxes
        return  sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class RandomFlip(object):
    """Randomly flip the input image with a probability of 0.5."""
    def __init__(self, flag=0.5):
        self.flag = flag

    def __call__(self, sample):
        if random.uniform(0.0, 1.0) > self.flag:
            image = sample['image']
            image = cv2.flip(image, random.randint(0, 1))
            sample['image'] = image
        return sample

