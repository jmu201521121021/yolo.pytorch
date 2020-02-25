import cv2
import random
import numpy as np
import torch
import numbers
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
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).long()}

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

class Normalize(object):
    """" Normalize a tensor image with mean and standard deviation
    Given mean and std, if not None, this transform will normalize the input ``torch.*Tensor`` i.e.
    ``input = (input - mean) / std``
    else
    ``input = input / 255
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            image : Tensor image of size (C, H, W) to be normalized.
        """
        image = sample["image"] 
        if self.mean is None or self.std is None: 
            image.div_(255) 
   	else:
	    image.sub_(self.mean).div_(self.std) 
        sample["image"] = image 
	return sample 

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Scale(ResizeImage):
    """
    Note: This transform is deprecated in favor of Resize.
    """
    def __init__(self, *args, **kwargs):
        super(Scale, self).__init__(*args, **kwargs)

class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, output_size):
        if isinstance(output_size, numbers.Number):
            self.output_size = (int(output_size), int(output_size))
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        th, tw = self.output_size
        i = int(round(h-th / 2.))
        j = int(round(w-tw / 2.))
        return image[i:i+th, j:j+tw]

class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
