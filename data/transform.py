import cv2
import random
import numpy as np
import torch
import numbers
import collections

import transform_func as F

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
        sample['image'] = image[i:i+th, j:j+tw]
        return sample

class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value at the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self, padding, fill=0, padding_mode='constant'):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        image = sample['image']
        if isinstance(self.padding, int):
            pad_left = pad_right = pad_top = pad_bottom = self.padding
        if isinstance(self.padding, collections.Sequence) and len(self.padding) == 2:
            pad_left = pad_right = self.padding[0]
            pad_top = pad_bottom = self.padding[1]
        if isinstance(self.padding, collections.Sequence) and len(self.padding) == 4:
            pad_left = self.padding[0]
            pad_right = self.padding[1]
            pad_top = self.padding[2]
            pad_bottom = self.padding[3]

        if self.padding_mode == 'contant':
            if isinstance(self.fill, int):
                fill_r = fill_g = fill_b = self.fill
            else:
                fill_r = self.fill[0]
                fill_g = self.fill[1]
                fill_b = self.fill[2]
            img_b, img_g, img_r = cv2.split(image)
            img_b = np.pad(img_b, ((pad_top, pad_bottom), (pad_left, pad_right)), self.padding_mode,
                           constant_values=((fill_b, fill_b), (fill_b, fill_b)))
            img_g = np.pad(img_g, ((pad_top, pad_bottom), (pad_left, pad_right)), self.padding_mode,
                           constant_values=((fill_g, fill_g), (fill_g, fill_g)))
            img_r = np.pad(img_r, ((pad_top, pad_bottom), (pad_left, pad_right)), self.padding_mode,
                           constant_values=((fill_r, fill_r), (fill_r, fill_r)))
            image = cv2.merge([img_b, img_g, img_r])
        else:
            if len(image.shape) == 3:
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), self.padding_mode)
        sample['image'] = image
        return sample

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
    def __init__(self, outputsize, padding=0, pad_if_needed=False):
        if isinstance(outputsize, numbers.Number):
            self.size = (int(outputsize), int(outputsize))
        else:
            self.size = outputsize
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(image, output_size):
        h, w = image.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, tw, th

    def __call__(self, sample):
        image = sample['image']
        if self.padding > 0:
            pad_1 = Pad(self.padding)
            image = pad_1(sample)

        if self.pad_if_needed and image.shape[1] < self.size[1]:
            pad_2 = Pad([int((1 + self.size[1] - image.size[0]) / 2), 0])
            image = pad_2(sample)
        if self.pad_if_needed and image.shape[0] < self.size[0]:
            pad_3 = Pad([(0, int(1 + self.size[0] - image.size[1]) / 2)])
            image = pad_3(sample)

        i, j, h, w = self.get_params(image, self.size)
        sample['image'] = image[i:i+h, j:j+w]
        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        if random.random() < self.p:
            sample['image'] = cv2.flip(image, 1)
            return sample
        return sample

class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        if random.random() < self.p:
            sample['image'] = cv2.flip(image, 0)
            return sample
        return sample

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, image):
        return self.lambd(image)

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)
        return transform

    def __call__(self, sample):
        image = sample['image']
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        sample['image'] = transform(image)
        return sample


if __name__ == '__main__':
    norm1 = Normalize(mean=0, std=2)
    image_arr = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
    cv2.imshow("image", image_arr)
    cv2.waitKey(0)
    img = cv2.resize(image_arr, (200, 200))
    print(image_arr.shape)
    print(img.shape)
    sample = {'image': img, 'label': None}
    # new_sample = norm1(sample)

