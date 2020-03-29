import cv2
import random
import numpy as np
import torch
import numbers
import collections
from yolov3.structures import  Boxes

import data.function as F

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
        if "boxes" in sample:
            boxes = sample["boxes"]
            assert isinstance(boxes, np.ndarray), "not support format {}".format(type(boxes))
            boxes = torch.from_numpy(boxes).float()
            boxes = Boxes(boxes)
            sample["boxes"] = boxes
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        sample["image"] = torch.from_numpy(image).float()
        sample["label"] = torch.from_numpy(label).long()

        return  sample

class TensorToNumpy(object):
    """Convert Tensors in sample to ndarray."""
    def __call__(self, sample):
        image = sample["image"]
        isinstance(type(image), torch.Tensor), "format is not support {}".format(type(image))
        image = image.permute(1,2, 0).numpy()
        sample["image"] = image.astype(np.uint8)

        return sample

class RandomFlip(object):
    """Randomly flip the input image with a probability of 0.5."""
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if random.uniform(0.0, 1.0) <= self.probability:
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
            image.div_(255.0)
        else:
            image.sub_(self.mean).div_(self.std)
        sample["image"] = image
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crops the given CV Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, output_size=256):
        if isinstance(output_size, int):
            self.output_size = (int(output_size), int(output_size))
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        th, tw = self.output_size
        i = int(round(h-th) / 2.)
        j = int(round(w-tw )/ 2.)
        sample['image'] = image[i:i+th, j:j+tw]
        return sample

class Pad(object):
    """Pad the given CV Image on all sides with the given "pad" value.
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
    """Crop the given CV Image at a random location.
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
    """Horizontally flip the given CV Image randomly with a given probability.
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
    """Vertically flip the given CV Image randomly with a given probability.
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


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, probability=0.5):
        self.lower = lower
        self.upper = upper
        self.probability = probability
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, sample):
        image = sample["image"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype(np.float)

        if random.uniform(0.0, 1.0) <= self.probability:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        sample["image"] = image
        return  sample


class RandomHue(object):
    def __init__(self, delta=18.0, probability=0.5):
        assert delta>=0.0 and delta<=360.0
        self.delta = delta
        self.probability = probability

    def __call__(self, sample):
        image = sample["image"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype(np.float)

        if random.uniform(0.0, 1.0) <= self.probability:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        sample["image"] = image
        return sample


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, probability=0.5):
        self.lower = lower
        self.upper = upper
        self.probability = probability
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, sample):
        image = sample["image"]
        if random.uniform(0.0, 1.0) <= self.probability:
            alpha = random.uniform(self.lower, self.upper)
            image = image.astype(np.float)
            image *= alpha
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
            sample["image"] = image

        return sample

class RandomBrightness(object):
    def __init__(self, delta=32, probability=0.5):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.probability = probability

    def __call__(self, sample):
        image = sample["image"]
        if random.uniform(0.0, 1.0) <= self.probability:
            image = image.astype(np.float)
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)

        sample["image"] = image
        return  sample

class RandomBlur(object):
    """
    Random Blur Image
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        image = sample["image"]
        if random.uniform(0.0, 1.0) <= self.probability:
            blur_shift = random.randint(0,1)
            if blur_shift == 0:
                kernel = random.randint(1, 4)*2 + 1
                image = cv2.blur(image, (kernel, kernel))
            else:
                kernel = random.randint(1, 4)*2 + 1
                image = cv2.GaussianBlur(image, (kernel, kernel), kernel/2.0)
        sample["image"] = image
        return sample

class RandomNoise(object):
    """Applying gaussian noise on the given CV Image randomly with a given probability.
        Guassion noise
        Salt and pepper noise
        Args:
            probability (float): probability of the image being noised. Default value is 0.5
        """

    def __init__(self, probability=0.5):
        self.probability =probability

    def __call__(self, sample):
        image = sample["image"]
        if random.uniform(0.0, 1.0) <= self.probability:
            noise_shift = random.randint(0, 1)
            if noise_shift == 0:
                noise = np.random.normal(0, random.uniform(0.001, 0.004) ** 0.5, image.shape)
                image = noise*255 + image.astype(np.float)
                image = np.clip(image, 0, 255).astype(np.uint8)

            elif noise_shift == 1:
                SNR = random.uniform(0.75, 0.95)
                h, w, c = image.shape
                mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
                mask = np.repeat(mask, c, axis=2)
                image[mask == 1] = 255
                image[mask == 2] = 0

        sample["image"] = image
        return  sample

class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        CV Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (np.ndarray): Image to be converted to grayscale.
        Returns:
            np.ndarray: Randomly grayscaled image.
        """
        image = sample['image'].copy()

        if random.random() < self.p:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sample["image"] = image
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)



class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees_lower(int): lower degree
        degrees_upper(inr): upper degree
        probability(float): probability with rotation

        resample ({cv2.NEAREST, cv2.BILINEAR,cv2.BICUBIC}, optional):
            An optional resampling filter.
            If omitted, or if the image has mode "1" or "P", it is set to NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees_lower=-30, degrees_uper=30, probability = 0.5, resample='BILINEAR', expand=False, center=None):
        self.degrees_lower = degrees_lower
        self.degrees_upepr = degrees_uper
        self.probability = probability
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, sample):

        image = sample['image']
        if random.uniform(0.0, 1.0) <= self.probability:
            angle = random.randint(self.degrees_lower, self.degrees_upepr)
            print(angle)
            new_image = F.rotate(image, angle, self.resample, self.expand, self.center)
            sample['image'] = new_image

        return sample



if __name__ == '__main__':
    toTensor = ToTensor()
    tensorToNumpy = TensorToNumpy()
    resize = ResizeImage(width=256, height=256)
    centerCrop = CenterCrop(output_size=512)
    norm = Normalize()
    noise = RandomNoise(1.0)
    blur = RandomBlur(1.0)
    hue = RandomHue(probability=1.0)
    saturation = RandomSaturation(probability=1.0)
    contrast = RandomContrast(probability=1.0)
    brightness = RandomBrightness(probability=1.0)
    rotao = RandomRotation(probability=1.0)
    image_arr = cv2.imread("../../dataset/test_data/1.png")
    sample = {'image': image_arr, 'label': np.array(1)}
    import copy
    new_sample = copy.deepcopy(sample)
    new_sample = resize(new_sample)

    #
    # new_sample = centerCrop(new_sample)
    # new_sample = norm1(new_sample)
    # new_sample = noise(new_sample)
    # new_sample = blur(new_sample)
    # new_sample = saturation(new_sample)
    # new_sample = hue(new_sample)
    # new_sample = contrast(new_sample)
    # new_sample = brightness(new_sample)
    # new_sample = rotao(new_sample)

    #### tensor to numpy each
    # new_sample = toTensor(new_sample)
    # new_sample = norm(new_sample)
    # new_sample["image"] *= 255
    # new_sample = tensorToNumpy(new_sample)

    cv2.imshow("new_image", new_sample['image'])
    cv2.imshow("image", sample['image'])
    cv2.waitKey(-1)

