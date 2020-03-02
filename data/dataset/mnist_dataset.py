import os
import numpy as np
import struct
from data.dataset.base_dataset import BaseDataset
from data.dataset.imagenet import get_image_list
from data.dataset.mnist import Mnist

__all__ = ["BuildMnistDataset"]


class BuildMnistDataset(BaseDataset):
    """Build Mnist Dataset"""
    def __init__(self, cfg, training=True):
        super(BuildMnistDataset, self).__init__(cfg, training)
        # TODO: read mnist dataset
        self.setItems(cfg)

    def setItems(self, cfg):
        data_root = cfg.DATASET.DATA_ROOT + os.sep

        # if file train exist, not to produce .txt files and images
        if (not os.path.exists(data_root + os.sep + 'train')):
            m = Mnist(data_root)
            m.convert_to_img(True)
            m.convert_to_img(False)

        self.items = get_image_list(cfg.DATASET.DATA_ROOT, self.training)


if __name__  == "__main__":
    from yolov3.configs.default import get_default_config
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "/home/lin/mnist"
    dataset = BuildMnistDataset(cfg, training=False)
    for data in enumerate(dataset):
        print(data)