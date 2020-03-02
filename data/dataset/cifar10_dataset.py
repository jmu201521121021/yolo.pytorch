import os
import numpy as np

from data.dataset.base_dataset import BaseDataset
from data.dataset.build import DATASET_REGISTRY
from data.dataset.imagenet import get_image_list

__all__ = ["BuildCifar10Dataset"]


@DATASET_REGISTRY.register()
class BuildCifar10Dataset(BaseDataset):

    def __init__(self, cfg, training=True):
        super(BuildCifar10Dataset, self).__init__(cfg, training)
        self.setItems(cfg)

    def setItems(self, cfg):
        self.items = get_image_list(cfg.DATASET.DATA_ROOT, self.training)


if __name__ == "__main__":
    from yolov3.configs.default import get_default_config
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "/home/lrr/下载/yolo.pytorch/data/dataset"
    dataset = BuildCifar10Dataset(cfg, training=False)
    for i, data in enumerate(dataset):
        print(data)
        if i == 1:
            break
