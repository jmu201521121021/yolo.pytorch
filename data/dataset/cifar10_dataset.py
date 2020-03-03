import os
import numpy as np

from data.dataset.base_dataset import BaseDataset
from data.dataset.build import DATASET_REGISTRY
from data.dataset.imagenet import get_image_list

__all__ = ["BuildCifar10Dataset"]


@DATASET_REGISTRY.register()
class BuildCifar10Dataset(BaseDataset):
    """Build Cifar-10 Dataset"""
    def __init__(self, cfg, training=True):
        super(BuildCifar10Dataset, self).__init__(cfg, training)
        self.setItems(cfg)

    def setItems(self, cfg):
        self.items = get_image_list(cfg.DATASET.DATA_ROOT, self.training)
