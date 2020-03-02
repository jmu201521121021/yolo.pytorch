from data.dataset.base_dataset import BaseDataset
from data.dataset.imagenet import get_image_list

__all__ = ["BuildMnistDataset"]


class BuildMnistDataset(BaseDataset):
    """Build Mnist Dataset"""
    def __init__(self, cfg, training=True):
        super(BuildMnistDataset, self).__init__(cfg, training)
        # TODO: read mnist dataset
        self.setItems(cfg)

    def setItems(self, cfg):
        self.items = get_image_list(cfg.DATASET.DATA_ROOT, self.training)

