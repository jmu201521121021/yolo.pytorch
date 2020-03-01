
from data.dataset.base_dataset import  BaseDataset

__all__ = ["BuildCocoDataset"]

class BuildCocoDataset(BaseDataset):
    """ Build Coco Dataset"""
    def __init__(self, cfg, traning=True):
        super(BuildCocoDataset, self).__init__(cfg, traning)
        # TODO: read coco dataset
        self.setItems(cfg)

    def setItems(self, cfg):
        self.items = None

