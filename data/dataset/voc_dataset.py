from data.dataset.base_dataset import  BaseDataset

__all__ = ["BuildVocDataset"]

class BuildVocDataset(BaseDataset):
    """ Build Coco Dataset"""
    def __init__(self, cfg, traning=True):
        super(BuildVocDataset, self).__init__(cfg, traning)
        # TODO: read voc2007 and voc2012 dataset
        self.setItems(cfg)

        def setItems(self, cfg):
            self.items = None