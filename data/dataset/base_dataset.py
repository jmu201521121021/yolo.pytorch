
import  torch.utils.data as data
from  data.transform import *
import copy
__all__ = ["BaseDataset"]

class BaseDataset(data.Dataset):
    def __init__(self, cfg, training=True):
        transform_list = []
        if training:
            transform_list = cfg.DATASET.TRAIN_TRANSFORM
        else:
            transform_list = cfg.DATASET.TEST_TRANSFORM
        transform  = []
        for transform_name in transform_list:
            transform.append(eval(transform_name))
        self.transform = Compose(transform)
        self.training = training
        self.items = None


    def setItems(self, cfg):
        raise  NotImplemented

    def __getitem__(self, index):
        item = copy.deepcopy(self.items[index])
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)
