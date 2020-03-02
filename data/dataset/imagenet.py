import torch
import os
import numpy as np

from data.dataset.base_dataset import BaseDataset
from data.dataset.build import DATASET_REGISTRY

__all__ = ["BuildImageNetDataset"]

def get_image_list(data_root, is_train):
    """
    Load image's path and label from txt file

    Args:
        data_root(str): root of dataset

    Returns:
        img_list(list): all imagenet' image path and lable,
        every member of list is dict[str->str, str->float]
        eg. {'image_path': '1.jpg', 'label': 0}
    """
    assert not data_root is None, "The data_root is None, please use --data_root to set it"
    if is_train:
        txt_name = "train.txt"
        data_root = os.path.join(data_root, "train")
    else:
        txt_name = "val.txt"
        data_root = os.path.join(data_root, "val")
    txt_path = os.path.join(data_root, txt_name)
    assert os.path.exists(txt_path), "Can not find {}".format(txt_path)
    img_lists = []
    with open(txt_path, 'r') as f:
        for image_label in f.readlines():
            image_label = image_label.strip("\n").split(" ")
            image = image_label[0]
            label = np.array(image_label[1], dtype=np.float32)
            img_lists.append({
                'image_path': os.path.join(data_root, image),
                'label': label
            })
    return img_lists

@DATASET_REGISTRY.register()
class BuildImageNetDataset(BaseDataset):

    def __init__(self, cfg, training=True):
        super(BuildImageNetDataset, self).__init__(cfg, training)
        self.setItems(cfg)

    def setItems(self, cfg):
        self.items = get_image_list(cfg.DATASET.DATA_ROOT, self.training)

if __name__  == "__main__":
    from yolov3.configs.default import get_default_config
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "E:\workspaces\YOLO_PYTORCH\dataset\imagenet"
    dataset = BuildImageNetDataset(cfg, training=False)
    for data in dataset:
        print(data)



