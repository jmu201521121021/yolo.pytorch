import torch
import os
import numpy as np
import torch.utils.data as data

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

class ImageNetDataset(data.Dataset):
    def __init__(self, data_root, transform, is_train=True):
        self.items = get_image_list(data_root, is_train)
        self.transform = transform

    def __getitem__(self, index):
        item = self.items[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)
