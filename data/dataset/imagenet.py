import torch
import os
import torch.utils.data.Dataset as Dataset
def get_image_list(data_root):
    """
    Args:
        data_root(str): root of dataset

    Returns:
        img_list(list): all imagenet' image path and lable,
        every member of list is dict[str->str, str->int]
        eg. {'image_path': '1.jpg', 'label': 0}
    """
    img_lists = []
    # TODO: read img_lists

    return img_lists

class ImageNetDataset(Dataset):
    def __init__(self, data_root, transform):
        self.items = get_image_list(data_root)
        self.transform = transform

    def __getitem__(self, index):
        item = self.items[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)