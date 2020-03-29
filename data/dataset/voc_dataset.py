
import xml.etree.ElementTree as ET
import numpy as np
from data.dataset.base_dataset import  BaseDataset
from data.dataset.build import DATASET_REGISTRY

__all__ = ["BuildVocDataset"]

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(year, image_id, data_root):
    """
    which get a image path and boxes labels
    Args
        year(str):      voc years (2007 or 2012)
        image_id(str):  file id
        data_root(str): dataset root
    Returns
        item(dict):     image_path, boxes, label
    """
    in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml'%(data_root,year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    image_name = root.find('filename').text
    image_path = '%s/VOCdevkit/VOC%s/JPEGImages/%s'%(data_root,year, image_name)
    boxes = []
    labels = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        box = [float(xmlbox.find('xmin').text)-1,  float(xmlbox.find('ymin').text)-1, float(xmlbox.find('xmax').text)-1, float(xmlbox.find('ymax').text)-1]

        labels.append(cls_id)
        boxes.append(box)
    item = {
        "image_path":   image_path,
        "boxes":        np.array(boxes).reshape((-1, 4)),
        "label":        np.array(labels),
        "width":        width,
        "height":       height,
    }
    return  item

def get_voc_annotations(cfg):
    """
    Args
        cfg(easydict): config

    Returns
        items(list):  image_path, boxes, label of all image
    """
    data_root = cfg.DATASET.DATA_ROOT
    items = []
    for year, image_set in sets:
        image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (data_root, year, image_set)).read().strip().split()
        for image_id in image_ids:
            item = convert_annotation(year, image_id, data_root)
            items.append(item)
    return  items

@DATASET_REGISTRY.register()
class BuildVocDataset(BaseDataset):
    """ Build Coco Dataset"""
    def __init__(self, cfg, traning=True):
        super(BuildVocDataset, self).__init__(cfg, traning)
        self.setItems(cfg)

    def setItems(self, cfg):
        self.items = get_voc_annotations(cfg)


if __name__ == "__main__":
    from yolov3.configs.default import  get_default_config
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "../../../dataset/voc_dataset/"
    voc_dataset = BuildVocDataset(cfg)

