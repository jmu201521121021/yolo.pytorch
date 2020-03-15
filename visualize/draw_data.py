import cv2
import numpy as np

__all__ = ["draw_rectangle"]

def draw_rectangle(in_img, boxes, label=None, color=(0, 0, 255), thickness=1, line_type=1):
    """
    draw rectangle and text
    Args
        in_img(numpy.ndarray)   :input image
        boxes(numpy.ndarray)    : boxes shape (N ,4), every box is (x_min, y_min, x_max, y_max)
        color(tuple)            : color of line
        thickness(int)          : width of line
        line_typei(int)         : line type
    Returns
        img(numpy.ndarry)   : img
    """
    assert isinstance(in_img, np.ndarray), "image not support formet {}".format(type(in_img))
    assert boxes.shape[1] == 4, "boxes shape should (N, 4)"

    img = in_img.copy()
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color, thickness=thickness, lineType=line_type)
        if label is not  None:
            cv2.putText(img, str(label[i]), (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,0),2)
    return img


if __name__  == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    from data.dataset import BuildVocDataset
    from  yolov3.configs.default import get_default_config
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "../../dataset/voc_dataset/"
    cfg.DATASET.TRAIN_TRANSFORM = ["ReadImage()"]
    voc_dataset = BuildVocDataset(cfg)
    for item in voc_dataset:
        image = cv2.imread(item["image_path"])
        boxes = item["boxes"]
        label = item["label"]
        image = draw_rectangle(image, boxes, [classes[la] for la in label])
        cv2.imshow("image", image)
        cv2.waitKey(-1)