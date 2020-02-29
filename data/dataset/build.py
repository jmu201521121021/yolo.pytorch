
from  yolov3.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset
1. A :easyDict `
"""
__all__ = ["build_dataset"]

def build_dataset(cfg, training=True):
    """
    build a dataset from 'cfg.DATASET.DATASET_NAME'
    Args
        cfg(edict): config param
        training(bool): training or eval
    Returens
        an instance of: torch,utils.data.DATASET
    """
    dataset_name = cfg.DATASET.DATASET_NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg, training)

    return  dataset


if __name__  == "__main__":
    from data.dataset.imagenet import*
    from yolov3.configs.default import get_default_config
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "E:\workspaces\YOLO_PYTORCH\dataset\imagenet"
    cfg.DATASET.DATASET_NAME = "BuildImageNetDataset"
    dataset = build_dataset(cfg, training=True)
    print(len(dataset))
