
import os
import torch
import  logging
import random
import logging
import numpy as np
import torch
import  datetime
from  data.dataset.build import build_dataset
from yolov3.utils.logger import setup_logger

import torch.utils.data as data

__all__ = ["build_classifier_train_dataloader",
           "build_classifier_test_dataloader",
           "build_detect_train_dataloader",
           "build_detect_test_dataloader",
           "seed_all_rng"]

def build_classifier_train_dataloader(cfg):
    """
    build classifier dataloader with train
    Args
        cfg(edict):config param
    Returns
        an instance ; torch.utils.data.DataLoader
    """
    dataset = build_dataset(cfg, training=True)
    logger = logging.getLogger("yolov3")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        logger = setup_logger(output=cfg.LOG.LOG_DIR)

    logger.info("classifier dataset number of train : {}".format(len(dataset)))
    dataloader = data.DataLoader(dataset,
                                 shuffle=True,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                 drop_last=True,
                                 batch_size=cfg.SOLVER.BATCH_SIZE,
                                 worker_init_fn=worker_init_reset_seed,
    )
    return  dataloader

def build_classifier_test_dataloader(cfg):
    """
    build classifier dataloader with test
    Args
        cfg(edict):config param
    Returns
        an instance ; torch.utils.data.DataLoader

    """
    dataset = build_dataset(cfg, training=False)

    logger = logging.getLogger("yolov3")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        logger = setup_logger(output=cfg.LOG.LOG_DIR)
    logger.info("classifier dataset number of test : {}".format(len(dataset)))
    dataloader = data.DataLoader(dataset,
                                 shuffle=True,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                 drop_last=True,
                                 batch_size=cfg.SOLVER.BATCH_SIZE,
                                 )
    return  dataloader

def build_detect_train_dataloader(cfg):
    """
    build detector dataloader with train
    Args
        cfg(edict):config param
    Returns
        an instance ; torch.utils.data.DataLoader
    """
    dataset = build_dataset(cfg, training=True)
    logger = logging.getLogger("yolov3")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        logger = setup_logger(output=cfg.LOG.LOG_DIR)

    logger.info("detector dataset number of train : {}".format(len(dataset)))
    dataloader = data.DataLoader(dataset,
                                 shuffle=True,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                 drop_last=True,
                                 batch_size=cfg.SOLVER.BATCH_SIZE,
                                 collate_fn=trivial_batch_collator,
                                 worker_init_fn=worker_init_reset_seed,
    )
    return  dataloader

def build_detect_test_dataloader(cfg):
    """
    build detector dataloader with test
    Args
        cfg(edict):config param

    Returns
        an instance ; torch.utils.data.DataLoader
    """
    dataset = build_dataset(cfg, training=False)
    logger = logging.getLogger("yolov3")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        logger = setup_logger(output=cfg.LOG.LOG_DIR)

    logger.info("detector dataset number of test : {}".format(len(dataset)))
    dataloader = data.DataLoader(dataset,
                                 shuffle=True,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                 drop_last=True,
                                 batch_size=cfg.SOLVER.BATCH_SIZE,
                                 )
    return  dataloader

def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)

def trivial_batch_collator(batch):
    # TODO: detect data format
    images = []
    labels = []
    for i, sample in enumerate(batch):
        if "image" in sample:
            images.append(sample["image"])
        if "label" in sample:
            labels.append(sample["label"])
    batch = {"image": torch.stack(images, 0), "label": labels}
    return batch