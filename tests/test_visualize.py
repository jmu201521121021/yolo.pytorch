import numpy as np
import random
import unittest
import torch
from visualize.visualize_log import  TensorBoardWriter
class TestVisualizer(unittest.TestCase):
    def test_tensorboard_scalar_dict(self):
        tb_writer = TensorBoardWriter(log_dir="./tensorboard_log_test")
        for i in range(100):
            scalar_dict = { "train_loss": random.uniform(0.0, 1.0),
                            "reg_loss": random.uniform(0.0, 1.0),
                            "cls_loss": random.uniform(0.0, 1.0),
                            "confidence_loss": random.uniform(0.0, 1.0),
                            "test_loss":  random.uniform(0.0, 1.0),
                            "top_1": random.uniform(0.0, 1.0),
                            "top_5": random.uniform(0.0, 1.0),
                            "Map@0.5": random.uniform(0.0, 1.0),
            }
            tb_writer.write(scalar_dict, (i+1))
        tb_writer.close()

    def test_tensorboard_images_dict(self):
        tb_writer = TensorBoardWriter(log_dir="./tensorboard_log_test")
        for i in range(2):
            scalar_dict = {"images":  np.zeros((2,3, 80, 80), dtype=np.int),
                           "image_names": "test_images",
                           }
            tb_writer.write(scalar_dict, (i + 1))
        tb_writer.close()

    def test_tensorboard_sclar_images_dict(self):
        tb_writer = TensorBoardWriter(log_dir="./tensorboard_log_test")
        for i in range(2):
            scalar_dict = {"images": np.zeros((2, 3, 80, 80), dtype=np.int),
                           "image_names": "test_images",
                           "train_loss": random.uniform(0.0, 1.0),
                           "reg_loss": random.uniform(0.0, 1.0),
                           "cls_loss": random.uniform(0.0, 1.0),
                           "confidence_loss": random.uniform(0.0, 1.0),
                           "test_loss": random.uniform(0.0, 1.0),
                           "top_1": random.uniform(0.0, 1.0),
                           "top_5": random.uniform(0.0, 1.0),
                           "Map@0.5": random.uniform(0.0, 1.0),
                           }
            tb_writer.write(scalar_dict, (i + 1))
        tb_writer.close()