import torch
import  logging
import  os
from abc import  ABCMeta, abstractclassmethod
from yolov3.modeling import  build_model
from yolov3.layers import  ShapeSpec
from visualize.visualize_log import TensorBoardWriter
from yolov3.utils.logger import setup_logger

__all__ = ["BaseSolver"]

class BaseSolver(metaclass=ABCMeta):
    """
    Abstract base class for Train and Test Solver
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractclassmethod
    def train(self):
       pass

    @abstractclassmethod
    def build_dataloader(self):
        train_dataloader = None
        test_dataloader = None
        return  train_dataloader, test_dataloader

    def test(self,dataloader):
        pass

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()

        print(self.model)
        print("The number of parameters: {}".format(num_params))

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_model_dir, self.model_name + "_{}.pth".format(epoch)))

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // self.decay_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

