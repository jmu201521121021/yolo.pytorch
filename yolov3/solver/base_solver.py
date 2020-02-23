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

        #init
        # self.cfg = cfg
        # self.model_name        = cfg.SOLVER.MODEL_NAME
        # self.save_model_freq   = cfg.SOLVER.SAVE_MODEL_FREQ
        # self.save_model_dir    = cfg.SOLVER.SAVE_MODEL_DIR
        # self.start_epoch       = cfg.SOLVER.START_EPOCH
        # self.max_epoch         = cfg.SOLVER.MAX_EPOCH
        # self.print_log_freq    = cfg.SOLVER.PRINT_LOG_FREQ
        # self.test_freq         = cfg.SOLVER.TEST_FREQ
        # self.lr                = cfg.SOLVER.LR
        # self.decay_epoch       = cfg.SOLVER.DECAY_EPOCH
        #
        # input_shape =  ShapeSpec(channels=cfg.MODEL.PIXEL_MEAN )
        # self.model = build_model(cfg, input_shape)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  self.lr,
        #                                  momentum=self.cfg.SOLVER.MOMENTUM,
        #                                  weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        # # tensorboard
        # self.tensorbord_write = TensorBoardWriter(log_dir=self.cfg.LOG.TENSORBOARD_LOG_DIR)
        # # logger
        # self.logger = logging.getLogger("yolov3")
        # if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        #     setup_logger(output=self.cfg.LOG.LOG_DIR)


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

