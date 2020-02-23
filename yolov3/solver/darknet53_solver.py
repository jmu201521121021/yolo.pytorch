
import torch
import  os
import  logging
from yolov3.modeling import  build_backbone
from yolov3.layers import  ShapeSpec
from visualize.visualize_log import TensorBoardWriter
from yolov3.utils.logger import setup_logger
from  yolov3.solver.base_solver import BaseSolver
from data.dataloader import build_classifier_train_dataloader, build_classifier_test_dataloader

__all__ = ["TrainDarknet53Solver"]

class TrainDarknet53Solver(BaseSolver):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.model_name       = cfg.SOLVER.MODEL_NAME
        self.batch_size       = cfg.SOLVER.BATCH_SIZE
        self.save_model_freq  = cfg.SOLVER.SAVE_MODEL_FREQ
        self.save_model_dir   = cfg.SOLVER.SAVE_MODEL_DIR
        self.start_epoch      = cfg.SOLVER.START_EPOCH
        self.max_epoch        = cfg.SOLVER.MAX_EPOCH
        self.print_log_freq   = cfg.SOLVER.PRINT_LOG_FREQ
        self.test_freq        = cfg.SOLVER.TEST_FREQ
        self.lr               = cfg.SOLVER.LR
        self.decay_epoch      = cfg.SOLVER.DECAY_EPOCH
        self.gpu_ids          = cfg.SOLVER.GPU_IDS
        self.pretrained       = cfg.SOLVER.PRETRAINED
        self.cfg.DATASET.DATA_ROOT = "E:\workspaces\YOLO_PYTORCH\dataset\imagenet"
        self.cfg.DATASET.DATASET_NAME = "BuildImageNetDataset"
        self.cfg.MODEL.DARKNETS.NUM_CLASSES = 1000

        self.device = torch.device('cpu') if len(self.gpu_ids) == 0  else torch.device('cuda')

        if not  os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

        input_shape = ShapeSpec(channels=self.cfg.MODEL.PIXEL_MEAN)
        self.model = build_backbone(self.cfg, input_shape).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         self.lr,
                                         momentum=self.cfg.SOLVER.MOMENTUM,
                                         weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        # tensorboard
        self.tensorbord_write = TensorBoardWriter(log_dir=self.cfg.LOG.TENSORBOARD_LOG_DIR)
        # logger
        self.logger = logging.getLogger("yolov3")
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            self.logger = setup_logger(output=self.cfg.LOG.LOG_DIR)
        # loss
        self.logger.info("start train {}".format(self.model_name))
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self):

        train_dataloader, test_dataloader = self.build_dataloader()
        # multi gpu
        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model)
        if len(self.pretrained) >0:
            self.model.load_state_dict(torch.load(self.pretrained))
        # 5.train model and val model
        total_iter = (self.start_epoch - 1) * len(train_dataloader) * self.batch_size
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            for iter, input_data in enumerate(train_dataloader):
                img = input_data['image'].to(self.device)
                target = input_data['label'].to(self.device)
                total_iter += 1
                output = self.model(img)
                loss = self.criterion(output["linear"], target)
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # save model
            if (epoch + 1) % self.save_model_freq == 0:
                self.save_model(epoch)
            # validate model
            if (epoch + 1) % self.test_freq == 0:
                self.test(test_dataloader)

            self.adjust_learning_rate(epoch)

    def test(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            for i, input_data in enumerate(dataloader):
                img = input_data['image'].to(self.device)
                target = input_data['label'].to(self.device)

                # compute output
                output = self.model(img)['linear']
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.model.train()
        pass

    def build_dataloader(self):
        train_dataloader = build_classifier_train_dataloader(self.cfg)
        test_dataloader = build_classifier_test_dataloader(self.cfg)
        return train_dataloader, test_dataloader
