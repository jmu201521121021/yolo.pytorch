
import torch
import time
import  os
import  logging
from yolov3.modeling import  build_backbone
from yolov3.layers import  ShapeSpec
from yolov3.utils.logger import setup_logger
from  yolov3.solver.base_solver import BaseSolver
from data.dataloader import build_classifier_train_dataloader, build_classifier_test_dataloader
from yolov3.utils.events import EventStorage
from yolov3.evaluation import  ImagenetEvaluator, inference_on_dataset

__all__ = ["TrainDarknet53Solver"]

class TrainDarknet53Solver(BaseSolver):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.backbone_name    =cfg.MODEL.BACKBONE.NAME
        self.model_name       =self.backbone_name
        self.batch_size       = cfg.SOLVER.BATCH_SIZE
        self.save_model_freq  = cfg.SOLVER.SAVE_MODEL_FREQ
        self.save_model_dir   = cfg.SOLVER.SAVE_MODEL_DIR
        self.start_epoch      = cfg.SOLVER.START_EPOCH
        self.max_epoch        = cfg.SOLVER.MAX_EPOCH

        self.print_log_freq   = cfg.SOLVER.PRINT_LOG_FREQ
        self.test_freq        = cfg.SOLVER.TEST_FREQ
        self.train_freq        = cfg.SOLVER.TRAIN_VIS_ITER_FREQ
        self.lr               = cfg.SOLVER.LR
        self.decay_epoch      = cfg.SOLVER.DECAY_EPOCH
        self.gpu_ids          = cfg.SOLVER.GPU_IDS
        self.pretrained       = cfg.SOLVER.PRETRAINED

        # self.cfg.DATASET.DATA_ROOT = "E:\workspaces\YOLO_PYTORCH\dataset\imagenet"
        # self.cfg.DATASET.DATASET_NAME = "BuildImageNetDataset"
        # self.cfg.MODEL.DARKNETS.NUM_CLASSES = 1000

        self.dataset_name = self.cfg.DATASET.DATASET_NAME
        # evaluation
        self.evaluator = ImagenetEvaluator(self.dataset_name)

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
        # self.tensorbord_write = TensorBoardWriter(log_dir=self.cfg.LOG.TENSORBOARD_LOG_DIR)
        # logger
        self.logger = logging.getLogger("yolov3")
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            self.logger = setup_logger(output=self.cfg.LOG.LOG_DIR)
        # loss
        self.logger.info("start train {}".format(self))
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self):
        # 5.train model and val model
        self.before_train()
        with  EventStorage(self.start_epoch) as self.storage:
            for epoch in range(self.start_epoch, self.max_epoch + 1):
                self._train_iter = iter(self._train_dataloader)
                for _ in range(len(self._train_dataloader)):
                    self.run_step()
                    self.after_step()
                # save model
                if (epoch + 1) % self.save_model_freq == 0:
                    self.save_model(epoch)
                # validate model
                if (epoch + 1) % self.test_freq == 0:
                    self.test()
                self.adjust_learning_rate(epoch)
        self.after_train()

    def before_train(self):
        """
        Called before the first iteration.
        """
        self._train_dataloader, self._test_dataloader = self.build_dataloader()

        self.max_iter = self.max_epoch * len(self._train_dataloader)
        self._writers = self.build_writers()
        self.total_iter = (self.start_epoch - 1) * len(self._train_dataloader) * self.batch_size
        # multi gpu
        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model)
        if len(self.pretrained) > 0:
            self.model.load_state_dict(torch.load(self.pretrained))

        self.model.train(True)

    def after_train(self):
        """
        Called after the last iteration.
        """
        self.writer_close()
    def run_step(self):
        """
         Called  the  iteration.
        """
        metrics_dict = dict()
        start_time = time.perf_counter()
        input_data = next(self._train_iter)

        end_time = time.perf_counter() - start_time
        img = input_data['image'].to(self.device)
        target = input_data['label'].to(self.device)
        self.total_iter += 1
        output = self.model(img)
        loss = self.criterion(output["linear"], target)

        accuracy, pred = self.evaluator.process_single_batches(target, output["linear"], topk=(1, 5))
        metrics_dict["train_top1"] = accuracy["top1"]
        metrics_dict["train_top5"] = accuracy["top5"]
        metrics_dict["data_time"] = end_time
        metrics_dict["loss"] = loss

        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def after_step(self):
        """Called  the  after iteration."""
        self.storage.put_scalar("lr", self.optimizer.param_groups[0]["lr"])
        # log
        if self.total_iter % self.print_log_freq == 0:
            self.writers_write()
        self.storage.step()

    def test(self):
        self.evaluator.reset()
        accuracy = inference_on_dataset(self.model,self._test_dataloader, self.evaluator)
        self._write_metrics(accuracy)
        self.writers_write()
        self.storage.test_step()

    def build_dataloader(self):
        train_dataloader = build_classifier_train_dataloader(self.cfg)
        test_dataloader = build_classifier_test_dataloader(self.cfg)
        return train_dataloader, test_dataloader

