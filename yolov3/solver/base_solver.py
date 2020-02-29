import torch
import  os
from abc import  ABCMeta, abstractclassmethod
from yolov3.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from yolov3.utils.events import EventStorage
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
                    self.test(self._test_dataloader)
                self.adjust_learning_rate(epoch)

        self.after_train()

    @abstractclassmethod
    def build_dataloader(self):
        train_dataloader = None
        test_dataloader = None
        return  train_dataloader, test_dataloader

    def before_train(self):
        """
        Called before the first iteration.
        """

    def after_train(self):
        """
        Called after the last iteration.
        """
        self.writer_close()

    def run_step(self):
        """
         Called  the  iteration.
        """

    def after_step(self):
        """Called  the  after iteration."""

    def test(self):
        pass

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()

        print(self.model)
        print("The number of parameters: {}".format(num_params))

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_model_dir, self.model_name + "_epoch_{}.pth".format(epoch)))

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // self.decay_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
               CommonMetricPrinter(self.max_iter),
              JSONWriter(os.path.join(self.cfg.LOG.LOG_DIR, "metrics.json")),
              TensorboardXWriter(self.cfg.LOG.LOG_DIR),
            ]

        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.LOG.LOG_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.LOG.LOG_DIR),
        ]

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        if "data_time" in metrics_dict:
            self.storage.put_scalar("data_time", metrics_dict["data_time"])
            metrics_dict.pop("data_time")

        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        self.storage.put_scalar("total_loss", total_losses_reduced)
        if len(metrics_dict) > 1:
            self.storage.put_scalars(**metrics_dict)

    def writers_write(self):
        for writer in self._writers:
            writer.write()

    def writer_close(self):
        for writer in self._writers:
            writer.close()
