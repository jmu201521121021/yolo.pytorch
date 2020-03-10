
import torch
import torch.nn.functional  as F
import numpy as np
import logging
from collections import OrderedDict, defaultdict
from yolov3.evaluation.evaluation import DatasetEvaluator

class ImagenetEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        super().__init__()
        self._dataset_name = dataset_name
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = list()
        self._accuracy  = defaultdict(list)

    def process_single_batches(self, target, output, topk=(1, )):
        output = output.to(self._cpu_device)
        target = target.to(self._cpu_device)
        output = F.softmax(output, 1)
        accuracy = {}
        pred = None
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                acc = correct_k.mul_(100.0 / batch_size)
                accuracy["top" + str(k)] = acc

        return accuracy, pred

    def process(self, target, output, topk=(1,5)):
        target = target["label"]
        output = output["linear"]
        output = F.softmax(output, 1)
        output = output.to(self._cpu_device)
        target = target.to(self._cpu_device)
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            self._predictions.append(pred)
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                acc = correct_k.mul_(100.0 / batch_size)
                self._accuracy["top" + str(k)].append(acc)


    def evaluate(self):

        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., top1, top5)
                * value: a dict of {metric name: score}, e.g.: {"top1": 80}
        """
        self._logger.info(
            "Evaluating {} ".format(
                self._dataset_name,
            )
        )
        accuracy = {}
        for top_id, acc in self._accuracy.items():
            accuracy[top_id] = np.mean(np.array(acc))
        return accuracy