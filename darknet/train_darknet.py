import torch
import  os
import logging
import  shutil

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from darknet.config import get_parser
from data.dataloader import build_classifier_train_dataloader, build_classifier_test_dataloader
import data.transform as transforms
from yolov3.modeling import build_backbone
from yolov3.layers import ShapeSpec
from yolov3.configs.default import get_default_config


def data_loader(cfg):
    train_dataloader = build_classifier_train_dataloader(cfg)
    val_dataloader = build_classifier_test_dataloader(cfg)
    return train_dataloader, val_dataloader

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """
        save weight
    """
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth'))

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train():
    # TODO train darknet53, dataset: imagenet
    # 1.set cfg
    args = get_parser()
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "E:\workspaces\YOLO_PYTORCH\dataset\imagenet"
    cfg.DATASET.DATASET_NAME = "BuildImageNetDataset"
    cfg.MODEL.DARKNETS.NUM_CLASSES = 1000
    input_shape = ShapeSpec(channels=args.input_channel, width=args.input_width, height=args.input_height)
    device =torch.device('cpu') if args.gpu_id is None  else torch.device('cuda')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = logging.getLogger('darknet53 train')
    tensorboard_writer = SummaryWriter('tb_logs/darknet53_train')
    # 2.load data
    train_dataloader, val_dataloader = data_loader(cfg)

    # 3.set backbone
    model = build_backbone(cfg, input_shape).to(device)

    # multi gpu
    if args.gpu_id is not None:
        if  len(args.gpu_id) > 0:
            model = nn.DataParallel(model)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained_model_path))
    # 4.set loss cross entropy
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # 5.train model and val model
    total_iter = (args.start_epoch - 1) * len(train_dataloader) * args.batch_size
    for epoch in range(args.start_epoch, args.max_epoch+1):
        for iter, input_data in enumerate(train_dataloader):
            img = input_data['image'].to(device)
            target = input_data['label'].to(device)
            total_iter += 1
            output = model(img)
            loss = criterion(output["linear"], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # save model
        if(epoch + 1) % args.save_freq == 0:
            save_checkpoint(model.state_dict(), False, args.save_dir, 'darknet53_%d.pth'.format(epoch))
        # validate model
        if(epoch + 1) % args.val_freq == 0:
            val(val_dataloader, model, criterion, args, device)

        adjust_learning_rate(optimizer, epoch, args)


def val(val_loader, model, criterion, args, device):
    with torch.no_grad():
        model.eval()
        for i,input_data in enumerate(val_loader):
            img = input_data['image'].to(device)
            target = input_data['label'].to(device)

            # compute output
            output = model(img)['linear']
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
    model.train()

if __name__ == '__main__':
    train()