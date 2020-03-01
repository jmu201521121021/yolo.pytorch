from yolov3.solver.darknet53_solver import TrainDarknet53Solver
from yolov3.configs.default import get_default_config
from yolov3.configs.merge_config import merge_config
import argparse
__all__ = ["get_parser"]

def get_parser():
    """
        Train darnet53's config
    """
    parser = argparse.ArgumentParser(description='Pytorch train Imagenet for darknet53')
    parser.add_argument('--data_root', default="..\..\dataset\imagenet", type=str, help='path to dataset')
    parser.add_argument('--data_name', default="BuildImageNetDataset", type=str, help='name of dataset')
    parser.add_argument('--num_workers', default=8, type=int, help='thead numbers of load data')
    parser.add_argument('--num_classes', default=1000, type=int, help="classifier number")
    parser.add_argument('--decay_epoch', default=30, type=int, help='decay epoch of learning rate')
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')
    parser.add_argument('--max_epoch', default=5, type=int, help='max epoch')

    parser.add_argument('--backbone_name',default="build_darknet_backbone", type=str, help='name of model')
    parser.add_argument('--save_model_freq', default=1, type=int, help='frequency of save weight, default(every 1 epoch to save  weight)')
    parser.add_argument('--save_model_dir', default='./weights/', type=str, help='path of save weight')
    parser.add_argument('--print_log_freq', default=1, type=int, help='frequency of show logging, defualt(every 100 iteration to show logging)')
    parser.add_argument('--test_freq', default=1, type=int, help='frequency of val model, default(every 1 epoch to val)')

    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--gpu_ids', default="", type=str, help='gpu id,if multi gpu:[0, 1, ..], if cpu device, set None' )
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,help='weight decay (default: 1e-4)')
    parser.add_argument('--pretrained', default='', type=str, help='pre-trained model, if use pre-trained model, set pre-train model path')

    parser.add_argument('--train_transform', default=["ReadImage()",
                                            "CenterCrop(320)",
                                            "RandomFlip(0.5)",
                                            "RandomNoise(probability=0.4)",
                                            "RandomBlur(probability=0.4)",
                                            "RandomHue(probability=0.4)",
                                            "RandomSaturation(probability=0.4)",
                                            "RandomContrast(probability=0.4)",
                                            "RandomBrightness(probability=0.3)",
                                            "ResizeImage(256, 256)",
                                            "ToTensor()",
                                            "Normalize()" ,], type=list, help="train data transform")
    parser.add_argument('--test_transform', default=["ReadImage()","ResizeImage(256, 256)","ToTensor()", "Normalize()"], type=list, help="test data transform")

    return parser.parse_args()

def setup():
    cfg = get_default_config()
    args = get_parser()
    cfg = merge_config(args, cfg)
    return  cfg

def trainer():
    cfg = setup()
    trainer = TrainDarknet53Solver(cfg)
    trainer.train()

if __name__ == '__main__':
    trainer()