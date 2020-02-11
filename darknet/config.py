import argparse
__all__ = ["get_parser"]
def get_parser():
    """
        Train darnet53's config
    """
    parser = argparse.ArgumentParser(description='Pytorch train Imagenet for darknet53')
    parser.add_argument('--data_root', type=str, help='path to dataset')
    parser.add_argument('--input_channel', type=int, default=3, help='channel of input data')
    parser.add_argument('--input_width', type=int, default=256, help='width of input data')
    parser.add_argument('--input_height', type=int, default=256, help='height of input data ')
    parser.add_argument('--num_workers', default=8, type=int, help='thead numbers of load data')
    parser.add_argument('--decay_epoch', default=30, type=int, help='decay epoch of learning rate')
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')
    parser.add_argument('--max_epoch', default=100, type=int, help='max epoch')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of save weight, default(every 1 epoch to save  weight)')
    parser.add_argument('--save_dir', default='./weights/', type=str, help='path of save weight')
    parser.add_argument('--log_freq', default=100, type=int, help='frequency of show logging, defualt(every 100 iteration to show logging)')
    parser.add_argument('--val_freq', default=1, type=int, help='frequency of val model, default(every 1 epoch to val)')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--gpu_id', default=None, type=list, help='gpu id,if multi gpu:[0, 1, ..], if cpu device, set None' )
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,help='weight decay (default: 1e-4)')
    parser.add_argument('--pretrained', action='store_true',help='use pre-trained model')
    parser.add_argument('--pretrained_model_path', default='model_best.pth', help='pre-trained model path')

    return parser.parse_args()