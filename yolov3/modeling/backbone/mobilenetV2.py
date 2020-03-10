import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from conda import activate

from yolov3.layers import get_activate, get_norm, ShapeSpec
from yolov3.modeling.backbone import Backbone, BACKBONE_REGISTRY

from yolov3.layers import SELayer, DWConv

class Mobilenet(Backbone):
    """
    Implement mobileNetV2 ()
    """

@BACKBONE_REGISTRY.register()
def build_mobilenet_backbone(cfg, input_shape):
    """
    Create a mobilenet instance form config
    :param cfg:
    :param input_shape:
    :return:
        mobilenet: a : class:`mobilenet` instance
    """
    norm               = cfg.MODEL.MOBILENET.NORM
    activate           = cfg.MODEL.MOBILENET.ACTIVATE
    alpha              = cfg.MODEL.MOBILENET.ACTIVATE_ALPHA

    out_features       = cfg.MODEL.MOBILENET.OUT_FEATURES
    in_channels        = input_shape.channels
    num_classes        = cfg.MODEL.MOBILENET.NUM_CLASSES

    linear_kernel_size = cfg.MODEL.MOBILENET.LINEAR_KERBER_SIZE
    ratio              = cfg.MODEL.MOBILENET.RATIO

    if num_classes is not None:
        out_features = None
    return Mobilenet(in_channels, num_classes, out_features, norm, activate, alpha, ratio, linear_kernel_size)

