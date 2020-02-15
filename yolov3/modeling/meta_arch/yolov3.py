from typing import List
import torch
from torch import  nn
from .build import META_ARCH_REGISTRY
from yolov3.layers import ShapeSpec, ConvNormAV, get_activate, get_norm
from yolov3.modeling.anchor_generator import build_anchor_generator
from yolov3.modeling.backbone.build import build_backbone

__all__ = ["Yolov3", "Yolov3Head"]

@META_ARCH_REGISTRY.register()
class Yolov3(nn.Module):
    """
     Implement Yolov3 (https://arxiv.org/abs/1804.02767).
    """
    def __init__(self, cfg, input_shape:ShapeSpec):
        super(Yolov3, self).__init__()
        #init param
        self.in_features = cfg.MODEL.YOLOV3.IN_FEATURES

        #backbone
        self.backbone = build_backbone(cfg, input_shape)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        #head
        self.head = Yolov3Head(cfg, feature_shapes)

    def forward(self, x):

        features = self.backbone(x)
        features = [features[f] for f in self.in_features]
        yolo_layer_outs= self.head(features)
        return  yolo_layer_outs

class Yolov3Head(nn.Module):
    """
    The head used in yolov3 for object classification and box regression.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(Yolov3Head, self).__init__()
        norm = cfg.MODEL.DARKNETS.NORM
        activate = cfg.MODEL.DARKNETS.ACTIVATE
        alpha = cfg.MODEL.DARKNETS.ACTIVATE_ALPHA

        in_channels = [input_feature.channels for input_feature in input_shape]
        num_classes = cfg.MODEL.YOLOV3.NUM_CLASSES
        num_anchors = [3, 3, 3]#build_anchor_generator(cfg, input_shape).num_cell_anchors
        yolo_layers = []

        for idx, in_channel in enumerate(in_channels):
            yolo_layer = nn.Sequential(
                ConvNormAV(in_channel,
                           in_channel*2,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           norm=get_norm(norm, in_channel*2),
                           activate=get_activate(activate, alpha),
                           bias=False),
                nn.Conv2d(in_channel*2, (num_classes + 4+1+1) * num_anchors[idx],1, 1, bias=True),
            )
            # init
            for layer in yolo_layer.modules():
                if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

            self.add_module("yolo_layer_{}".format(idx), yolo_layer)
            yolo_layers.append(yolo_layer)

        self.yolo_layers = yolo_layers

    def forward(self, features):
        outs = []
        assert (len(features) == len(self.yolo_layers))

        for feature, yolo_layer in zip(features, self.yolo_layers):
            out = yolo_layer(feature)
            outs.append(out)

        return  outs