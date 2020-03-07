import torch
from torch import nn
from yolov3.layers import (
    ShapeSpec,
    ConvNormAV,
    FrozenBatchNorm2d,
    get_norm,
    get_activate
)
from yolov3.modeling.backbone.backbone import Backbone
from yolov3.modeling.backbone.build import BACKBONE_REGISTRY

# refer to https://github.com/longcw/yolo2-pytorch/blob/master/darknet.py

__all__ = [
    '_make_layers',
    'DarkNet19',
    'build_darknet19_backbone',
]

def _make_layers(in_channels, net_cfg, norm='BN', activate='LeakyReLU', alpha=0.1):
    layers = []
    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg, norm, activate, alpha)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(ConvNormAV(
                    in_channels,
                    out_channels,
                    kernel_size=ksize,
                    padding=(ksize - 1) // 2,
                    norm=get_norm(norm, out_channels),
                    activate=get_activate(activate, alpha),
                    bias=False
                ))
                # layers.append(net_utils.Conv2d(in_channels, out_channels,
                #     ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


class DarkNet19(Backbone):
    def __init__(self, in_channels, kargs, num_classes=None, out_features=None):
        super(DarkNet19, self).__init__()

        self.num_classes = num_classes
        if kargs is not None:
            norm = kargs['norm']
            activate = kargs['activate']
            alpha = kargs['alpha']
        else:
            norm = 'BN'
            activate = 'LeakyReLU'
            alpha = 0.1

        net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)]
        ]
        
        # darknet
        self.conv1s, c1 = _make_layers(in_channels, net_cfgs[0:5], norm, activate, alpha)
        self._out_feature_strides = {'conv1s': 1}
        self._out_feature_channels = {'conv1s': c1}
        self.conv2, c2 = _make_layers(c1, net_cfgs[5], norm, activate, alpha)
        self._out_feature_strides['conv2'] = 1
        self._out_feature_channels['conv2'] = c2
        name = 'conv2'
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Conv2d(c2, num_classes, kernel_size=1, stride=1),
                nn.AdaptiveAvgPool2d((1)),
                nn.Softmax(dim=0)
            )
            name = 'softmax'
        
        if out_features is None:
            out_features = [name]
        self._out_features = out_features

    def forward(self, x):
        outputs = {}
        x = self.conv1s(x)
        if 'conv1s' in self._out_features:
            outputs['conv1s'] = x
        x = self.conv2(x)
        if 'conv2' in self._out_features:
            outputs['conv2'] = x
        if self.num_classes is not None:
            x = self.classifier(x)
            x = x.view(x.size(0), -1)
            if 'softmax' in self._out_features:
                outputs['softmax'] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_darknet19_backbone(cfg, input_shape):
    kargs = None
    out_features = None
    num_classes = None
    in_channels = 3 if input_shape is None else input_shape.channels
    if cfg is not None:
        kargs = {
            'norm': cfg.MODEL.DARKNETS.NORM,
            'activate': cfg.MODEL.DARKNETS.ACTIVATE,
            'alpha': cfg.MODEL.DARKNETS.ACTIVATE_ALPHA,
        }
        out_features = cfg.MODEL.DARKNETS.OUT_FEATURES
        num_classes = cfg.MODEL.DARKNETS.NUM_CLASSES
        if num_classes is not None:
            out_features = None
    return DarkNet19(in_channels, kargs, out_features=out_features, num_classes=num_classes)

if __name__ == "__main__":
    net = build_darknet19_backbone(None, None)
    print(net)
