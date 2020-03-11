import torch.nn as nn
import torch
import fvcore.nn.weight_init as weight_init

from yolov3.layers import ConvNormAV, DWBnReluConv
from yolov3.layers import get_norm, get_activate, ShapeSpec
from yolov3.modeling.backbone import Backbone, BACKBONE_REGISTRY
from yolov3.configs.default import get_default_config

__all__ = ["MobileBase", "MobileNetV1", "build_mobilenetv1_backbone"]


class MobileBase(nn.Module):
    def __init__(self, input_channels,
                       output_channels,
                       stride,
                       expand_ratio):
        super(MobileBase, self).__init__()

        self.conv_dw = DWBnReluConv(input_channels,
                                    output_channels,
                                    stride=stride,
                                    expand_ratio=expand_ratio)

    def forward(self, x):
        out = self.conv_dw(x)
        return out


class MobileNetV1(Backbone):
    def __init__(self, input_channels=3,
                       num_classes=10,
                       out_features=None,
                       norm="BN",
                       activate="ReLU",
                       expand_ratio=1):
        super(MobileNetV1, self).__init__()

        self.num_classes = num_classes
        stem_out_channels = int(32 * expand_ratio)
        self._out_feature_stride = {"stem": 2}
        self._out_feature_channels = {"stem": stem_out_channels}
        out_channels = [128, 256, 512, 1024, 1024]
        self.stem = ConvNormAV(input_channels, stem_out_channels, 3, 2, 1,
                                     norm=get_norm(norm, stem_out_channels),
                                     activate=get_activate(activate, alpha=0.1))

        res1 = [MobileBase(32, 64, 1, expand_ratio),
               MobileBase(64, 128, 2, expand_ratio)]
        res2 = [MobileBase(128, 128, 1, expand_ratio),
               MobileBase(128, 256, 2, expand_ratio)]
        res3 = [MobileBase(256, 256, 1, expand_ratio),
               MobileBase(256, 512, 2, expand_ratio)]
        res4 = [MobileBase(512, 512, 1, expand_ratio),
               MobileBase(512, 512, 1, expand_ratio),
               MobileBase(512, 512, 1, expand_ratio),
               MobileBase(512, 512, 1, expand_ratio),
               MobileBase(512, 512, 1, expand_ratio),
               MobileBase(512, 1024, 2, expand_ratio)]
        res5 = [MobileBase(1024, 1024, 1, expand_ratio)]
        self.stages_and_names = []
        current_stride = self._out_feature_stride["stem"]
        stages = [res1, res2, res3, res4, res5]
        for i, blocks in enumerate(stages):
            stage = nn.Sequential(*blocks)
            name = "res"+str(i+1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            if i != (len(stages)-1):
                current_stride *= 2
            self._out_feature_stride[name] = current_stride
            self._out_feature_channels[name] = out_channels[i] * expand_ratio

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(int(1024 * expand_ratio), num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.constant_(self.linear.bias, 0)
            name = "linear"

        #classifier model
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_mobilenetv1_backbone(cfg, input_shape):
    """
    Create a MobileNet instance from config.

    Returns:
        MobileNet: a :class:`MobileNetV1` instance.
    """
    # if input_shape is None:
    #     input_shape = ShapeSpec(channels=3)
    # cfg = get_default_config()
    norm                = cfg.MODEL.MOBILENETV1.NORM
    activate            = cfg.MODEL.MOBILENETV1.ACTIVATE
    out_features        = cfg.MODEL.MOBILENETV1.OUT_FEATURES
    in_channels         = input_shape.channels
    num_classes         = cfg.MODEL.MOBILENETV1.NUM_CLASSES
    expand_ratio        = cfg.MODEL.MOBILENETV1.EXPAND_RATIO

    if num_classes is not None:
        out_features = None
    return MobileNetV1(in_channels, num_classes, out_features, norm, activate, expand_ratio)


if __name__ == "__main__":
    net = build_mobilenetv1_backbone(None, None)
    print(net)
