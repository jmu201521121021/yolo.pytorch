import torch
import torch.nn as nn
from yolov3.layers import get_activate, ShapeSpec
from yolov3.modeling.backbone import Backbone, BACKBONE_REGISTRY



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activate="ReLU", alpha=0):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activate(activate, alpha)
        )


class Bottlenecks(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, activate="ReLU", alpha=0):
        super(Bottlenecks, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, activate=activate, alpha=alpha))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, activate=activate, alpha=alpha),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobilenetV2(Backbone):
    """
    Implement mobileNetV2 (https://arxiv.org/abs/1801.04381)
    """
    def __init__(self,
                 in_channels=32,
                 num_classes=10,
                 activate="ReLU",
                 out_features=None,
                 alpha=0,
                 width_mult=1.0,
                 round_nearest=1,
                 ):
        """
        :param num_classes: Number of classes
        :param width_mult: Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
        :param inverted_residual_setting:
        :param round_nearest: Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        :param block: Module specifying inverted residual building block for mobilenet
        """
        super(MobilenetV2, self).__init__()
        self.num_classes = num_classes
        self._out_feature_stride = {"stem": 2}
        self._out_feature_channels = {"stem": 32}
        input_channel = in_channels
        out_channels = [24, 32, 64, 160, 320]
        last_channel = 1280

        res1 = [
            Bottlenecks(in_channels=32, out_channels=16, stride=1, expand_ratio=1, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=16, out_channels=24, stride=2, expand_ratio=6, activate="ReLU", alpha=0)
        ]

        res2 = [
            Bottlenecks(in_channels=24, out_channels=24, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=24, out_channels=32, stride=2, expand_ratio=6, activate="ReLU", alpha=0)
        ]

        res3 = [
            Bottlenecks(in_channels=32, out_channels=32, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=32, out_channels=32, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=32, out_channels=64, stride=2, expand_ratio=6, activate="ReLU", alpha=0),
        ]

        res4 = [
            Bottlenecks(in_channels=64, out_channels=64, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=64, out_channels=64, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=64, out_channels=64, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=64, out_channels=96, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=96, out_channels=96, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=96, out_channels=96, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=96, out_channels=160, stride=2, expand_ratio=6, activate="ReLU", alpha=0),
        ]

        res5 = [
            Bottlenecks(in_channels=160, out_channels=160, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=160, out_channels=160, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
            Bottlenecks(in_channels=160, out_channels=320, stride=1, expand_ratio=6, activate="ReLU", alpha=0),
        ]

        self.stages_and_names = []
        current_stride = self._out_feature_stride["stem"]
        stages = [res1, res2, res3, res4, res5]

        # build first layer
        self.stem = nn.Sequential(ConvBNReLU(input_channel, 32, stride=2, activate=activate, alpha=alpha))

        # build inverted residual blocks
        for i, blocks in enumerate(stages):
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            if i != (len(stages) - 1):
                current_stride *= 2
            self._out_feature_stride[name] = current_stride
            self._out_feature_channels[name] = out_channels[i]

        # build last several layers
        nn.Sequential(ConvBNReLU(out_channels[4], last_channel, kernel_size=1, activate=activate, alpha=alpha))

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(320, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.constant_(self.linear.bias, 0)
            name = "linear"

        # classifier model
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        self._initialize_weights()

    # weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
def build_mobilenetV2_backbone(cfg, input_shape):
    """
    Create a mobilenet instance form config
    :param cfg:
    :param input_shape:
    :return:
        mobilenet: a : class:`mobilenet` instance
    """

    activate = cfg.MODEL.MOBILENETV2.ACTIVATE
    alpha = cfg.MODEL.GHOSTNET.ACTIVATE_ALPHA
    out_features = cfg.MODEL.MOBILENETV2.OUT_FEATURES
    in_channels = input_shape.channels
    num_classes = cfg.MODEL.MOBILENETV2.NUM_CLASSES

    if num_classes is not None:
        out_features = None
    return MobilenetV2(in_channels, num_classes, activate, out_features, alpha=alpha)

if __name__ == "__main__":
    net = build_mobilenetV2_backbone(2, 10)
    print(net)