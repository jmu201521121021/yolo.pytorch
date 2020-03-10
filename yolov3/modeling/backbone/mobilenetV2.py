import torch
import torch.nn as nn
from yolov3.layers import get_activate
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

class Mobilenet_V2(Backbone):
    """
    Implement mobileNetV2 (https://arxiv.org/abs/1801.04381)
    """
    def __init__(self,
                 in_channels=32,
                 num_classes=1000,
                 activate="ReLU",
                 out_features=None,
                 alpha=0,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        :param num_classes: Number of classes
        :param width_mult: Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
        :param inverted_residual_setting:
        :param round_nearest: Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        :param block: Module specifying inverted residual building block for mobilenet
        """
        super(Mobilenet_V2, self).__init__()

        if block is None:
            block = Bottlenecks
        input_channel = in_channels
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # build first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        out_features = [ConvBNReLU(3, input_channel, stride=2, activate=activate, alpha=alpha)]

        # build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                out_features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # build last several layers
        out_features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, activate=activate, alpha=alpha))

        # make it nn.Sequential
        self.out_features = nn.Sequential(*out_features)

        # build classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
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
        x = self.out_features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


@BACKBONE_REGISTRY.register()
def build_mobilenetV2_backbone(cfg, input_shape):
    """
    Create a mobilenet instance form config
    :param cfg:
    :param input_shape:
    :return:
        mobilenet: a : class:`mobilenet` instance
    """
    activate = "ReLU"
    alpha = 0

    out_features = None
    in_channels = 32
    num_classes = 1000

    if num_classes is not None:
        out_features = None
    return Mobilenet_V2(in_channels, num_classes, activate, out_features, alpha=alpha)

if __name__ == "__main__":
    net = build_mobilenetV2_backbone(2, 1000)
    print(net)
