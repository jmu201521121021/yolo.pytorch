
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov3.layers import get_activate, get_norm, ShapeSpec
from yolov3.modeling import Backbone

from yolov3.layers import SELayer, DWConv

class GhostModule(nn.Module):
    def __init__(self, input_channels,
                      output_channels,
                      ratio,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      use_bias=True,
                      linear_kernel=3,
                      activation="ReLU",
                      alpha=0,):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        # only support  output_channels % ratio == 0, this is different with paper
        assert output_channels % ratio == 0, "not support format with param s"
        self.middle_channels = int(output_channels // ratio)
        self.conv = nn.Conv2d(input_channels, self.middle_channels, kernel_size, stride, padding, bias=use_bias)
        self.linear_transfer = nn.Conv2d(self.middle_channels, (output_channels - self.middle_channels), linear_kernel, 1,
                                         padding=int(linear_kernel/2), groups=self.middle_channels, bias=False)
        if activation is not  None:
            self.activation = get_activate(activation, alpha)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        # when ratio = 1, then equal simple conv2d
        if self.ratio == 1:
            return  x

        if self.activation is not  None:
            x = self.activation(x)
        linear_out = self.linear_transfer(x)
        if self.activation is not  None:
            linear_out = self.activation(linear_out)

        out = torch.cat((x, linear_out), 1)
        return out

class Bottlenecks(nn.Module):
    def __init__(self, input_channels,
                        output_channels,
                        expand_ratio,
                        ratio,
                        linear_size=3,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        use_bias=False,
                        norm = "BN",
                        activation="ReLU",
                        alpha=0,
                        use_se=False,
                        ):
        super(Bottlenecks, self).__init__()

        middle_channels = input_channels * expand_ratio
        self.ghost_module0 = nn.Sequential(GhostModule(input_channels, middle_channels, ratio, kernel_size,1, padding, linear_kernel=linear_size, use_bias=use_bias),
                                           get_activate(activation, alpha),
                                           get_norm(norm, middle_channels),
                                           )
        self.ghost_module1 = nn.Sequential(GhostModule(middle_channels, output_channels, ratio, kernel_size, 1, padding, linear_kernel=linear_size, use_bias=use_bias),
                                            get_norm(norm, output_channels),
                                           )
        self.dw_stride2x2 = nn.Conv2d(middle_channels, middle_channels, kernel_size, stride, padding, groups=middle_channels) if stride == 2 else None
        self.stride2x2_res = DWConv(input_channels, output_channels, kernel_size, stride, padding) if stride == 2 else None
        self.se_layer = SELayer(middle_channels, 4) if use_se else None

    def forward(self, x):
        shortcut = x
        # ghost_module1
        out = self.ghost_module0(x)
        # stride = 2
        if self.dw_stride2x2 is not  None:
            shortcut = self.stride2x2_res(shortcut)
            out = self.dw_stride2x2(out)
        #SE
        if self.se_layer is not None:
            out = self.se_layer(out)
        # ghost_module2
        out = self.ghost_module1(out)
        # add
        out = out + shortcut
        return  out

class Ghostnet(Backbone):
    """
    Implement ghostNet (https://arxiv.org/abs/1911.11907).
    """
    def __init__(self, input_channels=3,
                        num_classes=1000,
                        norm="BN",
                        activation="ReLU",
                        alpha=0,
                        ratio=2,
                        linear_size=3,
                        expand_ratio_channel=1.0):
        super(Ghostnet, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(input_channels, 16, 3, 2 ,1),
                                  get_norm(norm, 16),
                                  get_activate(activation, alpha))

        res1 = [Bottlenecks(16, 16, 1.0, ratio,linear_size=linear_size, activation=activation),
                     Bottlenecks(16, 24, 48 / 16, ratio,stride=2,linear_size=linear_size, activation=activation),]

        res2 = [Bottlenecks(24, 24, 72 / 24, ratio, linear_size=linear_size, activation=activation),
                     Bottlenecks(24, 40, 72 / 24, ratio, stride=2, use_se=True, linear_size=linear_size, activation=activation),]

        res3 = [Bottlenecks(40, 40, 120 / 40, ratio, use_se=True,linear_size=linear_size, activation=activation),
                    Bottlenecks(40, 80, 240 / 40, ratio, stride=2, linear_size=linear_size, activation=activation), ]

        res4 = [Bottlenecks(80, 80, 200 / 80, ratio, linear_size=linear_size, activation=activation),
                     Bottlenecks(80, 80, 184 / 80, ratio, linear_size=linear_size, activation=activation),
                     Bottlenecks(80, 80, 184 / 80, ratio, linear_size=linear_size, activation=activation),
                     Bottlenecks(80, 112, 480 / 80, ratio, use_se=True, linear_size=linear_size,activation=activation),
                     Bottlenecks(112, 112, 672 / 112, ratio, use_se=True, linear_size=linear_size, activation=activation),
                     Bottlenecks(112, 160, 672 / 112, ratio, use_se=True,stride=2, linear_size=linear_size, activation=activation),]

        res5 = [Bottlenecks(160, 160, 960 / 160, ratio, linear_size=linear_size, activation=activation),
                     Bottlenecks(160, 160, 960 / 160, ratio,  use_se=True, linear_size=linear_size, activation=activation),
                     Bottlenecks(160, 160, 960 / 160, ratio, linear_size=linear_size, activation=activation),
                     Bottlenecks(160, 160, 960 / 160, ratio, use_se=True, linear_size=linear_size, activation=activation),
                     nn.Conv2d(160, 960, 3, 1, 1, bias=True)
                   ]
        self.res1 = nn.Sequential(*res1)
        self.res2 = nn.Sequential(*res2)
        self.res3 = nn.Sequential(*res3)
        self.res4 = nn.Sequential(*res4)
        self.res5 = nn.Sequential(*res5)

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.conv_last = nn.Conv2d(960, 1280, 1, 1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)

        out = self.global_avg(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return  out

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
def build_ghostnet_backbone(cfh, input_shape):
    """
    Create a ghostnet instance from config.

    Returns:
        ghostnet: a :class:`ghostnet` instance.
    """

if __name__ == "__main__":
    se_layer = SELayer(32, 16)
    ghost_module = GhostModule(3, 32, 4)
    print(ghost_module)
    x = torch.randn(1, 3, 256, 256)
    out = ghost_module(x)
    se_out = se_layer(out)
    x_bls = torch.randn(1, 16, 256, 256)
    bottlenecks = Bottlenecks(16, 32, 2, 2, stride=2, use_se=True)
    print(bottlenecks)
    out_bls = bottlenecks(x_bls)

    print(out_bls.size())
    print(out.size(), se_out.size())
