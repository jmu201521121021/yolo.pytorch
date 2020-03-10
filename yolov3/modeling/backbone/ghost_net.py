
import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init

from yolov3.layers import get_activate, get_norm, ShapeSpec
from yolov3.modeling.backbone import Backbone, BACKBONE_REGISTRY

from yolov3.layers import SELayer, DWConv

__all__ = ["GhostModule", "Ghostnet", "build_ghostnet_backbone" ]
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

        middle_channels = int(input_channels * expand_ratio)
        self.ghost_module0 = nn.Sequential(GhostModule(input_channels, middle_channels, ratio, kernel_size,1, padding, linear_kernel=linear_size, use_bias=use_bias),
                                           get_activate(activation, alpha),
                                           get_norm(norm, middle_channels),
                                           )
        self.ghost_module1 = nn.Sequential(GhostModule(middle_channels, output_channels, ratio, kernel_size, 1, padding, linear_kernel=linear_size, use_bias=use_bias),
                                            get_norm(norm, output_channels),
                                           )
        self.dw_stride2x2 = nn.Conv2d(middle_channels, middle_channels, kernel_size, stride, padding, groups=middle_channels) if stride == 2 else None
        self.res_block = None if input_channels == output_channels else  DWConv(input_channels, output_channels, kernel_size, stride, padding)
        self.se_layer = SELayer(middle_channels, 4) if use_se else None

    def forward(self, x):
        if self.res_block is not  None:
            shortcut = self.res_block(x)
        else:
            shortcut = x
        # ghost_module1
        out = self.ghost_module0(x)
        # stride = 2
        if self.dw_stride2x2 is not  None:
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
                        out_features=None,
                        norm="BN",
                        activation="ReLU",
                        alpha=0,
                        ratio=2,
                        linear_size=3,
                        expand_ratio_channel=1.0):
        super(Ghostnet, self).__init__()

        self.num_classes = num_classes
        self._out_feature_strides = {"stem": 2}
        self._out_feature_channels = {"stem": 16}
        res_out_channels = [24, 40, 80, 160, 960]
        self.stem = nn.Sequential(nn.Conv2d(input_channels, 16, 3, 2 ,1, bias=False),
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
        self.stages_and_names = []
        current_stride = self._out_feature_strides["stem"]
        stages = [res1, res2, res3, res4, res5]
        for i, blocks in enumerate(stages):
            stage = nn.Sequential(*blocks)
            name = "res" + str(i+1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            if i != (len(stages)-1):
                current_stride = current_stride * 2
            self._out_feature_strides[name] =  current_stride
            self._out_feature_channels[name] = res_out_channels[i]

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.conv_last = nn.Sequential(nn.Conv2d(960, 1280, 1, 1),
                                           get_norm(norm, 1280),
                                           get_activate(activation, alpha),)

            self.linear = nn.Linear(1280, num_classes)
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
            x = self.conv_last(x)
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
def build_ghostnet_backbone(cfg, input_shape):
    """
    Create a ghostnet instance from config.

    Returns:
        ghostnet: a :class:`ghostnet` instance.
    """
    norm                = cfg.MODEL.GHOSTNET.NORM
    activate            = cfg.MODEL.GHOSTNET.ACTIVATE
    alpha               = cfg.MODEL.GHOSTNET.ACTIVATE_ALPHA

    out_features        = cfg.MODEL.GHOSTNET.OUT_FEATURES
    in_channels         = input_shape.channels
    num_classes         = cfg.MODEL.GHOSTNET.NUM_CLASSES

    linear_kernel_size  =  cfg.MODEL.GHOSTNET.LINEAR_KERBER_SIZE
    ratio               =  cfg.MODEL.GHOSTNET.RATIO

    if num_classes is not None:
        out_features = None
    return Ghostnet(in_channels, num_classes, out_features, norm, activate, alpha, ratio, linear_kernel_size)
