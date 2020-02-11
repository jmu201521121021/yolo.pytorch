import torch
from torch import nn

from yolov3.layers import (
    ShapeSpec,
    ConvNormAV,
    FrozenBatchNorm2d,
    get_norm,
    get_activate,
)
from  yolov3.modeling.backbone.backbone import Backbone
from yolov3.modeling.backbone.build import BACKBONE_REGISTRY

__all__ = [ "DarkNetBlockBase",
            "BottleneckBlock",
            "BasicPool",
            "BasicStem",
            "make_stage",
            "DarkNet",
            "build_darknet_backbone",
            ]

class DarkNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

class BottleneckBlock(DarkNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        num_groups=1,
        norm="BN",
        activate = 'PReLU',
        alpha = 0.1,
        dilation=1,
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super(BottleneckBlock, self).__init__(in_channels, out_channels, 1)
        stride_1x1, stride_3x3 = (1, 1)

        self.conv_norm_av1 = ConvNormAV(
            in_channels,
            bottleneck_channels,
            kernel_size = 1,
            stride= stride_1x1,
            norm = get_norm(norm, bottleneck_channels),
            activae= get_activate(activate, alpha),
            bias=False,
        )

        self.conv_norm_av2 = ConvNormAV(
            bottleneck_channels,
            out_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, out_channels),
            activae=get_activate(activate, alpha),
            bias=False
        )

    def forward(self, x):
        shortcut = x
        out = self.conv_norm_av1(x)

        out = self.conv_norm_av2(out)

        out = shortcut + out

        return out

class BasicPool(DarkNetBlockBase):

    def __init__(self, in_channels, out_channels, norm="BN", activate="PReLU", alpha=0.1, stride=2):
        super(BasicPool, self).__init__(in_channels, out_channels, stride=stride)
        self.conv_norm_av =  ConvNormAV(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
            activae=get_activate(activate, alpha)
        )

    def forward(self, x):
        return  self.conv_norm_av(x)

class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN", activate="PReLU", alpha=0.1):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super(BasicStem, self).__init__()
        self.out_channels_num = out_channels
        self.conv_bn_av1 = ConvNormAV(
            in_channels,
            out_channels//2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels//2),
            activae=get_activate(activate, alpha)
        )
        self.conv_bn_av2 = ConvNormAV(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
            activae=get_activate(activate, alpha)
        )

    def forward(self, x):
        x = self.conv_bn_av1(x)
        x = self.conv_bn_av2(x)
        return x

    @property
    def out_channels(self):
        return self.out_channels_num

    @property
    def stride(self):
        return 2  # = stride 2 conv

def make_stage(block_class, num_blocks, **kwargs):
    """
    Create a darknet stage by creating many blocks.

    Args:
        block_class (class): BottleneckBlock
        num_blocks (int):
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """

    blocks = []
    kwargs["out_channels"] = kwargs["out_channels"] // 2
    for i in range(num_blocks):
        blocks.append(block_class( **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    blocks.append( BasicPool(
            kwargs["in_channels"],
            kwargs["out_channels"]*2,
            norm=kwargs["norm"],
            activate=kwargs["activate"],
            alpha=kwargs["alpha"]
        ))
    return blocks

class DarkNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(DarkNet, self).__init__()

        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                if isinstance(block, DarkNetBlockBase):
                    curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * 2
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

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
def build_darknet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.DARKNETS.NORM
    activate = cfg.MODEL.DARKNETS.ACTIVATE
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DARKNETS.STEM_OUT_CHANNELS,
        norm=norm,
        activate=activate,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    out_features        = ["res4", "res5", "res6"]#cfg.MODEL.RESNETS.OUT_FEATURES
    depth               =  cfg.MODEL.DARKNETS.DEPTH
    num_groups          =  cfg.MODEL.DARKNETS.NUM_GROUPS
    width_per_group     =  cfg.MODEL.DARKNETS.WIDTH_PER_GROUP
    bottleneck_channels =  num_groups * width_per_group
    in_channels         =  cfg.MODEL.DARKNETS.STEM_OUT_CHANNELS
    out_channels        =  cfg.MODEL.DARKNETS.RES2_OUT_CHANNELS
    num_classes         =  cfg.MODEL.DARKNETS.NUM_CLASSES

    num_blocks_per_stage = {53:[1, 2, 8, 8, 4],}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res4":4, "res5":5, "res6": 6}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = 1
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "activate": activate,
            "alpha": 0.1,
            "dilation": dilation,
        }
        stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        bottleneck_channels *= 2
        out_channels = out_channels * 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)

        if num_classes is not None:
            out_features = None

    return DarkNet(stem, stages, out_features=out_features, num_classes=num_classes)

if __name__ == '__main__':
    # x = torch.rand(32, 32, 128, 128)
    # block = BottleneckBlock(
    #     32,
    #     32,
    #     16
    # )
    # print(block)
    # out = block(x)
    # print(out.size())

    #DARK
    net = build_darknet_backbone(None, None)
    print(net)