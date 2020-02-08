# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from yolov3.layers import ShapeSpec
from yolov3.utils.registry import Registry

from yolov3.modeling.backbone.backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`yolov3.layers.ShapeSpec`, which contains the input shape specification.

It must returns an instance of :class:`Backbone`.
"""


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
