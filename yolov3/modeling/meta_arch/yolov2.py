import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .build import META_ARCH_REGISTRY
from yolov3.layers import (
    ShapeSpec,
    ConvNormAV,
    FrozenBatchNorm2d,
    get_norm,
    get_activate
)
from yolov3.modeling.anchor_generator import build_anchor_generator
from yolov3.modeling.backbone.build import build_backbone
from yolov3.modeling.backbone.darknet19 import _make_layers
from yolov3.structures import Boxes
from yolov3.structures import pairwise_iou


__all__ = ['Yolov2', 'Yolov2Head']


# https://blog.csdn.net/qq_17550379/article/details/78948839
def _reorg_layer(x, stride):
    b, c, h, w = x.size()
    x = x.view(b, c, h / stride, stride, w / stride,
               stride).transpose(3, 4).contiguous()
    x = x.view(b, c, h / stride * w / stride,
               stride * stride).transpose(2, 3).contiguous()
    x = x.view(b, c, stride * stride, h / stride,
               w / stride).transpose(1, 2).contiguous()
    x = x.view(b, stride * stride * c, h /
               stride, w / stride).contiguous()
    return x


def yolo_to_bbox(bbox_pred, anchors, H, W):
    bsize = bbox_pred.shape[0]
    num_anchors = anchors.shape[0]
    bbox_out = np.zeros((bsize, H * W, num_anchors, 4))

    for b in range(bsize):
        for row in range(H):
            for col in range(W):
                ind = row * W + col
                for a in range(self.num_anchors):
                    cx = (bbox_pred_np[b, ind, a, 0] + col) / W
                    cy = (bbox_pred_np[b, ind, a, 1] + row) / H
                    bw = bbox_pred[b, ind, a, 2] * anchors[a][0] / W * 0.5
                    bh = bbox_pred[b, ind, a, 3] * anchors[a][1] / H * 0.5

                    bbox_out[b, ind, a, 0] = cx - bw
                    bbox_out[b, ind, a, 1] = cy - bh
                    bbox_out[b, ind, a, 2] = cx + bw
                    bbox_out[b, ind, a, 3] = cy + bh
    return bbox_out


def bbox_ious(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    intersec = np.zeros((N, K))

    for k in range(K):
        qbox_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[n, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[n, 1]) + 1
                )
                if ih > 0:
                    box_area = (
                        (boxes[k, 2] - boxes[k, 0] + 1) *
                        (boxes[k, 3] - boxes[k, 1] + 1)
                    )
                    inter_area = iw * ih
                    intersec[n, k] = inter_area / \
                        (qbox_area + box_area - inter_area)
    return intersec


@META_ARCH_REGISTRY.register()
class Yolov2(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(Yolov2, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.YOLOV2.IN_FEATURES

        self.backbone = build_backbone(cfg, input_shape)

        backbone_shape = self.backbone.output_shape()

        self.num_classes = cfg.MODEL.YOLOV2.NUM_CLASSES
        self.num_anchors = build_anchor_generator(
            cfg, input_shape).num_cell_anchors

        # head
        self.head = Yolov2Head(cfg, backbone_shape)

        self.to(self.device)

    def forward(self, batched_inputs):
        if self.training:
            images = [x['image'].to(self.device) for x in batched_inputs]
            images = torch.stack(images, 0)
            if 'boxes' in batched_inputs:
                gt_boxes = [x['boxes'].to(self.device) for x in batched_inputs]
                gt_classes = [x['label'].to(self.device)
                              for x in batched_inputs]
        else:
            images = batched_inputs
        features = self.backbone(images)
        bbox_pred, iou_pred, score_pred, prob_pred = self.head(features)

        if self.training:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()
            bsize, hw, _, _ = bbox_pred_np.shape

            _classes = np.zeros(
                (bsize, hw, self.num_anchors, self.num_classes))
            _classes_mask = np.zeros((bsize, hw, self.num_anchors, 1))

            _ious = np.zeros((bsize, hw, self.num_anchors, 1))
            _ious_mask = np.zeros((bsize, hw, self.num_anchors, 1))

            _boxes = np.zeros((bsize, hw, self.num_anchors, 4))
            _boxes[:, :, 0:2] = 0.5
            _boxes[:, :, 2:4] = 1.0
            _box_mask = np.zeros((bsize, hw, self.num_anchors, 1)) + 0.01

            anchors = np.asarray(
                [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)

            W = images.size(2) / 32
            H = images.size(3) / 32

            bbox_np = yolo_to_bbox(
                np.ascontiguousarray(bbox_pred_np, dtype=np.float),
                anchors,
                H, W
            )

            bbox_np[:, :, :, 0::2] *= W * 32
            bbox_np[:, :, :, 1::2] *= H * 32

            gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)
            bbox_np_b = np.reshape(bbox_np, (-1, 4))

            ious = bbox_ious(
                np.ascontiguousarray(bbox_np_b, dtype=np.float),
                np.ascontiguousarray(gt_boxes_b, dtype=np.float)
            )
            best_ious = np.max(ious, axis=1).reshape(_ious_mask.shape)
            


class Yolov2Head(nn.Module):
    def __init__(self, cfg, input_shape):
        super(Yolov2Head, self).__init__()
        net_cfgs = [
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]
        c1 = input_shape['conv1s'].channels
        c2 = input_shape['conv2'].channels
        self.conv3, c3 = _make_layers(c2, net_cfgs[0])
        self.stride = 2
        self.conv4, c4 = _make_layers(
            c1 * self.stride * self.stride + c3, net_cfgs[1])

        self.num_classes = cfg.MODEL.YOLOV2.NUM_CLASSES
        self.num_anchors = build_anchor_generator(
            cfg, input_shape).num_cell_anchors
        out_channels = num_anchors * (num_classes + 5)
        self.conv5 = ConvNormAV(c4, out_channels, activate=None)
        self.global_average_pool = nn.AvgPool2d((1, 1))

    def forward(self, x):
        conv1s = x['conv1s']
        conv2 = x['conv2']
        conv3 = self.conv3(conv2)
        conv1s_reorg = _reorg_layer(conv1s, self.stride)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        conv4 = self.conv4(cat_1_3)
        conv5 = self.conv5(conv4)
        global_average_pool = self.global_average_pool(conv5)
        bsize, _, h, w = global_average_pool.size()
        global_average_pool_reshaped = global_average_pool.permute(
            0, 2, 3, 1).contiguous().view(bsize, -1, self.num_anchors, self.num_classes + 5)
        xy_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])

        score_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(
            score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)
        return bbox_pred, iou_pred, score_pred, prob_pred
