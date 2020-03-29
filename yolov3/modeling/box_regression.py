# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Tuple
import torch

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


__all__ = ["Box2BoxTransform"]


@torch.jit.script
class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
        return pred_boxes


    def get_yolo_deltas(self, cell_anchors, target_boxes, net_w, net_h, ft_w, ft_h):
        """
        Get box regression transformation deltas (tx, ty, tw, th) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(cell_anchors, torch.Tensor), type(cell_anchors)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        target_widths = (target_boxes[:, 2] - target_boxes[:, 0]) * net_w
        target_heights = (target_boxes[:, 3] - target_boxes[:, 1]) * net_h
        target_ctr_x = (target_boxes[:, 0] + 0.5 * target_widths) * ft_w
        target_ctr_y = (target_boxes[:, 1] + 0.5 * target_heights) * ft_h


        tx = target_ctr_x - torch.floor(target_ctr_x)
        ty = target_ctr_y - torch.floor(target_ctr_y)

        tw = torch.log(target_widths / cell_anchors[0])
        th = torch.log(target_heights / cell_anchors[1])

        deltas = torch.stack((tx, ty, tw, th), dim=1)
        return deltas

    def apply_yolo_deltas(self, deltas_xy, deltas_wh, boxes, net_w, net_h, ft_w, ft_h):
        """
        Apply transformation `deltas` (tx, ty, tw, th) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N,k, 4), where k >= 1.
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas_xy.dtype)

        widths = boxes[:, :, :, 2]
        heights = boxes[:, :, :, 3]
        ctr_x = boxes[:, :, :,  0]
        ctr_y = boxes[:, :, :, 1]

        wx, wy, ww, wh = self.weights
        tx = deltas_xy[:, :, :, 0] / wx
        ty = deltas_xy[:, :, :, 1] / wy
        tw = deltas_wh[:, :, :, 0] / ww
        th = deltas_wh[:, :, :, 1] / wh

        # Prevent sending too large values into torch.exp()
        tw = torch.clamp(tw, max=self.scale_clamp)
        th = torch.clamp(th, max=self.scale_clamp)

        pred_ctr_x = (tx + ctr_x) / ft_w
        pred_ctr_y = (ty + ctr_y) / ft_h
        pred_w = torch.exp(tw) * widths / net_w
        pred_h = torch.exp(th) * heights / net_h

        N, HW, A, _ = deltas_xy.shape
        pred_boxes = torch.zeros(N, HW, A, 4)
        pred_boxes[:, :, :, 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, :, :, 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, :, :, 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, :, :, 3] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
