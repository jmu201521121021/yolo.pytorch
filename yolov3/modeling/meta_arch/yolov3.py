from typing import List
import torch
from torch import  nn
from .build import META_ARCH_REGISTRY
from yolov3.layers import ShapeSpec, ConvNormAV, get_activate, get_norm
from yolov3.modeling.anchor_generator import build_anchor_generator
from yolov3.modeling.backbone.build import build_backbone
from yolov3.structures import Boxes
from yolov3.structures import  pairwise_iou

__all__ = ["Yolov3", "Yolov3Head"]

@META_ARCH_REGISTRY.register()
class Yolov3(nn.Module):
    """
     Implement Yolov3 (https://arxiv.org/abs/1804.02767).
    """
    def __init__(self, cfg, input_shape:ShapeSpec):
        super(Yolov3, self).__init__()
        #init param
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.YOLOV3.IN_FEATURES

        #backbone
        self.backbone = build_backbone(cfg, input_shape)
        backbone_shape = self.backbone.output_shape() # p3 p4 p5
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        #head
        self.head = Yolov3Head(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        #TODO matcher

        self.to(self.device)

    def forward(self, batched_inputs):
        if self.training:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = torch.stack(images, 0)
            if 'boxes' in batched_inputs:
                gt_boxes = [x["boxes"].to(self.device) for x in batched_inputs]
                gt_classes = [x["label"].to(self.device) for x in batched_inputs]
        else:
            images = batched_inputs

        features = self.backbone(images)
        features = [features[f] for f in self.in_features]
        yolo_layer_outs= self.head(features)

        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_boxes, gt_classes)
            return self.losses(gt_classes, gt_anchors_reg_deltas, yolo_layer_outs)

        return  yolo_layer_outs

    def losses(self, gt_classes, gt_anchors_deltas, yolo_layer_out):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pass

    @torch.no_grad()
    def get_ground_truth(self, anchors, target_boxes,target_classes):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            target_boxes (list[Instances]): a list of N `Boxes`s. The i-th
                `Boxes` contains the ground-truth boxes
                for the i-th input image.  Specify `targets` during training only.
            target_classes (list[Instances]): a list of N `Boxes`s. The i-th
                `Boxes` contains the ground-truth label
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, gt_boxes_per_image, gt_classes_per_image in zip(anchors, target_boxes, target_classes):
            match_quality_matrix = pairwise_iou(gt_boxes_per_image, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(gt_boxes_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = gt_classes_per_image[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_boxes.tensor
                )

                gt_classes_i = gt_classes_per_image[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, box_cls, box_delta, anchors, image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        #TODO
        pass



class Yolov3Head(nn.Module):
    """
    The head used in yolov3 for object classification and box regression.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(Yolov3Head, self).__init__()
        norm = cfg.MODEL.DARKNETS.NORM
        activate = cfg.MODEL.DARKNETS.ACTIVATE
        alpha = cfg.MODEL.DARKNETS.ACTIVATE_ALPHA

        in_channels = [input_feature.channels for input_feature in input_shape]
        num_classes = cfg.MODEL.YOLOV3.NUM_CLASSES
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        yolo_layers = []

        for idx, in_channel in enumerate(in_channels):
            yolo_layer = nn.Sequential(
                ConvNormAV(in_channel,
                           in_channel*2,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           norm=get_norm(norm, in_channel*2),
                           activate=get_activate(activate, alpha),
                           bias=False),
                nn.Conv2d(in_channel*2, (num_classes + 4+1+1) * num_anchors[idx],1, 1, bias=True),
            )
            # init
            for layer in yolo_layer.modules():
                if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

            self.add_module("yolo_layer_{}".format(idx), yolo_layer)
            yolo_layers.append(yolo_layer)

        self.yolo_layers = yolo_layers

    def forward(self, features):
        outs = []
        assert (len(features) == len(self.yolo_layers))

        for feature, yolo_layer in zip(features, self.yolo_layers):
            out = yolo_layer(feature)
            outs.append(out)

        return  outs