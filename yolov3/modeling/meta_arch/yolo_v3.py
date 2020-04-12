from typing import List
import torch
from torch import  nn
import torch.nn.functional as F
from yolov3.modeling.meta_arch.build import META_ARCH_REGISTRY
from yolov3.layers import ShapeSpec, ConvNormAV, get_activate, get_norm
from yolov3.modeling.anchor_generator import build_anchor_generator
from yolov3.modeling.backbone.build import build_backbone
from yolov3.structures import Boxes, pairwise_iou
# from yolov3.modeling.matcher import MatcherYOLO
from yolov3.modeling.box_regression import Box2BoxTransform
from yolov3.modeling.losses import CELossNoSoftmax
__all__ = ["Yolov3", "Yolov3Head"]

def permute_to_N_HW_A_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N(,xHxW),A, K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, H*W, -1, K)  # Size=(N, HW, A,K)
    return tensor

class YOLOLayer(nn.Module):
    """
    yolo layer
    """
    def __init__(self, ignore_thresh, num_classes,  net_w, net_h, anchor_num_per_ft):
        super(YOLOLayer,self).__init__()
        self.ignore_thresh = ignore_thresh
        #self.matcher = MatcherYOLO(ignore_thresh)

        self.anchor_num_per_ft = anchor_num_per_ft
        self.num_classes = num_classes

        self.box2BoxTransform = Box2BoxTransform((1.0, 1.0, 1.0, 1.0))
        self.net_w = net_w
        self.net_h = net_h
        # self.reg_criterion = torch.nn.MSELoss()
        # self.cls_criterion = torch.nn.CrossEntropyLoss()
        # self.confidence_criterion = torch.nn.CrossEntropyLoss()

        self.criterion_reg = nn.MSELoss(reduction="none")
        self.criterion_cls = CELossNoSoftmax(reduction="none")
        self.criterion_confidence = nn.BCELoss(reduction="none")


    def setTarget(self, gt_boxes, gt_classes):
        self.gt_boxes = gt_boxes
        self.gt_classes = gt_classes

    def forward(self, yolo_layer_out, cell_anchors, anchors=None):
        pre_box_xy = []
        pre_box_wh = []
        pre_confidence = []
        pre_cls = []
        feature_map_w = []
        feature_map_h = []

        for feature_map in yolo_layer_out:
            feature_map_h.append(feature_map.shape[2])
            feature_map_w.append(feature_map.shape[3])

            feature_map = permute_to_N_HW_A_K(feature_map, (4 + 1 + self.num_classes))
            pre_box_xy.append(torch.sigmoid(feature_map[:, :, :, 0:2]))
            pre_box_wh.append(feature_map[:, :, :, 2:4])
            pre_confidence.append(torch.sigmoid(feature_map[:, :, :, 4:5]))
            pre_cls.append(torch.sigmoid(feature_map[:, :, :, 5:]))

        if self.training:
            loss = 0
            # N X M, N  number of ground truth, M number of cell anchors
            anchor_matched_idxs_ft = self.get_anchor_match_index(cell_anchors, self.gt_boxes)
            anchor_matched_idxs = torch.cat(anchor_matched_idxs_ft, 0)
            anchor_gt =[cell_anchors[0].tensor[index] for index in anchor_matched_idxs]
            anchor_gt = torch.stack(anchor_gt, 0)
            # cofidence delta


            # reg cls delta
            min_anchor_idx = 0
            max_anchor_idx = 0
            feature_index = 0

            delta_xy_batch = []
            delta_wh_batch = []
            delat_cls_batch = []
            gt_anchors_reg_delta_batch = []
            gt_anchor_cls_batch = torch.cat([copy.deepcopy(self.gt_classes.view(-1)) for _ in range(len(yolo_layer_out))], 0)
            gt_scale_batch = torch.cat([copy.deepcopy(self.get_gt_scale()) for _ in range(len(yolo_layer_out))], 0).view(-1, 1)
            gt_scale_batch = gt_scale_batch.repeat(1, 2)
            conf_batch = []
            delat_weight_batch = []
            conf_weight_batch = []


            for pre_box_xy_per_ft, pre_box_wh_per_ft, pre_cls_per_ft, pre_confidence_ft,\
                    ft_w, ft_h, anchor_per_ft in zip(pre_box_xy, pre_box_wh, pre_cls, pre_confidence, feature_map_w, feature_map_h, anchors):

                max_anchor_idx = self.anchor_num_per_ft[feature_index] + max_anchor_idx
                anchor_matched_ft_idxs = (anchor_matched_idxs >= min_anchor_idx) & \
                                          (anchor_matched_idxs < max_anchor_idx)

                delta_xy, delta_wh, delta_cls = self.get_box_cls_pre(pre_box_xy_per_ft, pre_box_wh_per_ft, pre_cls_per_ft, ft_w, ft_h,
                                     anchor_matched_idxs_ft, min_anchor_idx, max_anchor_idx)
                delta_xy_batch.append(delta_xy)
                delta_wh_batch.append(delta_wh)
                delat_cls_batch.append(delta_cls)

                delta_weight = torch.zeros(anchor_matched_ft_idxs.size())
                delta_weight[anchor_matched_ft_idxs] = 1

                delat_weight_batch.append(delta_weight)

                gt_anchors_reg_deltas = self.get_ground_truth(anchor_gt, self.gt_boxes, ft_w, ft_h)
                confidence_weight = self.get_confidence_weight(pre_box_xy_per_ft, pre_box_wh_per_ft, anchor_per_ft,
                                                                ft_w, ft_h)
                gt_anchors_reg_delta_batch.append(gt_anchors_reg_deltas)
                conf_weight_batch.append(confidence_weight)
                conf_batch.append(pre_confidence_ft.view(-1))

                min_anchor_idx = max_anchor_idx
                feature_index += 1

            delta_xy_batch = torch.cat(delta_xy_batch, 0)
            delta_wh_batch = torch.cat(delta_wh_batch, 0)
            delat_cls_batch = torch.cat(delat_cls_batch, 0)
            gt_anchors_reg_delta_batch = torch.cat(gt_anchors_reg_delta_batch, 0)

            delat_weight_batch = torch.cat(delat_weight_batch, 0)
            conf_weight_batch = torch.cat(conf_weight_batch, 0)
            conf_batch = torch.cat(conf_batch, 0)

            reg_loss = self.criterion_reg(delta_xy_batch * gt_scale_batch, gt_anchors_reg_delta_batch[:, :2] * gt_scale_batch) + \
                        self.criterion_reg(delta_wh_batch * gt_scale_batch, gt_anchors_reg_delta_batch[:, 2:] * gt_scale_batch)
            reg_loss = torch.sum(torch.sum(reg_loss * delat_weight_batch[:,None], dim=1) / 2.0) / torch.sum(delat_weight_batch)

            print("pre cls {} gt cls{}".format(delat_cls_batch.shape, gt_anchor_cls_batch.shape))
            cls_loss = torch.sum(self.criterion_cls(delat_cls_batch, gt_anchor_cls_batch) * delat_weight_batch) / torch.sum(delat_weight_batch)

            gt_conf_batch = torch.zeros(conf_batch.shape[0]).to(conf_batch)

            conf_loss = torch.sum(self.criterion_confidence(conf_batch, gt_conf_batch) * conf_weight_batch) / torch.sum(conf_weight_batch)

            loss = reg_loss + cls_loss + conf_loss

            return loss
    @torch.no_grad()
    def get_anchor_match_index(self, cell_anchors, target_boxes):
        anchor_matched_idxs = []
        for anchors_per_image, gt_boxes_per_image in zip(cell_anchors, target_boxes):
            match_quality_matrix = pairwise_iou(gt_boxes_per_image, anchors_per_image)
            matched_vals, matches = match_quality_matrix.max(dim=1)
            anchor_matched_idxs.append(matches)

        return anchor_matched_idxs

    @torch.no_grad()
    def get_ground_truth(self, cell_anchors, target_boxes, ft_w, ft_h):
        target_boxes = Boxes.cat(target_boxes).tensor
        gt_box_delta = self.box2BoxTransform.get_yolo_deltas(cell_anchors,target_boxes, self.net_w, self.net_h, ft_w, ft_h)
        return gt_box_delta

    @torch.no_grad()
    def get_confidence_weight(self, pre_box_xy_per_ft, pre_box_wh_per_ft, anchor_per_ft, ft_w, ft_h):
        pre_boxes = self.box2BoxTransform.apply_yolo_deltas(pre_box_xy_per_ft, pre_box_wh_per_ft, anchor_per_ft,
                                                            self.net_w, self.net_h, ft_w, ft_h)
        N, HW, A, _ = pre_boxes.shape
        # print("N:{},HW:{} K:{}".format(N,HW, A))
        weights = []
        for  pre_box_per_image, gt_boxes_per_image in zip(pre_boxes, self.gt_boxes):
            pre_box_per_image = Boxes(pre_box_per_image.view(-1, 4))
            match_quality_matrix = pairwise_iou(pre_box_per_image, gt_boxes_per_image)
            matched_vals, matches = match_quality_matrix.max(dim=1)
            indexes = (matched_vals < self.ignore_thresh)
            weight = torch.zeros_like(matched_vals)
            weight[indexes] = 1
            weights.append(weight)
            print(match_quality_matrix.shape)

        weights = torch.cat(weights, 0)
        return weights

    @torch.no_grad()
    def get_gt_scale(self):
        gt_boxes = Boxes.cat(self.gt_boxes)
        return 2 - gt_boxes.area()

    def get_box_cls_pre(self, pre_xy, pre_wh, pre_cls, ft_w, ft_h, anchor_idxes, min_anchor_idx, max_anchor_idx):
        """
        Args:
            pre_xy(Tensor.Float): x, y of pre boxes
            pre_xy(Tensor.Float): w, h of pre boxes
            pre_cls(Tensor.Float): cls of per boxes
            ft_w(int): width of feature map
            ft_h(int): height of feature map
            anchor_idx(list): anchor
            min_anchor_idx(int)
            max_anchor_idx(int)
        """
        delta_xy =[]
        delta_wh = []
        delta_cls = []

        for num_batch, gt_boxes_per_image in enumerate(self.gt_boxes):
            delta_xy_per_image = torch.zeros(gt_boxes_per_image.tensor.shape[0], 2)
            delta_wh_per_image = torch.zeros(gt_boxes_per_image.tensor.shape[0], 2)
            delta_cls_per_image =  torch.zeros(gt_boxes_per_image.tensor.shape[0], self.num_classes)

            anchor_idx = anchor_idxes[num_batch]
            ctr_xy =  (gt_boxes_per_image.tensor[:, :2] + gt_boxes_per_image.tensor[:, 2:]) / 2.0
            ctr_x = torch.floor(ctr_xy[:, 0] * ft_w).to(torch.int)
            ctr_y = torch.floor(ctr_xy[:, 1] * ft_h).to(torch.int)

            image_idx = 0
            for i, j , c in zip(ctr_x, ctr_y, anchor_idx):
                if c >= min_anchor_idx and c < max_anchor_idx:
                    c = c - min_anchor_idx
                    delta_xy_per_image[image_idx, :] = pre_xy[num_batch, i*ft_w+ j, c, :]
                    delta_wh_per_image[image_idx, :] = pre_wh[num_batch, i*ft_w+ j, c, :]
                    delta_cls_per_image[image_idx,:] = pre_cls[num_batch, i*ft_w+ j, c, :]
                image_idx += 1

            delta_xy.append(delta_xy_per_image)
            delta_wh.append(delta_wh_per_image)
            delta_cls.append(delta_cls_per_image)

        delta_xy = torch.stack(delta_xy, 0).view(-1, 2)
        delta_wh = torch.stack(delta_wh, 0).view(-1, 2)
        delta_cls = torch.stack(delta_cls, 0).view(-1,self.num_classes)

        return  delta_xy, delta_wh, delta_cls


    def get_pre_box(self):
        pass
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
        # TODO
        pass


# @META_ARCH_REGISTRY.register()
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

        self.num_classes =  cfg.MODEL.YOLOV3.NUM_CLASSES


        self.to(self.device)

    def forward(self, images):
        features = self.backbone(images)
        features = [features[f] for f in self.in_features]
        yolo_layer_outs= self.head(features)
        anchors = self.anchor_generator(features)


        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, self.gt_boxes, self.gt_classes)
            return self.losses(gt_classes, gt_anchors_reg_deltas, yolo_layer_outs)

        return  yolo_layer_outs




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


if __name__ == "__main__":
    import copy

    from yolov3.configs.default import get_default_config
    from yolov3.modeling.anchor_generator import YoloAnchorGenerator

    num_images = 2
    cfg = get_default_config()
    input_shape = [ShapeSpec(stride=32), ShapeSpec(stride=16), ShapeSpec(stride=8)]
    # anchor_generator = DefaultAnchorGenerator(cfg,input_shape)
    anchor_generator = YoloAnchorGenerator(cfg, input_shape)
    features = [torch.zeros(num_images, 75, 8, 8), torch.zeros(num_images, 75, 16, 16), torch.zeros(num_images, 75, 32, 32)]
    anchors, cell_anchors = anchor_generator.forward(features)
    print("anchor_num:{}".format(anchor_generator.num_cell_anchors))


    gt_boxes = Boxes(torch.Tensor([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.3, 0.3]]))
    gt_boxes = [gt_boxes, gt_boxes]
    gt_cls = torch.Tensor([[1, 2], [1,2]]).to(torch.long)

    yolo_layer = YOLOLayer(0.5, 20, 416, 416, anchor_generator.num_cell_anchors)
    yolo_layer.setTarget(gt_boxes, gt_cls)

    # feature_sizes = [8, 16, 32]

    # feature_maps = []
    # for feature_size in feature_sizes:
    #     feature_maps.append(torch.randn(num_images, 255, feature_size, feature_size))

    # cell_anchors = Boxes(torch.Tensor([[0, 0, 3, 3],
    #                                   [1, 1, 7, 7],
    #                                   [0, 0, 3, 3],
    #                                   [1, 1, 5, 5],
    #                                   [0, 0, 3, 3],
    #                                   [1, 1, 5, 5],
    #                                   [0, 0, 3, 3],
    #                                   [1, 1, 11, 11],
    #                                   [0, 0, 3, 3],
    #                                   ]))
    # cell_anchors = [copy.deepcopy(cell_anchors) for _ in range(num_images)]

    yolo_layer(features, cell_anchors, anchors)




