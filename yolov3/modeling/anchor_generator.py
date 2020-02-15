import  torch
from torch import  nn
from  typing import  List
from  yolov3.layers import  ShapeSpec
from yolov3.utils import  Registry

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
ANCHOR_GENERATOR_REGISTRY.__doc__ = ""

@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        pass

    def _calculate_anchors(self, sizes, aspect_ratios):
       pass
    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_cell_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = None
        pass
        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                height / width.

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        pass
        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        anchors = []
        return anchors

def build_anchor_generator(cfg, input_shape:List[ShapeSpec]):
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)
