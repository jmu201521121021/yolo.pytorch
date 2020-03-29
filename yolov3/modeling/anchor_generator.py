import  torch
import  copy
from torch import  nn
from  typing import  List
from  yolov3.layers import  ShapeSpec
from yolov3.utils.registry import  Registry
from yolov3.structures import  Boxes

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
ANCHOR_GENERATOR_REGISTRY.__doc__ = ""

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(size, stride, offset, device):
    """
    transfor anchor location to image location
    """
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y

@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        self.sizes = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        self.strides = [x.stride for x in input_shape]
        self.offset = cfg.MODEL.ANCHOR_GENERATOR.OFFSET

        self.num_features = len(self.strides)

        self.cell_anchors = self._calculate_anchors(self.sizes)

    def _calculate_anchors(self, sizes=(((10,13), (16,30), (33,23)),((30,61),  (62,45),  (59,119)),((116,90),  (156,198),  (373,326)))):
        # If one size  is specified and there are multiple feature
        # maps, then we "broadcast" anchors of that single size
        # over all feature maps.
        assert len(sizes) == self.num_features
        cell_anchors = [self.generate_cell_anchors(size) for size in sizes]
        return BufferList(cell_anchors)

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

        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        """
         Args:
            grid_sizes(list[int]):  Each int is the size of feature map

        Returns:
            list[Tensor]: Each Tensor is the anchor of feature map
        """

        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def generate_cell_anchors(self, sizes=((10,13), (16,30), (33,23))):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.

        Returns:
            Tensor of shape (len(sizes) ), 4) storing anchor boxes
                in XYXY format.
        """
        anchors = []
        for anchor_w_h in sizes:
            w = anchor_w_h[0]
            h = anchor_w_h[1]
            x0, y0, x1, y1 = -w/2, -h/2, w/2, h/2
            anchors.append([x0, y0, x1, y1])

        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        num_images = len(features[0]) # batch size
        grid_sizes = [ feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes = Boxes(anchors_per_feature_map)
            anchors_in_image.append(boxes)

        anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        return anchors

class YoloAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        self.sizes = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        self.offset = cfg.MODEL.ANCHOR_GENERATOR.OFFSET
        self.strides = [x.stride for x in input_shape]
        self.num_features = len(self.strides)

        self.cell_anchors = self._calculate_anchors(self.sizes)

    def _calculate_anchors(self, sizes=(((10,13), (16,30), (33,23)),((30,61),  (62,45),  (59,119)),((116,90),  (156,198),  (373,326)))):
        # If one size  is specified and there are multiple feature
        # maps, then we "broadcast" anchors of that single size
        # over all feature maps.
        assert len(sizes) == self.num_features
        cell_anchors = [self.generate_cell_anchors(size) for size in sizes]
        return BufferList(cell_anchors)

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

        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        """
         Args:
            grid_sizes(list[int]):  Each int is the size of feature map

        Returns:
            list[Tensor]: Each Tensor is the anchor of feature map, x_min, x_max, w, h
        """

        anchors = []
        for size, base_anchors in zip(grid_sizes, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, 1, self.offset, base_anchors.device)
            shift_w_h = shift_x.new_full(
                shift_x.size(), 0, dtype=torch.float
            )
            shifts = torch.stack((shift_x, shift_y, shift_w_h, shift_w_h), dim=1)
            shifts = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
            anchors.append(shifts)
        return anchors

    def generate_cell_anchors(self, sizes=((10,13), (16,30), (33,23))):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.

        Returns:
            Tensor of shape (len(sizes) ), 4) storing anchor boxes
                in XYXY format.
        """
        anchors = []
        for anchor_w_h in sizes:
            w = anchor_w_h[0]
            h = anchor_w_h[1]
            anchors.append([0, 0, w, h])

        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        num_images = len(features[0]) # batch size
        grid_sizes = [ feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors= []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes =  [copy.deepcopy(anchors_per_feature_map) for _ in range(num_images)]
            boxes = torch.stack(boxes, 0)
            anchors.append(boxes)

        cell_anchors = [Boxes(cell_anchors) for cell_anchors in self.cell_anchors]
        cell_anchors = Boxes.cat(cell_anchors)

        #anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        cell_anchors = [copy.deepcopy(cell_anchors) for _ in range(num_images)]

        return anchors, cell_anchors

def build_anchor_generator(cfg, input_shape:List[ShapeSpec]):
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)

## DEBUG
if __name__ == '__main__':
    from yolov3.configs.default import get_default_config
    cfg = get_default_config()
    input_shape = [ShapeSpec(stride=32), ShapeSpec(stride=16), ShapeSpec(stride=8)]
    # anchor_generator = DefaultAnchorGenerator(cfg,input_shape)
    anchor_generator = YoloAnchorGenerator(cfg, input_shape)
    print(len(anchor_generator.cell_anchors))
    features = [torch.zeros(2, 78,8, 8), torch.zeros(2, 78, 16, 16), torch.zeros(2, 78, 32, 32)]
    anchors, cell_anchors = anchor_generator.forward(features)
    print("anchor_num:".format(anchor_generator.num_cell_anchors))
    for anchor in anchors:
        print(len(anchor))
    #print(anchors)
