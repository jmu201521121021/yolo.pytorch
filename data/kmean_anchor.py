import os
__all__ = ["KmeanAnchor"]

class KmeanAnchor:
    def __init__(self, net_width, net_height, k, out_dir="./anchor_out"):
        """
        Args:
            net_width(int): width of net input
            net_height(int): height of net input
            k(int): k of kmean
        """
        self.net_width = net_width
        self.net_height = net_height
        self.k = k
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def kmean_gen_anchor_size(self, boxes_size_list):
        """

        Args
            boxes_size_list(list): all boxes size of dataste, every data of list,
            eg.[x_min, y_min, x_max, y_max] , value in [0,1]
        """
        anchor_size_list = []
        # todo kmean

        self.save_anchor_size(anchor_size_list)
        pass

    def save_anchor_size(self, anchor_list):
        """
        save anchor size
        Args
            anchor_list(list): all anchor size, every data  of list,
            eg. [7, 10]->[w, h]
        """
        file_name = "anchor_size_k_{}.txt".format(self.k)
        with open(os.path.join(self.out_dir, file_name), 'w') as fp:
            for anchor_size in anchor_list:
                line = str(anchor_size[0]) + "," + str(anchor_size[1]) + "\n"
                fp.write(line)

if __name__ == "__main__":
    gen_anchor_size = KmeanAnchor(618, 618, 9)
    gt_boxes_size_list = []
    gen_anchor_size.kmean_gen_anchor_size(gt_boxes_size_list)