import os
import numpy as np
import random
from data.dataset.voc_dataset import get_voc_annotations, BuildVocDataset
from yolov3.configs.default import get_default_config

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

    def kmean_gen_anchor_size(self, boxes_size_list, eps=0.00005, iterations=250):
        """

        Args
            boxes_size_list(list): all boxes size of dataset, every data of list,
            eg.[x_min, y_min, x_max, y_max] , value in [0,1]
        """

        # todo kmean
        N = boxes_size_list.shape[0]
        random_indices = [random.randrange(N) for i in range(self.k)]

        centroids = boxes_size_list[random_indices]
        num = centroids.shape[0]

        distance_sum_pre = -1
        assignments_pre = -1 * np.ones(N, dtype=np.int64)
        iteration = 0

        while True:
            iteration += 1
            distances = []

            for i in range(N):
                distance = 1 - self.IOU(boxes_size_list[i], centroids)
                distances.append(distance)
            distances_array = np.array(distances, np.float32)
            assignments = np.argmin(distances_array, axis=1)
            distances_sum = np.sum(distances_array)
            centroid_sums = np.zeros(centroids.shape, np.float32)
            for i in range(N):
                centroid_sums[assignments[i]] += boxes_size_list[i]
            for j in range(num):
                centroids[j] = centroid_sums[j] / (np.sum(assignments == j))
            diff = abs(distances_sum - distance_sum_pre)

            print("iteration: {},distance: {}, diff: {}, avg_IOU: {}".format(iteration, distances_sum, diff,
                                                                               np.sum(1 - distances_array) / (N * num)))

            if (assignments == assignments_pre).all():
                print("cluster result is no more change.\n")
                break
            if diff < eps:
                print("eps is arrived.\n")
                break
            if iteration > iterations:
                print("iteration is arrived.\n")
                break
            # record previous iter
            distance_sum_pre = distances_sum
            assignments_pre = assignments.copy()
        self.save_anchor_size(centroids)


    def save_anchor_size(self, anchor_list):
        """
        save anchor size
        Args
            anchor_list(list): all anchor size, every data  of list,
            eg. [7, 10]->[w, h]
        """
        file_name = "anchor_size_k_{}.txt".format(self.k)
        with open(os.path.join(self.out_dir, file_name), 'w') as fp:
            w = anchor_list[:, 0]
            index = np.argsort(w)
            for anchor_size in anchor_list[index]:
                line = str(anchor_size[0]) + "," + str(anchor_size[1]) + "\n"
                fp.write(line)

    def IOU(self, annotation_array, centroids):
        all_iou = []
        w, h = annotation_array
        for centroid in centroids:
            c_w, c_h = centroid
            if c_w >= w and c_h >= h:
                iou = w * h / (c_w * c_h)
            elif c_w >= w and c_h <= h:
                iou = w * c_h / (w * h + (c_w - w) * c_h)
            elif c_w <= w and c_h >= h:
                iou = c_w * h / (w * h + (c_h - h) * c_w)
            else:
                iou = (c_w * c_h) / (w * h)
            all_iou.append(iou)
        return np.array(all_iou, np.float32)


if __name__ == "__main__":
    gen_anchor_size = KmeanAnchor(416, 416, 9)
    gt_boxes_size_list = []
    cfg = get_default_config()
    cfg.DATASET.DATA_ROOT = "./dataset"
    for i, item in enumerate(get_voc_annotations(cfg)):
        for box in item["boxes"]:
            w = (box[2] - box[0])/item["width"]
            h = (box[3] - box[1])/item["height"]
            gt_boxes_size_list.append((w, h))
            # print(i, ' ', w, ' ', h)
        # print(item)
        # if i == 2:
        #     break
    # gt_boxes_size_list = [
    #     [0.1, 0.2],
    #     [0.5, 0.6],
    #     [0.11, 0.22],
    #     [0.51, 0.62],
    #     [0.13, 0.23],
    #     [0.53, 0.63],
    #     [0.14, 0.25],
    #     [0.53, 0.65],
    #     [0.16, 0.25],
    #     [0.56, 0.66],
    #     [0.17, 0.24],
    #     [0.57, 0.68],
    #     [0.14, 0.23],
    #     [0.18, 0.26],
    #     [0.54, 0.66],
    #     [0.16, 0.28],
    #     [0.16, 0.27],
    #     [0.57, 0.65]
    # ]
    gt_boxes_w_h = np.array(gt_boxes_size_list)
    gen_anchor_size.kmean_gen_anchor_size(gt_boxes_w_h)

