import os
import cv2
import torchvision.datasets.mnist as mnist

class Mnist(object):
    def __init__(self, root="/home/lin/mnist/"):
        """
        :param root: mnist files root
        :return:
        """
        self.root = root
        self.train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )

        self.test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )

    def convert_to_img(self, train=True):
        """
        :param train: True => form train images and train .txt files
                      False => form test(val) images and text(val) .txt files
         :return:
        """
        if (train):
            data_path = self.root + 'train' + os.sep
            if (not os.path.exists(data_path)):
                os.makedirs(data_path)
            f = open(data_path + 'train.txt', 'w')
            """make ten files"""
            self.makeFiles(data_path)
            for i, (img, label) in enumerate(zip(self.train_set[0], self.train_set[1])):
                img_path = data_path + str(label)[7] + os.sep + str(i) + '.jpg'
                new_image = cv2.cvtColor(img.numpy(), cv2.COLOR_GRAY2BGR)
                cv2.imwrite(img_path, new_image)
                f.write(str(label)[7] + os.sep + str(i) + '.jpg' + ' ' + str(label)[7] + '\n')
            f.close()
        else:
            data_path = self.root + 'val' + os.sep
            if (not os.path.exists(data_path)):
                os.makedirs(data_path)
            f = open(data_path + 'val.txt', 'w')
            """make ten files"""
            self.makeFiles(data_path)
            for i, (img, label) in enumerate(zip(self.test_set[0], self.test_set[1])):
                img_path = data_path + str(label)[7] + os.sep + str(i) + '.jpg'
                new_image = cv2.cvtColor(img.numpy(), cv2.COLOR_GRAY2BGR)
                cv2.imwrite(img_path, new_image)
                f.write(str(label)[7] + os.sep + str(i) + '.jpg' + ' ' + str(label)[7] + '\n')
            f.close()

    def makeFiles(self, data_path):
        """
        :param data_path: File's paths
        :return:
        """
        for i in range(10):
            file_name = data_path + str(i)
            if not os.path.exists(file_name):
                os.mkdir(file_name)

