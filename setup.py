#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from os import path
from setuptools import find_packages, setup
import torch


torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "yolov3", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version

setup(
    name="yolov3",
    version=get_version(),
    author="jmucv",
    url="https://github.com/jmu2015211211021/yolo.pytorch",
    description="yolov3 base pytorch "
    "platform for object detection .",
    packages=find_packages(exclude=("yolov3", "darknet", "data")),
    python_requires=">=3.6",
    install_requires=[
        "termcolor>=1.1",
        "Pillow==6.2.2",  # torchvision currently does not work with Pillow 7
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore",
    ],
)
