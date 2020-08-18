# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# -----------------------------------------------------------------------------
# Config definition of imagenet pretrained model path
# -----------------------------------------------------------------------------


from yacs.config import CfgNode as CN

MODEL_PATH = dict()
MODEL_PATH = {
    "bninception": "./backbones/bn_inception-52deb4733.pth",
    "resnet50": "./backbones/pytorch_resnet50.pth",
    "googlenet": "./backbones/googlenet-1378be20.pth",
}

MODEL_PATH = CN(MODEL_PATH)
