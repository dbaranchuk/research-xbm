# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch


class XBM:
    def __init__(self, cfg):
        self.K = cfg.XBM.SIZE
        self.feats = torch.zeros(self.K, 128).cuda()
        self.random_feats = torch.nn.functional.normalize(torch.randn(3*self.K, 128).cuda(), p=2, dim=1)
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda() - 1
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    @property
    def is_empty(self):
        return self.ptr == 0 and self.targets[0].item() == -1

    def get(self):
        if self.is_full:
            return self.feats, self.random_feats, self.targets
        else:
            return self.feats[:self.ptr], self.random_feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        if len(targets) > self.K:
            feats = feats[:self.K]
            targets = targets[:self.K]

        q_size = len(targets)
        if self.ptr + q_size > self.K:
            remainder = self.K - self.ptr
            self.feats[self.ptr:] = feats[:remainder]
            self.targets[self.ptr:] = targets[:remainder]

            self.feats[: q_size - remainder] = feats[remainder:]
            self.targets[: q_size - remainder] = targets[remainder:]
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets

        self.ptr += q_size
        self.ptr = self.ptr % self.K
