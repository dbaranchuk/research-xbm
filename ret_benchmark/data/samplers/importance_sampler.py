# encoding: utf-8

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import copy
import random
from collections import defaultdict

import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import Sampler


class Scorer(nn.Module):
    def __init__(self, model):
        super(Scorer, self).__init__()
        self.model = copy.deepcopy(model)
        self.model.training = False

    @torch.no_grad()
    def forward(self, x):
        x = self.model.backbone(x)
        x = self.model.head.fc(x)
        return x.norm(2, dim=-1)


class ImportanceSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (BaseDataSet).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances, max_iters):
        self.label_index_dict = dataset.label_index_dict
        self.batch_size = batch_size
        self.K = num_instances
        self.num_labels_per_batch = self.batch_size // self.K
        self.max_iters = max_iters
        self.labels = list(self.label_index_dict.keys())

        self.scorer = None
        self.dataset = dataset

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"|Sampler| iters {self.max_iters}| K {self.K}| M {self.batch_size}|"

    def update_scorer(self, model):
        self.scorer = Scorer(model)

    def _prepare_batch(self):
        batch_idxs_dict = defaultdict(list)
        count = list()
        t0 = time.time()

        for label in self.labels:
            idxs = copy.deepcopy(self.label_index_dict[label])
            if len(idxs) % self.K != 0:
                idxs.extend(
                    np.random.choice(
                        idxs, size=self.K - len(idxs) % self.K, replace=True
                    )
                )
            if self.scorer is not None and len(idxs) > self.K:
                with torch.no_grad():
                    x = torch.stack([self.dataset[idx][0] for idx in idxs])
                    scores = self.scorer(x).cpu()
                U = torch.rand_like(scores)
                gumbel_noise = torch.log(-torch.log(U + 1e-20) + 1e-20)
                sampled_idxs = torch.topk(scores + gumbel_noise, self.K, largest=True)[1].tolist()
                batch_idxs_dict[label] = [sampled_idxs]
            else:
                random.shuffle(idxs)
                batch_idxs_dict[label] = [
                    idxs[i * self.K : (i + 1) * self.K] for i in range(len(idxs) // self.K)
                ]

            count.append(len(batch_idxs_dict[label]))
        count = np.array(count)
        avai_labels = copy.deepcopy(self.labels)
        if self.scorer is not None:
            print(f"_prepare_data with IS {time.time() - t0}s")
        return batch_idxs_dict, avai_labels, count

    def __iter__(self):
        batch_idxs_dict, avai_labels, count = self._prepare_batch()
        for _ in range(self.max_iters):
            batch = []
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels, count = self._prepare_batch()

            selected_labels = np.random.choice(
                avai_labels, self.num_labels_per_batch, False, count / count.sum()
            )
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                batch.extend(batch_idxs)
                label_idx = avai_labels.index(label)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.pop(label_idx)
                    count = np.delete(count, label_idx)
                else:
                    count[label_idx] = len(batch_idxs_dict[label])
            yield batch