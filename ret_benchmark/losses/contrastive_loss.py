from __future__ import absolute_import

import torch
from torch import nn
import numpy as np
from ret_benchmark.losses.registry import LOSS
from ret_benchmark.utils.log_info import log_info


@LOSS.register("contrastive_loss")
class ContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveLoss, self).__init__()
        self.margin = 0.5
        self.total_pos_freqs = []
        self.total_neg_freqs = []
        self.neg_topk = cfg.XBM.NEG_TOPK

    def forward(self, inputs_col, targets_col, inputs_row, target_row, neg_inputs_row=None):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        if neg_inputs_row is not None:
            neg_sim_mat = torch.matmul(inputs_col, neg_inputs_row.t())
        epsilon = 1e-5
        loss = list()

        neg_count, pos_count = list(), list()
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            if neg_inputs_row is not None:
                neg_pair_ = neg_sim_mat[i]
            else:
                neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

            if inputs_col.shape[0] != inputs_row.shape[0] and self.neg_topk > 0:
                top_idxs = torch.topk(neg_pair_, min(self.neg_topk, len(neg_pair_)), largest=True)[1]
                neg_pair = neg_pair_[top_idxs]
            else:
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            if len(pos_pair_) > 0:
                if inputs_col.shape[0] != inputs_row.shape[0]:
                    pos_loss = 6 * torch.sum(-pos_pair_ + 1) / len(pos_pair_)  # TODO: name 6
                else:
                    pos_loss = torch.sum(-pos_pair_ + 1)
                pos_count.append(len(pos_pair_))
            else:
                pos_loss = 0

            if len(neg_pair) > 0:
                if inputs_col.shape[0] != inputs_row.shape[0]:
                    neg_loss = 15 * torch.sum(neg_pair) / len(neg_pair)  # TODO: name 13
                else:
                    neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = 0

            loss.append(pos_loss + neg_loss)

        if inputs_col.shape[0] == inputs_row.shape[0]:
            prefix = "batch_"
        else:
            log_info['memory_pos_loss'] = pos_loss.item() if pos_loss != 0 else 0
            log_info['memory_neg_loss'] = neg_loss.item() if neg_loss != 0 else 0
            prefix = "memory_"

        if len(pos_count) != 0:
            log_info[f"{prefix}average_pos"] = sum(pos_count) / len(pos_count)
        else:
            log_info[f"{prefix}average_pos"] = 0

        if len(neg_count) != 0:
            log_info[f"{prefix}average_neg"] = sum(neg_count) / len(neg_count)
        else:
            log_info[f"{prefix}average_neg"] = 0

        log_info[f"{prefix}non_zero"] = len(neg_count)
        if inputs_col.shape[0] == inputs_row.shape[0]:
            print(loss)
        loss = sum(loss) / n  # / all_targets.shape[1]
        return loss
