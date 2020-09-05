import torch
from torch import nn

from ret_benchmark.modeling.registry import HEADS
from ret_benchmark.utils.init_methods import weights_init_kaiming


@HEADS.register("linear_norm")
class LinearNorm(nn.Module):
    def __init__(self, cfg):
        super(LinearNorm, self).__init__()
        self.bn1 = nn.BatchNorm1d(cfg.MODEL.HEAD.IN_CHANNELS)
        self.fc1 = nn.Linear(cfg.MODEL.HEAD.IN_CHANNELS, cfg.MODEL.HEAD.IN_CHANNELS)
        self.fc1.apply(weights_init_kaiming)

        self.bn2 = nn.BatchNorm1d(cfg.MODEL.HEAD.IN_CHANNELS)
        self.fc2 = nn.Linear(cfg.MODEL.HEAD.IN_CHANNELS, cfg.MODEL.HEAD.DIM)
        self.fc2.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
