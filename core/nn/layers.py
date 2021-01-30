"""Network components."""

import torch.nn as nn
from torch.nn.utils import spectral_norm

from .utils import NoneLayer


def get_normalization_1d(norm_type, features):
    if norm_type == 'bn':
        return nn.BatchNorm1d(features)
    if norm_type == 'in':
        return nn.InstanceNorm1d(features, affine=True)
    if norm_type == 'ln':
        return nn.LayerNorm(normalized_shape=features)
    if norm_type == 'gn':
        return nn.GroupNorm(num_groups=features//10, num_channels=features)
    if norm_type == 'none':
        return NoneLayer()


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, normalization, use_spectral, bias=False):
        super().__init__()
        self.module = nn.Linear(in_dim, out_dim, bias=bias)
        if use_spectral:
            self.module = spectral_norm(self.module)
        self.block = nn.Sequential(
            get_normalization_1d(normalization, out_dim),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        x = self.module(x)
        x = self.block(x)
        return x
