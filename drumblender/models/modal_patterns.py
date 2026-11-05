"""Temporal sparse dictionary for contextual modal features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalModalSAE(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        features: int = 256,
        top_k: int = 8,
        kernel_size: int = 9,
    ):
        super().__init__()
        if not 1 <= top_k <= features or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                "Require 1 <= top_k <= features and a positive odd kernel."
            )
        self.features, self.top_k = features, top_k
        self.encoder = nn.Conv1d(
            channels, features, kernel_size, padding=kernel_size // 2
        )
        self.dictionary = nn.Parameter(
            torch.randn(features, channels, kernel_size) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(channels))

    def encode(self, hidden, mask):
        b, k, t, c = hidden.shape
        h = hidden.masked_fill(~mask[..., None], 0)
        pre = self.encoder(h.reshape(b * k, t, c).transpose(1, 2)).relu()
        values, indices = pre.topk(self.top_k, dim=1)
        sparse = torch.zeros_like(pre).scatter(1, indices, values)
        return sparse.transpose(1, 2).reshape(b, k, t, -1) * mask[..., None]

    def decode(self, sparse):
        b, k, t, features = sparse.shape
        weights = (
            self.dictionary
            / self.dictionary.flatten(1).norm(dim=1).clamp_min(1e-6)[:, None, None]
        )
        h = F.conv_transpose1d(
            sparse.reshape(b * k, t, features).transpose(1, 2),
            weights,
            self.bias,
            padding=self.dictionary.shape[-1] // 2,
        )
        return h.transpose(1, 2).reshape(b, k, t, -1)

    def forward(self, hidden, mask, rho=None):
        sparse = self.encode(hidden, mask)
        reconstructed = self.decode(sparse).masked_fill(~mask[..., None], 0)
        weights = mask if rho is None else rho * mask
        denominator = (weights.sum() * hidden.shape[-1]).clamp_min(1e-12)
        loss = (
            (reconstructed - hidden.detach()).square() * weights[..., None]
        ).sum() / denominator
        return loss, sparse

    def edit(self, hidden, mask, feature: int, amount: float, *, sparse=None):
        if not 0 <= feature < self.features or not -1 <= amount <= 4:
            raise ValueError("Feature index out of range or amount outside [-1,4].")
        if amount == 0:
            return hidden
        if sparse is None:
            sparse = self.encode(hidden, mask)
        b, k, t, c = hidden.shape
        atom = self.dictionary[feature : feature + 1]
        atom = atom / atom.norm().clamp_min(1e-6)
        delta = (
            F.conv_transpose1d(
                (sparse[..., feature] * amount).reshape(b * k, 1, t),
                atom,
                padding=atom.shape[-1] // 2,
            )
            .transpose(1, 2)
            .reshape(b, k, t, c)
        )
        return (hidden + delta).masked_fill(~mask[..., None], 0)
