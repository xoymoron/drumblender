"""Condition the existing filtered-noise path on contextual modal features.

Reference residual features are cached before editing. Re-encoding source minus
edited modal would encourage cancellation of the requested edit and is forbidden
by the reference API. Conditioning learns correlations, not counterfactual truth.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from drumblender.upgrades.encoders.dac_style import DACStyleEncoder


class ModalConditionedNoise(nn.Module):
    def __init__(self, modal_channels: int = 64, noise_channels: int = 64,
                 bands: int = 128):
        super().__init__()
        self.encoder = DACStyleEncoder(
            input_channels=1, hidden_channels=16, output_channels=noise_channels,
            strides=(2, 2, 4, 8), transpose_output=True,
        )
        self.condition = nn.Sequential(
            nn.Linear(modal_channels, noise_channels), nn.GELU(),
            nn.Linear(noise_channels, noise_channels * 2),
        )
        self.output = nn.Sequential(
            nn.Linear(noise_channels, noise_channels), nn.GELU(),
            nn.Linear(noise_channels, bands),
        )
        nn.init.constant_(self.output[-1].bias, -5.0)

    def encode_reference(self, residual):
        # The strided waveform encoder needs a minimum input extent. Padding is
        # confined to feature extraction; targets and spectral losses stay valid-length.
        if residual.shape[-1] < 256:
            residual = F.pad(residual, (0, 256 - residual.shape[-1]))
        return self.encoder(residual)

    def decode(self, source_features, modal_hidden, modal_mask):
        # Pool only to provide noise context; the modal branch retains its modes.
        context = (modal_hidden * modal_mask[..., None]).sum(1)
        context = context / modal_mask.sum(1).clamp_min(1)[..., None]
        context = F.interpolate(context.transpose(1, 2),
                                size=source_features.shape[1], mode="linear",
                                align_corners=False).transpose(1, 2)
        gain, bias = self.condition(context).chunk(2, dim=-1)
        conditioned = source_features * (1 + 0.5 * gain.tanh()) + 0.5 * bias.tanh()
        # Positive filter amplitudes prevent sign ambiguities in the noise path.
        return F.softplus(self.output(conditioned))

    @staticmethod
    def link(reference_bands, edited_bands, amount: float):
        if not 0 <= amount <= 1:
            raise ValueError("noise_link must be in [0,1].")
        if amount == 0:
            return reference_bands
        delta = (edited_bands.clamp_min(1e-8).log()
                 - reference_bands.clamp_min(1e-8).log()).clamp(-1.4, 1.4)
        return reference_bands * (amount * delta).exp()
