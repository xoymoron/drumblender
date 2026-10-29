"""Joint modal trajectories with typed inputs and permutation-equivariant mixing.

Frequencies use radians/sample, matching the feature cache and ModalSynth.
This module operates on a uniform, unpadded frame grid. The training task slices
each variable-length sample before calling it; silent observations remain masked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    """Local temporal processing without normalization across modes or time."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv1d(channels * 2, channels, 1),
        )

    def forward(self, x, mask):
        b, k, t, c = x.shape
        y = self.norm(x).masked_fill(~mask[..., None], 0)
        y = y.reshape(b * k, t, c).transpose(1, 2)
        y = self.pointwise(self.depthwise(y))
        y = y.transpose(1, 2).reshape(b, k, t, c)
        return (x + y).masked_fill(~mask[..., None], 0)


class ModeInteraction(nn.Module):
    """Inducing attention: preserve mode tokens while exchanging set context."""

    def __init__(self, channels: int, context_tokens: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(context_tokens, channels) * 0.02)
        self.read = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.write = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, mask):
        b, k, t, c = x.shape
        sequence = self.norm(x).permute(0, 2, 1, 3).reshape(b * t, k, c)
        valid = mask.permute(0, 2, 1).reshape(b * t, k)
        sequence = sequence.masked_fill(~valid[..., None], 0)
        # MultiheadAttention cannot softmax an entirely masked key set. A zero
        # dummy key handles silence; masking the output removes its contribution.
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~safe_valid.any(dim=1)
        queries = self.queries[None].expand(b * t, -1, -1)
        context, _ = self.read(
            queries, sequence, sequence,
            key_padding_mask=~safe_valid, need_weights=False,
        )
        delta, _ = self.write(sequence, context, context, need_weights=False)
        delta = delta.reshape(b, t, k, c).permute(0, 2, 1, 3)
        return (x + delta).masked_fill(~mask[..., None], 0)


@dataclass
class ModalEncoding:
    hidden: torch.Tensor                 # [B,K,Tc,C]
    mask: torch.Tensor                   # [B,K,Tc], observation support
    reference: torch.Tensor              # [B,3,K,T]
    observed: torch.Tensor               # [B,K,T]
    log_center: torch.Tensor             # [B,K,1], angular frequency
    gain: torch.Tensor                   # [B,1,1]


class NeuralModalAutoencoder(nn.Module):
    """Separate input stems, joint patterns, and physically typed output heads."""

    def __init__(self, channels: int = 64, context_tokens: int = 8):
        super().__init__()
        if channels < 16 or channels % 8 or context_tokens < 1:
            raise ValueError("channels must be >=16 and divisible by 8; context_tokens >0")
        self.channels = channels
        attr_width = channels * 3 // 8

        def stem():
            return nn.Sequential(
                nn.Conv1d(3, attr_width, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(attr_width, attr_width, 5, stride=2, padding=2),
            )

        self.frequency_stem = stem()
        self.amplitude_stem = stem()
        self.phase_stem = nn.Sequential(nn.Linear(2, channels // 4), nn.GELU())
        self.input_projection = nn.Linear(channels, channels)
        self.temporal = nn.ModuleList([TemporalBlock(channels, d) for d in (1, 2)])
        self.interactions = nn.ModuleList([
            ModeInteraction(channels, context_tokens) for _ in range(2)
        ])
        self.decoder_temporal = TemporalBlock(channels, 2)
        self.decoder_interaction = ModeInteraction(channels, context_tokens)
        self.output = nn.Linear(channels, 2)
        self.phase_output = nn.Linear(channels, 2)
        nn.init.constant_(self.output.bias, -3.0)
        with torch.no_grad():
            self.output.bias[0] = 0.0
            self.phase_output.bias.copy_(torch.tensor([1.0, 0.0]))

    @staticmethod
    def _visibility(observed, probability):
        if probability == 0:
            return observed
        b, k, t = observed.shape
        keep = torch.rand(b * k, 1, max(1, math.ceil(t / 8)),
                          device=observed.device) >= probability
        keep = F.interpolate(keep.float(), size=t, mode="nearest").bool()
        return observed & keep.reshape(b, k, t)

    def encode(self, params, mask_probability: float = 0.0):
        if params.ndim != 4 or params.shape[1] != 3 or min(params.shape[2:]) < 1:
            raise ValueError("Expected nonempty modal parameters [B,3,K,T].")
        if not 0 <= mask_probability < 1:
            raise ValueError("mask_probability must be in [0,1).")
        params = params.float()
        frequencies, amplitudes, phase = params.unbind(1)
        observed = amplitudes > 0
        b, k, t = frequencies.shape
        gain = amplitudes.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        log_frequency = frequencies.clamp_min(1e-7).log()
        center = (log_frequency * observed).sum(-1, keepdim=True)
        center = center / observed.sum(-1, keepdim=True).clamp_min(1)
        log_amplitude = (amplitudes / gain).clamp_min(1e-7).log()

        def typed_input(value):
            visible = self._visibility(observed, mask_probability)
            # The two masks distinguish missing peaks from training corruption.
            return torch.stack((value.masked_fill(~visible, 0),
                                observed.float(), visible.float()), dim=2).reshape(b * k, 3, t)

        f = self.frequency_stem(typed_input(log_frequency / 8))
        a = self.amplitude_stem(typed_input(log_amplitude / 8))
        tc = f.shape[-1]
        p = torch.stack((phase[..., 0].cos(), phase[..., 0].sin()), -1)
        p = self.phase_stem(p).reshape(b * k, -1, 1).expand(-1, -1, tc)
        h = torch.cat((f, a, p), 1).transpose(1, 2).reshape(b, k, tc, -1)
        # Match the two stride-2 convolution windows, including the last frame.
        mask = observed.float().reshape(b * k, 1, t)
        for _ in range(2):
            mask = F.max_pool1d(mask, 5, stride=2, padding=2)
        mask = mask.reshape(b, k, tc).bool()
        h = self.input_projection(h).masked_fill(~mask[..., None], 0)
        for temporal, interaction in zip(self.temporal, self.interactions):
            h = interaction(temporal(h, mask), mask)
        return ModalEncoding(h, mask, params, observed, center, gain)

    def decode(self, hidden, state: ModalEncoding):
        b, k, tc, c = hidden.shape
        t = state.reference.shape[-1]
        h = self.decoder_interaction(
            self.decoder_temporal(hidden, state.mask), state.mask
        )
        trajectories = self.output(h).reshape(b * k, tc, 2).transpose(1, 2)
        trajectories = F.interpolate(trajectories, size=t, mode="linear", align_corners=False)
        trajectories = trajectories.reshape(b, k, 2, t)
        frequency = (state.log_center + trajectories[:, :, 0].clamp(-8, 8)).exp()
        frequency = frequency.clamp(1e-7, math.pi * 0.999)
        amplitude = state.gain * trajectories[:, :, 1].clamp(-16, 4).exp()
        amplitude = amplitude.masked_fill(~state.observed, 0)
        frequency = torch.where(state.observed, frequency, state.reference[:, 0])
        pooled = (h * state.mask[..., None]).sum(2)
        pooled = pooled / state.mask.sum(2, keepdim=True).clamp_min(1)
        circular = self.phase_output(pooled)
        # A tiny bias keeps atan2 defined even for a degenerate output vector.
        circular = circular + circular.new_tensor([1e-6, 0.0])
        phase = torch.atan2(circular[..., 1], circular[..., 0])
        phase = phase[..., None].expand(-1, -1, t)
        return torch.stack((frequency, amplitude, phase), 1)

    def forward(self, embedding=None, params=None):
        """Legacy modal_autoencoder signature; standalone tasks also use encode/decode."""
        if params is None:
            params = embedding
        state = self.encode(params)
        return self.decode(state.hidden, state), state.hidden

    @staticmethod
    def apply_edit(reference, baseline_decoded, edited_decoded,
                   max_octaves: float = 1.0, max_gain_db: float = 12.0):
        """Anchor changes to a fitted reference; silent frames remain silent."""
        df = (edited_decoded[:, 0].clamp_min(1e-7).log()
              - baseline_decoded[:, 0].clamp_min(1e-7).log())
        da = (edited_decoded[:, 1].clamp_min(1e-8).log()
              - baseline_decoded[:, 1].clamp_min(1e-8).log())
        f = reference[:, 0] * df.clamp(-max_octaves * math.log(2), max_octaves * math.log(2)).exp()
        a = reference[:, 1] * da.clamp(-max_gain_db * math.log(10) / 20,
                                      max_gain_db * math.log(10) / 20).exp()
        dp = edited_decoded[:, 2, :, :1] - baseline_decoded[:, 2, :, :1]
        dp = torch.atan2(dp.sin(), dp.cos())
        phase = reference[:, 2] + dp
        phase = torch.atan2(phase.sin(), phase.cos())
        f = torch.where(reference[:, 1] > 0, f.clamp(1e-7, math.pi * 0.999), reference[:, 0])
        return torch.stack((f, a, phase), 1)
