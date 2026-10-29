"""Staged Lightning training and reference-cached modal/noise inference.

Reconstruction examples do not identify the correct noise for an arbitrary
creative edit. Noise linking is therefore opt-in and never trained to cancel an
edit by reconstructing the unchanged source waveform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from drumblender.models.modal_control import ModalEncoding, NeuralModalAutoencoder
from drumblender.models.modal_patterns import TemporalModalSAE
from drumblender.models.modal_conditioned_noise import ModalConditionedNoise
from drumblender.synths.modal import ModalSynth
from drumblender.synths.noise import NoiseGenerator


def spectral_loss(prediction, target):
    """Training MR-STFT on valid audio only; use a smaller FFT for short clips.

This is a training objective, not a replacement for the repository's evaluation
metrics. center=False avoids artificial boundary samples in these spectra.
"""
    length = target.shape[-1]
    if prediction.shape != target.shape or length < 1:
        raise ValueError("Spectral loss requires matching, nonempty waveforms.")
    if length < 4:
        return F.l1_loss(prediction, target)
    sizes = [n for n in (512, 1024, 2048) if n <= length]
    if not sizes:
        sizes = [2 ** int(math.log2(length))]
    losses = []
    for n in sizes:
        window = torch.hann_window(n, device=target.device, dtype=torch.float32)

        def spectrum(x):
            return torch.stft(x.float().reshape(-1, length), n, hop_length=max(1, n // 4),
                              window=window, center=False, return_complex=True).abs()

        p, y = spectrum(prediction), spectrum(target)
        convergence = (p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1).clamp_min(1e-4)
        log_difference = (p.clamp_min(1e-5).log() - y.clamp_min(1e-5).log()).abs().mean()
        losses.append(convergence.mean() + log_difference)
    return torch.stack(losses).mean()


def parameter_losses(prediction, target):
    """Anchor only real observations, with an energy-weighted circular phase loss."""
    observed = target[:, 1] > 0
    scale = target[:, 1].amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    weight = (target[:, 1] / scale).sqrt() * observed
    denominator = weight.sum().clamp_min(1)
    f_error = (prediction[:, 0].clamp_min(1e-7).log()
               - target[:, 0].clamp_min(1e-7).log()).abs()
    a_error = ((prediction[:, 1] / scale).clamp_min(1e-7).log()
               - (target[:, 1] / scale).clamp_min(1e-7).log()).abs()
    phase_weight = weight.sum(-1)
    phase_error = 1 - (prediction[:, 2, :, 0] - target[:, 2, :, 0]).cos()
    return {
        "frequency": (f_error * weight).sum() / denominator,
        "amplitude": (a_error * weight).sum() / denominator,
        "phase": (phase_error * phase_weight).sum() / phase_weight.sum().clamp_min(1),
    }


@dataclass
class ModalReference:
    encoding: ModalEncoding
    parameters: torch.Tensor
    noise_features: torch.Tensor | None
    noise_bands: torch.Tensor | None
    num_samples: int


class ModalControlTask(pl.LightningModule):
    """Four explicit stages: modal, noise, joint, patterns."""

    def __init__(self, stage: str = "modal", sample_rate: int = 48000,
                 channels: int = 64, context_tokens: int = 8,
                 features: int = 256, top_k: int = 8,
                 learning_rate: float = 1e-4, joint_modal_lr_scale: float = 0.1,
                 mask_probability: float = 0.15,
                 modal_audio_weight: float = 0.1,
                 anchor_weight: float = 1.0, residual_weight: float = 0.1):
        super().__init__()
        if (sample_rate <= 0 or learning_rate <= 0 or not 0 < joint_modal_lr_scale <= 1
                or not 0 <= mask_probability < 1
                or min(modal_audio_weight, anchor_weight, residual_weight) < 0):
            raise ValueError("Invalid training rates, loss weights or sample rate.")
        self.save_hyperparameters()
        self.modal_autoencoder = NeuralModalAutoencoder(channels, context_tokens)
        self.patterns = TemporalModalSAE(channels, features, top_k)
        self.noise_autoencoder = ModalConditionedNoise(channels)
        self.modal_synth = ModalSynth()
        self.noise_synth = NoiseGenerator(window_size=256)
        self.register_buffer("pattern_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("noise_updates", torch.tensor(0, dtype=torch.long))
        self.set_stage(stage)

    def set_stage(self, stage):
        if stage not in ("modal", "noise", "joint", "patterns"):
            raise ValueError("stage must be modal, noise, joint or patterns")
        self.stage = stage
        self.hparams.stage = stage
        for module, enabled in (
            (self.modal_autoencoder, stage in ("modal", "joint")),
            (self.noise_autoencoder, stage in ("noise", "joint")),
            (self.patterns, stage == "patterns"),
        ):
            module.requires_grad_(enabled)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "stage"):
            if self.stage not in ("modal", "joint"):
                self.modal_autoencoder.eval()
            if self.stage not in ("noise", "joint"):
                self.noise_autoencoder.eval()
            if self.stage != "patterns":
                self.patterns.eval()
        return self

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        groups = []
        for module, rate in (
            (self.modal_autoencoder, lr * (self.hparams.joint_modal_lr_scale if self.stage == "joint" else 1)),
            (self.noise_autoencoder, lr), (self.patterns, lr),
        ):
            parameters = [p for p in module.parameters() if p.requires_grad]
            if parameters:
                groups.append({"params": parameters, "lr": rate})
        return torch.optim.AdamW(groups, weight_decay=1e-6)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["modal_control_stage"] = self.stage

    def _noise_bands(self, features, hidden, mask, length):
        bands = self.noise_autoencoder.decode(features, hidden, mask)
        # Match the actual OLA hop, regardless of the encoder's boundary rounding.
        frames = math.ceil(length / self.noise_synth.hop_size) + 1
        return F.interpolate(bands.transpose(1, 2), size=frames,
                             mode="linear", align_corners=False).transpose(1, 2)

    def _render_noise(self, bands, length, seed=None):
        excitation = None
        if seed is not None:
            generator = torch.Generator(device=bands.device).manual_seed(seed)
            excitation = torch.rand(
                bands.shape[0], bands.shape[1], self.noise_synth.window_size,
                generator=generator, device=bands.device, dtype=bands.dtype,
            ) * 2 - 1
        return self.noise_synth(bands, length, noise_source=excitation).unsqueeze(1)

    def _example_loss(self, audio, params, training):
        probability = self.hparams.mask_probability if training and self.stage == "modal" else 0.0
        state = self.modal_autoencoder.encode(params, probability)
        if self.stage == "patterns":
            loss, sparse = self.patterns(state.hidden.detach(), state.mask)
            used = (sparse.detach() > 0).any(dim=(0, 1, 2)).float().mean()
            return loss, {"sae": loss, "feature_fraction": used}

        prediction = self.modal_autoencoder.decode(state.hidden, state)
        anchors = parameter_losses(prediction, params)
        anchor = anchors["frequency"] + anchors["amplitude"] + 0.1 * anchors["phase"]
        length = audio.shape[-1]
        modal = self.modal_synth(prediction, length)
        stats = dict(anchors)
        if self.stage in ("modal", "joint"):
            with torch.no_grad():
                teacher = self.modal_synth(params, length)
            modal_loss = spectral_loss(modal, teacher)
            loss = self.hparams.anchor_weight * anchor + self.hparams.modal_audio_weight * modal_loss
            stats["modal_audio"] = modal_loss
        else:
            loss = modal.new_zeros(())

        if self.stage in ("noise", "joint"):
            # No backpropagation from residual encoding or conditioning into the
            # modal backbone. Its raw-audio gradient comes only through modal audio.
            residual = audio - modal.detach()
            features = self.noise_autoencoder.encode_reference(residual)
            bands = self._noise_bands(features, state.hidden.detach(), state.mask, length)
            noise = self._render_noise(bands, length, seed=None if training else 0)
            mixture_loss = spectral_loss(modal + noise, audio)
            residual_loss = spectral_loss(noise, residual.detach())
            loss = loss + mixture_loss + self.hparams.residual_weight * residual_loss
            stats.update(mixture_audio=mixture_loss, residual_audio=residual_loss)
        return loss, stats

    def _step(self, batch, training):
        losses, statistics = [], {}
        for i in range(batch["audio"].shape[0]):
            length = int(batch["lengths"][i])
            frames = int(batch["frame_lengths"][i])
            modes = int(batch["mode_counts"][i])
            audio = batch["audio"][i:i + 1, :, :length].float()
            params = batch["params"][i:i + 1, :, :modes, :frames].float()
            loss, stats = self._example_loss(audio, params, training)
            losses.append(loss)
            for key, value in stats.items():
                statistics.setdefault(key, []).append(value.detach())
        loss = torch.stack(losses).mean()
        prefix = "train" if training else "validation"
        self.log(f"{prefix}/loss", loss, on_step=training, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(losses))
        for key, values in statistics.items():
            self.log(f"{prefix}/{key}", torch.stack(values).mean(), on_step=False,
                     on_epoch=True, sync_dist=True, batch_size=len(losses))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, True)
        if self.stage == "patterns":
            self.pattern_updates.add_(1)
        elif self.stage in ("modal", "joint"):
            # Any backbone update invalidates a previously fitted dictionary.
            self.pattern_updates.zero_()
        if self.stage in ("noise", "joint"):
            self.noise_updates.add_(1)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, False)

    @torch.no_grad()
    def encode_reference(self, audio, params):
        """Analyze one unpadded sample once, before any creative intervention."""
        if self.training:
            raise RuntimeError("Call eval() before caching an inference reference.")
        if audio.ndim != 3 or audio.shape[:2] != (1, 1) or params.shape[0] != 1:
            raise ValueError("Reference caching expects a single mono sample.")
        state = self.modal_autoencoder.encode(params)
        prediction = self.modal_autoencoder.decode(state.hidden, state)
        features = bands = None
        if self.noise_updates.item() > 0:
            modal = self.modal_synth(prediction, audio.shape[-1])
            features = self.noise_autoencoder.encode_reference(audio - modal)
            bands = self._noise_bands(features, state.hidden, state.mask, audio.shape[-1])
        return ModalReference(state, prediction, features, bands, audio.shape[-1])

    @torch.no_grad()
    def edit_reference(self, reference, feature=0, amount=0.0, noise_link=0.0, seed=0):
        if self.training:
            raise RuntimeError("Call eval() before editing.")
        if not 0 <= noise_link <= 1:
            raise ValueError("noise_link must be in [0,1].")
        if amount != 0 and self.pattern_updates.item() == 0:
            raise RuntimeError("Train patterns on the final backbone before editing.")
        state = reference.encoding
        hidden = self.patterns.edit(state.hidden, state.mask, feature, amount)
        if amount == 0:
            parameters = reference.parameters
        else:
            decoded = self.modal_autoencoder.decode(hidden, state)
            parameters = self.modal_autoencoder.apply_edit(reference.parameters,
                                                           reference.parameters, decoded)
        modal = self.modal_synth(parameters, reference.num_samples)
        noise = torch.zeros_like(modal)
        bands = reference.noise_bands
        if bands is not None:
            if noise_link != 0 and amount != 0:
                candidate = self._noise_bands(reference.noise_features, hidden,
                                              state.mask, reference.num_samples)
                bands = self.noise_autoencoder.link(bands, candidate, noise_link)
            noise = self._render_noise(bands, reference.num_samples, seed)
        return {"audio": modal + noise, "modal_audio": modal, "noise_audio": noise,
                "params": parameters, "noise_bands": bands}
