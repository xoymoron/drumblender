"""
torchmetrics
"""
from typing import Any
import hashlib
import importlib
import json
import math
from pathlib import Path

import torch
import torchaudio
from torchmetrics import Metric


DEFAULT_METRICS_CONFIG = Path(__file__).resolve().parents[1] / "cfg/metrics/drumblender_metrics.yaml"


def load_evaluation_metrics(config_path=None):
    """Use one metric configuration for tensor evaluation and WAV rescoring."""
    import yaml

    path = Path(config_path) if config_path is not None else DEFAULT_METRICS_CONFIG
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))

    def instantiate(value):
        if isinstance(value, dict):
            if "class_path" in value:
                module, name = value["class_path"].rsplit(".", 1)
                cls = getattr(importlib.import_module(module), name)
                return cls(**instantiate(value.get("init_args", {})))
            return {key: instantiate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [instantiate(item) for item in value]
        return value

    metrics = instantiate(spec)
    required = {"lsd", "flux_onset", "mr_stft", "mr_log_mel", "band_spectral", "log_rms_envelope"}
    if not isinstance(metrics, torch.nn.ModuleDict) or not required.issubset(metrics):
        raise ValueError(f"Evaluation config must define {sorted(required)}")
    config_sha256 = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    return metrics, config_sha256


@torch.no_grad()
def score_reconstruction(metrics, pred, target, sample_rate=None):
    """Score one valid, unpadded mono example; never use the training objective."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[:2] != (1, 1):
        raise ValueError("Expected matching [1, 1, samples] prediction and target")
    if pred.shape[-1] == 0 or not (torch.isfinite(pred).all() and torch.isfinite(target).all()):
        raise ValueError("Evaluation requires nonempty, finite audio")
    result = {}
    for name, metric in metrics.items():
        expected_rate = getattr(metric, "sample_rate", None)
        if sample_rate is not None and expected_rate is not None and sample_rate != expected_rate:
            raise ValueError(f"{name} expects {expected_rate} Hz, received {sample_rate} Hz")
        if name == "mr_stft":
            # Auraloss centers its STFT with reflect padding around the real clip.
            min_samples = max(metric.fft_sizes) // 2 + 1
            if pred.shape[-1] < min_samples:
                raise ValueError(f"MR-STFT requires at least {min_samples} samples")
        if isinstance(metric, Metric):
            metric.reset()
        value = metric(pred, target)
        if isinstance(metric, Metric):
            metric.reset()
        values = value if isinstance(value, dict) else {name: value}
        for key, item in values.items():
            if item.numel() != 1 or not torch.isfinite(item).all():
                raise ValueError(f"Nonfinite or nonscalar evaluation metric: {key}")
            result[f"test/{key}"] = float(item.detach().cpu())
    return result


def evaluation_score_keys(metrics):
    """Every scalar output expected from the configured evaluation suite."""
    keys = [f"test/{name}" for name in metrics if name != "band_spectral"]
    keys.extend(f"test/band_{kind}_{name}" for name, _, _ in metrics["band_spectral"].bands
                for kind in ("lsd", "sc"))
    return keys


def audio_pair_fingerprint(bundle_dir, source_filename):
    """Invalidate cached scores if either exported WAV changes (or disappears)."""
    relative = Path(str(source_filename).replace("\\", "/"))
    if relative.anchor or ".." in relative.parts:
        raise ValueError(f"Unsafe audio path: {source_filename}")
    digest = hashlib.sha256()
    for kind in ("recon", "target"):
        path = Path(bundle_dir) / kind / relative
        digest.update(kind.encode())
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing")
    return digest.hexdigest()


class LogSpectralDistance(Metric):
    """
    Log Spectral Distance (LSD) metric.

    Implementation based on https://arxiv.org/abs/1909.06628
    """

    full_state_update = False

    def __init__(
        self, n_fft=8092, hop_size=64, eps: float = 1e-8, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("lsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.eps = eps

    def _log_spectral_power_mag(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft, device=x.device),
            return_complex=True,
            pad_mode="constant",
        )
        return torch.log(torch.square(torch.abs(X)) + self.eps)

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape
        assert x.ndim == 3 and x.shape[1] == 1, "Only mono audio is supported"
        x = x.squeeze(1)
        y = y.squeeze(1)

        X = self._log_spectral_power_mag(x)
        Y = self._log_spectral_power_mag(y)

        lsd = torch.mean(torch.square(X - Y), dim=-2)

        lsd = torch.mean(torch.sqrt(lsd), dim=-1)

        self.lsd += torch.sum(lsd)
        self.count += lsd.shape[0]

    def compute(self) -> torch.Tensor:
        return self.lsd / self.count


class MFCCError(Metric):
    """
    MFCC Error
    """

    full_state_update = False

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("mfcc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape
        assert x.ndim == 3 and x.shape[1] == 1, "Only mono audio is supported"
        if x.shape[-1] <= self.n_fft // 2:
            raise ValueError(f"MFCC requires more than {self.n_fft // 2} samples")
        x = x.squeeze(1)
        y = y.squeeze(1)

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length},
        ).to(device=x.device, dtype=x.dtype)
        X = mfcc(x)
        Y = mfcc(y)
        mae = torch.mean(torch.abs(X - Y), dim=(-2, -1))

        self.mfcc += torch.sum(mae)
        self.count += mae.shape[0]

    def compute(self) -> torch.Tensor:
        return self.mfcc / self.count


class MultiResolutionLogMelDistance(torch.nn.Module):
    """Mean absolute natural-log mel magnitude distance across resolutions."""

    def __init__(
        self,
        sample_rate: int = 48000,
        fft_sizes=(512, 2048, 8192),
        hop_sizes=(128, 512, 2048),
        n_mels=(40, 80, 128),
        f_min: float = 20.0,
        f_max: float = 20000.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(n_mels)) or not fft_sizes:
            raise ValueError("Log-mel resolutions must have matching nonempty lengths")
        if not 0 <= f_min < f_max <= sample_rate / 2:
            raise ValueError("Invalid mel frequency range")
        self.sample_rate = sample_rate
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes)
        self.eps = eps
        for i, (n_fft, n_mel) in enumerate(zip(fft_sizes, n_mels)):
            fb = torchaudio.functional.melscale_fbanks(
                n_fft // 2 + 1, f_min, f_max, n_mel, sample_rate,
                norm="slaney", mel_scale="slaney",
            )
            self.register_buffer(f"mel_filter_{i}", fb)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distances = []
        for i, (n_fft, hop) in enumerate(zip(self.fft_sizes, self.hop_sizes)):
            window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            fb = getattr(self, f"mel_filter_{i}")
            def log_mel(audio):
                spectrum = torch.stft(
                    audio[:, 0], n_fft=n_fft, hop_length=hop,
                    window=window, return_complex=True, pad_mode="constant",
                ).abs().transpose(1, 2)
                return torch.log(torch.matmul(spectrum, fb).clamp_min(self.eps))
            distances.append(torch.mean(torch.abs(log_mel(pred) - log_mel(target))))
        return torch.stack(distances).mean()


class BandSpectralErrors(torch.nn.Module):
    """Per-band log-power LSD and magnitude SC, from one STFT.

    SC divides by target-band energy. A band absent from the reference can
    therefore produce a large, informative error; report tail quantiles.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 4096,
        hop_size: int = 256,
        bands=(("low", 20, 250), ("mid", 250, 4000), ("high", 4000, 20000)),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.eps = eps
        names = set()
        self.bands = []
        for name, lo, hi in bands:
            if name in names or not 0 <= lo < hi <= sample_rate / 2:
                raise ValueError(f"Invalid or duplicate spectral band: {name}")
            start = max(0, math.ceil(lo * n_fft / sample_rate))
            stop = min(n_fft // 2 + 1, math.ceil(hi * n_fft / sample_rate))
            if start >= stop:
                raise ValueError(f"Empty spectral band: {name}")
            self.bands.append((name, start, stop))
            names.add(name)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        window = torch.hann_window(self.n_fft, device=pred.device, dtype=pred.dtype)
        def magnitude(audio):
            return torch.stft(
                audio[:, 0], n_fft=self.n_fft, hop_length=self.hop_size,
                window=window, return_complex=True, pad_mode="constant",
            ).abs()
        x, y = magnitude(pred), magnitude(target)
        scores = {}
        for name, start, stop in self.bands:
            xa, ya = x[:, start:stop], y[:, start:stop]
            log_difference = torch.log(xa.square() + self.eps) - torch.log(ya.square() + self.eps)
            scores[f"band_lsd_{name}"] = log_difference.square().mean(dim=1).sqrt().mean()
            scores[f"band_sc_{name}"] = torch.linalg.vector_norm(xa - ya, dim=(1, 2)).div(
                torch.linalg.vector_norm(ya, dim=(1, 2)).clamp_min(self.eps)
            ).mean()
        return scores


class LogRMSEnvelopeError(torch.nn.Module):
    """Mean absolute difference of framewise log RMS envelopes."""

    def __init__(self, frame_size: int = 1024, hop_size: int = 256, eps: float = 1e-5):
        super().__init__()
        if frame_size <= 0 or hop_size <= 0 or eps <= 0:
            raise ValueError("Frame, hop, and floor must be positive")
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def envelope(audio):
            if audio.shape[-1] < self.frame_size:
                raise ValueError(f"Log RMS requires at least {self.frame_size} samples")
            power = torch.nn.functional.avg_pool1d(
                audio.square(), kernel_size=self.frame_size, stride=self.hop_size
            )
            return 0.5 * torch.log(power + self.eps ** 2)
        return torch.mean(torch.abs(envelope(pred) - envelope(target)))


class SpectralFluxOnsetError(Metric):
    """
    Error between spectral flux onset signals
    """

    full_state_update = False

    def __init__(self, n_fft=1024, hop_size=64, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.n_fft = n_fft
        self.hop_size = hop_size

    def _onset_signal(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Input must be of shape (batch, channels, length)"
        assert x.shape[1] == 1, "Input must be mono"

        x = x.squeeze(1)
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft, device=x.device),
            return_complex=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
        )
        # STFT is [batch, frequency, time]; Eq. (3) differentiates time.
        flux = torch.diff(torch.abs(X), dim=-1)
        flux = (flux + torch.abs(flux)) / 2
        flux = torch.square(flux)
        flux = torch.sum(flux, dim=1)

        return flux

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape
        assert x.ndim == 3 and x.shape[1] == 1, "Only mono audio is supported"
        if x.shape[-1] < self.hop_size:
            raise ValueError(f"Spectral flux requires at least {self.hop_size} samples")

        x = self._onset_signal(x)
        y = self._onset_signal(y)

        onset_error = torch.mean(torch.abs(x - y), dim=-1)

        self.error += torch.sum(onset_error)
        self.count += onset_error.shape[0]

    def compute(self) -> torch.Tensor:
        return self.error / self.count
