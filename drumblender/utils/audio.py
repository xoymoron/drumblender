"""
Audio utility functions
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
from einops import repeat

from drumblender.utils.device import resolve_device


def preprocess_audio_file(
    input_file: Path,
    output_file: Path,
    sample_rate: int,
    num_samples: Optional[int] = None,
    mono: bool = True,
    filter_silent_all: bool = True,
    silent_all_threshold_db: float = -75.0,
    remove_start_silence: bool = True,
    start_silence_threshold_db: float = -60.0,
    remove_end_silence: bool = True,
    tail_silence_threshold_db: float = -70.0,
    tail_peak_ratio: float = 0.001,
    min_tail_silence_ms: float = 50.0,
    frame_size: int = 256,
    hop_size: int = 256,
    max_duration_sec: Optional[float] = 14.0,
    device: Union[str, torch.device] = "cpu",
):
    """Convert one WAV file into the canonical training-audio representation.

    The variable-length pipeline is: load, select one channel when requested,
    resample, reject all-silent audio, trim leading/trailing silence, reject
    samples longer than the configured limit, and save. No loudness or peak
    normalization is applied. Signal operations use ``device`` (cpu, cuda,
    cuda:N, or auto); decoding and saving always use CPU tensors.
    """
    compute_device = resolve_device(device)
    waveform, orig_freq = torchaudio.load(input_file)
    assert waveform.ndim == 2, "Expecting a 2D tensor, channels x samples"
    waveform = waveform.to(compute_device)

    if mono:
        waveform = select_highest_rms_channel(waveform)
        assert waveform.shape[0] == 1, "Expecting a mono signal"

    if orig_freq != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=sample_rate
        ).to(device=compute_device, dtype=waveform.dtype)
        waveform = resampler(waveform)

    if filter_silent_all:
        if is_entirely_silent(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=silent_all_threshold_db,
        ):
            raise ValueError(f"silent_all: below {silent_all_threshold_db}dB")

    if remove_start_silence:
        waveform = cut_start_silence(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=start_silence_threshold_db,
        )

    if remove_end_silence:
        min_tail_silence_samples = int((min_tail_silence_ms / 1000.0) * sample_rate)

        waveform = cut_end_silence(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=tail_silence_threshold_db,
            min_silence_samples=min_tail_silence_samples,
            peak_ratio=tail_peak_ratio,
        )

    if max_duration_sec is not None and (num_samples is None):
        max_len = int(max_duration_sec * sample_rate)
        if waveform.shape[1] > max_len:
            raise ValueError(
                f"too_long: {waveform.shape[1]} samples > {max_len} samples"
            )

    if num_samples is not None and waveform.shape[1] != num_samples:
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]
        else:
            num_pad = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, num_pad))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_file, waveform.cpu(), sample_rate)


def select_highest_rms_channel(waveform: torch.Tensor) -> torch.Tensor:
    """Return the physical input channel with the largest RMS amplitude.

    ``torch.argmax`` deterministically selects the first channel when RMS
    values tie. A one-channel waveform is returned unchanged.
    """
    assert waveform.ndim == 2, "Expecting a 2D tensor, channels x samples"
    if waveform.shape[0] == 1:
        return waveform

    channel_rms = torch.sqrt(torch.mean(waveform.square(), dim=1))
    channel_index = int(torch.argmax(channel_rms).item())
    return waveform[channel_index : channel_index + 1, :]


def generate_sine_wave(
    frequency, num_samples, sample_rate, stereo: bool = False
) -> torch.Tensor:
    """Generate a sine wave."""
    n = torch.arange(num_samples)
    x = torch.sin(frequency * 2 * torch.pi * n / sample_rate)
    x = repeat(x, "n -> c n", c=2 if stereo else 1)
    return x


def first_non_silent_sample(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
) -> Union[int, None]:
    """
    Returns the index of the first non-silent sample in a waveform.
    Implementation based on Essentia StartStopCut.
    """
    assert x.ndim == 1, "Expecting a 1D tensor"
    frames = torch.split(x, frame_size)
    thrshold_power = float(np.power(10.0, threshold_db / 10.0))

    for i, frame in enumerate(frames):
        power = torch.inner(frame, frame) / frame.shape[-1]
        if power > thrshold_power:
            return i * hop_size
    return None


def cut_start_silence(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
) -> torch.Tensor:
    """
    Removes silent samples from the beginning of a waveform.
    """
    assert x.ndim == 2, "Expecting (channels, num_samples)"

    start_samples = []
    for channel in x:
        start_sample = first_non_silent_sample(
            channel, frame_size=frame_size, hop_size=hop_size, threshold_db=threshold_db
        )
        if start_sample is not None:
            start_samples.append(start_sample)

    if len(start_samples) == 0:
        # No frame cleared the onset threshold.
        raise ValueError(f"Entire wavfile below threshold level {threshold_db}dB")

    return x[:, min(start_samples) :]


def is_entirely_silent(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -75.0,
) -> bool:
    """
    True if ALL frames are below threshold_db (power dB).
    x: [C, T]
    """
    assert x.ndim == 2, "Expecting (channels, num_samples)"
    _, T = x.shape
    if T == 0:
        return True

    thr_power = float(np.power(10.0, threshold_db / 10.0))
    num_frames = int(math.ceil(T / hop_size))

    for i in range(num_frames):
        start = i * hop_size
        end = min(start + frame_size, T)
        frame = x[:, start:end]
        if frame.numel() == 0:
            continue
        power = (frame * frame).mean(dim=1)  # [C]
        if bool(torch.any(power > thr_power).item()):
            return False

    return True


def cut_end_silence(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
    min_silence_samples: int = 0,
    peak_ratio: float = 0.02,
) -> torch.Tensor:
    """
    Removes silent samples from the end of a waveform.

    Cut condition:
      - trailing region length >= min_silence_samples
      - tail peak <= peak_ratio * global_peak
      - tail frames (mean power) are below threshold_db (power dB)
    """
    assert x.ndim == 2, "Expecting (channels, num_samples)"
    _, T = x.shape
    if T == 0:
        return x

    global_peak = float(x.abs().max().item())
    if global_peak <= 1e-12:
        raise ValueError("Entire wavfile is (near) zero")

    thr_power = float(np.power(10.0, threshold_db / 10.0))

    num_frames = int(math.ceil(T / hop_size))
    frame_starts = [i * hop_size for i in range(num_frames)]

    silent_flags = []
    for start in frame_starts:
        end = min(start + frame_size, T)
        frame = x[:, start:end]
        if frame.numel() == 0:
            silent = True
        else:
            power = (frame * frame).mean(dim=1)  # [C]
            silent = bool(torch.all(power <= thr_power).item())
        silent_flags.append(silent)

    # Find the last non-silent frame.
    last_non_silent_idx = None
    for i in range(num_frames - 1, -1, -1):
        if not silent_flags[i]:
            last_non_silent_idx = i
            break

    if last_non_silent_idx is None:
        raise ValueError(f"Entire wavfile below threshold level {threshold_db}dB")

    candidate_cut = min(frame_starts[last_non_silent_idx] + frame_size, T)
    trailing_len = T - candidate_cut
    if trailing_len < min_silence_samples:
        return x

    tail_peak = (
        float(x[:, candidate_cut:].abs().max().item()) if candidate_cut < T else 0.0
    )
    if tail_peak <= peak_ratio * global_peak:
        return x[:, :candidate_cut]

    return x
