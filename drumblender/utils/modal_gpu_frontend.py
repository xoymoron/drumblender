"""CUDA FFT front end for the existing CPU modal peak tracker.

Only spectral transforms run on CUDA. Each bounded block returns to NumPy so
the peak fitting, association, and exported features keep their CPU semantics.
"""

import numpy as np
from scipy import signal
import torch
import torch.nn.functional as functional


def cqt_coefficients(
    reduced,
    sample_rate,
    centers,
    frequencies,
    lengths,
    width,
    transform_size,
    decimation,
    device,
    batch_size,
):
    """Evaluate the same centered CQT kernels as the SciPy path in CUDA batches."""
    spectrum = torch.fft.fft(torch.as_tensor(reduced, device=device), n=transform_size)
    positions = centers / decimation
    left = np.floor(positions).astype(np.int64)
    fraction = positions - left
    first_index = torch.as_tensor(
        width // 2 + np.clip(left, 0, len(reduced) - 1), device=device
    )
    second_index = torch.as_tensor(
        width // 2 + np.clip(left + 1, 0, len(reduced) - 1), device=device
    )
    fraction = torch.as_tensor(fraction, dtype=torch.float32, device=device)
    output = np.empty((len(centers), len(frequencies)), np.complex64)

    for start in range(0, len(frequencies), batch_size):
        stop = min(start + batch_size, len(frequencies))
        kernels = np.zeros((stop - start, width), np.complex64)
        for row, index in enumerate(range(start, stop)):
            frequency = frequencies[index]
            length = lengths[index]
            offsets = np.arange(length) - length // 2
            window = signal.windows.hann(length, sym=True)
            offset = (width - length) // 2
            kernels[row, offset : offset + length] = (
                2
                * window
                / window.sum()
                * np.exp(2j * np.pi * frequency * offsets / sample_rate)
            )

        kernels = torch.as_tensor(kernels, device=device)
        convolved = torch.fft.ifft(
            torch.fft.fft(kernels, n=transform_size, dim=-1) * spectrum,
            dim=-1,
        )
        first = convolved.index_select(-1, first_index)
        second = convolved.index_select(-1, second_index)
        frequency = torch.as_tensor(
            frequencies[start:stop], dtype=torch.float32, device=device
        )
        rotation = torch.exp(2j * torch.pi * frequency / sample_rate)[:, None]
        phase = torch.exp(
            2j * torch.pi * frequency[:, None] * fraction[None] / sample_rate
        )
        coefficients = (
            (1 - fraction)[None] * first + fraction[None] * second / rotation
        ) * phase
        output[:, start:stop] = coefficients.transpose(0, 1).cpu().numpy()
    return output


def stft_blocks(audio, centers, size, frame_block_size, device):
    """Yield centered, normalized STFT blocks on CPU after CUDA transforms."""
    audio = torch.as_tensor(audio, device=device)
    padded = functional.pad(audio, (size // 2, size // 2))
    windows = padded.unfold(0, size, 1)
    window = torch.hann_window(size, periodic=True, device=device)
    rotation = torch.where(
        torch.arange(size // 2 + 1, device=device) % 2 == 0, 1.0, -1.0
    )
    for start in range(0, len(centers), frame_block_size):
        stop = min(start + frame_block_size, len(centers))
        locations = np.rint(centers[start:stop]).astype(np.int64)
        indices = torch.as_tensor(locations, device=device)
        spectra = torch.fft.rfft(windows.index_select(0, indices) * window, dim=-1)
        spectra *= (2 / window.sum()) * rotation
        yield start, locations, spectra.cpu().numpy()
