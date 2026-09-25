from typing import Optional

import torch
from einops import rearrange


class ModalSynth(torch.nn.Module):
    """
    Modal synthesis with given frequencies, amplitudes, and optional phase
    Amplitude interpolates linearly. Frequency interpolates only when both
    neighboring frames contain an active observation.
    """

    def forward(self, params: torch.Tensor, num_samples: int):
        """
        params: [nb,num_params,num_modes,num_frames], expected parameters
                are: frequency, amplitude, and phase (optional)
        num_samples: number of samples to generate
        """
        assert params.ndim == 4, "Expected 4D tensor"
        assert params.size()[1] in [2, 3], "Expected 2 or 3 parameters"

        params = torch.chunk(params, params.size()[1], dim=1)
        params = [p.squeeze(1) for p in params]

        # Pass the phase if it was given, otherwise None
        phase = None
        if len(params) == 3:
            phase = params[2]

        y = modal_synth(params[0], params[1], num_samples, phase)
        y = rearrange(y, "b n -> b 1 n")
        return y


def modal_synth(
    freqs: torch.Tensor,
    amps: torch.Tensor,
    num_samples: int,
    phase: Optional[torch.Tensor] = None,
    mode_chunk_size: int = 32,
) -> torch.Tensor:
    """
    Synthesizes a modal signal from a set of frequencies, phases, and amplitudes.

    Args:
        freqs: A 3D tensor of frequencies in angular frequency of shape
            (batch_size, num_modes, num_frames)
        amps: A 3D tensor of amplitudes of shape (batch_size, num_modes, num_frames)
        num_samples: Number of samples in the output signal
        phase: Initial sine phase in radians, read from the first frame.
        mode_chunk_size: Bound inference workspace by rendering this many modes
            at a time. Autograd still retains intermediates during training.
    """
    batch_size, num_modes, _ = freqs.shape
    assert freqs.shape == amps.shape
    if num_samples < 1 or mode_chunk_size < 1:
        raise ValueError("num_samples and mode_chunk_size must be positive.")

    # Legacy caches use 0 or fmin in silent frames. The value at an unobserved
    # frame is irrelevant to the track, but linear sample interpolation can
    # otherwise turn it into an audible pitch sweep.
    freqs = _fill_inactive_frequencies(freqs, amps)
    active = amps != 0
    frame_count = freqs.shape[-1]
    frame_position = (
        torch.arange(num_samples, device=freqs.device, dtype=freqs.dtype) + 0.5
    ) * frame_count / num_samples - 0.5
    left = frame_position.floor().long().clamp(0, frame_count - 1)
    right = (left + 1).clamp(max=frame_count - 1)
    y = freqs.new_zeros((batch_size, num_samples))
    for start in range(0, num_modes, mode_chunk_size):
        stop = start + mode_chunk_size
        chunk_frequencies = freqs[:, start:stop]
        chunk_active = active[:, start:stop]
        w = torch.nn.functional.interpolate(
            chunk_frequencies, size=num_samples, mode="linear", align_corners=False
        )
        active_left = chunk_active.index_select(-1, left)
        active_right = chunk_active.index_select(-1, right)
        # The carrier stays at the observed endpoint while amplitude fades to
        # or rises from zero. A missing peak can bridge track matching, but it
        # cannot contribute an interpolated frequency to audible samples.
        w = torch.where(
            active_left & ~active_right,
            chunk_frequencies.index_select(-1, left),
            w,
        )
        w = torch.where(
            ~active_left & active_right,
            chunk_frequencies.index_select(-1, right),
            w,
        )
        a = torch.nn.functional.interpolate(
            amps[:, start:stop], size=num_samples, mode="linear", align_corners=False
        )
        phase_env = torch.cumsum(w, dim=-1)
        if phase is not None:
            phase_env = phase_env + phase[:, start:stop, :1]
        y = y + torch.sum(a * torch.sin(phase_env), dim=1)
    return y


def _fill_inactive_frequencies(freqs: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
    """Hold the last observed frequency in gaps and fill leading frames.

    Activity is independent of frequency values, so this also repairs cached
    fmin padding. Fully silent modes use zero frequency. Gather/interpolation
    keep gradients connected to the observed frequency values. This stored
    placeholder does not define interpolation through missing observations;
    modal_synth applies the active-endpoint rule at sample resolution.
    """
    frames = freqs.shape[-1]
    if frames < 1:
        raise ValueError("Modal parameters must contain at least one frame.")
    active = amps != 0
    index = torch.arange(frames, device=freqs.device).expand_as(freqs)
    previous = torch.where(active, index, -1).cummax(dim=-1).values
    following = (
        torch.where(active, index, frames).flip(-1).cummin(dim=-1).values.flip(-1)
    )
    source = torch.where(previous >= 0, previous, following).clamp(0, frames - 1)
    filled = freqs.gather(-1, source)
    return torch.where(
        active.any(dim=-1, keepdim=True), filled, torch.zeros_like(filled)
    )
