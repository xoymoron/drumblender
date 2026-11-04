from numbers import Real

import torch
from torch.nn.utils.rnn import pad_sequence

from drumblender.utils.lengths import validate_lengths


def pad_audio_params_collate(batch):
    """
    Supports items of:
      - (audio, length)
      - (audio, params)
      - (audio, params, length)
    where audio is a mono Tensor [1, T] or [T],
    params is Tensor or dict-like, and length is int. Modal tensors [P, M, F]
    return (audio, params, lengths, frame_lengths); other inputs return three
    entries. Frame counts include valid silent trajectory frames.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if any(len(item) != len(batch[0]) for item in batch):
        raise ValueError("Batch items must have the same structure")

    if len(batch[0]) == 3:
        audios, params, lengths = zip(*batch)
    elif len(batch[0]) == 2:
        audios, params = zip(*batch)
        if isinstance(params[0], Real) or (torch.is_tensor(params[0]) and params[0].ndim == 0):
            lengths = params
            params = (None,) * len(batch)
        else:
            lengths = [a.shape[-1] for a in audios]
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

    audios_1d = []
    checked_lengths = []
    for a, length in zip(audios, lengths):
        if not torch.is_tensor(a) or a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[0] != 1):
            raise ValueError("Expected nonempty mono audio [T] or [1,T]")
        if not torch.isfinite(a).all():
            raise ValueError("Audio must contain only finite samples")
        checked = validate_lengths([length], batch_size=1, max_length=a.shape[-1], device=a.device)
        checked_lengths.append(int(checked[0]))
        if a.dim() == 2 and a.shape[0] == 1:
            audios_1d.append(a.squeeze(0))
        else:
            audios_1d.append(a)
    lengths = torch.tensor(checked_lengths, dtype=torch.long)

    padded = pad_sequence(audios_1d, batch_first=True)
    padded = padded.unsqueeze(1)

    frame_lengths = None
    if params[0] is None:
        if any(p is not None for p in params):
            raise ValueError("All items must consistently include or omit parameters")
        params_out = None

    elif isinstance(params[0], dict):
        if any(not isinstance(p, dict) or p.keys() != params[0].keys() for p in params):
            raise ValueError("Parameter dictionaries must have the same keys")
        out = {}
        for k in params[0].keys():
            vals = [p[k] for p in params]
            if torch.is_tensor(vals[0]):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = torch.tensor(vals)
        params_out = out

    elif torch.is_tensor(params[0]):
        p0 = params[0]
        if any(not torch.is_tensor(p) or p.ndim != p0.ndim for p in params):
            raise ValueError("Parameter tensors must have matching shape axes")
        if p0.dim() == 3:
            if any(p.shape[:2] != p0.shape[:2] or min(p.shape) < 1 for p in params):
                raise ValueError("Modal tensors must have the same parameter/mode shape and nonempty frames")
            frame_lengths = torch.tensor([p.shape[-1] for p in params], dtype=torch.long)
            flats = []
            PM = p0.shape[0] * p0.shape[1]
            for p in params:
                flats.append(p.reshape(PM, p.shape[-1]).transpose(0, 1))
            padded_flat = pad_sequence(flats, batch_first=True)
            padded_flat = padded_flat.transpose(1, 2)
            params_out = padded_flat.reshape(len(params), p0.shape[0], p0.shape[1], -1)
        else:
            if any(p.shape != p0.shape for p in params):
                raise ValueError("Parameter tensors must have matching shape")
            params_out = torch.stack(list(params), dim=0)
    else:
        raise TypeError("Parameters must be tensors, dictionaries, or None")

    if frame_lengths is not None:
        return padded, params_out, lengths, frame_lengths
    return padded, params_out, lengths
