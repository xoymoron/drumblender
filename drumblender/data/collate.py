import torch
from torch.nn.utils.rnn import pad_sequence


def pad_audio_params_collate(batch):
    """
    Supports items of:
      - (audio, length)
      - (audio, params)
      - (audio, params, length)
    where audio is Tensor [..., T] (typically [1, T] or [T]),
    params is Tensor or dict-like, and length is int.
    """

    if len(batch[0]) == 3:
        audios, params, lengths = zip(*batch)
        lengths = torch.tensor(lengths, dtype=torch.long)
    elif len(batch[0]) == 2:
        audios, params = zip(*batch)
        if isinstance(params[0], int):
            lengths = torch.tensor(params, dtype=torch.long)
            params = (None,) * len(batch)
        else:
            lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

    audios_1d = []
    for a in audios:
        if a.dim() == 2 and a.shape[0] == 1:
            audios_1d.append(a.squeeze(0))
        elif a.dim() == 1:
            audios_1d.append(a)
        else:
            audios_1d.append(a.reshape(-1))

    padded = pad_sequence(audios_1d, batch_first=True)
    padded = padded.unsqueeze(1)

    if params[0] is None:
        params_out = None

    elif isinstance(params[0], dict):
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
        if p0.dim() == 3:
            flats = []
            PM = p0.shape[0] * p0.shape[1]
            for p in params:
                flats.append(p.reshape(PM, p.shape[-1]).transpose(0, 1))
            padded_flat = pad_sequence(flats, batch_first=True)
            padded_flat = padded_flat.transpose(1, 2)
            params_out = padded_flat.reshape(len(params), p0.shape[0], p0.shape[1], -1)
        else:
            params_out = torch.stack(list(params), dim=0)

    return padded, params_out, lengths
