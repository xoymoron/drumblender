"""Modal-control batches carry both audio lengths and exact feature lengths."""
from __future__ import annotations

from functools import partial

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from drumblender.data.audio import AudioWithParametersDataset


def collate_modal_control(batch):
    """Pad storage only: the task removes padding before encoding and rendering."""
    audio, params, lengths = zip(*batch)
    for waveform, tracks, length in batch:
        if waveform.ndim != 2 or waveform.shape[0] != 1 or length != waveform.shape[-1] or length < 1:
            raise ValueError("Expected nonempty mono audio and its actual unpadded length.")
        if tracks.ndim != 3 or tracks.shape[0] != 3 or min(tracks.shape[1:]) < 1:
            raise ValueError("Expected modal tracks [3,K,T] with nonempty mode/frame axes.")
        if not torch.isfinite(waveform).all() or not torch.isfinite(tracks).all():
            raise ValueError("Audio and modal features must be finite.")
        active = tracks[1] > 0
        if (tracks[1] < 0).any() or ((tracks[0] <= 0) & active).any() or ((tracks[0] >= torch.pi) & active).any():
            raise ValueError("Require nonnegative amplitudes and active frequencies in (0,pi) radians/sample.")
    max_audio = max(lengths)
    max_frames = max(p.shape[-1] for p in params)
    max_modes = max(p.shape[-2] for p in params)
    return {
        "audio": torch.stack([F.pad(x, (0, max_audio - x.shape[-1])) for x in audio]),
        "params": torch.stack([F.pad(p, (0, max_frames - p.shape[-1],
                                         0, max_modes - p.shape[-2])) for p in params]),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "frame_lengths": torch.tensor([p.shape[-1] for p in params], dtype=torch.long),
        "mode_counts": torch.tensor([p.shape[-2] for p in params], dtype=torch.long),
    }


class ModalControlDataModule(pl.LightningDataModule):
    """Use existing split rules; never truncate modes or rescale padded frames."""

    def __init__(self, feature_root: str, processed_root: str,
                 sample_rate: int = 48000, batch_size: int = 2,
                 num_workers: int = 4, seed: int = 20020420):
        super().__init__()
        self.feature_root, self.processed_root = feature_root, processed_root
        self.sample_rate, self.batch_size = sample_rate, batch_size
        self.num_workers, self.seed = num_workers, seed
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative.")

    def setup(self, stage=None):
        factory = partial(
            AudioWithParametersDataset, data_dir=self.feature_root,
            audio_dir=self.processed_root, meta_file="metadata.json",
            sample_rate=self.sample_rate, num_samples=None, seed=self.seed,
            split_strategy="sample_pack", expected_num_modes=None, normalize=False,
        )
        self.train_dataset = factory(split="train")
        self.val_dataset = factory(split="val")
        if not len(self.train_dataset) or not len(self.val_dataset):
            raise ValueError("Train/validation split is empty; provide enough samples per pack.")

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, collate_fn=collate_modal_control,
            persistent_workers=self.num_workers > 0, pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)
