"""Provide LightningDataModule wrappers for datasets."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Type

import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset as TorchDataset

from drumblender.data.audio import AudioDataset
from drumblender.data.bucketing import BucketingBatchSampler
from drumblender.data.collate import pad_audio_params_collate

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AudioDataModule(pl.LightningDataModule):
    """Load local preprocessed audio through Lightning.

    Raw preprocessing runs through scripts/preprocess_datasets_pure.py.
    data_dir_unprocessed records source provenance; no downloads run here.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        dataset_class: Type[TorchDataset] = AudioDataset,
        dataset_kwargs: Optional[Dict] = None,
        meta_file="metadata.json",
        data_dir="dataset/audio-only",
        data_dir_unprocessed="dataset-unprocessed/audio-only",
        sample_rate=48000,
        num_samples: Optional[int] = None,
        use_bucketing: bool = True,
        bucket_boundaries: Optional[list[int]] = None,
        drop_last: bool = False,
        seed: int = 20261106,
        skip_prepare_data: bool = True,
        num_buckets: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_cls = dataset_class
        self.dataset_kwargs = dataset_kwargs or {}
        self.meta_file = meta_file
        self.data_dir = Path(data_dir)
        self.data_dir_unprocessed = Path(data_dir_unprocessed)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.use_bucketing = use_bucketing
        self.bucket_boundaries = bucket_boundaries
        self.num_buckets = num_buckets
        self.drop_last = drop_last
        self.seed = seed
        if not skip_prepare_data:
            raise ValueError(
                "Automatic dataset preparation is removed; preprocess audio and "
                "extract modal features with the scripts before training."
            )
        self.skip_prepare_data = skip_prepare_data

    def prepare_data(self) -> None:
        """Require the local dataset and metadata before starting a run."""
        dataset_meta = self.data_dir / self.meta_file
        if not self.data_dir.is_dir() or not dataset_meta.is_file():
            raise FileNotFoundError(
                f"Preprocessed dataset is missing. Expected: {dataset_meta}"
            )

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        args = [self.data_dir, self.meta_file, self.sample_rate, self.num_samples]
        if stage in ("fit", None):
            self.train_dataset = self.dataset_cls(
                *args,
                split="train",
                **self.dataset_kwargs,
            )
            self.val_dataset = self.dataset_cls(
                *args,
                split="val",
                **self.dataset_kwargs,
            )

            # Cache valid lengths when the dataset does not provide them.
            if self.use_bucketing and not hasattr(self.train_dataset, "lengths"):
                try:
                    lengths = []
                    for i in range(len(self.train_dataset)):
                        item = self.train_dataset[i]
                        audio = item[0]
                        valid_length = item[-1] if isinstance(item[-1], int) else None
                        lengths.append(
                            valid_length
                            if valid_length is not None
                            else audio.shape[-1]
                        )
                    self.train_dataset.lengths = lengths
                except Exception:
                    self.train_dataset.lengths = None
        elif stage == "validate":
            self.val_dataset = self.dataset_cls(
                *args,
                split="val",
                **self.dataset_kwargs,
            )
        elif stage == "test":
            self.test_dataset = self.dataset_cls(
                *args,
                split="test",
                **self.dataset_kwargs,
            )

    def _distributed_state(self):
        """Keep all sampler sharding here instead of Lightning's replacement."""
        world_size = 1
        global_rank = 0
        current_epoch = 0
        if getattr(self, "trainer", None) is not None:
            world_size = int(getattr(self.trainer, "world_size", 1) or 1)
            global_rank = int(getattr(self.trainer, "global_rank", 0) or 0)
            current_epoch = int(getattr(self.trainer, "current_epoch", 0) or 0)
            connector = getattr(self.trainer, "_accelerator_connector", None)
            if world_size > 1 and getattr(connector, "replace_sampler_ddp", False):
                raise ValueError(
                    "AudioDataModule shards distributed loaders itself. Set "
                    "Trainer(replace_sampler_ddp=False) to prevent Lightning "
                    "from replacing its duration-aware batch sampler."
                )
        return world_size, global_rank, current_epoch

    def train_dataloader(self):
        lengths = getattr(self.train_dataset, "lengths", None)
        world_size, global_rank, current_epoch = self._distributed_state()

        if self.use_bucketing and lengths is not None:
            batch_sampler = BucketingBatchSampler(
                lengths=lengths,
                batch_size=self.batch_size,
                boundaries=self.bucket_boundaries,
                num_buckets=self.num_buckets,
                shuffle=True,
                drop_last=self.drop_last,
                seed=self.seed,
                num_replicas=world_size,
                rank=global_rank,
            )
            batch_sampler.set_epoch(current_epoch)

            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=pad_audio_params_collate,
                pin_memory=True,
            )

        # Use ordinary shuffled batches when lengths are unavailable.
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            sampler.set_epoch(current_epoch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=pad_audio_params_collate,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return self._evaluation_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._evaluation_dataloader(self.test_dataset)

    def _evaluation_dataloader(self, dataset):
        """Keep every evaluation item and minimize padding with length sorting.

        Distributed evaluation repeats batches/items as needed for equal step
        counts. Use one device when each item must be evaluated exactly once.
        """
        world_size, global_rank, _ = self._distributed_state()
        lengths = getattr(dataset, "lengths", None)
        if self.use_bucketing:
            if lengths is None:
                lengths = []
                for index in range(len(dataset)):
                    item = dataset[index]
                    valid_length = item[-1] if isinstance(item[-1], int) else None
                    lengths.append(
                        valid_length if valid_length is not None else item[0].shape[-1]
                    )
                dataset.lengths = lengths
            return DataLoader(
                dataset,
                batch_sampler=BucketingBatchSampler(
                    lengths,
                    self.batch_size,
                    boundaries=[],
                    shuffle=False,
                    drop_last=False,
                    num_replicas=world_size,
                    rank=global_rank,
                ),
                num_workers=self.num_workers,
                collate_fn=pad_audio_params_collate,
                pin_memory=True,
            )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=False,
                seed=self.seed,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=pad_audio_params_collate,
            pin_memory=True,
            drop_last=False,
        )


class ModalDataModule(AudioDataModule):
    """Load an existing modal feature cache through Lightning.

    Modal extraction settings describe the cache and single-file inference.
    Extract features separately with scripts/build_modal_features.py.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        dataset_class: Type[TorchDataset] = AudioDataset,
        dataset_kwargs: Optional[Dict] = None,
        meta_file="metadata.json",
        data_dir="dataset/modal",
        data_dir_unprocessed="dataset-unprocessed/modal",
        sample_rate=48000,
        num_samples: Optional[int] = None,
        num_modes=1,
        min_length=10,
        threshold=-80.0,
        hop_length=64,
        fmin=20,
        n_bins=96,
        bins_per_octave=12,
        diff_threshold=5.0,
        save_modal_audio=True,
        use_bucketing: bool = True,
        bucket_boundaries: Optional[list[int]] = None,
        drop_last: bool = False,
        seed: int = 20261106,
        skip_prepare_data: bool = True,
        num_buckets: int = 8,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_class=dataset_class,
            dataset_kwargs=dataset_kwargs,
            meta_file=meta_file,
            data_dir=data_dir,
            data_dir_unprocessed=data_dir_unprocessed,
            sample_rate=sample_rate,
            num_samples=num_samples,
            use_bucketing=use_bucketing,
            bucket_boundaries=bucket_boundaries,
            drop_last=drop_last,
            seed=seed,
            skip_prepare_data=skip_prepare_data,
            num_buckets=num_buckets,
        )
        self.num_modes = num_modes
        self.min_length = min_length
        self.threshold = threshold
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.diff_threshold = diff_threshold
        self.save_modal_audio = save_modal_audio
