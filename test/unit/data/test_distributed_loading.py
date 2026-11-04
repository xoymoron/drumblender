"""Exercise manually sharded loaders through Lightning's data connector."""

from collections import Counter

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loops.utilities import _set_sampler_epoch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import Dataset

from drumblender.data import AudioDataModule


class IndexedAudio(Dataset):
    def __init__(self, size=13):
        self.lengths = [32] * size

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return torch.full((1, 32), float(index)), 32


@pytest.fixture
def distributed_trainer():
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        replace_sampler_ddp=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.strategy.cluster_environment.set_world_size(2)
    yield trainer
    trainer.strategy.cluster_environment.set_global_rank(0)


@pytest.mark.parametrize("use_bucketing", [True, False])
@pytest.mark.parametrize(
    ("stage", "mode"),
    [
        ("train", RunningStage.TRAINING),
        ("val", RunningStage.VALIDATING),
        ("test", RunningStage.TESTING),
    ],
)
def test_distributed_loaders_shard_once_and_keep_equal_steps(
    distributed_trainer, use_bucketing, stage, mode
):
    data = AudioDataModule(batch_size=3, use_bucketing=use_bucketing)
    data.trainer = distributed_trainer
    setattr(data, f"{stage}_dataset", IndexedAudio())
    per_rank = []
    step_counts = []
    for rank in range(2):
        distributed_trainer.strategy.cluster_environment.set_global_rank(rank)
        loader = getattr(data, f"{stage}_dataloader")()
        prepared = distributed_trainer._data_connector._prepare_dataloader(
            loader, mode=mode
        )
        batches = list(prepared)
        step_counts.append(len(batches))
        per_rank.append(
            [int(value) for batch in batches for value in batch[0][:, 0, 0]]
        )

    assert step_counts == [3, 3]
    assert all(len(indices) < 13 for indices in per_rank)
    counts = Counter(index for indices in per_rank for index in indices)
    assert set(counts) == set(range(13))
    # Padding repeats a whole batch for bucketing, one item for ordinary sampling.
    assert sum(counts.values()) == (16 if use_bucketing else 14)


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_distributed_auto_replacement_fails_before_corrupting_batches(
    distributed_trainer, stage
):
    distributed_trainer._accelerator_connector.replace_sampler_ddp = True
    data = AudioDataModule(batch_size=3)
    data.trainer = distributed_trainer
    setattr(data, f"{stage}_dataset", IndexedAudio())
    with pytest.raises(ValueError, match="replace_sampler_ddp=False"):
        getattr(data, f"{stage}_dataloader")()


@pytest.mark.parametrize("use_bucketing", [True, False])
def test_lightning_epoch_update_changes_distributed_training_order(
    distributed_trainer, use_bucketing
):
    data = AudioDataModule(batch_size=4, use_bucketing=use_bucketing)
    data.trainer = distributed_trainer
    data.train_dataset = IndexedAudio(32)
    loader = distributed_trainer._data_connector._prepare_dataloader(
        data.train_dataloader(), mode=RunningStage.TRAINING
    )
    combined = CombinedLoader(loader, mode="max_size_cycle")

    def indices():
        return [int(value) for batch in loader for value in batch[0][:, 0, 0]]

    _set_sampler_epoch(combined, 0)
    first = indices()
    _set_sampler_epoch(combined, 1)
    second = indices()
    _set_sampler_epoch(combined, 0)
    assert first != second
    assert indices() == first
    assert len(first) == len(second) == 16
