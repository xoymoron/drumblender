"""Check coverage, duration-dependent boundaries, and evaluation batches."""

from collections import Counter

import pytest
import torch
from torch.utils.data import Dataset

from drumblender.data import AudioDataModule, ModalDataModule
from drumblender.data.bucketing import BucketingBatchSampler


def test_default_sampler_keeps_incomplete_buckets():
    sampler = BucketingBatchSampler([1, 2, 3, 10, 11], 2, [5])
    indices = [i for batch in sampler for i in batch]
    assert Counter(indices) == Counter(range(5))


def test_default_boundaries_follow_length_quantiles():
    sampler = BucketingBatchSampler(
        [1, 2, 3, 4, 5, 6, 7, 8], batch_size=2, num_buckets=4
    )
    assert sampler.boundaries == [2, 4, 6]
    assert len(sampler) == 4
    assert Counter(i for batch in sampler for i in batch) == Counter(range(8))


def test_repeated_quantiles_do_not_create_empty_buckets():
    sampler = BucketingBatchSampler([7, 7, 7], batch_size=2, num_buckets=8)
    assert sampler.boundaries == []
    assert sorted(i for batch in sampler for i in batch) == [0, 1, 2]


def test_nonshuffled_batches_are_sorted_by_length():
    sampler = BucketingBatchSampler(
        [20, 1, 19, 2], batch_size=2, boundaries=[], shuffle=False
    )
    assert list(sampler) == [[0, 2], [3, 1]]


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_datamodule_batches_keep_every_item_and_reduce_padding(stage):
    class VariableAudio(Dataset):
        lengths = [20, 1, 19, 2, 3]

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, index):
            length = self.lengths[index]
            return torch.full((1, length), float(index + 1)), length

    data = AudioDataModule(batch_size=2, num_workers=0)
    setattr(data, f"{stage}_dataset", VariableAudio())
    batches = list(getattr(data, f"{stage}_dataloader")())
    ids = [int(value) - 1 for batch in batches for value in batch[0][:, 0, 0]]
    assert Counter(ids) == Counter(range(5))
    if stage != "train":
        assert [batch[2].tolist() for batch in batches] == [[20, 19], [3, 2], [1]]
        allocated = sum(batch[0].shape[0] * batch[0].shape[-1] for batch in batches)
        assert allocated == 47


@pytest.mark.parametrize("data_class", [AudioDataModule, ModalDataModule])
def test_default_datamodule_does_not_truncate_long_examples(data_class, tmp_path):
    class LongAudio(Dataset):
        def __init__(self, data_dir, meta_file, sample_rate, num_samples, **kwargs):
            self.target = num_samples

        def __len__(self):
            return 1

        def __getitem__(self, index):
            length = 100_001
            if self.target is not None:
                length = min(length, self.target)
            return torch.ones(1, length), length

    data = data_class(data_dir=tmp_path, dataset_class=LongAudio)
    data.setup("test")
    batch = next(iter(data.test_dataloader()))
    assert batch[2].tolist() == [100_001]


def test_distributed_sampler_keeps_coverage_and_equal_steps():
    samplers = [
        BucketingBatchSampler([1, 2, 3, 4, 5], 2, [], num_replicas=2, rank=rank)
        for rank in range(2)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert len(batches[0]) == len(batches[1])
    assert set(i for per_rank in batches for batch in per_rank for i in batch) == set(
        range(5)
    )
