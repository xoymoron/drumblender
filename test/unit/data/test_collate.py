import numpy as np
import pytest
import torch

from drumblender.data.collate import pad_audio_params_collate


@pytest.mark.parametrize("length", [np.int64(3), torch.tensor(3)])
def test_collate_recognizes_integer_scalar_audio_lengths(length):
    audio, params, lengths = pad_audio_params_collate([(torch.ones(1, 5), length)])
    assert audio.shape == (1, 1, 5)
    assert params is None
    assert lengths.tolist() == [3]


@pytest.mark.parametrize("length", [0, -1, 6, 2.5, True])
def test_collate_rejects_invalid_valid_length(length):
    with pytest.raises((TypeError, ValueError)):
        pad_audio_params_collate([(torch.ones(1, 5), None, length)])


def test_collate_rejects_stereo_instead_of_concatenating_channels():
    with pytest.raises(ValueError, match="mono"):
        pad_audio_params_collate([(torch.ones(2, 5), 5)])


def test_collate_does_not_reinterpret_different_modal_parameter_axes():
    items = [(torch.ones(1, 5), torch.zeros(3, 2, 4), 5),
             (torch.ones(1, 5), torch.zeros(2, 3, 4), 5)]
    with pytest.raises(ValueError, match="shape"):
        pad_audio_params_collate(items)


def test_collate_rejects_empty_batches():
    with pytest.raises(ValueError, match="empty"):
        pad_audio_params_collate([])
