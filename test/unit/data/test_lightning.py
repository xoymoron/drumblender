"""
Tests for the data.lightning module.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from drumblender.data import AudioDataModule
from drumblender.data import AudioWithParametersDataset
from drumblender.data import ModalDataModule
from drumblender.data.collate import pad_audio_params_collate

TESTED_MODULE = "drumblender.data.lightning"


def test_waveform_only_collation_preserves_valid_lengths():
    audio, params, lengths = pad_audio_params_collate(
        [(torch.ones(1, 5), 5), (torch.ones(1, 3), 3)]
    )
    assert params is None
    torch.testing.assert_close(lengths, torch.tensor([5, 3]))
    torch.testing.assert_close(audio[1], torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]))


@pytest.mark.parametrize("datamodule_cls", [AudioDataModule, ModalDataModule])
def test_local_dataset_requires_metadata(datamodule_cls, tmp_path):
    data = datamodule_cls(data_dir=tmp_path, meta_file="metadata.json")
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        data.prepare_data()


@pytest.mark.parametrize("datamodule_cls", [AudioDataModule, ModalDataModule])
def test_local_dataset_is_ready_without_preprocessing(datamodule_cls, tmp_path):
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    data = datamodule_cls(data_dir=tmp_path, meta_file="metadata.json")
    data.prepare_data()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metadata.json"]


def test_audio_datamodule_init():
    AudioDataModule()


def processed_metadata(filename: str):
    # Test metadata to mock a processed dataset
    data = AudioDataModule()
    expected_filename = Path(data.data_dir).joinpath(data.meta_file)
    if filename.name != expected_filename:
        raise FileNotFoundError

    metadata = {}
    for i in range(100):
        metadata[i] = {
            "filename": f"kick_{i}.wav",
            "sample_pack_key": "pack_a",
            "type": "electro",
        }

    return metadata


@pytest.fixture
def kick_datamodule(fs, mocker):
    # Mock the dataset directory and metadata file
    data = AudioDataModule()
    fs.create_dir(data.data_dir)
    fs.create_file(Path(data.data_dir).joinpath(data.meta_file))
    mocker.patch("drumblender.data.audio.json.load", side_effect=processed_metadata)
    return AudioDataModule()


def test_audio_datamodule_setup_train(kick_datamodule):
    kick_datamodule.setup("fit")
    assert len(kick_datamodule.train_dataset) == 80
    assert len(kick_datamodule.val_dataset) == 10
    with pytest.raises(AttributeError):
        kick_datamodule.test_dataset


def test_audio_datamodule_setup_val(kick_datamodule):
    kick_datamodule.setup("validate")
    assert len(kick_datamodule.val_dataset) == 10
    with pytest.raises(AttributeError):
        kick_datamodule.train_dataset
    with pytest.raises(AttributeError):
        kick_datamodule.test_dataset


def test_audio_datamodule_setup_test(kick_datamodule):
    kick_datamodule.setup("test")
    assert len(kick_datamodule.test_dataset) == 10
    with pytest.raises(AttributeError):
        kick_datamodule.train_dataset
    with pytest.raises(AttributeError):
        kick_datamodule.val_dataset


def test_audio_datamodule_train_data(kick_datamodule, mocker):
    # Test that the train data loader works and returns the correct shape
    kick_datamodule.setup("fit")
    train_loader = kick_datamodule.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    mocker = mocker.patch(
        "drumblender.data.audio.torchaudio.load",
        return_value=(
            torch.rand(1, kick_datamodule.num_samples),
            kick_datamodule.sample_rate,
        ),
    )

    batch = next(iter(train_loader))
    assert batch[0].shape == (
        kick_datamodule.batch_size,
        1,
        kick_datamodule.num_samples,
    )


def test_modal_datamodule_init():
    data = ModalDataModule()
    assert isinstance(data, ModalDataModule)
    assert isinstance(data, AudioDataModule)


def processed_modal_metadata(filename: str):
    # Test metadata to mock a processed dataset
    data = ModalDataModule()
    expected_filename = Path(data.data_dir).joinpath(data.meta_file)
    if filename.name != expected_filename:
        raise FileNotFoundError

    metadata = {}
    for i in range(100):
        metadata[i] = {
            "filename": f"kick_{i}.wav",
            "filename_modal": f"kick_{i}_modal.wav",
            "features": f"kick_{i}.pt",
            "num_samples": data.num_samples,
            "sample_pack_key": "pack_a",
            "type": "electro",
        }

    return metadata


def kick_modal_datamodule(fs, mocker, **kwargs):
    # Mock the dataset directory and metadata file
    data = ModalDataModule(**kwargs)
    fs.create_dir(data.data_dir)
    fs.create_file(Path(data.data_dir).joinpath(data.meta_file))
    mocker.patch(
        "drumblender.data.audio.json.load", side_effect=processed_modal_metadata
    )
    return ModalDataModule(**kwargs)


def test_modal_datamodule_setup_train(fs, mocker):
    dm = kick_modal_datamodule(fs, mocker)
    dm.setup("fit")
    assert len(dm.train_dataset) == 80
    assert len(dm.val_dataset) == 10
    with pytest.raises(AttributeError):
        dm.test_dataset


def test_modal_datamodule_setup_val(fs, mocker):
    dm = kick_modal_datamodule(fs, mocker)
    dm.setup("validate")
    assert len(dm.val_dataset) == 10
    with pytest.raises(AttributeError):
        dm.test_dataset
    with pytest.raises(AttributeError):
        dm.train_dataset


def test_modal_datamodule_setup_test(fs, mocker):
    dm = kick_modal_datamodule(fs, mocker)
    dm.setup("test")
    assert len(dm.test_dataset) == 10
    with pytest.raises(AttributeError):
        dm.val_dataset
    with pytest.raises(AttributeError):
        dm.train_dataset


def test_modal_datamodule_train_data(fs, mocker):
    # Test that the train data loader works and returns the correct shape
    dm = kick_modal_datamodule(fs, mocker)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    _ = mocker.patch(
        "drumblender.data.audio.torchaudio.load",
        return_value=(
            torch.rand(1, dm.num_samples),
            dm.sample_rate,
        ),
    )

    audio_batch, params, lengths = next(iter(train_loader))
    assert params is None
    assert audio_batch.shape == (dm.batch_size, 1, dm.num_samples)
    torch.testing.assert_close(lengths, torch.full((dm.batch_size,), dm.num_samples))


def test_modal_datamodule_audio_param_dataset_train(fs, mocker):
    dm = kick_modal_datamodule(
        fs,
        mocker,
        batch_size=8,
        dataset_class=AudioWithParametersDataset,
        dataset_kwargs={"parameter_key": "features"},
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    # Mock loading audio
    mock_audio_load = mocker.patch(
        "drumblender.data.audio.torchaudio.load",
        return_value=(
            torch.rand(1, dm.num_samples),
            dm.sample_rate,
        ),
    )

    # Mock loading parameters
    mock_feature_load = mocker.patch(
        "drumblender.data.audio.torch.load",
        return_value=torch.rand(3, 4, 10),
    )

    audio_batch, parameters, lengths = next(iter(train_loader))

    assert audio_batch.shape == (dm.batch_size, 1, dm.num_samples)
    torch.testing.assert_close(lengths, torch.full((dm.batch_size,), dm.num_samples))
    assert parameters.shape == (dm.batch_size, 3, 4, 10)

    assert mock_audio_load.call_count == dm.batch_size
    assert mock_feature_load.call_count == dm.batch_size
