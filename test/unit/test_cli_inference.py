"""Single-file inference with inline and referenced data configurations."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from drumblender import cli


def test_short_clip_inference_keeps_the_original_sample_and_frame_counts(monkeypatch, mocker):
    class EchoModel(torch.nn.Module):
        def forward(self, audio, params):
            assert params.shape == (1, 3, 1, 4)
            return audio

    settings = dict(sample_rate=8000, num_samples=None, num_modes=1,
                    hop_length=32, fmin=1000, n_bins=12, bins_per_octave=12,
                    min_length=1, threshold=-90.0, diff_threshold=5.0)
    mocker.patch.object(cli.model_utils, "load_model",
                        return_value=(EchoModel(), SimpleNamespace(**settings)))
    monkeypatch.setattr("sys.argv", ["drumblender-synth", "cfg.yaml", "model.ckpt",
                                     "in.wav", "out.wav"])
    audio = (torch.arange(100) * (2 * torch.pi * 1500 / 8000)).sin()[None]
    mocker.patch.object(cli.torchaudio, "load", return_value=(audio, 8000))
    save = mocker.patch.object(cli.torchaudio, "save")
    cli.inference()
    torch.testing.assert_close(save.call_args.args[1], audio)


@pytest.mark.parametrize("referenced", [False, True])
@pytest.mark.parametrize("fixed_length", [None, 4096])
def test_inference_uses_resolved_data_config(
    tmp_path, monkeypatch, mocker, referenced, fixed_length
):
    settings = dict(
        sample_rate=48000, num_samples=fixed_length, num_modes=2, hop_length=256
    )
    data = {"class_path": "drumblender.data.ModalDataModule", "init_args": settings}
    data_path = tmp_path / "data.yaml"
    data_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = tmp_path / "model.yaml"
    config.write_text(
        yaml.safe_dump({"data": data_path.name if referenced else data}),
        encoding="utf-8",
    )

    class EchoModel(torch.nn.Module):
        def forward(self, audio, params):
            assert not self.training
            assert not torch.is_grad_enabled()
            return audio

    model = EchoModel()
    loader = mocker.patch.object(
        cli.model_utils,
        "load_model",
        return_value=(model, SimpleNamespace(**settings)),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["drumblender-synth", str(config), "model.ckpt", "in.wav", "out.wav"],
    )
    source = torch.ones(1, 2048)
    mocker.patch.object(cli.torchaudio, "load", return_value=(source, 48000))
    save = mocker.patch.object(cli.torchaudio, "save")

    class Analyzer:
        def __init__(self, sample_rate, num_modes, hop_length):
            assert (sample_rate, num_modes, hop_length) == (48000, 2, 256)

        def __call__(self, audio):
            return tuple(torch.ones(1, 2, 9) for _ in range(3))

    monkeypatch.setattr(cli, "CQTModalAnalysis", Analyzer)
    cli.inference()

    loader.assert_called_once_with(str(config), "model.ckpt", include_data=True)
    saved = save.call_args.args
    assert saved[0] == "out.wav" and saved[2] == 48000
    expected = (
        source if fixed_length is None else torch.nn.functional.pad(source, (0, 2048))
    )
    torch.testing.assert_close(saved[1], expected)


def test_checkpoint_loader_resolves_nested_data_from_other_directory(
    tmp_path, monkeypatch, mocker
):
    config = Path(__file__).resolve().parents[2] / "cfg/06_modal.yaml"
    checkpoint_loader = mocker.patch.object(
        cli.model_utils.DrumBlender, "load_from_checkpoint", return_value=object()
    )
    monkeypatch.chdir(tmp_path)
    _, data = cli.model_utils.load_model(str(config), "model.ckpt", include_data=True)
    assert data.sample_rate == 48000
    assert data.num_modes == 64
    assert data.num_samples is None
    assert data.dataset_kwargs["split_strategy"] == "sample_pack"
    checkpoint_loader.assert_called_once()
