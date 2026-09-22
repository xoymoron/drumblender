"""CPU/CUDA checks for raw preprocessing on the target GPU servers."""

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from drumblender.utils.audio import preprocess_audio_file
from drumblender.utils.device import check_device, resolve_device


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
]


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_auto_and_explicit_cuda_indices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("auto") == torch.device("cuda:1")
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(ValueError, match="index 2"):
        resolve_device("cuda:2")


@pytest.mark.parametrize("name", ["mps", "invalid"])
def test_unsupported_device_is_rejected(name):
    with pytest.raises(ValueError, match="Use cpu"):
        resolve_device(name)


def test_incompatible_cuda_build_fails_preflight(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no kernel image is available")

    monkeypatch.setattr(torch, "ones", fail)
    with pytest.raises(RuntimeError, match="Cannot execute CUDA kernels"):
        check_device(torch.device("cuda:0"))


@pytest.mark.parametrize("device", DEVICES)
def test_preprocessing_resamples_and_saves_on_selected_device(tmp_path, device):
    sample_rate = 16000
    tone = torch.sin(2 * torch.pi * 440 * torch.arange(sample_rate) / sample_rate)
    source = tmp_path / "source.wav"
    torchaudio.save(source, torch.stack([tone * 0.1, tone * 0.7]), sample_rate)

    outputs = []
    for name, selected_device in (("reference", "cpu"), ("selected", device)):
        output_path = tmp_path / f"{name}.wav"
        preprocess_audio_file(source, output_path, 48000, device=selected_device)
        waveform, output_rate = torchaudio.load(output_path)
        assert output_rate == 48000
        assert waveform.shape[0] == 1
        assert waveform.square().mean().sqrt() > 0.4
        outputs.append(waveform)

    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_preprocessing_preserves_rejections_and_unfaded_tail(tmp_path, device):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    torchaudio.save(source, torch.zeros(1, 1000), 1000)
    with pytest.raises(ValueError, match="silent_all"):
        preprocess_audio_file(source, output, 1000, device=device)

    torchaudio.save(source, torch.full((1, 2000), 0.5), 1000)
    with pytest.raises(ValueError, match="too_long"):
        preprocess_audio_file(source, output, 1000, max_duration_sec=1, device=device)

    waveform = torch.tensor([[0.8, 0.8, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0]])
    torchaudio.save(source, waveform, 1000)
    preprocess_audio_file(
        source,
        output,
        1000,
        frame_size=4,
        hop_size=4,
        min_tail_silence_ms=1.0,
        device=device,
    )
    saved, _ = torchaudio.load(output)
    torch.testing.assert_close(saved, waveform[:, :4], rtol=0, atol=1e-4)
