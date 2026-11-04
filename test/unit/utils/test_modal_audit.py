"""Boundary normalization, long-phase accuracy and portable cache identity."""

import numpy as np
import pytest
import torch

from drumblender.synths.modal import modal_synth
from drumblender.utils.modal_gpu_frontend import stft_blocks
from scripts.build_modal_features import stable_id


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cqt_observed_gain_on_complex_tone(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    from scipy.fft import next_fast_len
    from drumblender.utils.modal_gpu_frontend import cqt_coefficients

    rate, count, frequency, width = 8000, 1024, 100, 129
    centers = np.array([0., 511., 1023.])
    audio = (.5 * np.exp(2j*np.pi*frequency*np.arange(count)/rate)).astype(np.complex64)
    coefficients = cqt_coefficients(audio, rate, centers, np.array([frequency]),
                                    np.array([width]), width, next_fast_len(count+width-1),
                                    1, device, 1)
    np.testing.assert_allclose(np.abs(coefficients[:, 0]), 1, atol=2e-6)


def test_existing_metadata_requires_resume(tmp_path, monkeypatch):
    import sys
    import pytest
    from scripts import build_modal_features as builder

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    metadata = output / "metadata.json"
    metadata.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        ["build", "--processed_root", str(source), "--out_dir", str(output)],
    )
    with pytest.raises(FileExistsError, match="resume"):
        builder.main()
    assert metadata.read_text() == "{}"


def test_cpu_and_torch_stft_boundary_gain_agree():
    from drumblender.utils.modal_analysis_new import CQTModalAnalysis

    model = CQTModalAnalysis(8000, short_window=256, long_window=256)
    frames = [np.empty((0, 6), np.float32) for _ in range(2)]
    captured = []
    model._peaks = lambda values, *args: (
        captured.append(values.copy()) or np.empty((0, 6), np.float32)
    )
    model._add_stft_peaks(
        np.ones(1024, np.float32), np.array([0.0, 512.0]), 0, frames, 256, 1
    )
    _, _, expected = next(
        stft_blocks(np.ones(1024, np.float32), np.array([0.0, 512.0]), 256, 2, "cpu")
    )
    np.testing.assert_allclose(captured, expected, atol=2e-6)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Long-tone phase precision check; no files are written."
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rate, count = 48000, 48000 * 20
    for hz in (5000, 12300):
        w = torch.tensor([[[2 * np.pi * hz / rate]]], device=args.device)
        result = modal_synth(w, torch.ones_like(w), count)[0].double()
        reference = torch.sin(
            torch.arange(1, count + 1, device=args.device, dtype=torch.float64)
            * w.double().item()
        )
        for name, region in (("first", slice(0, rate)), ("last", slice(-rate, None))):
            snr = 10 * torch.log10(
                reference[region].square().sum()
                / (result[region] - reference[region]).square().sum()
            )
            print(hz, name, float(snr))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stft_boundary_uses_observed_window_gain(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    audio = np.ones(1024, np.float32)
    _, _, spectra = next(stft_blocks(audio, np.array([0.0, 512.0]), 256, 2, device))
    np.testing.assert_allclose(spectra[:, 0].real, 2, atol=2e-6)


@pytest.mark.parametrize("frequency", [55, 447, 1234])
def test_steady_tone_boundary_amplitude(frequency):
    from drumblender.utils.modal_analysis_new import CQTModalAnalysis

    rate = 48000
    time = np.arange(rate) / rate
    audio = (0.5 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    result = CQTModalAnalysis(rate, num_modes=1).analyze(audio)
    assert 0.9 < result.amplitudes[0, 0] / 0.5 < 1.1


def test_long_float32_phase_has_no_tail_precision_loss():
    rate, seconds = 48000, 20
    frequency = torch.tensor([[[2 * np.pi * 12300 / rate]]], requires_grad=True)
    amplitude = torch.ones_like(frequency, requires_grad=True)
    actual = modal_synth(frequency, amplitude, rate * seconds)
    reference = torch.sin(
        torch.arange(1, rate * seconds + 1, dtype=torch.float64)
        * frequency.detach().double().item()
    )
    error = (actual[0].double() - reference).square()
    assert error[-rate:].mean() < 1e-12
    actual.square().mean().backward()
    assert torch.isfinite(frequency.grad).all()
    assert torch.isfinite(amplitude.grad).all()


def test_cache_id_is_independent_of_path_separator():
    assert stable_id("pack/sub/sample.wav") == stable_id(r"pack\sub\sample.wav")


def test_empty_original_cache_has_centered_frame_count(tmp_path, monkeypatch):
    import sys
    import json
    import soundfile as sf
    from scripts import build_modal_features as builder

    source = tmp_path / "source"
    source.mkdir()
    sf.write(source / "silent.wav", np.zeros(128), 8000)
    output = tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--processed_root",
            str(source),
            "--out_dir",
            str(output),
            "--modal_backend",
            "original",
            "--sample_rate",
            "8000",
            "--fmin",
            "1000",
            "--n_bins",
            "12",
            "--hop_length",
            "32",
            "--no_tqdm",
        ],
    )
    builder.main()
    (item,) = json.loads((output / "metadata.json").read_text()).values()
    assert torch.load(output / item["feature_file"], weights_only=True).shape[-1] == 5


def test_shard_merge_ignores_progress_display_setting(tmp_path, monkeypatch):
    import json
    import sys
    import soundfile as sf
    from scripts import merge_modal_shards

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    for index in range(2):
        name = f"{index}.wav"
        sf.write(source / name, np.zeros(32), 8000)
        shard = output / f"shard_{index}"
        shard.mkdir(parents=True)
        (shard / "audio.wav").write_bytes((source / name).read_bytes())
        torch.save(torch.zeros(3, 1, 2), shard / "feature.pt")
        config = dict(
            processed_root=str(source),
            num_shards=2,
            shard_index=index,
            meta_name="metadata.json",
            num_modes=1,
            sample_rate=8000,
            no_tqdm=bool(index),
        )
        (shard / "modal_config.json").write_text(json.dumps(config))
        (shard / "metadata.json").write_text(
            json.dumps(
                {stable_id(name): dict(filename="audio.wav", feature_file="feature.pt")}
            )
        )
    monkeypatch.setattr(
        sys, "argv", ["merge", "--out_dir", str(output), "--num_shards", "2"]
    )
    merge_modal_shards.main()
    assert len(json.loads((output / "metadata.json").read_text())) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("backend", ["hybrid", "cqt", "stft"])
def test_cuda_boundary_matches_cpu(backend):
    from drumblender.utils.modal_analysis_new import CQTModalAnalysis

    rate = 48000
    audio = (0.5 * np.sin(2 * np.pi * 447 * np.arange(rate) / rate)).astype(np.float32)
    cpu = CQTModalAnalysis(rate, backend=backend, num_modes=1).analyze(audio)
    gpu = CQTModalAnalysis(
        rate, backend=backend, num_modes=1, compute_device="cuda"
    ).analyze(audio)
    np.testing.assert_allclose(
        cpu.amplitudes[:, :3], gpu.amplitudes[:, :3], rtol=0.02, atol=0.005
    )
