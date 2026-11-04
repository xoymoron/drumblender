"""Exercise short CQT extraction using real audio and feature caches."""

import json
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts import build_modal_features


def extract(tmp_path, monkeypatch, backend, length, *options):
    source = tmp_path / "source"
    output = tmp_path / "cache"
    source.mkdir(exist_ok=True)
    waveform = 0.5 * np.sin(np.arange(length) * 2 * np.pi * 1500 / 8000)
    sf.write(source / "hit.wav", waveform, 8000, subtype="FLOAT")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_modal_features", "--processed_root", str(source),
            "--out_dir", str(output), "--modal_backend", backend,
            "--sample_rate", "8000", "--hop_length", "32", "--fmin", "1000",
            "--n_bins", "12", "--bins_per_octave", "12", "--num_modes", "1",
            "--min_length", "1", "--no_tqdm", *options,
        ],
    )
    build_modal_features.main()
    metadata = json.loads((output / "metadata.json").read_text())
    return output, metadata


@pytest.mark.parametrize("backend", ["original", "legacy"])
def test_short_cqt_retry_keeps_original_audio_clock(tmp_path, monkeypatch, backend):
    output, metadata = extract(tmp_path, monkeypatch, backend, 100)
    item, = metadata.values()
    waveform, _ = sf.read(output / item["filename"])
    params = torch.load(output / item["feature_file"], weights_only=True)
    assert len(waveform) == item["num_samples"] == item["analysis_num_samples"] == 100
    assert params.shape == (3, 1, 4)
    assert torch.any(params[1] != 0)
    config = json.loads((output / "modal_config.json").read_text())
    assert config["short_cqt_boundary_policy"] == "constant_on_reflect_failure"


@pytest.mark.parametrize("backend", ["original", "legacy"])
def test_long_cqt_is_unchanged_by_short_clip_retry(tmp_path, monkeypatch, backend):
    first = tmp_path / "enabled"
    second = tmp_path / "disabled"
    first.mkdir()
    second.mkdir()
    enabled, enabled_meta = extract(first, monkeypatch, backend, 320)
    disabled, disabled_meta = extract(second, monkeypatch, backend, 320, "--no_pad_short")
    a, = enabled_meta.values()
    b, = disabled_meta.values()
    assert a["analysis_num_samples"] == b["analysis_num_samples"] == 320
    torch.testing.assert_close(
        torch.load(enabled / a["feature_file"], weights_only=True),
        torch.load(disabled / b["feature_file"], weights_only=True),
        rtol=0, atol=0,
    )


def test_explicit_padding_is_saved_with_the_analysis_length(tmp_path, monkeypatch):
    output, metadata = extract(tmp_path, monkeypatch, "original", 100, "--pad_to", "320")
    item, = metadata.values()
    assert sf.info(output / item["filename"]).frames == 320
    assert item["num_samples"] == item["analysis_num_samples"] == 320
