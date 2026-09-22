"""CLI checks that do not require audio libraries or GPU installations."""

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.preprocess_datasets_pure import Config, parse_args, validate_config


ROOT = Path(__file__).resolve().parents[2]


def test_direct_script_help_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preprocess_datasets_pure.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert "cuda:N" in result.stdout


def test_preprocessing_accepts_paths_options_and_device(tmp_path):
    cfg = parse_args([
        "--input", str(tmp_path / "raw"),
        "--output", str(tmp_path / "processed"),
        "--rejected", str(tmp_path / "rejected"),
        "--logs", str(tmp_path / "logs"), "--device", "cuda:1",
        "--sample-rate", "44100", "--no-mono", "--no-remove-end-silence",
    ])
    assert cfg.raw_root == tmp_path / "raw"
    assert cfg.processed_root == tmp_path / "processed"
    assert cfg.rejected_root == tmp_path / "rejected"
    assert cfg.logs_root == tmp_path / "logs"
    assert cfg.device == "cuda:1"
    assert cfg.num_samples is None
    assert cfg.max_duration_sec == 14.0
    assert cfg.sample_rate == 44100
    assert cfg.mono is False
    assert cfg.remove_end_silence is False


@pytest.mark.parametrize("nested", [False, True])
def test_preprocessing_refuses_outputs_inside_source(tmp_path, nested):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = Config(
        raw_root=raw, processed_root=raw / "output" if nested else raw,
        rejected_root=tmp_path / "rejected", logs_root=tmp_path / "logs",
    )
    with pytest.raises(ValueError, match="must not overlap"):
        validate_config(cfg)


def test_preprocessing_default_device_is_cpu(tmp_path):
    roots = [tmp_path / name for name in ("raw", "processed", "rejected", "logs")]
    roots[0].mkdir()
    config = parse_args([
        "--input", str(roots[0]), "--output", str(roots[1]),
        "--rejected", str(roots[2]), "--logs", str(roots[3]),
    ])
    assert config.device == "cpu"
