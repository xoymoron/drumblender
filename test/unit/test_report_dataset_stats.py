"""Tests for the standalone processed-dataset statistics report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def load_report_module():
    script_path = Path(__file__).parents[2] / "scripts" / "report_dataset_stats.py"
    spec = importlib.util.spec_from_file_location("report_dataset_stats", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_report_dataset_summarizes_processed_packs(tmp_path):
    processed_root = tmp_path / "processed"
    (processed_root / "pack_a").mkdir(parents=True)
    (processed_root / "pack_b").mkdir(parents=True)
    sf.write(
        processed_root / "pack_a" / "kick.wav",
        np.full(480, 0.5, dtype=np.float32),
        480,
    )
    sf.write(
        processed_root / "pack_b" / "snare.WAV",
        np.full(240, 0.25, dtype=np.float32),
        480,
    )

    module = load_report_module()
    report = module.report_dataset(processed_root)

    assert report["overall"]["file_count"] == 2
    assert report["overall"]["duration_seconds"]["mean"] == pytest.approx(0.75)
    assert report["overall"]["duration_seconds"]["median"] == pytest.approx(0.75)
    assert report["overall"]["duration_seconds"]["min"] == pytest.approx(0.5)
    assert report["overall"]["duration_seconds"]["max"] == pytest.approx(1.0)
    assert report["overall"]["total_duration_seconds"] == pytest.approx(1.5)
    assert report["overall"]["rms_dbfs"]["mean"] == pytest.approx(-9.0309, abs=1e-3)
    assert report["overall"]["peak_dbfs"]["median"] == pytest.approx(-9.0309, abs=1e-3)
    assert report["sample_rates"] == {"480": 2}
    assert report["packs"]["pack_a"]["file_count"] == 1
    assert report["packs"]["pack_a"]["total_duration_seconds"] == pytest.approx(1.0)
    assert report["packs"]["pack_b"]["duration_seconds"]["mean"] == pytest.approx(0.5)

    output_dir = tmp_path / "statistics"
    module.write_report(report, output_dir)
    saved_report = json.loads((output_dir / "dataset_statistics.json").read_text())
    assert saved_report["overall"]["total_duration_seconds"] == pytest.approx(1.5)
    assert "Total duration" in (output_dir / "dataset_statistics.md").read_text()
