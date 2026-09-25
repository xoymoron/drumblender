"""Summarize audio-duration and level distributions in a processed WAV dataset.

The input tree is treated as a collection of sample packs: its first relative
directory component is the pack name. The report never changes audio files.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf


QUANTILES = {"p10": 0.10, "p90": 0.90, "p99": 0.99}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the processed dataset directory and report destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Processed dataset root containing WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive dataset_statistics.json and .md.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=65_536,
        help="Frames decoded per read; lower values reduce peak RAM use.",
    )
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    return args


def list_wavs(root: Path) -> list[Path]:
    """Return all WAV files below ``root`` in deterministic order."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def pack_name(root: Path, path: Path) -> str:
    """Use the first directory below the dataset root as the sample-pack name."""
    relative_path = path.relative_to(root)
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "(root)"


def dbfs(amplitude: float) -> Optional[float]:
    """Convert a non-negative linear amplitude to dBFS, or None for silence."""
    if amplitude <= 0.0:
        return None
    return float(20.0 * math.log10(amplitude))


def inspect_wav(path: Path, block_size: int) -> dict[str, object]:
    """Read one WAV in blocks and return duration, RMS, and peak measurements."""
    with sf.SoundFile(str(path)) as audio_file:
        sample_rate = int(audio_file.samplerate)
        frame_count = len(audio_file)
        channel_count = int(audio_file.channels)
        square_sum = 0.0
        sample_count = 0
        peak = 0.0
        for block in audio_file.blocks(
            blocksize=block_size,
            dtype="float64",
            always_2d=True,
        ):
            square_sum += float(np.square(block).sum())
            sample_count += int(block.size)
            if block.size:
                peak = max(peak, float(np.abs(block).max()))

    rms = math.sqrt(square_sum / sample_count) if sample_count else 0.0
    return {
        "sample_rate": sample_rate,
        "channel_count": channel_count,
        "duration_seconds": frame_count / sample_rate,
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
    }


def numeric_summary(values: Iterable[float]) -> Optional[dict[str, float]]:
    """Return common distribution statistics, or None when no values exist."""
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return None
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        **{
            name: float(np.quantile(array, quantile))
            for name, quantile in QUANTILES.items()
        },
    }


def summarize_records(
    records: list[dict[str, object]], error_count: int
) -> dict[str, object]:
    """Aggregate successfully decoded files into one overall or pack summary."""
    durations = [float(record["duration_seconds"]) for record in records]
    rms_values = [
        float(record["rms_dbfs"])
        for record in records
        if record["rms_dbfs"] is not None
    ]
    peak_values = [
        float(record["peak_dbfs"])
        for record in records
        if record["peak_dbfs"] is not None
    ]
    return {
        "file_count": len(records),
        "decode_error_count": error_count,
        "silent_file_count": sum(record["rms_dbfs"] is None for record in records),
        "total_duration_seconds": float(sum(durations)),
        "duration_seconds": numeric_summary(durations),
        "rms_dbfs": numeric_summary(rms_values),
        "peak_dbfs": numeric_summary(peak_values),
        "sample_rates": {
            str(rate): count
            for rate, count in sorted(
                Counter(int(record["sample_rate"]) for record in records).items()
            )
        },
        "channel_counts": {
            str(channels): count
            for channels, count in sorted(
                Counter(int(record["channel_count"]) for record in records).items()
            )
        },
    }


def report_dataset(root: Path, block_size: int = 65_536) -> dict[str, object]:
    """Scan a processed dataset and return overall and pack-level summaries."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    records: list[dict[str, object]] = []
    records_by_pack: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    errors_by_pack: Counter[str] = Counter()

    for path in list_wavs(root):
        current_pack = pack_name(root, path)
        relative_path = path.relative_to(root)
        try:
            record = inspect_wav(path, block_size)
        except Exception as error:
            errors.append(
                {
                    "path": str(relative_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            errors_by_pack[current_pack] += 1
            continue
        records.append(record)
        records_by_pack[current_pack].append(record)

    pack_names = sorted(set(records_by_pack) | set(errors_by_pack))
    overall = summarize_records(records, len(errors))
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root),
        "wav_files_found": len(records) + len(errors),
        "overall": overall,
        "sample_rates": overall["sample_rates"],
        "channel_counts": overall["channel_counts"],
        "packs": {
            name: summarize_records(records_by_pack[name], errors_by_pack[name])
            for name in pack_names
        },
        "decode_errors": errors,
    }


def format_seconds(seconds: float) -> str:
    """Render a duration as seconds and a readable hours-minutes-seconds value."""
    hours, remainder = divmod(seconds, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return f"{hours:02.0f}:{minutes:02.0f}:{seconds:06.3f} ({total_seconds:.3f} s)"


def format_value(value: Optional[float], suffix: str = "") -> str:
    """Render nullable numeric values consistently in Markdown tables."""
    return "N/A" if value is None else f"{value:.3f}{suffix}"


def format_summary_pair(
    summary: Optional[dict[str, float]], first: str, second: str, suffix: str
) -> str:
    """Render two named statistics from one nullable distribution summary."""
    if summary is None:
        return "N/A"
    first_value = format_value(summary[first], suffix)
    second_value = format_value(summary[second], suffix)
    return f"{first_value} / {second_value}"


def markdown_report(report: dict[str, object]) -> str:
    """Render the JSON report into a concise human-readable Markdown summary."""
    overall = report["overall"]
    assert isinstance(overall, dict)
    duration = overall["duration_seconds"]
    rms = overall["rms_dbfs"]
    peak = overall["peak_dbfs"]
    lines = [
        "# Processed Dataset Statistics",
        "",
        f"- Input: `{report['input_root']}`",
        f"- Generated (UTC): {report['generated_at_utc']}",
        f"- WAV files found: {report['wav_files_found']}",
        f"- Decode errors: {overall['decode_error_count']}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Files | {overall['file_count']} |",
        "| Total duration | "
        f"{format_seconds(float(overall['total_duration_seconds']))} |",
        "| Duration mean / median | "
        f"{format_summary_pair(duration, 'mean', 'median', ' s')} |",
        "| Duration min / max | "
        f"{format_summary_pair(duration, 'min', 'max', ' s')} |",
        "| RMS mean / median | "
        f"{format_summary_pair(rms, 'mean', 'median', ' dBFS')} |",
        "| Peak mean / median | "
        f"{format_summary_pair(peak, 'mean', 'median', ' dBFS')} |",
        f"| Silent files | {overall['silent_file_count']} |",
        f"| Sample rates | {overall['sample_rates']} |",
        f"| Channel counts | {overall['channel_counts']} |",
        "",
        "RMS and peak are calculated per file from the stored processed waveform. "
        "Their overall mean and median therefore weight each file equally; "
        "they are not LUFS measurements.",
        "",
        "## By Pack",
        "",
        "| Pack | Files | Total duration | Duration mean / median | "
        "Duration min / max | RMS mean / median | Peak median | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    packs = report["packs"]
    assert isinstance(packs, dict)
    for name, summary in packs.items():
        assert isinstance(summary, dict)
        duration = summary["duration_seconds"]
        rms = summary["rms_dbfs"]
        peak = summary["peak_dbfs"]
        row = {
            "name": name,
            "files": summary["file_count"],
            "total": format_seconds(float(summary["total_duration_seconds"])),
            "duration_mean_median": format_summary_pair(
                duration, "mean", "median", " s"
            ),
            "duration_min_max": format_summary_pair(duration, "min", "max", " s"),
            "rms_mean_median": format_summary_pair(rms, "mean", "median", " dBFS"),
            "peak_median": (
                format_value(peak["median"], " dBFS") if peak else "N/A"
            ),
            "errors": summary["decode_error_count"],
        }
        lines.append(
            "| {name} | {files} | {total} | {duration_mean_median} | "
            "{duration_min_max} | {rms_mean_median} | {peak_median} | "
            "{errors} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def write_text_atomic(path: Path, content: str) -> None:
    """Write a complete report before atomically replacing an older one."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_report(report: dict[str, object], output_dir: Path) -> None:
    """Write JSON and Markdown versions of a completed dataset report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        output_dir / "dataset_statistics.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomic(output_dir / "dataset_statistics.md", markdown_report(report))


def main(argv: Optional[list[str]] = None) -> None:
    """Run the statistics report from the command line."""
    args = parse_args(argv)
    report = report_dataset(args.input, args.block_size)
    output_dir = args.output_dir.expanduser().resolve()
    write_report(report, output_dir)
    overall = report["overall"]
    assert isinstance(overall, dict)
    print(
        f"[done] WAV files={overall['file_count']}, "
        f"decode errors={overall['decode_error_count']}"
    )
    print(
        "  total duration: "
        f"{format_seconds(float(overall['total_duration_seconds']))}"
    )
    print(f"  report directory: {output_dir}")


if __name__ == "__main__":
    main()
