"""Estimate full-dataset modal extraction time from duration-stratified WAVs.

Only analysis calls are timed: WAV decoding, analyzer initialization, feature
serialization, and plotting are excluded. No features are written.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch

from drumblender.utils.modal_analysis import CQTModalAnalysis as LegacyAnalysis
from drumblender.utils.modal_analysis_new import CQTModalAnalysis as NewAnalysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed_root", type=Path, default=Path("../dataset/processed")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("../dataset/modal_features/processed_modal_flat/metadata.json"),
    )
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--target_count", type=int, default=17000)
    parser.add_argument("--legacy_modes", type=int, default=64)
    parser.add_argument("--new_modes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/modal_new_review/batch_timing.json"),
    )
    args = parser.parse_args()
    if args.sample_count < 8 or args.target_count < 1:
        parser.error("Use at least eight sampled files and a positive target count")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    population = sorted(
        [
            (item["orig_relpath"], int(item["num_samples"]))
            for item in metadata.values()
        ],
        key=lambda row: row[1],
    )
    sample_rate = 48000
    groups = np.array_split(np.arange(len(population)), 8)
    generator = np.random.default_rng(args.seed)
    selected = []
    for group_id, group in enumerate(groups):
        count = max(1, round(args.sample_count * len(group) / len(population)))
        for index in generator.choice(
            group, size=min(count, len(group)), replace=False
        ):
            selected.append((group_id, int(index)))

    torch.set_num_threads(1)
    legacy = LegacyAnalysis(
        sample_rate,
        hop_length=256,
        fmin=20,
        n_bins=240,
        bins_per_octave=24,
        min_length=10,
        num_modes=args.legacy_modes,
        threshold=-80,
        diff_threshold=5,
        verbose=False,
    )
    modern = NewAnalysis(
        sample_rate,
        hop_length=256,
        fmin=20,
        n_bins=240,
        bins_per_octave=24,
        min_length=4,
        num_modes=args.new_modes,
        threshold=-90,
        diff_threshold=5,
    )

    def measure(index):
        relative, expected_samples = population[index]
        waveform, rate = sf.read(args.processed_root / relative, dtype="float32")
        if (
            rate != sample_rate
            or waveform.ndim != 1
            or len(waveform) != expected_samples
        ):
            raise ValueError(f"Unexpected format or metadata for {relative}")
        tensor = torch.from_numpy(waveform[None])
        # Legacy reflect padding cannot handle the shortest valid samples.
        minimum = legacy.cqt.kernel_width // 2 + 1
        if tensor.shape[-1] < minimum:
            tensor = torch.nn.functional.pad(tensor, (0, minimum - tensor.shape[-1]))
        with torch.no_grad():
            started = perf_counter()
            old_result = legacy(tensor)
            old_seconds = perf_counter() - started
        started = perf_counter()
        new_result = modern.analyze(waveform)
        new_seconds = perf_counter() - started
        return dict(
            source=relative,
            duration_seconds=len(waveform) / sample_rate,
            legacy_seconds=old_seconds,
            new_seconds=new_seconds,
            legacy_modes=old_result[0].shape[1],
            new_modes=len(new_result.frequencies),
        )

    # Warm both analyzers without counting initialization or the first call.
    measure(selected[0][1])
    rows = []
    for ordinal, (group_id, index) in enumerate(selected, 1):
        row = measure(index)
        row["duration_group"] = group_id
        rows.append(row)
        print(
            f"{ordinal}/{len(selected)} {row['duration_seconds']:.2f}s audio: "
            f"legacy {row['legacy_seconds']:.3f}s, NEW {row['new_seconds']:.3f}s",
            flush=True,
        )

    group_means = []
    for group_id, group in enumerate(groups):
        subset = [row for row in rows if row["duration_group"] == group_id]
        group_means.append(
            dict(
                population=len(group),
                sampled=len(subset),
                legacy_mean_seconds=float(
                    np.mean([row["legacy_seconds"] for row in subset])
                ),
                new_mean_seconds=float(np.mean([row["new_seconds"] for row in subset])),
            )
        )
    weights = np.array([group["population"] for group in group_means], dtype=float)
    weights /= weights.sum()
    legacy_mean = float(
        np.dot(weights, [group["legacy_mean_seconds"] for group in group_means])
    )
    new_mean = float(
        np.dot(weights, [group["new_mean_seconds"] for group in group_means])
    )
    durations = np.array([length / sample_rate for _, length in population])
    summary = dict(
        population_count=len(population),
        sample_count=len(rows),
        population_duration_quantiles_seconds=np.quantile(
            durations, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
        ).tolist(),
        population_mean_duration_seconds=float(durations.mean()),
        legacy_mode_ceiling=args.legacy_modes,
        new_mode_ceiling=args.new_modes,
        estimated_legacy_seconds_per_file=legacy_mean,
        estimated_new_seconds_per_file=new_mean,
        estimated_new_over_legacy_ratio=new_mean / legacy_mean,
        target_count=args.target_count,
        estimated_legacy_hours=args.target_count * legacy_mean / 3600,
        estimated_new_hours=args.target_count * new_mean / 3600,
        scope="CPU analysis only; excludes WAV I/O, output serialization, initialization, and dataset contention",
        groups=group_means,
        files=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"groups", "files"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
