"""Compare NEW modal CPU and CUDA analysis on processed WAVs.

This is a server-side pilot before a full extraction. Both modes use the same
tracking settings; only CQT/STFT spectral transforms change compute device.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from drumblender.utils.modal_analysis_new import CQTModalAnalysis
from scripts.build_modal_features import list_wavs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed_root", type=Path, default=Path("../dataset/processed")
    )
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5152845)
    parser.add_argument("--num_modes", type=int, default=128)
    parser.add_argument("--fast", action="store_true", help="Measure the 16 ms fast extraction preset.")
    parser.add_argument("--output", type=Path, default=Path("modal_device_timing.json"))
    args = parser.parse_args()
    if args.sample_count < 2:
        parser.error("--sample_count must be at least 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment.")

    files = list_wavs(args.processed_root)
    if len(files) < args.sample_count:
        raise ValueError("Fewer processed WAVs than requested benchmark samples.")
    durations = sorted((sf.info(path).frames, path) for path in files)
    # Equal-count duration strata avoid always selecting the single longest
    # outlier, which can dominate a small sample's corpus-time projection.
    edges = np.linspace(0, len(durations), args.sample_count + 1).astype(int)
    rng = np.random.default_rng(args.seed)
    indices = [
        rng.integers(edges[index], edges[index + 1])
        for index in range(args.sample_count)
    ]
    selected = [durations[index][1] for index in indices]
    torch.set_num_threads(1)
    options = dict(
        num_modes=args.num_modes,
        hop_length=768 if args.fast else 256,
        refine=not args.fast,
        fast=args.fast,
        max_gap=0 if args.fast else 2,
    )
    cpu = CQTModalAnalysis(48000, compute_device="cpu", **options)
    gpu = CQTModalAnalysis(48000, compute_device="cuda", **options)

    def run(path, analyzer):
        waveform, rate = sf.read(path, dtype="float32")
        if waveform.ndim != 1 or rate != 48000:
            raise ValueError(f"Expected 48 kHz mono preprocessed WAV: {path}")
        started = perf_counter()
        result = analyzer.analyze(waveform)
        if analyzer.compute_device == "cuda":
            torch.cuda.synchronize()
        return (
            perf_counter() - started,
            len(result.frequencies),
            len(waveform) / rate,
            result.stage_seconds,
        )

    # Exclude one-time FFT setup and CUDA context creation from file timings.
    run(selected[0], cpu)
    run(selected[0], gpu)
    results = []
    progress = tqdm(selected, desc="benchmark", unit="file", dynamic_ncols=True)
    for index, path in enumerate(progress, 1):
        cpu_seconds, cpu_modes, seconds, cpu_stages = run(path, cpu)
        gpu_seconds, gpu_modes, _, gpu_stages = run(path, gpu)
        row = dict(
            source=str(path),
            audio_seconds=seconds,
            cpu_seconds=cpu_seconds,
            gpu_seconds=gpu_seconds,
            cpu_modes=cpu_modes,
            gpu_modes=gpu_modes,
            cpu_stages=cpu_stages,
            gpu_stages=gpu_stages,
        )
        results.append(row)
        tqdm.write(
            f"{index}/{len(selected)} {seconds:.2f}s: "
            f"CPU {cpu_seconds:.3f}s, CUDA {gpu_seconds:.3f}s"
        )
        progress.set_postfix(cpu=f"{cpu_seconds:.2f}s", cuda=f"{gpu_seconds:.2f}s")

    summary = dict(
        sample_count=len(results),
        cpu_total_seconds=sum(row["cpu_seconds"] for row in results),
        gpu_total_seconds=sum(row["gpu_seconds"] for row in results),
        scope="Analysis only; excludes file reading, output writing, and two-process contention",
        fast=args.fast,
        sampling="random within equal-count duration strata",
        seed=args.seed,
        dataset_files=len(files),
        files=results,
    )
    summary["gpu_over_cpu_ratio"] = (
        summary["gpu_total_seconds"] / summary["cpu_total_seconds"]
    )
    summary["projected_cpu_hours"] = (
        summary["cpu_total_seconds"] / len(results) * len(files) / 3600
    )
    summary["projected_single_gpu_hours"] = (
        summary["gpu_total_seconds"] / len(results) * len(files) / 3600
    )
    summary["projected_two_gpu_ideal_hours"] = (
        summary["projected_single_gpu_hours"] / 2
    )
    summary["stage_totals_seconds"] = {}
    for device in ("cpu", "gpu"):
        totals = {}
        for row in results:
            for name, seconds in row[f"{device}_stages"].items():
                totals[name] = totals.get(name, 0.0) + seconds
        summary["stage_totals_seconds"][device] = totals
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"CPU {summary['cpu_total_seconds']:.1f}s, "
        f"CUDA {summary['gpu_total_seconds']:.1f}s, "
        f"CUDA/CPU {summary['gpu_over_cpu_ratio']:.2f}; "
        f"projected two GPU ideal {summary['projected_two_gpu_ideal_hours']:.2f}h "
        f"(analysis only; small sample); {args.output}"
    )
    print(
        "CUDA stage totals (s): "
        + ", ".join(
            f"{name}={seconds:.1f}"
            for name, seconds in sorted(
                summary["stage_totals_seconds"]["gpu"].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
    )


if __name__ == "__main__":
    main()
