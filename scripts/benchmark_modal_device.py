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

from drumblender.utils.modal_analysis_new import CQTModalAnalysis
from scripts.build_modal_features import list_wavs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed_root", type=Path, default=Path("../datasets/processed")
    )
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--num_modes", type=int, default=128)
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
    indices = np.linspace(0, len(durations) - 1, args.sample_count).astype(int)
    selected = [durations[index][1] for index in indices]
    torch.set_num_threads(1)
    cpu = CQTModalAnalysis(48000, num_modes=args.num_modes, compute_device="cpu")
    gpu = CQTModalAnalysis(48000, num_modes=args.num_modes, compute_device="cuda")

    def run(path, analyzer):
        waveform, rate = sf.read(path, dtype="float32")
        if waveform.ndim != 1 or rate != 48000:
            raise ValueError(f"Expected 48 kHz mono preprocessed WAV: {path}")
        started = perf_counter()
        result = analyzer.analyze(waveform)
        if analyzer.compute_device == "cuda":
            torch.cuda.synchronize()
        return perf_counter() - started, len(result.frequencies), len(waveform) / rate

    # Exclude one-time FFT setup and CUDA context creation from file timings.
    run(selected[0], cpu)
    run(selected[0], gpu)
    results = []
    for index, path in enumerate(selected, 1):
        cpu_seconds, cpu_modes, seconds = run(path, cpu)
        gpu_seconds, gpu_modes, _ = run(path, gpu)
        row = dict(
            source=str(path),
            audio_seconds=seconds,
            cpu_seconds=cpu_seconds,
            gpu_seconds=gpu_seconds,
            cpu_modes=cpu_modes,
            gpu_modes=gpu_modes,
        )
        results.append(row)
        print(
            f"{index}/{len(selected)} {seconds:.2f}s: "
            f"CPU {cpu_seconds:.3f}s, CUDA {gpu_seconds:.3f}s",
            flush=True,
        )

    summary = dict(
        sample_count=len(results),
        cpu_total_seconds=sum(row["cpu_seconds"] for row in results),
        gpu_total_seconds=sum(row["gpu_seconds"] for row in results),
        scope="Analysis only; excludes file reading, output writing, and two-process contention",
        files=results,
    )
    summary["gpu_over_cpu_ratio"] = (
        summary["gpu_total_seconds"] / summary["cpu_total_seconds"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"CPU {summary['cpu_total_seconds']:.1f}s, "
        f"CUDA {summary['gpu_total_seconds']:.1f}s, "
        f"CUDA/CPU {summary['gpu_over_cpu_ratio']:.2f}; {args.output}"
    )


if __name__ == "__main__":
    main()
