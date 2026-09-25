"""Measure offline analysis and whole-clip CPU synthesis, excluding file I/O.

This measures throughput, not audio callback deadlines or plugin latency.
The deterministic fixture contains sound throughout its full duration.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import torch

from drumblender.synths.modal import modal_synth
from drumblender.utils.modal_analysis_new import CQTModalAnalysis


def dense_fixture(seconds, sample_rate):
    """Dense resonances with independent, nonmonotonic amplitude trajectories."""
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    audio = np.zeros(len(time), np.float64)
    frequencies = np.geomspace(45, min(18000, sample_rate * 0.45), 112)
    for index, frequency in enumerate(frequencies):
        envelope = np.exp(-time / (3 + index % 12))
        envelope *= 0.75 + 0.25 * np.cos(2 * np.pi * (0.4 + index % 7) * time)
        audio += envelope * np.sin(2 * np.pi * frequency * time + index * 0.7)
    generator = np.random.default_rng(20260925)
    audio += 0.15 * generator.standard_normal(len(time)) * np.exp(-time / 5)
    return (audio / np.max(np.abs(audio)) * 0.8).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--num_modes", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/modal_new_review/benchmark.json")
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.runs < 1 or args.threads < 1:
        parser.error("seconds, runs, and threads must be positive")
    torch.set_num_threads(args.threads)
    audio = dense_fixture(args.seconds, args.sample_rate)
    analyzer = CQTModalAnalysis(args.sample_rate, num_modes=args.num_modes)
    result = analyzer.analyze(audio)
    cold_seconds = result.analysis_seconds
    analysis_times, synthesis_times = [], []
    for _ in range(args.runs):
        result = analyzer.analyze(audio)
        analysis_times.append(result.analysis_seconds)
        frequencies, amplitudes, phases = [
            torch.from_numpy(array[None]) for array in result.parameters()
        ]
        frequencies = frequencies * (2 * torch.pi / args.sample_rate)
        started = perf_counter()
        with torch.no_grad():
            output = modal_synth(frequencies, amplitudes, len(audio), phases)
        synthesis_times.append(perf_counter() - started)
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite synthesized output")
    summary = dict(
        platform=platform.platform(),
        processor=platform.processor(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        torch=torch.__version__,
        torch_threads=args.threads,
        fft_workers=1,
        fixture="112 dense AM/decaying sinusoids plus noise; no zero-padding",
        duration_seconds=len(audio) / args.sample_rate,
        sample_rate=args.sample_rate,
        selected_modes=len(result.frequencies),
        candidates_before_limit=result.candidates_before_limit,
        cold_analysis_seconds=cold_seconds,
        warm_analysis_seconds=analysis_times,
        synthesis_seconds=synthesis_times,
        median_analysis_rtf=float(np.median(analysis_times) / args.seconds),
        median_synthesis_rtf=float(np.median(synthesis_times) / args.seconds),
        scope="Whole-clip CPU throughput only; not a causal analysis or plugin callback benchmark",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
