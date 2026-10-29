"""Compare full NEW analysis to a trusted pre-refactor Python source snapshot.

The reference is imported as executable Python. No FAST settings are used.
Run both versions in this same process; do not time concurrent extraction jobs.
Report exact parameter/mask equality as well as alternating warm-run timings.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf

from drumblender.utils.modal_analysis_new import CQTModalAnalysis, render_modal


def load_reference(path):
    spec = importlib.util.spec_from_file_location("modal_benchmark_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_cases(seconds):
    rate = 48000
    time = np.arange(rate) / rate
    phase = 2 * np.pi * np.cumsum(180 - 100 * time) / rate
    wave = 0.3 * np.sin(phase)
    wave += 0.1 * np.sin(2 * np.pi * 1000 * time)
    wave += 0.1 * np.sin(2 * np.pi * 1020 * time + 0.3)
    burst = (time >= 0.2) & (time < 0.22)
    wave[burst] += 0.4 * np.sin(2 * np.pi * 6500 * time[burst])
    yield "glide_beating_hf_burst", wave.astype(np.float32), rate
    yield "silence", np.zeros(rate // 2, np.float32), rate
    if seconds > 0:
        # Keep the stress signal definition shared with the existing benchmark.
        from scripts.benchmark_modal_analysis import dense_fixture

        yield f"dense_{seconds:g}s", dense_fixture(seconds, rate), rate


def compare(reference, audio, rate, runs):
    models = [reference.CQTModalAnalysis(rate), CQTModalAnalysis(rate)]
    # Warm both implementations before recording timings, then alternate order
    # to reduce systematic cache/thermal bias. Constructor and I/O are excluded.
    results = [model.analyze(audio) for model in models]
    timings = [[], []]
    stages = [[], []]
    for run in range(runs):
        for index in ([0, 1] if run % 2 == 0 else [1, 0]):
            results[index] = models[index].analyze(audio)
            timings[index].append(results[index].analysis_seconds)
            stages[index].append(results[index].stage_seconds)
    fields = {}
    for key in (
        "frequencies",
        "amplitudes",
        "phases",
        "observed",
        "confidence",
        "source",
        "frame_times",
        "scores",
    ):
        old, new = getattr(results[0], key), getattr(results[1], key)
        fields[key] = bool(np.array_equal(old, new))
    for key in ("candidates_before_limit", "candidates_before_score_gate"):
        fields[key] = getattr(results[0], key) == getattr(results[1], key)
    old_wave = reference.render_modal(results[0], len(audio), rate)
    new_wave = render_modal(results[1], len(audio), rate)
    fields["rendered_waveform"] = bool(np.array_equal(old_wave, new_wave))
    old_median, new_median = (float(np.median(values)) for values in timings)
    return dict(
        sample_rate=rate,
        duration_seconds=len(audio) / rate,
        selected_modes=len(results[1].frequencies),
        exact=fields,
        all_exact=all(fields.values()),
        baseline_seconds=timings[0],
        optimized_seconds=timings[1],
        baseline_median_seconds=old_median,
        optimized_median_seconds=new_median,
        speedup=old_median / new_median,
        baseline_stage_seconds=stages[0],
        optimized_stage_seconds=stages[1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dense_seconds", type=float, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/modal_optimization/comparison.json"),
    )
    args = parser.parse_args()
    if args.runs < 1 or args.dense_seconds < 0:
        parser.error("runs must be positive and dense_seconds nonnegative")
    reference = load_reference(args.reference)
    report = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        optimized_sha256=hashlib.sha256(
            Path(sys.modules[CQTModalAnalysis.__module__].__file__).read_bytes()
        ).hexdigest(),
        settings=dict(
            backend="hybrid", fast=False, refine=True, hop_length=256, num_modes=128
        ),
        cases=[],
    )

    def record(name, audio, rate):
        print(f"Analyzing {name}: {len(audio) / rate:.3f}s", flush=True)
        result = compare(reference, audio, rate, args.runs)
        result["name"] = name
        report["cases"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"exact={result['all_exact']}, {result['baseline_median_seconds']:.3f}s -> "
            f"{result['optimized_median_seconds']:.3f}s ({result['speedup']:.2f}x)",
            flush=True,
        )

    for path in args.inputs:
        audio, rate = sf.read(path, dtype="float32")
        if audio.ndim != 1:
            parser.error(f"Expected mono input: {path}")
        record(str(path), audio, rate)
    for name, audio, rate in synthetic_cases(args.dense_seconds):
        record(name, audio, rate)
    if not all(item["all_exact"] for item in report["cases"]):
        raise SystemExit("Output equivalence failed; inspect the saved report")


if __name__ == "__main__":
    main()
