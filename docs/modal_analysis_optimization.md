# Full NEW optimization — 2026-10-29

This refactor speeds up the full NEW analyzer. Run the existing NEW entry point
without `--fast`; no new mode or dependency is required. CQT, both STFTs, hop 256,
complex fitting, short-window HF envelope measurement, phase refinement, matching
and selection rules are retained. No parameters are retuned to obtain a speedup.

## Why retain the transforms?

A profile of the 7.317-second local target sample spent 0.23 seconds in the CQT
stage and 4.92 seconds in the two STFT stages. Much of the latter was downstream
Python/numerical work: complex peak fitting alone took 3.37 seconds. These are
instrumented profile times, not uninstrumented benchmark results. Removing CQT
would sacrifice LF information while addressing only a small part of this case's
cost. The refactor therefore changes execution of the peak operations first.

## Changes

- Evaluate the frequency-axis median background only at detected peak locations.
  Nearest-edge padding, odd kernel width and the contrast definition are identical.
- Assemble Hann-response offsets for all eligible close-peak groups in a frame,
  then evaluate them in one vectorized call. Keep the same individual `lstsq`
  systems, rcond, conditioning check and amplitude acceptance range.
- Identify eligible groups by boundary indices. Collect fitted coefficients and
  update amplitudes/phases together instead of allocating temporary arrays for
  every group. Groups larger than four still follow the existing behavior.
- Construct cross-view duplicate relations once per frame. Preserve the same
  selection order, strict comparison and greedy suppression. For more than 1024
  candidates, use the original row-at-a-time relation to avoid large quadratic
  allocations; the normal 128-mode setting uses bounded small matrices.
- Replace tiny per-track NumPy clipping calls with scalar bounds, explicitly
  preserving float64 addition for custom bounds under NumPy 2 scalar promotion.

The CPU peak code is also shared by the CUDA FFT frontend. CUDA throughput and
cross-device equality were not measured on this CPU-only local Torch runtime.

## Validation and reproduction

The pre-change source is preserved locally at
`analysis/modal_optimization/baseline.py`. It is a benchmark reference, not a
second production backend. A trusted reference source can be supplied to:

```bash
python -m scripts.benchmark_modal_equivalence --reference analysis/modal_optimization/baseline.py --inputs analysis/web_compare_4v4/audio/s01_target.wav analysis/web_compare_4v4/audio/s02_target.wav analysis/web_compare_4v4/audio/s08_target.wav --runs 3 --output analysis/modal_optimization/final_comparison.json
```

The tool imports that Python reference, warms both versions, alternates execution
order, and reports full-output equality plus individual run times and medians.
It includes a glide/beating/HF-burst signal, silence and a dense 15-second signal.
Run it without concurrent CPU extraction jobs. Source hashes and library versions
are included in the JSON. No GPU speed or full-corpus speed is inferred from it.

The broader comparison in `analysis/modal_optimization/comparison.json` checked
eight real targets plus those three synthetic cases: frequencies, amplitudes,
phases, observation masks, sources, confidence, scores, frame times, candidate
counts and reference-rendered waveform were all exactly equal. The final comparison
rechecks three real cases and the three synthetic cases after the scalar-promotion
guard. Separate small reference comparisons covered duplicate merging with
0/1/256/1024/1100 candidates and complex fitting at 1024/2048/8192 FFT sizes.

Existing modal regressions: **25 passed**. The first run had an unrelated pytest
temporary-directory permission error; the successful run used a workspace-local
temporary directory and disabled pytest's cache provider:

```bash
python -m pytest test/unit/utils/test_modal_analysis_new.py test/unit/utils/test_modal_analysis.py test/unit/synths/test_modes.py --confcutdir=test/unit -o addopts= -p no:cacheprovider --basetemp=analysis/modal_optimization/pytest_tmp_final -q
```

Local measurements use Python 3.13, NumPy 2.4.3, SciPy 1.17.1 on Windows. The
project's supported Python 3.10–3.12 training environment and server GPU hardware
were not exercised. Exact equality is evidence for the checked inputs, not a
claim of exhaustive equality over every possible waveform/configuration.

## Final measured timings

Three alternating warm measurements per implementation, median wall seconds,
same CPU/process, full hybrid NEW with all refinement enabled:

| Input | Before | After | Speedup |
|---|---:|---:|---:|
| s01 target, 0.309 s | 0.517 s | 0.253 s | 2.04x |
| s02 target, 2.209 s | 1.626 s | 0.794 s | 2.05x |
| s08 target, 7.317 s | 4.373 s | 2.335 s | 1.87x |
| Dense synthetic, 15 s | 12.464 s | 6.890 s | 1.81x |

All six final cases (including glide/beating/burst and silence) had exact output
and waveform equality. Short/simple inputs can see much smaller gains: the broad
run's 43 ms input was effectively unchanged at about 31–32 ms. The optimized
15-second case remains an offline analysis workload, not a plugin callback.
Production extraction uses these changes automatically through
`python -m scripts.build_modal_features_new`; omit `--fast` to retain full NEW.
