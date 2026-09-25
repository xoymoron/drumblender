# NEW modal analysis validation — 2026-09-25

## Scope

This records frontend/synthesizer checks. No network was trained and no claim of
improved end-to-end two-branch reconstruction is made. Spectrograms were visually
inspected; listening evaluation remains for the researcher.

## Regressions and integration

```bash
python -m pytest test/unit/utils/test_modal_analysis_new.py test/unit/utils/test_modal_analysis.py test/unit/synths/test_modes.py --confcutdir=test/unit -o addopts= -q
```

Result: **25 passed**. The isolated test invocation bypasses the unrelated global
W&B fixture and coverage settings. It is not a full project test run.

Coverage includes silence, unequal-mode batches, two simultaneous modes, resolved
beating, delayed starts, gaps that do not count as observations, expired tracks,
more than 64 surviving modes, cached fmin padding, differentiable chunked
synthesis, renderer/frame alignment, strong attack plus quiet tonal tail, a short
HF burst, the lower CQT boundary, and separate default NEW/legacy cache paths.
New regressions verify that a zero-amplitude gap neither invents a frequency
curve in the export nor makes an audible carrier sweep during amplitude fades;
the NumPy reference renderer and Torch synthesizer agree around such a gap.

The cached-padding, quiet-tail, short-HF, and LF-boundary regressions were observed
failing before their corresponding fixes. The short HF check uses a 20 ms,
6.5 kHz burst: the corrected peak amplitude exceeds 0.2 and its half-max width
stays below 55 ms, with a short-window envelope source recorded. Windowing still
limits exact attack/offset recovery.

A two-file dataset export completed with no failures and produced fixed
`[3, 128, frames]` features, `metadata.json`, and `modal_config.json`. A repeated
run with `--resume` skipped both complete files. The feature
builder, inspector, and benchmark have runnable module entry points. Modified
Python files were formatted, and the targeted diff whitespace check passed.

## CPU measurement

- CPU: Intel Core Ultra 5 228V, Windows 11.
- CPU Torch 2.7.1, one Torch thread; SciPy FFT uses one worker.
- Input: 48 kHz, 15 seconds, 112 deterministic resonances with amplitude
  modulation and decays plus noise. Sound spans the entire clip; no silent padding.
- Output: 128 selected tracks.
- Warm analysis, two runs: **20.23, 18.48 seconds**; median RTF **1.290**.
- Torch synthesis, two runs: **2.55, 2.93 seconds**; median RTF **0.183**.
- First analysis call: 20.57 seconds. These timings exclude file I/O.

The final benchmark ran separately from the spectrogram export. Measurements are
from this machine and include normal OS variation. Whole-clip throughput is not
a plugin audio-callback deadline measurement. Analysis is noncausal and still
expensive; a sample-loading cache remains necessary for an interactive plugin.

Validation used an isolated Python 3.13.11 / NumPy 2.4.3 / SciPy 1.17.1 environment
because it was available locally. The project's declared Python 3.10–3.12 and
NumPy 1.26.4 training environment was not exercised in this run.

The reproducible command and raw timings are:

```bash
python -m scripts.benchmark_modal_analysis --seconds 15 --num_modes 128 --runs 2 --threads 1
```

Raw output: `analysis/modal_new_review/benchmark.json`.

### Existing-file extraction estimate

The local processed population has **13,446** WAVs, mean duration **1.386 s**;
its 90th-percentile duration is **3.312 s** and maximum **12.763 s**. A separate,
warmed, duration-stratified sample of 64 WAVs compared the old analyzer capped
at 64 with NEW capped at 128. Duration-stratum weighted means were **1.358 s**
per file for legacy and **0.751 s** for NEW: NEW/legacy **0.553**. If 17,000
target files have a similar duration and complexity mix, analysis calls alone
project to **6.41 h** legacy and **3.55 h** NEW on this CPU. WAV I/O, feature
writing, startup, other CPU load, and training are excluded. A previous 32-file
draw gave **4.84 h** versus **4.10 h**, illustrating the variation from a small
sample. Neither draw is a guaranteed total runtime or a plugin callback test.

The script and file-level timing record are:

```bash
python -m scripts.estimate_modal_batch_time --sample_count 64 --target_count 17000
```

`analysis/modal_new_review/batch_timing.json`.

## Visual inspection outputs

```bash
python -m scripts.inspect_modal_analysis analysis/web_compare_4v4/audio/s01_target.wav analysis/web_compare_4v4/audio/s02_target.wav analysis/web_compare_4v4/audio/s08_target.wav --compare_legacy --legacy_modes 128 --output analysis/modal_new_review
```

Both analyzers receive a mode ceiling of 128. They do not have matched detection
threshold calibration, and actual retained counts can differ. Legacy rendering
also uses the corrected padding behavior. These are modal-only comparisons.

Open `analysis/modal_new_review/index.html` for the three input cases. Each case
has an LF and full-band view, original/legacy/NEW/residual WAVs, and `tracks.npz`.
The two HTML buttons reload the selected sample's existing media or the report
page; fresh analysis requires rerunning the inspector command. The four audio
labels and the four image rows now spell out which signal is being played.
The LF tail truncation seen in the first revision was corrected; the long sample
now retains its approximately 72/117/166 Hz resonances through the tail. The
residual still contains tonal content and phase mismatch, so it is not a validated
noise decomposition. The final audio and visual judgment belongs to the researcher.

## Fresh review fixes

A separate read-only review identified and the implementation addressed:

1. Long-window precedence smeared short HF envelopes: isolated HF anchors now
   use short-window amplitude/phase measured at their fine frequency.
2. Frequencies just above fmin were missed: CQT guard bins and frequency-range
   checks after interpolation preserve boundary peaks.
3. NEW shared the legacy default cache directory: backend-specific output
   defaults and a parser regression now keep them separate.
4. Legacy zero-amplitude patience frames could produce a frequency sweep while
   amplitude interpolated across the same interval. NEW exports the last
   observed frequency as an inactive placeholder, and the synthesizer holds the
   observed endpoint across active/zero transitions. Two active frequency
   observations still interpolate as before. This synth repair also applies to
   legacy cached features.
5. A loose **-40 dB relative track-score** gate removes some low-scoring valid
   tracks before the mode ceiling. This is not a physical-mode classifier. In
   spot checks a tonal cowbell had 46 candidates above the gate from 408 valid
   tracks, while a 4.37-second cymbal had 1,292 above the gate from 6,872;
   the cymbal still hits the 128-mode ceiling. No learned noise branch or
   reconstruction metric was evaluated.

No changes were committed. Existing unrelated metrics/report edits were preserved.
