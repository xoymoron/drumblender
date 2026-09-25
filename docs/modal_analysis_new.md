# NEW modal analysis

The implementation is `drumblender/utils/modal_analysis_new.py` (lowercase, as
tracked by Git). It replaces the previous median-subtracted CQT tracker. The
class name `CQTModalAnalysis` and the three-parameter feature interface remain.
The existing `modal_analysis.py` stays available as the legacy/ampzero baseline.

## Signal path

1. **Observe multiple resolutions.** The default `hybrid` backend keeps CQT
   from 20 Hz to approximately 700 Hz, at 24 bins/octave. Anti-aliased decimation
   and FFT convolution reduce its CPU cost. Full-rate Hann STFTs with 8192 and
   2048 samples provide complementary narrowband and short-window observations.
   At 48 kHz these windows are about 171 and 43 ms. All three views use one frame
   grid, with a nominal 256-sample hop.
2. **Estimate peaks in the original complex spectra.** Log-magnitude parabolic
   interpolation estimates frequency. Local contrast helps reject weak maxima;
   median subtraction does not alter the reconstructed amplitude. STFT peaks
   receive complex Hann-lobe fits, jointly for groups of up to four close peaks.
   Ill-conditioned fits retain their original observations.
   Isolated HF anchors use the short STFT to remeasure amplitude and phase at
   the fine frequency estimate; crowded pairs retain the fine-window envelope.
   This prevents long-window priority from stretching an isolated short hit.
3. **Combine observations without averaging different modes together.** When
   two resolutions observe the same neighborhood, the narrower-band observation
   takes precedence. A coarse peak cannot replace a resolved pair by their mean.
4. **Track in Hz.** Bounded frequency prediction, a Hz/percentage gate, a soft
   phase cost, and one-to-one greedy matching associate peaks. Missing frames
   retain their timestamps. A track expires after more than two missing frames;
   it cannot be revived after an arbitrary silence. This is not a globally
   optimal assignment algorithm, and ambiguous crossings can still split tracks.
5. **Filter actual observations.** Defaults require four observed frames, a
   two-frame observed streak, and an observed fraction of at least 0.5 within
   the track span. Gaps are never counted as observations. These are permissive
   engineering defaults, not proof of a physical eigenmode. Overlapping analysis
   windows also mean consecutive frames are not independent observations.
6. **Export synthesis parameters.** Small, bounded phase-derived corrections
   refine frequency. Amplitudes retain arbitrary rises, falls, and beating;
   there is no exponential-decay or harmonic prior. An energy/confidence score
   ranks tracks. The default removes tracks with scores more than 40 dB below
   the strongest track in the sample, then applies the mode ceiling. This loose
   relative floor is a heuristic, not a calibrated probability or a learned
   noise/modal separation. Use `min_relative_score_db=None` in Python to disable it.

No NN training or noise-branch modification is included. Waveform residuals in
the inspection report are `original - modal`, including any phase error; they
must not be interpreted as a ground-truth noise target.

## Frequency padding and phases

Observation validity and synthesis padding are separate concerns:

- Missing observations have zero amplitude, false `observed`, zero confidence,
  and source `-1`.
- Missing frames carry the last observed frequency only as a safe placeholder;
  leading frames carry the first. They do not define a frequency curve.
- `drumblender/synths/modal.py` repairs inactive frequency padding from older
  caches. At sample rate, a segment fading from an active frame to zero holds
  the active frequency; a segment rising from zero uses the upcoming active
  frequency. Frequency interpolates only between two active frames. This keeps
  independently interpolated amplitude from making a pitch sweep audible.
  Existing cached features can sound different around these boundaries.
- Phase is a fitted **initial sine phase**, repeated along the frame dimension
  for compatibility. It is not a sequence of freely reset frame phases.
- Frame centers match `ModalSynth`'s `align_corners=False` interpolation. The
  reference renderer uses the same mapping. Frequencies returned by analysis
  are **Hz**; saved training features use **radians/sample**.

`analyze(mono_numpy)` returns `ModalAnalysisResult`, including frequencies,
amplitudes, phases, observation masks, confidence, source IDs, frame times,
scores, candidate count before the mode ceiling, and elapsed time. Source IDs
are 0=CQT, 1=long STFT, 2=short STFT, and 3=long-STFT frequency with a short-STFT
envelope/phase. `analyzer(batch)` retains the tuple interface
with shapes `[B, M, F]`; Torch input returns CPU float32 tensors.

## Defaults and mode counts

NEW defaults to **128 modes maximum**, not 128 required modes. Use 192 or 256
for a denser instrument if visual inspection supports it. The Python API accepts
`num_modes=None` to retain every valid track; dataset exports use a fixed ceiling
for batching. Per-frame candidate work is separately bounded by `max_peaks`
(default `max(256, 2 * num_modes)`). There is no guarantee that more modes improve
the modal/noise decomposition.

The calibrated amplitude floor is the larger of **-90 dBFS** and **-80 dB relative
to the waveform peak**. A stricter attack-relative floor removed audible LF tails
in an early revision and was relaxed. These analysis thresholds differ from the
preprocessing silence threshold. Legacy nnAudio magnitudes have different scaling,
so matching the numeric thresholds does not establish a fair sensitivity match.

Training configurations currently containing `expected_num_modes: 64` must be
changed to the new export ceiling (for example, 128). Otherwise the loader will
truncate the new features. Existing experiment configurations are not silently
changed. Rebuild NEW features in a separate output directory and use their
`modal_config.json` to record the settings; do not mix them with old features.
Also align the data module's `num_modes` and any per-mode parameter bank such as
`ModalAmpParameters.num_modes`. A checkpoint with a 64-mode learned bank is not
automatically a compatible 128-mode checkpoint. The NEW builder defaults to
`../datasets/modal_features/processed_modal_new128`, separate from the legacy path.

## Commands

Run from the repository root in the supported Python 3.10–3.12 environment:

```bash
python -m pip install -e ".[modal_new]"
python -m scripts.build_modal_features_new --processed_root ../datasets/processed --out_dir ../datasets/modal_features/processed_modal_new128 --num_modes 128 --checkpoint_every 100
python -m scripts.inspect_modal_analysis path/to/mono_sample.wav --output analysis/modal_review --num_modes 128
```

The builder stores `metadata.json` every 100 newly processed files and once at
the end. If the run stops, repeat the same builder command with `--resume` to
skip completed feature/WAV pairs. Resume checks `modal_config.json` and refuses
changed analysis settings. The saved `active_modes` count may be lower than the
fixed 128-slot tensor shape. The old dataset is left in its original directory.

For a legacy comparison, additionally install the `modal` extra and append
`--compare_legacy`. It defaults to legacy 64 versus NEW 128; use
`--legacy_modes 128` for an equal-ceiling comparison. This is a frontend comparison,
not an end-to-end trained-checkpoint evaluation. The legacy comparison uses the
repaired synthesizer, so it does not reproduce the old frequency-padding artifact.

The inspection output contains:

- `index.html`: LF/full-band spectrograms and original/modal/residual WAV players.
- `comparison.png`: identical dB reference and color limits within each view.
- `tracks.npz`: Hz trajectories and masks, including silent padding metadata.
- `summary.json`: analysis/render time, selected modes, and unnormalized RMS.
- Float WAVs: no normalization, limiting, or clipping on export.

The HTML has a sample selector and a **reload selected sample** button. Reload
re-reads WAVs and PNG from disk; it does not rerun modal analysis. Rerun the
inspector with the desired WAV(s), then click **reload report** to see the new
sample list. Its four audio cards mean Original, legacy modal-only, NEW
modal-only, and Original minus NEW waveform residual. The residual is not a
trained noise output. Rebuild only the HTML layout from existing summaries with:

```bash
python -m scripts.inspect_modal_analysis --refresh_html --output analysis/modal_new_review
```

`--backend stft` and `--backend cqt` permit frontend comparisons without training.
Dataset export uses `--modal_backend` for the same choice. `--no_refine` disables
complex-lobe fitting and phase-based frequency correction. The historical `.txt`
driver now forwards to the maintained builder; its unused Kalman/masking switches
and parallel-worker CLI are retired rather than silently ignored.

## CPU scope

An optional `--compute_device cuda` path now runs the CQT and STFT spectral
transforms on one CUDA GPU while keeping peak fitting and tracking on CPU.
Two-GPU extraction uses separate file shards and a metadata merge. The complete
Conda, benchmark, extraction, resume, and merge commands are in
`docs/modal_gpu_server.md`. No 3090 timing or GPU-versus-CPU output comparison
has been measured on this local machine.

Analysis is centered and noncausal. A 20 Hz CQT filter at 24 bins/octave still
spans about 1.7 seconds of audio; FFT convolution reduces computation, not the
required context. For a sample-loading plugin, analyze/cache parameters when a
sample is loaded, then render/control the oscillator bank during playback.

STFT work is processed in blocks. The differentiable Torch synthesizer renders
32 modes at a time to bound whole-clip inference workspace; training autograd
retains its required intermediates. The NumPy reference renderer handles one mode
at a time. Neither renderer is a finished stateful plugin audio callback.

For a full-dataset time estimate from the existing processed WAVs and legacy
metadata, use:

```bash
python -m scripts.estimate_modal_batch_time --processed_root ../datasets/processed --target_count 17000 --sample_count 64
```

This times CPU analysis calls, excluding I/O and feature writing. It compares
legacy's 64-mode setting with NEW's 128-mode setting and reports duration strata,
per-file results, and a dataset-scale estimate in JSON.

```bash
python -m scripts.benchmark_modal_analysis --seconds 15 --num_modes 128 --runs 3 --threads 1
```

This produces a deterministic 15-second mixture with 112 resonances, nonmonotonic
amplitudes and noise throughout the clip, and measures analysis and whole-clip
Torch synthesis separately. RTF < 1 means throughput faster than the clip's
duration; it does **not** establish callback latency, polyphonic capacity, or
hard realtime safety. Run the benchmark without other concurrent CPU tasks.

## Known limits

- Finite windows cannot uniquely resolve arbitrarily close/short modes. LF
  onset timing remains blurred by long windows; beating below available frequency
  resolution may appear in a single amplitude envelope.
- Sparse observations, overlapping noise peaks and source changes can fragment
  or misassociate tracks. The mode ceiling can discard real weak structure.
- A magnitude spectrogram cannot verify phase accuracy or residual whiteness.
  Inspect the waveforms by listening as well. Reconstruction improvement for a
  retrained two-branch model has not been established by these frontend checks.
