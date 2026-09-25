# Modal Analysis Rewrite Implementation Plan

**Goal:** Replace `modal_analysis_NEW.py` with CPU-oriented hybrid modal analysis, preserve real observation filtering, and eliminate frequency sweeps caused by silent padding.

**Architecture:** Use a low-rate FFT CQT for LF and chunked complex STFTs for complementary observations. Track peaks in Hz with bounded gaps, fit local sinusoidal spectra, and export the existing frequency/amplitude/phase interface plus observation metadata. Repair inactive frequencies in the synthesizer so cached legacy features are also safe.

**Spec:** The approved design and the user's implementation request in this task. No exponential or monotonic decay prior; support 15-second samples, configurable mode counts above 64, English documentation, and visual inspection without training.

**Execution:** Implement inline in the user's current checkout. Existing edits to reconstruction metrics and reports are unrelated and must be preserved. No commit is requested.

## Review Focus

- Silent placeholders must not count as peak observations.
- Multiple simultaneous peaks and empty frames must preserve their identities and timestamps.
- Silent frequency padding must not cause an audible downward sweep.
- Close modes, beating, LF glides, and short HF resonances must remain representable.
- Batches with different mode counts and 15-second inputs must have bounded memory use.

## Tasks

- [x] Add regression tests for observation filtering, timing, beating, mode limits, and synthesizer padding; observe their failure.
- [x] Implement the hybrid analyzer and document its numerical conventions and CPU limits.
- [x] Repair inactive frequency handling independently in the synthesizer and cover cached legacy inputs.
- [x] Wire a NEW dataset entry point and add a standalone spectrogram/audio inspection tool.
- [x] Run focused tests, inspect real sample plots, and measure warm CPU analysis/synthesis time on 15-second signals.
- [x] Review the diff, polish English comments and usage documentation, and report measured limits without claiming live streaming support.

## Execution record

- Baseline regressions failed on the old NEW constructor and on cached fmin
  padding in the synthesizer. Replacement and padding repair passed those checks.
- An explicit quiet-tail regression failed with the first attack-relative floor;
  the defaults were relaxed to -90 dBFS and -80 dB relative to input peak.
- Final fresh review identified short HF envelope smearing, a hole immediately
  above the CQT fmin boundary, and shared legacy/NEW default output paths.
  All three were addressed. HF and LF regressions were observed failing before
  their repairs. The builder parser now exposes backend-specific defaults for
  an explicit cache-path regression.
- Focused result: 22 tests passed (NEW, existing legacy analyzer, existing modal
  synthesizer). Formatting and targeted diff whitespace checks passed.
- Execution choice: use the user's checkout as explicitly requested, preserve
  existing unrelated changes, and leave changes uncommitted.
- Scope choice: keep noncausal analysis at sample-load time. Whole-clip synthesis
  throughput is measured separately and does not certify plugin callback safety.
- No end-to-end retraining, listening judgment, or noise-branch validation is
  claimed. The existing sample comparison is a visual inspection artifact.
