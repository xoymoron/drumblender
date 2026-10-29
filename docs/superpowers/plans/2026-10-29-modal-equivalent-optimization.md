# Full NEW modal analysis optimization

Goal: speed up the complete NEW algorithm without FAST's reduced frame grid or
skipped refinements. User authorized direct implementation and delegated the
choice of optimization approach. Preserve unrelated working-tree changes.

Design: retain CQT, both STFT windows, all thresholds, observation rules, phase
refinement, short-window envelopes and mode limits. Profile first; vectorize
repeated numerical work while retaining peak ordering and least-squares rules.
No new runtime dependencies. Check outputs against a frozen pre-change analyzer,
including masks, sources, selected tracks and rendered audio. Time the two
versions in the same process, separately from profiling and other CPU work.

- [x] Snapshot current analyzer and profile a real long sample.
- [x] Optimize peak-local contrast, grouped fitting and duplicate comparisons.
- [x] Compare baseline/current complete outputs on real and synthetic inputs.
- [x] Run existing modal regressions and report measured speed/limitations.

Baseline snapshot: analysis/modal_optimization/baseline.py (local benchmark input,
not an alternative production backend). Real sample profile: CQT 0.23 s,
long STFT 2.75 s, short STFT 2.17 s, fitting 3.37 s across both windows. These
are instrumented profile times, not a promised production speedup.
