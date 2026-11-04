# Reconstruction evaluation

The Lightning evaluation path is `drumblender test`: LightningCLI loads the
checkpoint, `AudioDataModule.test_dataloader()` selects test audio, and
`DrumBlender.test_step()` logs metrics. `scripts/reconstruction_metrics.sh` and
`scripts/reconstruction_modal.sh` use that path with the unlabeled custom corpus.
The modal-only wrapper requires a modal checkpoint. Label-specific acoustic,
electronic and instrument evaluations are no longer part of this workflow.

For this repository's checkpoint reconstruction evaluation, run one export on
the unfiltered test split from the project root:

```bash
pip install -e .

python scripts/export_recon_wavs.py \
  --config cfg/05_all_parallel.yaml \
  --ckpt path/to/checkpoint.ckpt \
  --data-config cfg/data/custom_all.yaml \
  --split test \
  --seed 20261106 \
  --synthesis-seed 20261106 \
  --output-dir logs/my_reconstruction_test
```

The export saves paired reconstruction and target WAVs, per-file metrics,
whole-test and top-level-pack statistics, `summary_table.tex`, and
`dashboard.html` in `logs/my_reconstruction_test/evaluation/`. The table has
MR-STFT, LSD, and temporal SF. Detailed statistics also include frequency-band
LSD, log-RMS envelope error, and MFCC error. These
diagnostics appear in per-file CSV, overall/per-pack statistics, and the
dashboard; they do not add columns to the TeX table.
To evaluate only selected packs, add one or more `--sample-pack-key PACK`
options. Pack filtering follows the recorded `sample_pack_key`-local 80/10/10
split, so test membership is unchanged. Use `--split-manifest PATH` to replay
a previous export's sample IDs and order.

The split seed and synthesis base seed are both 20261106. Synthesis derives a
repeatable seed for each sample ID, so pack filtering does not change its noise.

Split membership uses each complete recorded pack identifier, including nested
components. Reports group files by their top-level source folder, so all nested
Splice packs appear under `splice`. Use the same metadata, seed, and split
settings as training. An explicit manifest fixes the selected IDs and order.
Export requires a new or empty output directory to keep runs from mixing;
rebuild an existing bundle's reports with `compile_results.py` instead.

Detailed reports include count, mean, population standard deviation, SEM,
median, P5/P25/P75/P95/P99, minimum, and maximum. SEM uses sample standard
deviation divided by sqrt(N); it is blank/null for N=1, where uncertainty
cannot be estimated. The overall mean weights each file equally, whereas
`pack_macro.csv` weights each top-level pack mean equally. SEM does not account
for dependence among samples within a pack.

Spectral, envelope, and MFCC metrics score the recorded audio length. The
current MR-STFT setup needs more than 1024 samples; shorter clips raise an
error instead of receiving a score changed by extra silence.
Log-RMS uses complete frames plus one final partial frame when needed, averaging
only real samples in that tail frame. This fixes previously ignored tail samples;
rerun `compile_results.py BUNDLE --recompute` for existing scores.

`test/mfcc` is frame-aligned mean absolute error over MFCC coefficients 1–39.
It uses 128 Slaney mel bands over 20–20,000 Hz, a 2048-sample Hann window,
128-sample hop, natural-log mel power, and an orthonormal DCT-II. C0 is
excluded to emphasize spectral shape; RMS-envelope error separately measures
level differences. The transform is constructed once and reused. This score
is raw coefficient MAE, not speech-style MCD in dB, and does not use DTW.

To rebuild reports or rescore previously exported WAV pairs without running
checkpoint inference again:

```bash
python scripts/compile_results.py logs/my_reconstruction_test
python scripts/compile_results.py logs/my_reconstruction_test --recompute
```

For an older bundle, copy the updated metric YAML before recomputing:

```bash
cp cfg/metrics/drumblender_metrics.yaml logs/my_reconstruction_test/configs/evaluation_metrics.yaml
python scripts/compile_results.py logs/my_reconstruction_test --recompute --device cuda
```

Rescoring defaults to CUDA with CPU fallback; use `--device cpu` explicitly
when needed. All evaluation metrics use local signal processing and require
no additional model weights or downloads.

`compile_results.py` also accepts a parent directory containing multiple
exported bundles, provided they use the same metric configuration. The export
saves targets by default; `--no-save-target` intentionally skips report
generation because a later WAV-pair rescore would then be impossible. After
changing metric code, use `--recompute` to refresh scores from the saved WAVs.
