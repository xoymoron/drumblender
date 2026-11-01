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
pip install -e ".[evaluation]"

python scripts/export_recon_wavs.py \
  --config cfg/05_all_parallel.yaml \
  --ckpt path/to/checkpoint.ckpt \
  --data-config cfg/data/custom_all.yaml \
  --split test \
  --output-dir logs/my_reconstruction_test
```

The export saves paired reconstruction and target WAVs, per-file metrics,
whole-test and top-level-pack statistics, `summary_table.tex`, and
`dashboard.html` in `logs/my_reconstruction_test/evaluation/`. The table has
MR-STFT, LSD, and temporal SF; the detailed statistics also include log-mel,
frequency-band LSD/SC, log-RMS envelope error, MFCC error, and paired timbre
distance. Both new diagnostics appear in per-file CSV, overall/per-pack
statistics, and the dashboard; they do not add columns to the TeX table.
To evaluate only selected
packs, add one or more `--sample-pack-key PACK` options. Pack filtering is
applied after the train/validation/test split, so the same test membership is
used in both cases. Use `--split-manifest PATH` to replay a previous export's
sample IDs and order.

Spectral, envelope, and MFCC metrics score the recorded audio length. The
current MR-STFT setup needs more than 1024 samples; shorter clips raise an
error instead of receiving a score changed by extra silence.

`test/mfcc` is frame-aligned mean absolute error over MFCC coefficients 1–39.
It uses 128 Slaney mel bands over 20–20,000 Hz, a 2048-sample Hann window,
128-sample hop, natural-log mel power, and an orthonormal DCT-II. C0 is
excluded to emphasize spectral shape; RMS-envelope error separately measures
level differences. The transform is constructed once and reused. This score
is raw coefficient MAE, not speech-style MCD in dB, and does not use DTW.

`test/paired_timbre` is cosine distance between CLAP style vectors. Each vector
concatenates channel means and population standard deviations over spatial
tokens from every block in the first three audio stages (10 blocks for this
checkpoint). The representation follows the Huang-style approach studied in
[Tian et al. (2025)](https://arxiv.org/abs/2507.07764), using the frozen
`laion/clap-htsat-unfused` model at the revision pinned in the metric YAML.
This specific checkpoint/preprocessing combination defines our metric; it is
not claimed to reproduce the paper's reported timbre-alignment scores.

CLAP requires a fixed 10-second input. Only this learned metric pads each
short window with zeros; padding is part of its definition and is not assumed
to be score-neutral. Hits are never repeated, gain is not normalized, and long
clips are split into aligned, nonoverlapping windows without dropping the tail.
Window distances are averaged with weights proportional to real sample counts.
The backbone is loaded on the first score, remains frozen in eval mode, and is
shared across bundles during rescoring. Its front-end covers 50–14,000 Hz, so
retain the full-band spectral diagnostics for low- and high-frequency fidelity.

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
when needed. Model weights use the Hugging Face cache and are downloaded on
first use. Rebuilding reports from a valid score cache does not load CLAP.

`compile_results.py` also accepts a parent directory containing multiple
exported bundles, provided they use the same metric configuration. The export
saves targets by default; `--no-save-target` intentionally skips report
generation because a later WAV-pair rescore would then be impossible. After
changing metric code, use `--recompute` to refresh scores from the saved WAVs.
