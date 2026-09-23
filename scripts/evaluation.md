# Reconstruction evaluation

The original DrumBlender test path is `drumblender test`: LightningCLI loads the
checkpoint, `AudioDataModule.test_dataloader()` selects test audio, and
`DrumBlender.test_step()` logs metrics. The original
`scripts/reconstruction_metrics*.sh` and `scripts/reconstruction_modal.sh` run
that path with the paper's data configurations. The original
`scripts/compile_results.py` assembled the resulting `metrics.csv` files into
paper tables. Those historical shell commands are retained for reference;
they use the paper's private dataset and original evaluation protocol.

For this repository's checkpoint reconstruction evaluation, run one export on
the unfiltered test split from the project root:

```bash
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
frequency-band LSD/SC, and log-RMS envelope error. To evaluate only selected
packs, add one or more `--sample-pack-key PACK` options. Pack filtering is
applied after the train/validation/test split, so the same test membership is
used in both cases. Use `--split-manifest PATH` to replay a previous export's
sample IDs and order.

To rebuild reports or rescore previously exported WAV pairs without running
checkpoint inference again:

```bash
python scripts/compile_results.py logs/my_reconstruction_test
python scripts/compile_results.py logs/my_reconstruction_test --recompute
```

`compile_results.py` also accepts a parent directory containing multiple
exported bundles, provided they use the same metric configuration. The export
saves targets by default; `--no-save-target` intentionally skips report
generation because a later WAV-pair rescore would then be impossible.
