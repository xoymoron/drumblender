# Paired-WAV evaluation

Run from the repository root. Pass one exported bundle or a directory containing
bundles:

```sh
python -m scripts.compile_results path/to/bundle --device cpu
```

A bundle contains `target/<source_filename>` and `recon/<source_filename>`
with matching sample rates and shapes. The manifest
`evaluation/per_file_loss.csv` identifies the pairs by `source_filename`.
Use `length` when the WAVs include right padding and `sample_pack_key`
to preserve grouping in reports. Pass `--recompute` to rescore the WAV pairs.
The metric settings live in [drumblender_metrics.yaml](../cfg/metrics/drumblender_metrics.yaml).

The evaluator consumes exported WAVs and does not instantiate models.

To generate a bundle from the current joint model, provide its checkpoint and
the same noise bank used for training. The exporter writes float WAVs with
the original common gain and a per-file branch diagnostics CSV.

```sh
python -m scripts.export_joint_resynthesis --checkpoint PATH_TO_JOINT_CHECKPOINT --data-dir PATH_TO_MODAL_CACHE --noise-bank assets/noise_bank_48k.pt --output runs/joint_test --branches
python -m scripts.compile_results runs/joint_test
```

`--branches` writes modal, noise, transient and T-off WAVs from the same
forward pass. The test split and deterministic item offsets are the defaults.
