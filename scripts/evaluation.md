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

Every export includes `sinusoidal/` with a direct synthesis of the analyzed
modal parameters at the original amplitude and frame timing. The evaluator
scores this folder against the same targets and adds a `Sinusoidal` row below
the model. Its per-clip scores and pack statistics have `_sinusoidal` in their
filenames. A missing sinusoidal folder is allowed for historical bundles.

`--branches` also writes modal, noise, transient and T-off WAVs from the same
forward pass. The `modal/` branch includes learned modal corrections. The test
split and deterministic item offsets are the defaults.

The default table columns are Model, MSS, LSD, LSD low, LSD mid, LSD high, SF,
and MFCC. MSS is the display name for `test/mr_stft`. Text and TeX tables use
two decimal places; CSV and JSON statistics retain their original precision.

For an OG checkpoint, run its exporter inside the OG checkout and pass this
repository's metric configuration. Then use this repository's evaluator:

```sh
python -m scripts.export_reconstructions --config PATH_TO_RESOLVED_OG_CONFIG --checkpoint PATH_TO_OG_CHECKPOINT --data-dir PATH_TO_OG_CQT_CACHE --metrics-config PATH_TO_MAIN_REPO/cfg/metrics/drumblender_metrics.yaml --output PATH_TO_OG_BUNDLE
# Run the next command from the main repository.
python -m scripts.compile_results PATH_TO_OG_BUNDLE
```

OG and joint exports use their respective modal caches and renderers. Each
Sinusoidal row therefore measures that export's analysis and synthesis path.
