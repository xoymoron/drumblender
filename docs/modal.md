# Modal analysis and feature caches

Run [audio preprocessing](../PROCESSING.md) first. The hybrid analyzer uses CQT
for low-frequency evidence and STFT windows for higher-frequency evidence and
short envelopes. The default analysis uses 48 kHz mono input, hop 256, and up to
128 modes. The CQT and STFT paths can also be selected individually.

## Extraction

```sh
python -m scripts.build_modal_features --processed_root ../dataset/processed --out_dir ../dataset/modal_features/hybrid128
```

For GPU workers, the launcher assigns each worker a shard and merges their
metadata after successful completion:

```sh
python -m scripts.build_modal_features_multi_gpu --gpus 0,1 --processed_root ../dataset/processed --out_dir ../dataset/modal_features/hybrid128_2gpu
```

Use `--resume` to continue an interrupted extraction with matching settings.
`--checkpoint_every` controls how often feature metadata is written to disk;
it does not load model weights.

## Cache and timing

- `modal_config.json` records the backend, analysis version, sample rate, hop,
  and extraction settings.
- `metadata.json` maps sample IDs to the original path, cached WAV and tensor,
  recorded sample-pack key, audio length, and analysis extent.
- Each feature tensor has shape `[3, modes, frames]`: angular frequency in
  radians/sample, linear amplitude, and initial sine phase in radians. Phase is
  repeated across frames. Silent or unused observations have zero amplitude.
- The stored frame count is `analysis_num_samples // hop_length + 1`. A padded
  analysis extent and the physical target length are tracked separately.
- `JointResynthesisDataset` preserves those clocks and converts frequency
  to FP32 cycles/sample on load, leaving the cache format unchanged. Splits
  are assigned within each recorded pack before selecting any subset.

See the [joint architecture specification](superpowers/specs/2026-11-05-joint-modal-noise-transient-design.md)
for interpolation, oscillator precision, and model inputs.
