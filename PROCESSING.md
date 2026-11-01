# Raw audio preprocessing

Raw samples live in `../rawdata`; the preprocessed WAVs live in
`../dataset/processed`. Both paths are relative to the repository root.

Run these commands from the repository root after installing the project
dependencies on the server. The script also supports direct execution from any
working directory using its absolute path. It does not install packages or
modify environments.

## Raw audio preprocessing

```bash
python scripts/preprocess_datasets_pure.py --input ../rawdata --output ../dataset/processed --rejected ../dataset/rejected --logs ../dataset/logs --device cpu
```

Use separate, non-overlapping directories. Pack/subfolder paths are preserved.
The existing rejection rules, highest-RMS channel selection, resampling and
silence trimming are unchanged. No fade-out or amplitude normalization is added.
Audio decoding/encoding runs on CPU; waveform operations use the selected device.

Sample rate, fixed output length, duration limit, channel selection, silence
thresholds, frame/hop sizes, and rejected-file copying can be changed through
CLI options. The default maximum duration is 20 seconds. A single -65 dB frame
threshold is used for both all-silent rejection and leading-silence trimming.
Trailing silence removal is disabled by default to preserve natural decay; use
`--remove-end-silence` only when a dataset needs explicit tail cleanup.

## Device selection

The preprocessing script accepts these `--device` values:

| Value | Behavior |
| --- | --- |
| `cpu` | CPU processing; the backward-compatible default. |
| `cuda` | Use the current visible CUDA GPU. |
| `cuda:1` | Use visible GPU index 1 (respects `CUDA_VISIBLE_DEVICES`). |
| `auto` | Use CUDA if PyTorch reports it available; otherwise use CPU. |

An explicit CUDA request fails when CUDA or the selected device is unavailable.
`auto` chooses the available device before processing; processing errors are
reported without retrying files on another device.
