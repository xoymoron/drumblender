# Raw audio preprocessing

Run these commands from the repository root after installing the project
dependencies on the server. The script also supports direct execution from any
working directory using its absolute path. It does not install packages or
modify environments.

## Raw audio preprocessing

```bash
python scripts/preprocess_datasets_pure.py --input /data/rawdata --output /data/processed --rejected /data/rejected --logs /data/logs --device cpu
```

Use separate, non-overlapping directories. Pack/subfolder paths are preserved.
The existing rejection rules, highest-RMS channel selection, resampling and
silence trimming are unchanged. No fade-out or amplitude normalization is added.
Audio decoding/encoding runs on CPU; waveform operations use the selected device.

Sample rate, fixed output length, duration limit, channel selection, silence
thresholds, frame/hop sizes, and rejected-file copying can be changed through
CLI options. Their defaults preserve the current raw-processing policy.

## Device selection

The preprocessing script accepts these `--device` values:

| Value | Behavior |
| --- | --- |
| `cpu` | CPU processing; the backward-compatible default. |
| `cuda` | Use the current visible CUDA GPU. |
| `cuda:1` | Use visible GPU index 1 (respects `CUDA_VISIBLE_DEVICES`). |
| `auto` | Use CUDA if PyTorch reports it available; otherwise use CPU. |

An explicit CUDA request fails before processing files when CUDA is unavailable
or its basic kernel check fails. `auto` does not hide an incompatible CUDA build
or retry processing failures on CPU. Each GPU server still needs a compatible
PyTorch/torchaudio build and NVIDIA driver; the Python source is shared.

CUDA does not guarantee faster preprocessing for short files: decoding, disk
I/O and synchronization may dominate. CPU/CUDA floating-point differences can
affect threshold-boundary decisions; bitwise identity is not promised.

## Server verification

With Torch, torchaudio, and pytest installed, run:

```bash
python -m pytest -o addopts= --confcutdir=test/unit test/unit/test_processing_cli.py test/unit/utils/test_processing_devices.py -q
```

These tests cover explicit/automatic device selection, CUDA preflight failures,
and CPU/CUDA resampling and trimming. CUDA cases are skipped on CPU-only hosts.
