# NEW modal extraction on two CUDA GPUs

Run these Linux shell commands from the repository root. The repository and
`../datasets/processed` must already be present on the server. CUDA mode runs
the CQT and two STFT FFT front ends on a GPU; decimation, peak fitting,
tracking, and feature serialization remain on CPU. A GPU can be slower on short
or peak-dense samples. There is no measured 3090 speedup until the server pilot
is run.

## 1. Create and activate an environment

```bash
conda create -n drumblender-modal python=3.10 -y
conda activate drumblender-modal
cd /path/to/drumblender
python -m pip install --upgrade pip
```

Python 3.10 is within this repository's declared `>=3.10,<3.13` range, and
PyTorch 2.7.1 provides Linux CUDA wheels for Python 3.10. Check
the NVIDIA driver with `nvidia-smi`. Install the CUDA 12.6 PyTorch 2.7.1 wheels
only if that driver supports them; the official PyTorch 2.7.1 archive also has
CUDA 11.8 wheels for older compatible drivers.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[modal_new]'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

The final command must report CUDA available and both RTX 3090 cards. The
project install needs network access to PyPI and the PyTorch wheel index. A
system CUDA Toolkit is not required by the binary wheel, but the NVIDIA driver
must work. Do not mix CPU-only and CUDA PyTorch wheels in this environment.

## 2. Check whether CUDA helps before the full run

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_modal_device --processed_root ../datasets/processed --sample_count 16 --output modal_device_timing.json
```

`gpu_over_cpu_ratio` below 1 means the CUDA front end was faster on the sampled
analysis calls. The script samples different WAV lengths, but 16 files cannot
guarantee full-dataset speed or equivalent results for every cymbal. Inspect
the reported CPU/GPU mode counts and a few modal outputs before full export.

## 3. Extract with both GPUs in one command

```bash
python -m scripts.build_modal_features_multi_gpu --gpus 0,1 --processed_root ../datasets/processed --out_dir ../datasets/modal_features/processed_modal_new128 --num_modes 128 --checkpoint_every 100
```

The Python launcher starts two child processes concurrently, assigns one GPU
to each, and gives each process alternating items from the same sorted WAV
list. Each writes to `shard_0` or `shard_1`. The launcher waits for both and
then runs the metadata merge automatically. The one terminal shows status;
per-shard detail is in `shard_0.log` and `shard_1.log` under `--out_dir`.

If a run stops, repeat the **same single command** with `--resume`. The
checkpoint records completed files every 100 new files and at clean exit. A
crash may therefore recompute files since the last checkpoint. If CUDA runs
out of memory, restart with `--gpu_cqt_batch_size 8` and a **new output
directory**: resume refuses changed analysis settings. To inspect the two
worker commands without starting them, append `--dry_run`.

## 4. Use the merged dataset

The automatic merge checks both configurations, every referenced file, and
whether all current processed WAVs appear exactly once. It writes the root
`metadata.json` with `shard_0/...` and `shard_1/...` paths. It does not copy
the audio or feature files. The training dataset can use this root directory
directly; set its expected mode count to 128. Keep both shard directories.

The original CPU path remains `python -m scripts.build_modal_features_new`
without `--compute_device cuda`. GPU output may differ slightly from CPU output
because CUDA and SciPy FFTs have different floating-point rounding. The 3090
throughput and GPU/CPU feature agreement have not been verified on this local
Windows machine, which has no CUDA device available to this workflow.
