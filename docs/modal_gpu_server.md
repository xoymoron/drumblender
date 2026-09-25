# NEW modal extraction on two CUDA GPUs

Run these Linux shell commands from the repository root. Raw WAVs are in
`../rawdata`; preprocessing writes `../dataset/processed`, which must be present
before modal extraction. CUDA mode runs
the CQT and two STFT FFT front ends on a GPU; decimation, peak fitting,
tracking, and feature serialization remain on CPU. A GPU can be slower on short
or peak-dense samples. The user's 16-file RTX 3090 pilot measured a CUDA/CPU
analysis-time ratio of 1.03 for the full-resolution settings.

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
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_modal_device --processed_root ../dataset/processed --sample_count 16 --output modal_device_timing.json
```

`gpu_over_cpu_ratio` below 1 means the CUDA front end was faster on the sampled
analysis calls. The script samples different WAV lengths, but 16 files cannot
guarantee full-dataset speed or equivalent results for every cymbal. Inspect
the reported CPU/GPU mode counts and a few modal outputs before full export.

## 3. Extract with both GPUs in one command

```bash
python -m scripts.build_modal_features_multi_gpu --gpus 0,1 --processed_root ../dataset/processed --out_dir ../dataset/modal_features/processed_modal_new128 --num_modes 128 --checkpoint_every 100
```

The Python launcher starts two child processes concurrently, assigns one GPU
to each, and gives each process alternating items from the same sorted WAV
list. Each writes to `shard_0` or `shard_1`. The launcher displays one tqdm
bar per GPU, then merges metadata automatically. Each shard also writes
`progress.json` (live counts and ETA), `run_stats.json` (mode counts and stage
timings), `failures.jsonl`, and a worker log. The merged output contains a
combined `run_stats.json`. A single CPU or CUDA extractor shows its own tqdm
bar and writes the same diagnostic files.

## Faster extraction with a separate output

The measured 16-file pilot averaged 1.97 seconds per CUDA analysis, implying
about 9.4 hours on one GPU or an ideal 4.7 hours on two GPUs for 17,173 files.
Those figures exclude file I/O and CPU contention, and length-quantile sampling
can over-weight long outliers. To target a shorter run, measure the fast preset
on a larger sample first:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_modal_device --processed_root ../dataset/processed --sample_count 128 --fast --output modal_device_timing_fast.json
python -m scripts.build_modal_features_multi_gpu --gpus 0,1 --processed_root ../dataset/processed --num_modes 128 --fast
```

If the fast benchmark still favors CPU, the same launcher can use separate CPU
processes with identical sharding, tqdm, checkpointing, and merge behavior:

```bash
python -m scripts.build_modal_features_multi_gpu --cpu_workers 2 --processed_root ../dataset/processed --num_modes 128 --fast --out_dir ../dataset/modal_features/processed_modal_fast128_cpu
```

Use a separate output directory for a CPU comparison. The launcher never mixes
CPU and CUDA feature files in one run.

`--fast` uses a 768-sample (16 ms) frame step, skips complex peak fitting,
phase refinement, and short-window remeasurement of isolated HF anchors.
It allows only consecutive-frame track continuation so a missing observation
cannot bridge a longer gap than in the original 256-sample setting.
It retains LF CQT, both STFT views, peak tracking, and the existing mode gates.
Its default output is `../dataset/modal_features/processed_modal_fast128` so it
cannot silently overwrite the full-resolution result. Short modes may be
missed, and frequency and attack curves may differ: inspect representative
spectrograms before using the full export for training. The benchmark's time
projection is analysis-only and assumes the sampled files represent the corpus.
To review individual fast analyses with the existing HTML and residual audio,
run `python -m scripts.inspect_modal_analysis --fast --max_files 8
../dataset/processed/your_pack_name`; it writes `analysis/modal_fast_review` by default.

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
because CUDA and SciPy FFTs have different floating-point rounding. Full-corpus
3090 throughput and GPU/CPU feature agreement remain unverified. One local
4.37-second cymbal WAV took 7.26 seconds at full resolution and 0.90 seconds
with the fast CPU preset; this is a profiling example, not a corpus estimate.
