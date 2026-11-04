# DrumBlender

Unlabeled percussion reconstruction and timbre-control experiments at 48 kHz.
Train/validation/test splits are made within each complete `sample_pack_key`
(including nested packs), with default ratios of 80/10/10 and seed `20261106`.
The same base seed is used by training, extraction, evaluation, audition and tests.
Epoch and sample seeds are derived from it; training RNG state advances normally.

## Research goals

Two directions were adopted on 2026-10-29:

- **A. Joint DSP synthesis:** jointly predict modal and noise/texture parameters,
  then render and sum their DSP outputs.
- **B. Modal-conditioned neural synthesis:** condition a neural waveform decoder
  on modal and audio/detail representations.

See [implementation goals](docs/research_implementation_goals.md) for the
comparison criteria and remaining work. Both are goals; neither is reported as
completed or quality-validated. The existing CLI's `joint` training stage
fine-tunes the current modal/noise model and does not implement target A.

## Installation

Use an isolated Python 3.10–3.12 environment. Install the CPU development profile
locally or the CUDA 12.6 profile on the two-GPU server:

```bash
python -m pip install -r requirements/local.txt
# On the training server instead:
python -m pip install -r requirements/server.txt
python -m pip check
```

See [environment setup and two-GPU training](docs/environment.md) for commands,
the `.venv` migration, effective batch sizes, and verification limits.

## Data and modal extraction

1. [Preprocess raw audio](PROCESSING.md) from `../rawdata` to
   `../dataset/processed`.
2. [Extract modal features](docs/modal.md#extraction), including parallel workers,
   checkpoint/resume and metadata merging.
3. [Review saved modal features](docs/modal.md#saved-feature-review) without rerunning
   either analyzer.

[cfg/data/custom_all.yaml](cfg/data/custom_all.yaml) selects the full unlabeled
corpus. `cfg/data/custom_pack_*.yaml` selects individual source packs.
Point `data_dir` at the feature cache used for the experiment and match
`expected_num_modes` to its mode count.

Variable-length training keeps each processed clip's full duration
(`num_samples: null`). The default preprocessing rejects clips longer than
20 seconds after silence trimming; it does not truncate them to that limit.
For the numbered ablations and modal-control path, cached waveform and analysis
lengths must agree.

The numbered ablations use duration quantile buckets (eight by default),
shuffle within and across buckets each epoch, and retain incomplete batches.
Evaluation sorts by descending length. Collation right-pads audio and modal
frames independently and returns `(audio, params, lengths, frame_lengths)`;
silent trajectory frames count as valid frames. Encoders and synthesizers
process groups with identical valid audio/frame extents before output padding
is restored. Losses average clip-level values computed on valid audio only.

The modal-control path instead carries lengths, frame counts, and mode counts
in a dictionary, slices each example before processing, and currently uses
ordinary shuffled training batches rather than duration buckets.

Numbered configs disable Lightning's automatic DDP sampler replacement because
their data module shards batches itself. Distributed loading may repeat items
to equalize rank step counts; use one device for evaluation without repeats.

## Training

```bash
drumblender fit -c cfg/05_all_parallel.yaml
```

All training paths use Adam with an initial learning rate of `1e-4` and no
weight decay. ReduceLROnPlateau monitors `validation/loss`, with patience 10,
factor 0.5, and minimum learning rate `6.25e-6`; its threshold uses the PyTorch
default. EarlyStopping monitors the same epoch-aggregated metric, with patience
20 and the default `min_delta=0`. Training runs for at most 150 epochs, with
batch size 8 per GPU, gradient accumulation over 8 batches, and seed `20261106`.
The nominal effective batch size is 64 on one GPU. These settings apply to each
modal-control stage as well as the six numbered ablations.

The six numbered configs retain modal/noise/transient ablations and use the
custom unlabeled corpus. Encoders live in `drumblender/models/`, their YAMLs in
`cfg/models/encoder/`, and training losses in `drumblender/loss.py` with YAMLs in
`cfg/loss/`. Batch size, gradient accumulation, and loss selection are defined
in the configs. The model config supplies the original SoundStream encoders.

The [modal-control training CLI](docs/modal.md#training-and-audition) uses the
`modal → noise → joint → patterns` sequence. That guide distinguishes the
existing implementation from the two research goals above.

## Reconstruction evaluation

Use [the evaluation workflow](scripts/evaluation.md) to export paired
target/reconstruction WAVs and generate whole-test and top-level-pack statistics,
tables and a dashboard.

The Lightning evaluation wrappers use the custom data config:

```bash
bash scripts/reconstruction_metrics.sh /path/to/run model_name
bash scripts/reconstruction_modal.sh /path/to/modal.ckpt
```

## Development

```bash
python -m pip install -e '.[dev,modal,modal_new]'
pre-commit install
ruff check .
pytest
```

`test/` contains functional and research regression tests. Checkpoints, feature
caches and evaluation results are local experiment artifacts.

Ruff checks Python errors, common bug patterns and import ordering; Black remains
the formatter. Both pre-commit and CI run Ruff with the rules in `pyproject.toml`.
Use `ruff check . --diff` to review suggested safe fixes, then `ruff check . --fix`
to apply them. Algorithm correctness, tensor shapes and audio quality still need
functional and research tests.
