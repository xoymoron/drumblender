# DrumBlender

Unlabeled percussion reconstruction and timbre-control experiments at 48 kHz.
Train/validation/test splits and evaluation reports use top-level sample packs.

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

Use Python 3.10–3.12 and the PyTorch/torchaudio versions declared in
[pyproject.toml](pyproject.toml), with a build appropriate for your machine.
Install the project and its modal-analysis dependencies:

```bash
python -m pip install -e '.[modal,modal_new]'
```

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

## Training

```bash
drumblender fit -c cfg/05_all_parallel.yaml
bash run.sh plain dac soundstream
```

The six numbered configs retain modal/noise/transient ablations and use the
custom unlabeled corpus. Encoders live in `drumblender/models/`, their YAMLs in
`cfg/models/encoder/`, and training losses in `drumblender/loss.py` with YAMLs in
`cfg/loss/`. Encoder and loss choices remain selectable through `run.sh`.

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
pytest
```

`test/` contains functional and research regression tests. Checkpoints, feature
caches and evaluation results are local experiment artifacts.
