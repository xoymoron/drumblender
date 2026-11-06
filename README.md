# DrumBlender research checkout

Research code for percussion resynthesis with an analysis-guided modal branch,
a band-based noise branch, and a modal-driven transient correction.

This checkout contains preprocessing, modal feature extraction, joint
reconstruction training, and paired-WAV evaluation. Build one carrier asset
and reuse it across training, resume, and export.

```sh
python -m scripts.build_noise_bank assets/noise_bank_48k.pt
python -m scripts.train_joint_resynthesis --data-dir PATH_TO_MODAL_CACHE --noise-bank assets/noise_bank_48k.pt --output runs/joint
```

The default config uses two CUDA devices, batch size 4 per GPU and 4
accumulation steps. The activation budget of 8192 MiB per GPU is a shape-based
estimate used to choose which clips to checkpoint. Use the
[server environment](docs/environment.md) for
training. The modal cache must contain `modal_config.json` and `metadata.json`.
Use `--audio-dir` if you want to read the matching processed source WAVs.

## Post-hoc knob audition

Use the selected joint checkpoint, its original noise bank, and the matching
modal cache and metadata. Each command writes to a new output directory.
Pass the same `--audio-dir` and `--meta-file` to `cache` and `audition` when
the reconstruction run used non-default paths.

```sh
python -m scripts.control_joint cache --checkpoint JOINT.ckpt --noise-bank BANK.pt --data-dir MODAL_CACHE --cache runs/control/h
python -m scripts.control_joint train --cache runs/control/h --output runs/control/sae
python -m scripts.control_joint audition --checkpoint JOINT.ckpt --noise-bank BANK.pt --data-dir MODAL_CACHE --cache runs/control/h --sae runs/control/sae/best.pt --output runs/control/audition
```

Open `runs/control/audition/index.html` locally and note up to two atom IDs.
The page compares RMS-matched WAVs of three validation hits per atom at
`-1`, `0`, and `+1`. Use `--strong` for the additional `+2` and `+4` renders.
The default shortlist is ten candidates, subject to activation coverage.
SAE training defaults to a 120-minute target and a 180-minute cap including
validation. Its log reports valid frames per second and the validation MSE
and dead-feature count. At FP32 and hop 128, the packed cache uses about
1.29 GiB per hour of source audio before metadata and partial frames.

- [Environment](docs/environment.md)
- [Audio preprocessing](PROCESSING.md)
- [Modal analysis and cache format](docs/modal.md)
- [Paired-WAV evaluation](scripts/evaluation.md)
- [Joint architecture specification](docs/superpowers/specs/2026-11-05-joint-modal-noise-transient-design.md)

The original DrumBlender implementation is credited to Shier et al.,
[*Differentiable Modelling of Percussive Audio with Transient and Spectral
Synthesis*](https://doi.org/10.61782/fa.2023.1093). The retained SoundStream-style
encoder and TCN derive from that implementation.
