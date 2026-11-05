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

The default config uses two CUDA devices, batch size 1 per GPU and 16
accumulation steps. Use the [server environment](docs/environment.md) for
training. The modal cache must contain `modal_config.json` and `metadata.json`.
Use `--audio-dir` if you want to read the matching processed source WAVs.

For post-hoc SAE fitting, call `model.eval().requires_grad_(False)` and extract
`ref.h_ref[:, None]` under `torch.no_grad()`. Pass `ref.rho[:, None]` as the
SAE loss weights. Cache the neutral sparse codes and pass them to
`TemporalModalSAE.edit`. Decode the edited sequence with the same reference
to reuse the strike FiLM and carrier offsets. The model returns normalized
branches and their common `gain`.

- [Environment](docs/environment.md)
- [Audio preprocessing](PROCESSING.md)
- [Modal analysis and cache format](docs/modal.md)
- [Paired-WAV evaluation](scripts/evaluation.md)
- [Joint architecture specification](docs/superpowers/specs/2026-11-05-joint-modal-noise-transient-design.md)

The original DrumBlender implementation is credited to Shier et al.,
[*Differentiable Modelling of Percussive Audio with Transient and Spectral
Synthesis*](https://doi.org/10.61782/fa.2023.1093). The retained SoundStream-style
encoder and TCN derive from that implementation.
