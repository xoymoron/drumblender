# Modal control implementation plan

> **2026-10-29 범위 갱신:** [현재 구현 목표](../../research_implementation_goals.md)에서 **A. Joint DSP synthesis**와 **B. Modal-conditioned neural synthesis**를 모두 채택했다. 아래 체크된 항목과 review record는 기존 modal control/conditioned-noise 코드에만 해당한다. 기존 `joint` stage를 새 목표 A의 완료로 간주하지 않으며, B의 full-waveform neural decoder 구현도 포함하지 않는다. 아래 실행 지침은 과거 구현 기록의 일부이며 A/B 세부 구현의 승인이나 실행 기록이 아니다.

> Execute inline with `superpowers:executing-plans`. The user explicitly approved implementation; preserve the current working tree and unrelated edits.

**Goal:** Implement a trainable modal pattern editor with staged modal/noise training and an optional modal-conditioned noise path.

**Architecture:** Typed modal stems, temporal/set interaction, temporal SAE, anchored parameter edits. Noise learns from a detached baseline residual and modal context; editing caches the source residual encoding. No loss compares edited audio against the unchanged source.

**Tech stack:** Python 3.10+, PyTorch 2.7, Lightning 1.9, existing DAC-style encoder and filtered-noise renderer.

**Spec:** `docs/modal_neural_control_design.md`, with the training addendum below.

## Training addendum

1. `modal`: train absolute modal reconstruction against extracted parameters and rendered modal audio; noise frozen.
2. `noise`: freeze modal backbone, fit noise using raw-mixture spectral loss plus a weak detached residual spectral target.
3. `joint`: low modal LR, retain parameter/modal-audio anchors, detach modal context and residual inputs to the noise encoder. Modal gradients come from its rendered branch and its anchors.
4. `patterns`: freeze the final reconstruction backbone; fit the temporal SAE. Run this AFTER joint training to avoid stale dictionary coordinates.

The neutral reference in the joint model is its fitted modal reconstruction, not the original extraction cache. Zero edits preserve that reconstruction. This explicitly replaces the earlier cache-only inference baseline, so training and inference do not use different decompositions.

Noise linking uses cached source residual features and changed modal context. `noise_link=0` preserves baseline noise bands; `noise_link=1` uses the condition-dependent change. This learns observational correlations, not ground-truth edited-noise behavior. Default is 0. Never recompute source-minus-edited-modal as noise input and never minimize edited-mixture loss against the unchanged source.

## Tasks

- [x] Implement `models/modal_control.py`: encode/decode/reference-delta API, observation masks, inducing attention, initial-phase consistency.
- [x] Implement `models/modal_patterns.py`: temporal SAE and fixed-support feature edits.
- [x] Implement `models/modal_conditioned_noise.py` and explicit noise excitation: fixed reference encoding, modal FiLM, repeatable audition.
- [x] Implement `data/modal_control.py`: explicit audio/frame lengths, no silent truncation of modes, dedicated collate.
- [x] Implement `tasks/modal_control.py`: losses, four stages, parameter freezing, low modal LR during joint fitting, reference cache and editing API.
- [x] Add training and audition CLIs, config, exact staged commands and limitations.
- [x] Review syntax, API wiring, gradient routing, lengths, zero edits and checkpoint transitions. Follow developer restriction: do not add/run tests without a test request; report runtime verification separately.

## Review focus

- Fully silent modes and frames must never create NaNs in attention or loss.
- Each sample must retain its actual modal frame grid; batch padding must not stretch tracks.
- Neutral rendering uses the same decomposition as noise training.
- Stages freeze both parameters and normalization/training behavior; SAE coordinates come from final backbone.
- Noise A/B audition reuses excitation; random noise changes must not be mistaken for control.

## Scope

Implement the modal editor and conditioning adapter using the existing noise renderer. The separately proposed complex-excitation noise architecture, VFM, a native plugin, and full realtime streaming remain separate work. No dataset-scale training is started on this workstation.

## Review record

- Python 3.10 AST parsing passed for eight touched Python modules. Whitespace diff check passed.
- Independent read-only review inspected the new modules and the existing dataset, DAC encoder and synthesizers; no confirmed important defect was reported.
- No tests were added or executed. The available local Python has no PyTorch. Actual tensor execution, checkpoint resume, DDP and learned audio behavior remain unverified.
- Design ruling: neutral control preserves the fitted two-branch baseline, replacing the earlier extraction-only reference, to keep train/inference decomposition consistent.
- Design ruling: noise coupling is optional and defaults to zero; paired edited targets do not exist, so observational conditioning is not presented as validated counterfactual control.
