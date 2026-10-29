# Neural modal control: training and audition

> **2026-10-29 범위 구분:** [현재 구현 목표](research_implementation_goals.md)는 **A. Joint DSP synthesis**와 **B. Modal-conditioned neural synthesis**를 모두 포함한다. 이 문서의 명령과 `modal -> noise -> joint -> patterns`는 기존 oscillator + filtered-noise 모델에만 해당한다. `joint`는 두 모듈을 fine-tune하는 stage이며 새 목표 A의 통합 parameter predictor 구현 완료를 뜻하지 않는다. B의 full-waveform neural decoder와 학습 명령은 아직 구현하지 않았다.

상태: 모델·학습·inference 코드 구현. 로컬 PyTorch가 없어 runtime, DDP, 학습 품질 및 realtime 성능은 미검증. Python 3.10 문법 검사와 source review만 수행했다.

## Noise는 언제 같이 바뀌어야 하나

Modal의 상대 frequency나 beating만 편집하면 noise를 고정하는 것이 유용할 수 있다. 반면 타격감이나 resonance/texture의 결합된 변화를 탐색하면 noise의 envelope와 spectrum도 함께 변하는 편이 자연스러울 수 있다. 모든 modal edit에 동일한 noise 변화가 정답인 것은 아니다.

이 구현은 다음 두 동작을 제공한다.

- `noise_link=0`: 기준 noise filter bands 유지. 기본값이다.
- `noise_link=1`: 편집된 modal hidden context에 따른 noise band 변화 적용. 중간 값은 변화량을 log-amplitude에서 보간한다.

원본 sample만으로 학습하므로 이 연동은 observational correlation에 기반한다. 특정 창의적 modal edit의 정답 noise를 supervision한 것은 아니다. 자동 연동이 좋아졌다는 주장은 청취 후에만 할 수 있다.

## 데이터와 branch 경로

```text
extracted modal P0 ── modal encoder ── joint hidden H ── modal decoder ── P_hat ── ModalSynth ── m
                                           │
                                      detach(context)
                                           │
original x - detach(m) ── DAC-style encoder ── FiLM-conditioned noise head ── NoiseGenerator ── n
                                                                                            │
                                                                          reconstruction = m+n
```

Residual `x-m`에는 noise뿐 아니라 modal 추출/복원 오차가 포함된다. Clean noise ground truth로 부르지 않는다. Noise는 기존 filtered random noise renderer를 사용한다. 별도 문서의 complex-excitation texture 모델은 이번 구현에 포함하지 않았다.

학습에서는 residual과 modal context를 detach한다. Noise loss가 encoder 입력이나 condition을 거쳐 modal representation을 임의로 움직이는 경로를 막고, modal은 자신의 synthesized audio와 parameter anchor에서 gradient를 받는다. 이것만으로 branch 역할 분리가 완벽해지는 것은 아니다.

## 학습 순서

| Stage | 학습하는 것 | Loss / 목적 |
|---|---|---|
| `modal` | Modal encoder/decoder | Extracted modal의 frequency/amplitude/initial phase + rendered modal MR-STFT. 일부 입력을 가리는 masked reconstruction 포함 |
| `noise` | Noise encoder + conditioning head | Modal 고정. `m+n` 대 원본 MR-STFT + 약한 `n` 대 detached residual MR-STFT |
| `joint` | Modal과 noise | 전체 mixture loss에 modal parameter/audio anchor 유지. Modal LR은 noise LR의 0.1배 |
| `patterns` | Temporal SAE만 | 최종 modal hidden feature의 sparse reconstruction. Modal/noise backbone 고정 |

Stage 간 이동은 weights만 읽고 optimizer를 새로 만드는 `--init_from`을 사용한다. 같은 stage의 중단된 학습은 `--resume`으로 optimizer/epoch까지 복원한다. SAE는 최종 backbone에 맞추므로 joint 뒤에 학습한다. Joint를 다시 돌렸으면 patterns도 다시 학습해야 한다.

아무런 noise 없이 modal control부터 보고 싶다면 `modal → patterns`도 가능하다. 두 branch를 사용하는 기본 경로는 위 네 단계다. Epoch 수는 실행 출발값이며 최적 학습 시간이라는 근거는 없다.

Loss 계수와 model width는 `cfg/modal_control.yaml`에서 바꾼다. Training MR-STFT는 512/1024/2048 중 실제 clip에 들어가는 FFT만 쓰며 아주 짧은 clip에는 더 작은 FFT를 쓴다. 평가 metric과 구별한다. 평가 WAV 길이를 늘리는 처리는 하지 않는다.

## Neutral의 정확한 의미

Joint 모델에서 neutral은 **학습된 modal + 그 modal을 기준으로 학습된 noise**다. 기존 extraction-only modal과 같다는 뜻이 아니다. 이전 설계의 extraction cache를 그대로 neutral로 사용하는 방식은 이 통합 학습에서는 변경했다. 학습 때의 modal을 inference에서 다른 값으로 바꾸면 noise residual의 기준이 어긋나기 때문이다.

Control amount 0에서는 이 fitted modal parameter를 직접 반환하고 noise bands도 그대로 쓴다. Nonzero edit는 baseline decoder와 edited decoder의 차이를 기준 modal에 적용한다. Mode 수, observation support, amp=0 규칙은 유지한다.

Inference에서 residual encoder는 **편집 전에 한 번** 실행한다. 편집 후 `x-edited_modal`을 재계산하여 noise에 넣지 않는다. 그런 경로는 요청한 변화를 noise가 상쇄할 수 있다. Edited output에 대해 unchanged source reconstruction loss를 걸지도 않는다.

## 서버 실행: Python 3.10, 3090 두 대

기존 DrumBlender 환경과 dependency를 사용한다. 추가 라이브러리는 없다. 아래 경로는 이제 추출을 완료한 NEW 128-mode cache를 사용한다. 기존 128 modes를 64개로 자르지 않는다.

```bash
conda activate drumblender
export CUDA_VISIBLE_DEVICES=0,1
export MODAL_ROOT=../dataset/modal_features/processed_modal_new128
```

1. Modal representation:

```bash
python -m scripts.train_modal_control --stage modal --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1
```

2. Frozen-modal noise fitting:

```bash
python -m scripts.train_modal_control --stage noise --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1 --init_from runs/modal_control/modal/checkpoints/last.ckpt
```

3. Joint fine-tuning:

```bash
python -m scripts.train_modal_control --stage joint --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1 --init_from runs/modal_control/noise/checkpoints/last.ckpt
```

4. Final-backbone pattern dictionary:

```bash
python -m scripts.train_modal_control --stage patterns --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1 --init_from runs/modal_control/joint/checkpoints/last.ckpt
```

학습 종료 시 best checkpoint 주소를 출력한다. 위 명령은 연결하기 쉬운 `last.ckpt` 예시이며, 실제 stage 이동에는 출력된 best checkpoint를 선택할 수 있다. 동일 stage의 resume은 다음과 같다.

```bash
python -m scripts.train_modal_control --stage joint --feature_root "$MODAL_ROOT" --devices 0,1 --resume runs/modal_control/joint/checkpoints/last.ckpt
```

GPU 한 대는 `--devices 1`, CPU는 `--accelerator cpu --devices 1`이다. `--devices 0,1`은 CUDA_VISIBLE_DEVICES 적용 후의 logical GPU ID다. Batch size 기본값 2는 GPU당 값이며 OOM이면 `--batch_size 1`을 사용한다. 긴 sample은 modal waveform의 differentiable rendering 때문에 메모리가 많이 필요할 수 있다. 전체 길이를 처리하며 몰래 crop하지 않는다.

새 task는 자체 optimizer와 exact-frame collate를 사용하므로 기존 `drumblender fit` config들과 섞지 않고 위 전용 CLI로 실행한다. 기존 reconstruction export CLI 대신 아래 audition CLI를 제공한다.

## Logging / checkpoint

`runs/modal_control/<stage>/`에 저장된다.

- `run_config.json`: config, 실행 인자, data 설정.
- `logs/version_*/metrics.csv`: total loss와 frequency/amplitude/phase, modal audio, mixture/residual audio 또는 SAE loss.
- `checkpoints/last.ckpt`: resume용.
- `checkpoints/epoch*.ckpt`: validation loss가 가장 낮은 checkpoint.

DDP는 Lightning이 실행하며 console progress bar를 사용한다. Validation noise는 고정 seed를 사용해 stochastic fluctuation을 줄인다. SAE의 `feature_fraction`은 batch 내 사용된 feature 비율이며 전체 corpus dead-feature 판정은 아니다.

## Audition

```bash
python -m scripts.audition_modal_control --checkpoint runs/modal_control/patterns/checkpoints/last.ckpt --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --amount 0.5 --noise_link 0 --seed 7 --device cuda:0 --output analysis/modal_control_fixed_noise
```

같은 sample/feature에서 연동을 비교하려면 seed를 유지하고 `--noise_link 1` 및 다른 output 폴더를 사용한다. `--sample_id`로 metadata의 key를, `--feature`로 dictionary ID를 직접 지정할 수 있다. 생략하면 seed로 sample을 뽑고 가장 활성화가 큰 feature를 선택한다. 강한 activation이 유용한 knob라는 뜻은 아니다.

출력:

- `original.wav`: 원본.
- `neutral_audio.wav` / `edited_audio.wav`: 전체 두 branch 결과.
- `neutral_modal_audio.wav` / `edited_modal_audio.wav`: modal만.
- `neutral_noise_audio.wav` / `edited_noise_audio.wav`: noise만.
- `modal_parameters.pt`: extracted/neutral/edited modal tensor.
- `report.json`: sample 이름/metadata, feature ID와 activation 후보, mode 수, amount/link/seed, output peak.

모든 WAV는 공통 amplitude 기준의 float WAV이며 개별 peak normalization을 하지 않는다. 같은 seed로 excitation을 재사용하므로 noise의 무작위 변화를 control 효과로 착각하는 일을 줄인다. 모든 파일은 해당 output 폴더에 같은 이름으로 저장되므로 비교 run에는 다른 output 폴더를 사용한다.

## 구현 범위와 남은 검증

Typed stems, temporal/set interaction, initial phase head, temporal SAE, joint/noise stages, optional linked noise, paired audition을 구현했다. VFM, native plugin, streaming renderer, source에 없는 feature의 additive injection 및 자동 semantic naming은 구현하지 않았다. 현재 feature gain edit는 source에서 활성화되지 않은 pattern을 새로 만들 수 없다.

첫 서버 실행에서 확인할 것은 loss/gradient의 finite 여부, 실제 학습 가능성, short/long sample의 memory, checkpoint transition와 DDP 동작이다. 이후 같은 checkpoint로 neutral fidelity, pattern sweep, fixed/linked noise를 청취한다. Noise 연동이 실패하면 `noise_link=0`으로 유지할 수 있고, counterfactual consistency에는 추가 데이터나 별도 학습 가정이 필요하다.
