# Transient-aware noise synthesis와 발견 가능한 latent control

작성: 2026-09-25. 상태: 연구 설계 제안. 모델 구현·학습·성능 검증 전이다.

## 1. 결정과 연구 질문

기존 sample을 분석해서 재구성하고, 그 sample의 정체성을 유지하면서 새로운 timbre를 탐색하는 악기를 만든다. 첫 구현은 두 audible branches를 유지한다.

\[
\hat x(u_m,u_n)=m(P(u_m))+n(z(u_n)).
\]

modal은 coherent resonances를 담당하고, noise는 stochastic texture뿐 아니라 broadband transient의 coherent microstructure까지 표현한다. 여기서 noise는 물리적으로 순수한 random process라는 뜻이 아니다. 더 정확한 내부 이름은 `texture`다. 별도 transient waveform을 더하지 않고, texture 내부의 excitation과 spectral envelope를 곱해서 하나의 waveform을 합성한다.

핵심 가설은 **빠른 continuous representation + source-conditioned complex excitation + 명시적인 spectral shaping**이 기존 filtered random noise보다 percussive reconstruction에 적합하며, 그 representation에서 사후적으로 유용한 control 방향을 찾을 수 있다는 것이다. 두 branch면 충분하다는 보장도, 모든 artifact가 사라진다는 보장도 없다.

이 모델은 explicit spectral DSP를 포함하는 hybrid neural synthesizer다. excitation이 충분히 자유로우면 일반 complex spectrogram autoencoder와 가까워진다. 따라서 `A=1`인 decoder와 반드시 비교해서 source–filter factorization이 실제로 fidelity/control에 도움이 되는지 검증한다. DSP 이름을 붙이는 것 자체를 기여로 삼지 않는다.

## 2. 현재 구현에서 확인한 것

| 현재 위치 | 확인 내용 | 설계에 미치는 영향 |
|---|---|---|
| `drumblender/upgrades/encoders/dac_style.py`, `cfg/upgrades/encoders/noise_dac_lstm_style.yaml` | DAC-inspired conv, stride product 128, 2-layer LSTM, 128-band output | pretrained DAC codec나 VQ가 아니다. encoder 이름만 교체해서 해결할 문제가 아니다. |
| `drumblender/synths/noise.py` | 예측 band magnitude로 impulse response를 만들고 매 forward 새로운 uniform noise를 filtering/overlap-add | seed 고정은 repeatability를 주지만 원음의 noise realization/phase를 복원하지 않는다. |
| `cfg/synths/noise.yaml` | window 256, implementation hop은 window/2 | control-rate와 synthesis 제약을 함께 봐야 한다. |
| `cfg/05_all_parallel.yaml` | modal, modal을 처리하는 parallel transient TCN, 별도로 더하는 noise | 현재 transient는 onset에만 제한된 branch가 아니다. encoder input과 synthesis wiring도 구분해야 한다. |
| `drumblender/tasks/drumblender.py` | public forward는 waveform; 최종 waveform 중심 loss | 새 task에 auxiliary outputs/loss 경로가 필요하다. public forward는 유지한다. |
| `drumblender/utils/modal_analysis_new.py`, `docs/modal_analysis_new.md` | hybrid CQT/STFT, confidence/observed/source metadata, 기본 최대 128 modes | metadata를 공유하되 residual을 clean noise target이라고 부르면 안 된다. |
| `scripts/build_modal_features.py` | 저장 feature는 `[3,M,F]` tensor | 분석 metadata가 학습 feature에 아직 연결되지 않았다. |
| `drumblender/data/collate.py` | metadata dict의 가변 시간축을 padding하지 못함 | 새 dataset/collate 경로를 추가한다. |
| `drumblender/utils/model.py` | loader parser 및 constructor introspection이 DrumBlender에 묶임 | 새 Lightning task를 평가/export하려면 loader도 연결해야 한다. |
| `scripts/export_recon_wavs.py` | params를 tensor로 가정 | structured modal metadata의 batch/device 변환을 지원해야 한다. |

현재 modal renderer는 inactive-frame frequency 처리와 initial phase integration을 수정한 상태다. 이를 과거 버그 상태로 설명하거나 되돌리지 않는다. 새 modal cache의 128 modes와 기존 config의 64 modes를 암묵적으로 섞지 않는다.

## 3. 문헌에서 가져올 것과 가져오지 않을 것

아래는 2026-09-25까지 확인한 primary sources다. venue나 최신성은 선택 근거의 일부이며, 우리 drum task에서 우월함을 입증하지 않는다.

| 연구 | 실제 관련성 | 적용과 경계 |
|---|---|---|
| [PLAUD, AIMC 2026](https://arxiv.org/abs/2608.13724) | NoiseBandNet 기반 latent-controlled performance instrument | 가장 가까운 선행연구. noise latent 악기 자체는 새롭지 않다. 원음 phase를 보존하는 percussion 편집을 별도로 검증해야 한다. |
| [Wavehax, TASLP; preprint 2024](https://arxiv.org/abs/2411.06807) | 2D convolution과 complex spectrum/iSTFT synthesis | TF topology를 함께 처리하는 decoder를 참고한다. speech harmonic prior를 noise에 그대로 옮기지 않는다. |
| [Revisiting Vocos, IWAENC 2026](https://arxiv.org/abs/2607.24323) | phase modeling에서 1D backbone의 한계를 분석 | iSTFT 사용만으로 phase가 해결되지 않는다. 2D 처리와 complex loss를 검증하고 consistency penalty는 ablation한다. |
| [Discovering and Steering Interpretable Concepts in Large Generative Music Models, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/88cf70805294cab2c8206e5b1803733a-Abstract-Conference.html) | MusicGen 내부 SAE로 언어·음악 이론에 없는 패턴도 발견 | 가장 직접적인 creative-knob 연결점. 생성 모델 결과를 drum editing 성능으로 전이 주장하지 않는다. labeling/CLAP는 필수가 아니다. |
| [SAEs Are Good for Steering—If You Select the Right Features, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.519/) | activation 설명과 output에 미치는 causal effect를 구분 | top-activation 예시만 보고 knob를 채택하지 말고 실제 intervention 결과로 선별한다. language 연구에서 가져오는 평가 원리다. |
| [NoiseBandNet, TASLP 2024](https://arxiv.org/abs/2307.08007) | fixed filtered-noise bank와 time-varying envelopes | 중요한 noise synthesis baseline. fixed noise가 source phase를 담는 것은 아니다. |
| [TexStat/TexEnv/TexDSP, DAFx 2025](https://arxiv.org/abs/2506.04073) | texture statistics와 envelope 기반 DDSP | diffuse tail 분석에 유용하다. time-invariant statistics를 attack timing/phase 주 loss로 쓰지 않는다. |
| [Modulation Discovery with DDSP, WASPAA 2025](https://christhetr.ee/mod_discovery/) | 저차원 modulation trajectory와 spline/LPF 선택 | smooth edit trajectory에 연결한다. reconstruction latent 전체를 느리게 만들지는 않는다. |
| [Re-Bottleneck, MLSP 2025](https://arxiv.org/html/2507.07867v2) | frozen AE latent를 사후에 다시 표현 | reconstruction과 control representation을 분리하는 참고. 정확한 inverse나 lossless reparameterization으로 부르면 안 된다. |
| [MusicRFM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4be1087b23fd298ba6fc19e0ba506aea-Abstract-Conference.html) | learned probes를 통한 music steering | label/preference가 생긴 뒤 refinement 후보. 처음부터 unsupervised knob discovery와 동일시하지 않는다. |

추가 비교 후보는 [ISAC, SampTA 2025](https://arxiv.org/abs/2505.07709)의 learned auditory filterbank다. 첫 구현에는 고정 STFT를 사용해 분석기 변화와 decoder 변화를 분리한다. diffusion/flow postfilter와 GAN은 perceptual improvement 가능성이 있지만 source의 정확한 phase를 보장하지 않으므로 첫 fidelity 실험의 필수 구성으로 두지 않는다.

## 4. 제안 architecture

```text
원본 x + 기준 modal parameters/metadata
       │
       ├── frozen reference modal renderer ───────────────── m_ref
       │
       └── r = x - m_ref
             │ short complex TF + long spectral context + confidence
             ▼
          local 2D encoder + coarse whole-sample context
             ▼
          continuous z [B,64,T]
             │                 ▲
             │          posthoc sparse delta control
             ▼
          2D complex excitation U + positive envelope A
             ▼
          n = scale × iSTFT(A ⊙ U)
             │
             └──────────────────────────────────────────── m + n
```

### 4.1 Analysis와 factorization

Prototype은 48 kHz mono, offline sample analysis다. streaming encoder나 live input latency를 지금 보장하지 않는다. 악기에서는 분석 결과를 cache하고 decoder/control을 반복 실행한다.

\[
s=\max(\operatorname{RMS}(x),10^{-4}),\quad
R=\operatorname{STFT}((x-m_{ref})/s).
\]

RMS는 valid samples만 사용한다. short STFT는 FFT 512, hop 64, periodic Hann, `center=True`, `pad_mode='constant'`, `normalized=True`다. 48 kHz에서 frame step은 1.33 ms지만 window duration은 10.67 ms다. hop을 temporal resolution과 동일시하지 않는다. long context는 FFT 2048, hop 256이며 실제 시간/주파수 좌표로 short grid에 정렬한다.

32 ERB-spaced anchors를 잇는 고정 linear interpolation matrix `Bf:[257,32]`를 만든다. log magnitude를 ridge projection하여 envelope target을 정한다.

\[
q^*=(B_f^TB_f+0.01I)^{-1}B_f^T\log(|R|+10^{-5}),
\quad A^*=\exp(\operatorname{clamp}(B_fq^*,\log10^{-4},8)),
\quad U^*=R/A^*.
\]

예측은 `A=exp(clamp(Bf q, log(1e-4),8))`, `N=A⊙U`, `n=s*iSTFT(N)`이다. `U`는 real/imag를 직접 출력한다. phase angle wrapping 문제를 피하며 DC/Nyquist imaginary part는 0으로 만든다. 원본 phase를 decoder에 복사하는 skip connection은 사용하지 않는다.

이 연산은 TF spectral multiplier다. 물리적 time-varying FIR convolution과 정확히 같다고 주장하지 않는다. `A,U`의 의미도 유일하게 identifiable하지 않다. envelope/weighted-excitation auxiliary loss는 정해진 factorization convention을 학습시키며 `A=1` ablation으로 실제 이점을 확인한다.

### 4.2 Encoder/decoder 초기 크기

| 위치 | 초기 설정 |
|---|---|
| Input | `[B,8,257,T]`: compressed Re/Im R, log|R|, compressed Re/Im modal STFT, confidence map, metadata-available map, aligned long log magnitude |
| Complex compression | `C(S)=S/(abs(S)+1e-5) * log1p(abs(S))`; real/imag를 두 channel로 사용 |
| Local encoder | frequency stride 2만 사용: 257→129→65→33; channels 32→48→64; 각 stage 2 ConvNeXt-style blocks; kernel `(5,7)`, expansion 2 |
| Local projection | frame마다 `[64,33]` flatten→128 |
| Global context | 16 frames masked average pool→2 Transformer blocks, width128, heads4, dropout0; 시간좌표 positional encoding; local time grid로 보간 후 더함 |
| Representation | Linear128→64, `[B,64,T]`; VQ/KL 없음 |
| Excitation decoder | Linear64→64×33; frequency interpolation33→65→129→257과 2D blocks; 최종2 channels→complex U |
| Envelope decoder | Conv1d64→128→32, kernel3; Bf를 통해 frequency interpolation |

64×750은 초당 48,000개의 실수 값이다. 첫 모델은 compression codec를 목표로 하지 않으며 size/bitrate 이점을 주장할 수 없다. bottleneck32와 context 제거를 후속 ablation한다. skip 없이 z로부터 source detail을 복원해야 한다.

Confidence map은 각 active mode의 confidence를 해당 frequency의 Hann-window spectral footprint로 배치한 뒤 겹치는 값의 maximum을 취한다. sample 시간좌표로 보간하되 missing endpoints를 valid observation으로 보간하지 않는다. legacy metadata 부재는 `available=0`으로 명시하며 confidence가 1이라고 가정하지 않는다. 본 실험은 metadata가 있는 cache를 사용한다.

### 4.3 Two branches 사이의 경계

첫 noise 학습에서는 modal renderer와 parameters를 고정한다. noise 모델이 modal 변경을 계속 상쇄하는 moving target을 피하기 위해서다. `x-m_ref`에는 modal 분석 오차도 들어가므로 supervised source separation이라고 부르지 않는다.

control 시에는 원본의 neutral analysis를 한 번만 수행한다. modal knob를 바꿀 때마다 `x-m(edited)`를 재분석하면 noise가 modal edit를 취소할 수 있다. 대신 cached z와 기준 modal을 유지하고 각각의 edit를 합성한다.

반대로 기준 residual이 modal phase error를 상쇄하고 있었다면 modal edit 이후 cancellation 관계가 깨질 수 있다. high-confidence tonal residual, modal/noise 간 상쇄 에너지, edit 전후 beating을 검사한다. residual의 modal-bin energy를 무조건 줄이는 penalty는 broadband attack을 훼손할 수 있으므로 기본 loss에 넣지 않는다.

## 5. 이름 없는 knob를 찾는 방법

FiLM은 conditioning을 전달하는 방법이다. 무엇을 control할지 발견하는 알고리즘은 아니다. 처음에는 좋은 reconstruction latent를 얻고 모델을 freeze한 뒤 SAE를 학습한다.

train-only mean/std로 z를 표준화한다. SAE는 64→512, nonnegative TopK8, unit-norm decoder columns, reconstruction MSE로 시작한다. dead/rare features는 사용 후보에서 제외하고 분포를 기록한다.

\[
a=E_{SAE}((z-\mu)/\sigma),\quad
a'_k=\max(0,a_k+\alpha\,q_{95,k}\,w_k(t)),
\]
\[
z'=z+\sigma\odot\{D_{SAE}(a')-D_{SAE}(a)\}.
\]

원래 z에 SAE의 **변화량만** 더하므로 SAE reconstruction error를 neutral output에 주입하지 않는다. α=0이면 같은 latent이고, deterministic decoder에서는 같은 waveform이다. nonzero edit는 phase도 바꿀 수 있다. phase를 항상 고정하는 것과 timbre를 유연하게 바꾸는 것을 동시에 보장하지 않는다.

`q95`는 train set의 positive activations에서 구한다. 후보 trajectory는 `w=1`과 `w=clip(a/q95,0,1)` 두 가지다. α grid는 `[-1,-0.5,0,0.5,1]`로 시작하지만 feature마다 validated range를 저장한다. 하나의 범위를 모든 sample에 안전하다고 선언하지 않는다.

후보 feature를 held-out samples에 실제로 적용해 audible effect, output smoothness, clipping/pre-echo, 비의도적 pitch/attack/level 변화, feature 간 중복을 측정한다. 이름과 CLAP score 없이도 할 수 있다. 최종 8–16개는 audition으로 선별한다. 큰 activation이나 예쁜 latent plot만으로 control이라고 부르지 않는다.

modal task의 `P(u)=P_base+[D(z,u)-D(z,0)]`와 같은 source-anchored delta 원리를 공유한다. modal은 mode-set topology, noise는 ordered TF topology를 사용한다. 두 encoder를 억지로 동일하게 만들지 않는다. noise-only와 modal-only knob를 먼저 검증한 뒤, 두 branch를 함께 움직이는 intentional macro를 추가할 수 있다.

## 6. Training과 loss

초기 engineering weights는 아래와 같다. 논문에서 검증된 최적값이 아니다. 각 항의 raw/weighted gradient scale을 기록해서 한 항이 지배하면 조정한다.

\[
L=L_{wave}+L_{complexMR}+L_{magMR}
 +0.2L_{env}+0.1L_{flux}+0.1L_A+0.1L_U.
\]

- `wave`: `(x_hat-x)/s`의 valid-sample L1.
- `complexMR`: FFT `[256,512,1024,2048,4096]`, hop N/4에서 target complex magnitude 평균으로 정규화한 complex absolute error. 저에너지 clip은 denominator floor 적용.
- `magMR`: 같은 grids에서 spectral convergence + log-magnitude L1의 평균.
- `env`: moving RMS windows `[48,240,960]` samples, log envelope L1, 동일 valid support만 비교.
- `flux`: log-magnitude의 시간축 difference L1. tensor `[B,F,T]`에서는 `dim=-1`.
- `A`: predicted log envelope와 convention target log A*의 L1.
- `U`: `A* * abs(U-U*)`의 평균을 mean|R|+floor로 정규화. silence bin의 arbitrary phase를 동일 가중하지 않는다.

각 clip을 true length로 잘라 STFT/loss 계산하는 reference 구현부터 만든다. mixed precision을 쓰더라도 FFT, complex multiply, phase-sensitive loss는 float32로 유지한다. zero-length는 reject하고 1-sample부터 짧은 clip을 지원한다. output length는 input valid length와 정확히 같다.

AdamW lr2e-4, betas(0.9,0.99), weight_decay1e-4, grad_clip1.0으로 시작한다. 첫 32-sample overfit은 full clips≤2s, batch4, max20k steps, seed3개다. 실패하면 GAN을 추가하기 전에 DSP closure, bottleneck, frame alignment, gradients를 확인한다. 이는 tiny-set diagnostic이며 generalization 결과가 아니다.

전체 training은 full-length reference를 기준으로 한다. 긴 clip의 memory 최적화가 필요할 때만 context pooling을 전체 clip에서 계산하고 local encoder/decoder를 overlap chunks로 실행한다. frame indices와 scale은 global이며 modal은 crop 전에 연속 합성한다. loss에는 chunk guard를 제외한다. chunk 결과가 full rendering과 일치하는 검증 없이 streaming 지원을 주장하지 않는다.

GAN/feature matching은 deterministic baseline이 안정된 다음 분리된 실험으로 추가한다. texture-statistics loss, phase-derivative loss, STFT consistency loss도 각각 추가 실험이며 한꺼번에 넣지 않는다.

## 7. 적은 수의 결정적인 실험

모든 주 비교에서 같은 split, sample lengths, modal cache, mode count, seed set을 사용한다. 새 modal 분석의 이점과 새 noise 모델의 이점을 섞지 않는다. `sample_pack` split만으로 unseen-pack generalization이라고 부르지 않는다.

| 순서 | 비교 | 판단할 질문 |
|---|---|---|
| D0 | oracle complex coefficients roundtrip; silence/impulse/flam/HF burst/long tail | DSP 구현이 representable waveform을 손상시키는가? 학습 가능성을 증명하는 실험은 아니다. |
| R0 | 기존 DAC+LSTM filtered noise, random/fixed realization; 기존 3-branch 참고 | 실제 baseline fidelity와 stochastic variance는 얼마인가? |
| R1 | 같은 modal + 제안 A⊙U, 기존 noise와 matched training | 학습된 source-conditioned excitation이 attack/phase 오류를 줄이는가? |
| R2 | R1 vs A=1 direct complex decoder | 명시적 envelope factorization이 reconstruction/control에 기여하는가? |
| R3 | R1 vs coarse-context 제거; 필요할 때만 N256/H32 | global context와 더 빠른 grid가 어떤 소리에서 필요한가? |
| C0 | random/PCA directions vs SAE delta, output-change magnitude matching | SAE가 더 유용하고 덜 중복된 knob를 주는가? |
| C1 | 같은 candidate directions의 additive route vs FiLM adapter | 발견 방향과 conditioning 방식의 효과를 구분할 수 있는가? |

R3와 C1은 각각 R1/C0 결과에서 필요성이 확인된 경우에만 실행한다. factorization이 도움이 없으면 그 결과를 받아들이고 direct complex texture decoder를 유지할 수 있다. 재현 가능한 arbitrary complex synthesis도 broad DDSP이지만, source–filter 해석을 기여로 내세우지는 않는다.

Reconstruction: 기존 MR-STFT protocol을 그대로 병행 보고하고, phase-sensitive complex distance, SNR/SI-SDR, attack timing, HF energy, tail envelope, pre-echo를 추가한다. silence의 SI-SDR은 undefined로 명시한다. known DSP latency 외 time alignment로 error를 숨기지 않는다. type별 결과와 worst 5%를 본다.

Listening: original, baseline, proposed를 무작위 순서로 loudness 조건을 명시해 비교한다. raw level metrics와 loudness-matched audition을 분리한다. 평균 점수 상승과 worst-case artifacts를 함께 판단한다.

Control: zero identity, 반복 재현성, small-step continuity, reverse sweep, held-out cross-sample transfer, output descriptors의 변화/보존, knob 중복을 평가한다. 단순 EQ/masking과 PCA를 baseline에 포함한다. unnamed knob는 특정 semantic target이 없으므로 모든 descriptor 변화를 leakage라고 부르지 않고, 보존 대상으로 정한 pitch/onset/level 및 artist가 의도한 변화와 구분한다.

## 8. 기여의 범위와 중단 기준

기여 후보는 (1) 두 branches 안에서 percussion phase와 broadband attack을 다루는 source-conditioned texture synthesis, (2) 좋은 reconstruction을 출발점으로 source-anchored unnamed controls를 발견·선별하는 방식, (3) modal/noise 상쇄와 실제 output selectivity를 포함한 평가다. 이 조합의 novelty는 추가 related-work 대조와 실험 후 주장한다.

단순 noise latent 악기, SAE 사용, iSTFT decoder, pretrained codec 제거 자체는 novelty claim으로 삼지 않는다. PLAUD와 music SAE steering은 특히 직접 비교해야 한다.

32-sample overfit에서 complex/waveform error가 줄지 않으면 architecture 확대보다 representation/optimization diagnosis를 먼저 한다. A=1이 우세하면 factorization을 고집하지 않는다. SAE가 PCA보다 유용하지 않으면 SAE 이름을 유지하기 위해 억지 knob를 선정하지 않는다. control이 modal cancellation을 깨뜨리면 해당 decomposition을 재검토하고 branch independence claim을 보류한다.

## 9. Global constraints

- Python >=3.10,<3.13; torch/torchaudio 2.7.1; pytorch-lightning 1.9.5를 유지한다.
- Prototype은 48 kHz mono, offline sample analysis와 cached control rendering이다.
- 기존 checkpoint/config/평가 protocol을 보존하고 새 architecture에는 새 class와 config를 사용한다.
- pretrained codec, VQ, text/CLAP supervision, 세 번째 audible transient branch는 초기 구현의 의존성이 아니다.
- 새로운 reconstruction 경로에는 raw waveform/complex-spectrum decoder skip을 두지 않는다.
- valid length를 보존하고 평가 대상 audio를 조용히 zero-pad하거나 tail-trim하지 않는다.
- 기존 작업 중인 파일과 사용자 변경을 덮어쓰거나 연구 baseline을 삭제하지 않는다.

구현 순서는 [implementation plan](../plans/2026-09-25-transient-aware-noise-control.md)에 있다.
