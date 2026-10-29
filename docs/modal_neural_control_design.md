# Joint temporal patterns for neural modal control

> **2026-10-29 목표 갱신:** [현재 구현 목표](research_implementation_goals.md)에서 **A. Joint DSP synthesis**와 **B. Modal-conditioned neural synthesis**를 모두 채택했다. 이 문서는 기존 modal encoder/SAE와 modal-conditioned filtered-noise 구현의 설계 기록이다. 아래 oscillator + noise 합산 구조를 프로젝트 전체의 유일한 목표로 해석하지 않는다. 기존 코드는 A의 재사용 기반이며 B의 전체 waveform decoder는 아니다. 공통 목표의 최신 정의는 연결된 문서를 따른다.

2026-09-25 수정. 아래 핵심 구조의 첫 구현과 통합 학습 CLI를 추가했다. Reconstruction, creative control, CPU realtime 성능은 검증 전이다. 기존 deterministic AE + global PCA 제안을 대체한다. 실제 학습 순서와 변경된 neutral 기준은 `docs/modal_control_training.md`를 따른다.

## 목적과 추천 구성

기존 percussive sample에서 추출한 modal trajectory를 출발점으로, dataset에서 학습한 frequency/amplitude/initial-phase의 공동 변화를 탐색한다. Text, CLAP, instrument annotation, harmonic alignment, exponential decay prior는 사용하지 않는다. Onset/tail에만 control을 제한하지 않는다.

추천: attribute별 stem → joint temporal/set representation → temporal sparse feature editing → shared modal decoder → 기존 DDSP renderer. UI는 learned pattern 선택과 amount이다. VFM은 더 큰 variation을 위한 확장 후보이며 첫 모델의 필수 부품이 아니다.

## 논문에서 확인한 내용과 적용 범위

| 원문 | 실제 내용 | 적용 판단 |
|---|---|---|
| [Interactive Neural Resonators, NIME 2023](https://nime.org/proceedings/2023/nime2023_81.pdf) | Shape, position, material parameters로 NN이 second-order resonator coefficients를 예측한다. | NN parameter prediction과 DSP rendering의 역할 분담. 물리 annotation 기반 control은 가져오지 않는다. |
| [PLAUD, 2026 preprint](https://arxiv.org/html/2608.13724) | NoiseBandNet 기반 synthesis, variational latent navigation과 performance 조작을 다룬다. | Playable latent와 reconstruction의 tradeoff. Modal-specific 방법은 아니다. |
| [TSPulse, 2025 preprint](https://arxiv.org/html/2505.13033v1) | Time/FFT dual-space masked reconstruction을 사용하는 작은 time-series model이다. | Masked temporal/attribute reconstruction을 참고한다. FFT domain과 modal frequency 채널은 서로 다른 개념이다. |
| [Object-Centric Latent Action Learning, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/39423/43384) | Object slots에서 inverse/forward dynamics를 학습한다. Relevant slot 선택에는 action label을 사용하는 probe가 있다. | 이미 추출한 mode의 identity와 관계를 유지한다. Percussion의 자연 감쇠를 agent action으로 간주하지 않는다. |
| [Words in Motion, ICLR 2025](https://arxiv.org/html/2406.11624) | Opposing motion attributes의 hidden-state 차이를 PCA로 모으고 SAE로 보완한다. | Hidden intervention을 참고한다. 전체 latent의 큰 분산 방향을 바로 creative knob로 해석하지 않는다. |
| [Discovering and Steering Interpretable Concepts in Large Generative Music Models, ICLR 2026](https://arxiv.org/html/2505.18186) | MusicGen hidden activations에 SAE를 학습하고 feature를 steering한다. | Unsupervised discovery와 audition. Language labeling/CLAP는 사용하지 않는다. Temporal dictionary는 우리의 확장 제안이다. |
| [SAEs Are Good for Steering — If You Select the Right Features, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.519/) | Input activation과 output steering 효과는 다를 수 있다. | 실제 decoded audio에 intervention하여 feature를 선별한다. |
| [Controlled Generation with Equivariant Variational Flow Matching, ICML 2025](https://proceedings.mlr.press/v267/eijkelboom25a.html) | Conditional training과 post-hoc Bayesian conditioning, molecular symmetries를 다룬다. | Reference와 learned-feature target을 조건으로 variation을 만드는 확장 후보. Molecular rotation/translation은 이식하지 않는다. |

PCA와 선형 dictionary는 최근 연구에서도 사용한다. Linear latent edit가 nonlinear decoder를 거치면 sample-dependent한 변화가 가능하지만, 이것만으로 creativity나 selectivity가 입증되지는 않는다. Unsupervised reconstruction만으로 의미 있는 knob가 고유하게 식별된다는 보장도 없다.

## 현재 코드에서 확인한 사실

- `drumblender/models/encoders.py:28`의 `ModalAmpParameters`는 sample pattern encoder가 아닌 mode-index별 고정 학습 amplitude multiplier이다.
- `cfg/06_modal.yaml`은 이 모듈을 사용한다. `drumblender/tasks/drumblender.py`에 새 modal module을 연결하고 별도 inference control API를 추가해야 한다.
- Cache는 `[3,K,T]`, batch는 `[B,3,K,T]`이다. Cache frequency는 radians/sample이며 분석 결과의 Hz와 구분한다.
- `drumblender/utils/modal_analysis_new.py`의 `ModalAnalysisResult.phases`는 initial sine phase를 모든 frame에 반복한 값이다. 측정한 framewise phase trajectory가 아니다.
- `drumblender/synths/modal.py`는 initial phase와 frequency 적분을 사용한다. Inactive endpoint를 관측 frequency처럼 보간하지 않는 규칙은 유지한다.
- 현재 synth는 full-buffer 경로다. Plugin에는 block 사이 oscillator state를 유지하는 경로가 필요하다.

## NN 구조

`B`: batch, `K`: mode 수, `T`: feature frame 수, `Tc=ceil(T/4)`. 아래 폭과 block 수는 구현 출발값이다.

| 구성 | Shape / layer | 역할 |
|---|---|---|
| Frequency stem | Observed log-frequency와 mask → shared temporal Conv1d → `[B,K,Tc,24]` | Pitch glide와 상대 frequency 변화 |
| Amplitude stem | 공통 sample gain으로 정규화한 log-amplitude와 mask → temporal Conv1d → `[B,K,Tc,24]` | Envelope, resurgence, modulation |
| Phase stem | Initial `(sin(phi0), cos(phi0))` → MLP → `[B,K,16]`, 시간축 broadcast | Circular initial phase |
| Joint tokens | Concatenate → 64-channel projection, residual temporal blocks | 한 mode의 attribute 관계와 시간 pattern |
| Mode interaction | 각 frame에서 8 inducing tokens를 사용하는 attention block 2개 | 여러 mode의 관계; 결과는 `[B,K,Tc,64]`로 돌아온다. |
| Shared decoder | Temporal blocks + 작은 mode interaction + temporal upsampling | Joint feature를 modal로 변환 |
| Output heads | Log-frequency `[B,K,T]`, log-amplitude `[B,K,T]`, circular initial phase `[B,K,2]` | 출력 단위는 구별하되 공통 hidden state로 결정 |

Mode interaction의 attention-score 비용은 대략 `O(Tc*K*8)`이며 전체 mode-time token의 quadratic attention은 사용하지 않는다. Projection/convolution 비용은 별도로 존재한다. 이 복잡도는 CPU realtime 보장이 아니다.

Mode-index embedding과 sample 간 frequency-rank alignment는 사용하지 않는다. Shared 함수와 masked set attention으로 mode permutation에 equivariant하게 만든다. Padding mode/time은 attention과 loss에서 제외한다.

기존 제안의 단일 masked-mean pooling은 제거한다. 개별 mode token을 유지하여 가까운 mode가 동일한 median-frequency query로 구별되지 않는 문제를 줄인다. Inducing tokens를 물리적인 8개 object로 해석하지 않는다.

Separate stems는 단위와 topology 차이에 대한 inductive bias다. Orthogonality loss, adversarial disentanglement, attribute별 독립 VAE로 frequency/amplitude/phase를 분리하지 않는다.

## Temporal pattern discovery와 control

Backbone을 학습한 뒤 고정하고 joint feature `H[B,K,Tc,64]`에 작은 convolutional SAE를 학습한다. Mode별로 weight를 공유하며 H에는 다른 mode의 context가 이미 들어 있다.

- Encoder: temporal Conv1d `64 → 256`, kernel 9, nonnegative activation, frame별 TopK 8.
- Sparse activation: `s[B,K,Tc,256]`.
- Decoder C: temporal convolutional dictionary. 각 feature가 9 latent frames와 64 hidden channels에 걸친 pattern을 가진다.
- Masked hidden reconstruction과 dictionary norm 관리를 사용한다. Dead/duplicate feature도 확인한다.

이는 기존 framewise SAE를 modal trajectory에 맞게 확장한 우리의 가설이다. 긴 pattern 전체가 한 atom에 담기는 것은 아니다. Atom의 반복 활성화와 temporal context로 장시간 변화를 표현한다. 실제 kernel 시간은 대략 `9*4*feature_hop/sample_rate`이므로 hop을 기준으로 선택한다.

선택 feature m의 강조/약화는 `s_u[...,m]=(1+u)*s_0[...,m]`, `u>=-1`로 시작한다. 이후 `H_u=H_0+C(s_u)-C(s_0)`를 사용하여 SAE reconstruction error를 원래 hidden state에 덮어쓰지 않는다. Reference의 TopK 선택을 고정하면 amount를 움직일 때 feature 집합이 갑자기 교체되지 않는다.

한 feature는 frequency와 amplitude 및 initial phase에 함께 관여할 수 있다. 예를 들어 여러 mode의 pitch glide와 amplitude resurgence를 함께 바꾸는 pattern을 기대할 수 있지만, 실제로 학습된 결과는 아니다. 같은 feature도 source activation과 decoder context에 따라 다른 결과를 낸다.

Multiplicative edit는 source에서 전혀 활성화되지 않은 feature를 새로 만들 수 없다. 더 큰 variation에서는 feature activation 범위에 맞춘 additive intervention을 별도로 audition할 수 있다. 유효 mode/time support 안에서 적용하더라도 source에 없던 pattern의 적합성은 보장하지 않는다. 새로운 activation sequence sampling은 아래 VFM 확장의 역할이다.

UI는 `Pattern 선택 + Amount`이다. 각 pattern에 높은 activation의 audio 예시와 실제 edit 예시를 보여준다. 처음부터 Metallic/Tension 등의 의미를 붙이지 않는다. 두 pattern을 함께 조작할 수 있지만 audible effect가 단순 합산된다고 가정하지 않는다.

## Reference 보존과 phase consistency

Frequency/amplitude decoder head는 log domain에서 다음과 같이 편집한다.

`log f_u = log f_0 + D_f(H_u) - D_f(H_0)`

`a_u = a_0 * exp(D_a(H_u) - D_a(H_0))`

Zero amount는 기준 parameter를 직접 반환한다. 통합 modal/noise 학습에서는 기준을 학습된 modal reconstruction으로 정한다. 원래 extraction cache와 동일하다는 뜻은 아니다. `a_0=0`은 유지하고 inactive frequency는 renderer의 active-endpoint 규칙을 따른다. 큰 edit의 gain/frequency/Nyquist 경계는 수치적으로 제한한다.

Phase head의 정규화 complex 출력 c를 이용하여

`phi0_u = wrap(phi0_0 + arg(c(H_u) * conj(c(H_0))))`

로 initial phase를 편집하고 이후 frequency를 연속 적분한다. 작은 phase-head vector norm에는 fallback이 필요하다. Frequency와 framewise phase를 서로 무관하게 출력하지 않는다. 두 값은 phase derivative 관계로 연결되어 있다.

Live playback에서는 initial-phase 변경을 다음 trigger부터 적용하는 것을 첫 구현의 동작으로 정한다. 재생 중 phase를 재설정하지 않는다. Frequency/amplitude 변화에는 control smoothing과 oscillator state 유지가 필요하다.

첫 모델은 mode 수와 activity support를 보존한다. 기존 curve의 움직임과 상대 energy/modulation은 수정할 수 있으나 mode birth/death 및 support 밖의 tail extension은 포함하지 않는다.

## 학습

1. Backbone을 clean reconstruction과 masked trajectory reconstruction으로 한 번 학습한다. Frequency block을 가리고 amplitude context로 복원하거나 그 반대를 수행한다. Training mask와 실제 missing-observation mask는 구분한다.
2. Backbone을 고정하고 cached hidden features로 SAE를 학습한다. 전체 DDSP 재학습은 아니지만 별도의 학습 단계라는 점은 명시한다.
3. 동일 checkpoint에서 feature amount sweep과 조합 audition으로 UI에 노출할 feature를 선별한다. 여러 backbone ablation을 선행 조건으로 요구하지 않는다.

Loss는 decoder의 절대 reconstruction에 건다. Neutral anchored output에만 loss를 걸면 입력이 통과해 NN 학습 신호가 없다.

기본 loss는 observed amplitude, amplitude-weighted log-frequency, energy-weighted circular initial phase, modal-target MR-STFT이다. Target waveform은 `ModalSynth(P0)`이다. Raw mixture를 modal 단독 target으로 사용하면 noise까지 modal로 설명하도록 압박할 수 있다. 이 학습이 extractor error를 개선한다고 주장하지 않는다.

필요하면 amplitude/frequency trajectory의 temporal FFT 차이를 보조 loss로 추가한다. 이는 modulation spectrum이며 raw-audio carrier spectrum이 아니다. 처음부터 별도 FFT encoder와 여러 scale의 loss를 필수 구성으로 쌓지 않는다. Phase-sensitive waveform/complex-STFT fitting은 frequency error와 결합하므로 optimization 난도가 올라갈 수 있다.

같은 target으로 돌아오게 하는 강한 latent-noise denoising은 필수로 넣지 않는다. Control하려는 변화에 decoder가 둔감해질 수 있기 때문이다. Masked input reconstruction과 control-direction invariance는 다른 목표다.

## VAE / VFM / CFG

VAE도 가능하지만 KL prior만으로 좋은 knob가 발견되지는 않는다. 이 첫 설계는 reference fidelity와 coupled-pattern intervention을 먼저 다룬다. Deterministic backbone + SAE는 generative prior가 아니며 prior sampling/OOD 안정성을 주장하지 않는다.

CG EVFM의 구체적 확장: frozen backbone의 joint latent trajectory에 작은 VFM prior를 학습하고, 선택 SAE feature의 목표 activation과 reference proximity로 conditioning한다. Instrument/text label은 필요 없다. 예를 들어 우리가 정한 energy를

`E(H)=lambda*||A_m(H)-target_m||^2 + mu*||H-H_reference||^2`

로 두고 `p(y|H) proportional to exp(-E(H))`를 사용하여 endpoint posterior를 보정할 수 있다. A_m은 learned feature이다. 이것은 원문의 audio 결과가 아닌 응용 가설이다. Reference proximity는 edit 크기를 제한하지만 지각적 selectivity를 보장하지 않는다. Hard TopK의 경계 때문에 guidance에는 differentiable activation surrogate를 고려하고 최종 feature와의 일관성을 확인해야 한다.

Permutation equivariance를 사용하려면 network뿐 아니라 prior/path/conditioning도 mode permutation을 일관되게 처리해야 한다. Continuous Gaussian VFM을 선택했다는 사실만으로 일반 flow matching보다 우월하다고 주장할 근거는 없다.

VFM은 variation 요청 때 background에서 후보를 생성해 cache하는 데 사용한다. ODE sampling과 guidance gradient를 audio callback에서 매번 수행하지 않는다. Seed를 고정한 condition sweep의 연속성과 CPU latency는 별도 확인이 필요하다.

CFG는 text 전용이 아니지만 conditional/unconditional prediction 사이의 guidance strength이다. Condition 자체가 재미있는 pattern인지 발견해 주지는 않는다. CG EVFM의 post-hoc Bayesian conditioning과도 동일한 방법이 아니다. 첫 모델에는 넣지 않는다.

## 평가와 contribution

Neutral identity, edited/reference curve, pitch drop, 맥놀이, 가까운 mode, silence gap, held-out audition을 같은 checkpoint로 확인한다. Feature activation이 높은 예시뿐 아니라 실제 intervention이 실패하는 예시도 기록한다. Uniform gain/pitch shift와 비교하여 변화가 단순한지 확인한다. Plugin 단계에서 load/encode, knob update, audio callback 시간을 분리 측정한다.

기여 후보는 existing percussive sample의 modal trajectory에 담긴 coupled temporal pattern을 발견하고 reference를 기준으로 재조합하는 control이다. SAE/VFM 사용 자체를 novelty로 주장하지 않는다. 실제 output에서 유용한 pattern과 보존 특성이 나타나는지 검증해야 한다.

## 향후 구현 파일

- `drumblender/models/modal_control.py`: typed stems, joint backbone, output heads, reference edit API.
- `drumblender/models/modal_patterns.py`: temporal SAE와 feature intervention.
- 별도 modal 학습 task/config: backbone reconstruction과 frozen-feature SAE fitting.
- `drumblender/synths/modal.py`: plugin용 stateful rendering.
- Inference viewer: pattern audition, amount sweep, reference 복귀.

Modal NN/SAE/conditioning adapter와 전용 staged training/audition 코드를 구현했다. 실행 방법과 미구현 범위는 `docs/modal_control_training.md`에 정리했다. Stateful plugin renderer와 VFM 확장은 아직 구현하지 않았다.
