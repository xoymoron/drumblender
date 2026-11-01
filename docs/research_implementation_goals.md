# Percussion reconstruction과 control: 두 구현 목표

업데이트: 2026-10-29. 상태: **사용자가 두 방향을 모두 구현 목표로 채택했다.** 이 문서는 현재 목표와 비교 기준을 정리하며, 두 architecture의 구현 완료나 성능 검증을 뜻하지 않는다. 세부 network, loss, 학습 순서는 후속 설계 대상이다.

## 1. 공동 목적과 결정 경위

Unlabeled percussion sample을 높은 fidelity로 reconstruction하고, 원본 sample에서 출발해 신기하고 재미있는 timbre 변화를 탐색하는 neural drum machine을 만든다. Control은 이름이 명확한 semantic attribute에 한정하지 않으며, onset/tail에만 제한하지 않는다.

이번 대화에서 구분한 내용은 다음과 같다.

1. DDSP는 미분 가능한 DSP 연산을 학습에 통합하는 방법론이다. Modal/sinusoidal 성분과 stochastic 성분을 모델링해 더하는 SMS 구조와 동일한 개념이 아니다.
2. DrumBlender의 oscillator + noise (+ transient) 경로는 SMS 계열의 structured synthesis 선택이다. Peak tracking은 time-varying frequency를 표현하지만, 물리 mode의 정확한 분리나 phase 정확성을 보장하지 않는다.
3. 두 개선 방향을 구분한다. **A는 synthesis parameter를 함께 예측한 뒤 DSP로 합성**하고, **B는 modal 정보를 conditioning으로 사용해 neural decoder가 전체 waveform을 생성**한다.
4. 사용자는 A와 B를 모두 채택했다. B를 주 연구 후보로 추천했던 assistant의 의견은 있었지만, A를 구현하지 않거나 단순 참고 대상으로만 남기기로 결정한 것은 아니다. 우열과 최종 논문 중심은 실험 후 판단한다.

Cymbal의 복잡성과 dense high-frequency 구조는 인정하며, 완벽한 cymbal 복원을 모든 진행의 필수 조건으로 삼지 않는다. 다만 어려운 sample을 평가에서 조용히 제외하지 않는다. Unlabeled라는 조건은 human semantic/physical annotation을 필수로 요구하지 않는다는 뜻이며, 추출 feature를 이용한 self-supervision과 청취 평가는 가능하다.

## 2. 두 목표의 경계

| 구분 | A. Joint DSP synthesis | B. Modal-conditioned neural synthesis |
|---|---|---|
| 입력 | Modal trajectories와 원본 audio/detail 정보 | Modal trajectories와 원본 audio/detail representation |
| Network의 책임 | 공통 context를 사용해 modal과 noise/texture synthesis parameter를 함께 예측 | 구분 가능한 modal/detail 입력을 함께 사용해 전체 audio를 복원 |
| 최종 합성 | Oscillator/modal DSP와 noise/texture DSP의 출력 합산 | 하나의 neural decoder가 최종 waveform 생성; 구현에 따라 complex spectrum과 inverse transform을 사용할 수 있음 |
| Modal 정보의 지위 | 실제 DSP 합성 parameter와 연결됨 | 편집 가능한 conditioning이며 oscillator 출력을 최종 audio에 반드시 더하지 않음 |
| 용어 | DDSP/SMS 계열의 structured synthesis | Modal-conditioned neural synthesis; SMS-informed / SMS-inspired라고 설명 가능 |
| 주된 검증 질문 | Joint parameter prediction이 별도 예측보다 reconstruction과 control에 이득을 주는가? | 높은 reconstruction을 유지하면서 modal conditioning이 실제로 유효한 control을 만드는가? |

이 표는 architecture 수준의 구분이다. 하나의 Python class로 묶였는지, 두 모델을 동시에 학습하는지, decoder head가 몇 개인지만으로 A/B를 분류하지 않는다. B를 단지 network가 differentiable하거나 iSTFT를 쓴다는 이유로 A와 같은 DDSP/SMS synthesis라고 부르지 않는다.

## 3. 목표 A: Joint DSP synthesis

개념적인 경로:

```text
modal observations + source audio/detail
                  -> joint context/network
                       -> modal parameter head -> oscillator/modal DSP -> m
                       -> noise/texture parameter head -> DSP synth   -> n
                                                                  output = m + n
```

- Modal과 noise/texture parameter를 함께 결정하도록 설계한다. 추출 modal을 그대로 렌더링하면서 noise encoder만 바꾸는 기존 방식과 구분한다.
- Frequency, amplitude, initial phase와 noise/texture parameter의 표현 및 제약은 명시한다. 기존 filtered-noise renderer의 유지/교체 여부는 아직 결정하지 않았다.
- Joint prediction은 기존 DSP renderer의 표현 범위를 자동으로 넓히지 않는다. Phase drift, broadband attack, branch 간 상쇄를 따로 확인한다.
- Control이 parameter상으로 분리돼도 output의 청각적 독립성이 보장되는 것은 아니다. 연관된 modal/texture 변화도 허용하며 실제 효과로 판단한다.
- 기존 modal encoder, temporal/set interaction, renderer와 control 도구는 재사용 후보이며, 그대로 합치는 것만으로 목표 A가 완성됐다고 보지 않는다.

## 4. 목표 B: Modal-conditioned neural synthesis

개념적인 경로:

```text
source x -> modal analysis -> modal representation p ----+
                                                       +-> neural decoder -> full waveform
source x -----------------> audio/detail representation z+
                                  modal/detail controls-+
```

- 추출 modal은 불완전한 관측값이자 editable conditioning이다. 분석 결과의 모든 peak를 정확한 물리 mode나 정답 oscillator waveform으로 강제하지 않는다.
- Modal과 audio/detail 입력의 출처를 구별하면서 decoder 안에서 interaction을 허용한다. Concat, FiLM, cross-attention 등 구체적 전달 방식은 아직 미정이다.
- Audio/detail representation에는 texture뿐 아니라 attack, phase detail, 누락된 resonance와 modal 분석 오차가 포함될 수 있다. 이를 clean noise latent라고 부르지 않는다. 원본 또는 residual 중 무엇을 encode할지도 후속 설계 대상이다.
- Source representation을 cache하고 그 기준에서 knob를 조작한다. Modal edit마다 원본을 다시 encode하거나 `x - edited_modal`을 새 residual로 넣어 edit를 취소하는 경로를 기본으로 두지 않는다.
- Decoder가 modal을 무시하거나 modal 정보를 detail latent에 중복 저장할 수 있다. 입력 type 구분만으로 role separation이나 유효한 control이 생겼다고 주장하지 않는다.
- Reconstruction과 conditioning 활용을 함께 다룰 학습 원리가 필요하다. Counterfactual edit의 정답은 원본 reconstruction pair만으로 주어지지 않는다. 완전한 disentanglement는 필수 목표로 확정하지 않았다.
- Learned decoder의 표현 자유도가 fidelity나 자연스러운 coupled control에 도움이 된다는 것은 검증할 가설이다. Phase/attack 문제가 자동으로 해결된다고 가정하지 않는다.

## 5. 현재 코드와 과거 문서의 지위

| 항목 | 현재 확인한 범위 | 이번 목표와의 관계 |
|---|---|---|
| 기존 `DrumBlender`와 `cfg/05_all_parallel.yaml` | 추출 modal, noise, transient의 structured synthesis 경로 | 기존 reconstruction reference로 보존 |
| `ModalControlTask`, typed modal encoder, temporal SAE, modal-conditioned noise, 전용 CLI | 코드가 있으며 training 문서는 runtime/DDP/학습 품질을 미검증으로 기록 | A의 재사용 기반. B의 전체 waveform decoder 구현은 아님 |
| 기존 CLI의 `modal -> noise -> joint -> patterns` | 기존 두-branch 모델의 staged training | `joint`는 학습 stage 이름이다. 목표 A의 통합 parameter prediction 구현 완료를 뜻하지 않음 |
| 2026-09-25 complex-excitation texture spec/plan | Frozen modal + `iSTFT(A * U)` texture의 두 audible branches 제안 | 이전 연구 후보로 보존. A/B의 필수 architecture나 B 구현으로 간주하지 않음 |
| 목표 A | 두 방향 중 하나로 채택 | 세부 설계·전용 구현·비교 검증이 남음 |
| 목표 B | 두 방향 중 하나로 채택 | 세부 설계·전용 구현·비교 검증이 남음 |

관련 문서:

- [기존 modal 분석·control 구현과 training / audition](modal.md)
- [과거 complex-excitation texture 제안](superpowers/specs/2026-09-25-transient-aware-noise-control-design.md)
- [과거 texture implementation plan](superpowers/plans/2026-09-25-transient-aware-noise-control.md)

기존 문서의 두 audible branches, frozen modal, SAE, 특정 width/loss와 CLI는 해당 구현/제안에 적용한다. 새 목표 B에 자동으로 상속하거나 이번 결정으로 이미 승인된 세부 설계처럼 취급하지 않는다.

## 6. 공통 구현 목표와 비교 기준

- 기존 48 kHz mono percussion, valid audio/frame lengths와 저장된 modal cache를 출발점으로 한다. 두 경로의 data split, 분석기 version, mode selection/count, normalization과 평가 길이를 맞추고 차이는 기록한다. 기존 encoder 교체용 hop/latent-width 계약을 B 전체의 필수 구조로 강제하지 않는다.
- 완료된 feature extraction은 재사용한다. 새 sidecar나 재분석이 필요하면 이유와 범위를 명시하며, architecture 효과와 분석기 교체 효과를 섞지 않는다.
- 기존 checkpoint/config와 사용자 변경을 보존하고 A/B를 구별할 config, checkpoint 식별자, export 경로를 둔다. 구체적 class/file 이름은 후속 설계에서 정한다.
- 비교에는 **A**, **B**, **B와 같은 neural decoder에서 modal conditioning을 제거한 대조군**을 포함한다. 기존 DrumBlender도 공통 reference로 유지한다. Parameter 수, latent 정보량, training/compute budget을 가능한 한 맞추고 맞출 수 없는 차이를 공개한다.
- Reconstruction은 neutral output으로 평가한다. 공통 spectral metrics와 청취를 기본으로, phase-sensitive error, attack, resonance와 tail/texture의 실패를 확인한다. 원본과 같은 source를 test에서 encode하는 것과 training sample을 암기하는 것을 구별한다.
- Control은 같은 reference에서 amount sweep, zero 복귀, 반복 재현성, small-step continuity와 held-out 적용을 확인한다. 이름 없는 knob에는 임의의 semantic 정답을 요구하지 않고, 실제 audible effect, 유용성, 중복과 artifact를 평가한다.
- B의 modal 활용은 conditioning 제거/교체와 fixed-detail-latent modal edit로 점검한다. Conditioning에 민감하다는 사실만으로 좋은 control이라고 판정하지 않는다.
- Learned knob의 이득은 simple DSP edits와 PCA/random directions를 통해 확인한다. 필요 시 output 변화량을 맞춰 비교하며, 큰 변화 자체를 좋은 control로 점수화하지 않는다. 모든 descriptor 변화를 leakage로 취급하지 않는다.
- Stochastic synthesis를 쓰면 neutral/edit에 같은 excitation 또는 seed를 사용하고 variation을 별도로 기록한다. `x - modal`을 clean noise ground truth로 간주하지 않는다.
- Reconstruction 결과와 control 결과는 분리 보고한다. A/B 중 하나가 우세하다는 주장이나 publishable novelty는 matched experiment와 related-work 대조 후에만 한다. Realtime/plugin 지원은 따로 측정한다.

## 7. 구현 진행 항목

- [x] A와 B의 구분 및 둘 다 채택한 결정을 구현 목표에 기록한다.
- [x] 기존 구현·과거 제안·새 목표를 문서에서 구별한다.
- [ ] A의 joint representation, output parameterization, renderer, 학습/편집 API를 설계하고 구현한다.
- [ ] B의 modal/detail representation, waveform decoder, conditioning 활용 학습과 편집 API를 설계하고 구현한다.
- [ ] 공통 subset에서 reconstruction 가능성과 modal intervention을 먼저 확인한다. Tiny-set fitting을 held-out 성능으로 보고하지 않는다.
- [ ] 동일 protocol에서 A/B 및 conditioning 대조군을 학습·평가하고 neutral/edited audition을 비교한다.
- [ ] 실패 예시와 함께 결과를 정리하고 최종 논문 중심 및 악기 구성은 그 뒤에 선택한다.

이번 문서 수정은 위 목표를 반영하는 작업이다. Model 코드 작성, 학습 실행 또는 최종 세부 architecture 확정을 수행한 것은 아니다.

## 8. 선행연구와 주장 경계

- [DDSP 공식 자료](https://storage.googleapis.com/ddsp/index.html), [DDSP-SFX](https://arxiv.org/abs/2309.08060): 여러 DSP component parameter의 joint prediction 자체를 새 기여로 주장하지 않는다.
- [Neural source-filter](https://arxiv.org/abs/1904.12088), [Learning Control of Neural Sound Effects Synthesis from Physically Inspired Models](https://arxiv.org/abs/2503.08806): 구조 정보나 control을 neural waveform synthesis에 전달하는 큰 원리도 선행연구가 있다. B의 단순 conditioning만으로 novelty를 확정하지 않는다.
- [PLAUD](https://arxiv.org/abs/2608.13724): playable latent와 bending의 직접적인 선행 사례다. 기존 sample에 연결된 높은 reconstruction과 유용한 control을 실험으로 보여줘야 하며, percussion 대상 또는 modal 사용 자체를 충분한 차별점으로 보지 않는다.

기여 후보는 **unlabeled 실제 percussion의 불완전한 modal 관측을 유효한 편집 수단으로 사용하면서 높은 reconstruction을 유지하는 방법**과, A/B의 차이를 입증하는 평가다. 이는 연구 가설이며 성립한 결과가 아니다.
