# Development TODO

This file tracks DrumBlender research and development items.

## Adopted implementation goals (2026-10-29)

현재 목표의 기준 문서는 [두 구현 목표](docs/research_implementation_goals.md)다.
**Joint DSP synthesis (A)**와 **Modal-conditioned neural synthesis (B)**를 모두 구현·비교한다.
B만 채택하거나 A를 단순 참고 대상으로만 남기기로 결정한 것이 아니다.

- [x] 두 방향의 합성 경계, 용어, 결정 경위와 기존 코드의 상태를 문서화한다.
- [ ] A: joint context에서 modal/noise parameter를 함께 예측하고 DSP 출력을 합산하는 경로를 설계·구현한다.
- [ ] B: modal conditioning과 audio/detail latent로 전체 waveform을 생성하는 neural decoder 및 conditioning 활용 학습을 설계·구현한다.
- [ ] 기존 DrumBlender와 A/B, B의 modal-conditioning 제거 대조군을 공통 protocol로 비교한다.
- [ ] Neutral reconstruction과 creative control을 분리 평가하고 held-out audition 및 실패 예시를 기록한다.

기존 `modal -> noise -> joint -> patterns`의 `joint`는 training stage다. 새 목표 A의 구현 완료를 뜻하지 않는다.
아래 분석기 조사 메모는 기존 경로/후보에 대한 기록이며 새 A/B 전체의 필수 architecture로 간주하지 않는다.

## Reference

- Modal feature entry points:
  - Legacy path: `drumblender/utils/modal_analysis.py` with
    `scripts/build_modal_features.py`.
  - New isolated path: `drumblender/utils/modal_analysis_NEW.py` with
    `scripts/build_modal_features_new.py`.

- Modal feature environment notes:
  - `modal_analysis_NEW.py` depends on `scipy.ndimage.median_filter`.
  - Modal feature extraction environments therefore need `scipy` in addition
    to the existing `torch` / `torchaudio` / `nnAudio` stack.

## CQT 개선 관련 자료조사
synchrosqueezed ridge analysis
FRI 기반 grid-free ridge estimation
EDSM/eaQHM model을 refiner로 사용

최신 방법이 보는 난제는 grid dependence, ridge jitter, short fake ridge, curved IF estimation failure, high noise/signal density에서의 mode 간 간섭 등이 있다. 

의미 있는 물리 mode가 아니라, 샘플 곳곳에 흩어진 지역적 에너지 maxima가 채택되는 점이 본질적 문제. FRI 논문도 제안법이 좋은 성능을 보였지만, spatial constraint를 쓰지 않으면 작은 oscillation이 남는다고 스스로 적고 있다. 이건 매우 중요한 힌트다. 즉, 개선의 핵심은 단순히 “더 좋은 transform” 하나로 끝나지 않고, 연속성/곡률/수명(duration)/에너지 통합량을 넣은 ridge selection objective가 반드시 같이 가야 한다는 뜻이다. 

### Synchrosqueezed ridge analysis - GPT가 본 best
CQT를 버리고, forward-time의 synchrosqueezed CWT(Continuous Wavelet Transform) 또는 synchrosqueezed STFT로 ridge 뽑기. 
기본적으로 AM-FM component의 instantaneous frequency ridge를 더 날카롭게 모으기 위해 나온 방법.
단순 spectrogram/CQT local maxima보다 곡선형 frequency movement에 훨씬 잘 맞는다. 
SCT(Synchrosqueezed Chirplet Transform): 2021년 chirplet transform이 chirp-rate 축에서 blur된다는 문제를 지적하고, SCT가 crossover IF에서도 더 높은 contrast와 이론적 보장을 준다고 말한다. 
2024년 high-order SCT에서 나아가 기존 SCT가 high chirp modulation에서 IF를 잘못 추정하는 문제를 고친다고 주장한다. 

ssqueezepy는 Python에서 CWT/STFT synchrosqueezing, ridge extraction, generalized Morse wavelets, GPU/병렬 가속을 제공해서 적용 쉽다. generalized Morse wavelet은 2022년 음악 신호 연구에서 Morlet/Gabor보다 더 analytic하고, 비정상(nonstationary) oscillatory music signal에서 amplitude/phase/frequency 해석 가능성을 높인다고 설명. 
percussive에서 HF가 복잡하고 scale-dependent인 경우 CQT의 고정 log-bin 체계보다 analytic wavelet 기반 ridge가 더 괜찮을지도. 
다만 raw ridge point 그대로가 아닌 전체 경로 최적화 필요. 누적 에너지, 수명, 곡률 penalty, 주파수 1차/2차 smoothness 등..
2024년 noisy scalogram ridge 논문이 ridge의 uniqueness와 noisy-clean deviation bound를 다루는 것도, ridge를 “그때그때의 peak”가 아니라 “연속적 object”로 다뤄야 한다는 쪽을 지지한다.

### FRI 기반 grid-free ridge estimation
FRI-TLS / FRI-SST
2022년 spectrogram column을 Dirac pulse stream이 TF kernel로 blur된 관측으로 보고, ridge 위치를 finite rate of innovation reconstruction으로 복원. 
최종 IF 추정치가 TF grid resolution에 의존하지 않는다 -> 떨림/여기저기 짧은/계단식 freq drop 해결
SOTA ridge detector + pseudo-Bayesian 방법과 비교해 성능 개선. Code Ocean 코드도 제공한

spatial constraint가 없어서 추정치가 true IF 주변에서 작게 oscillate -> track continuity penalty를 넣지 않으면 여전히 떨림. 따라서 synchrosqueezed TFR 위의 어려운 구간이나 HF dense region에서 grid-free local refiner로 쓰자

### EDSM / eaQHM
ridge 찾고 나서 refine하는 용도로 쓰기. 2024년 비교 논문에서 EDSM을 subspace 기반 exponentially damped sinusoid model, eaQHM을 local characteristics에 적응하는 non-parametric time-varying basis로 설명. 
eaQHM은 medium-to-large window에서 EDSM보다 낫고, 반대로 EDSM은 작은 window에서 더 강하다는 결과. 또 EDSM은 damping factor를 직접 모델링하므로, strike 이후 decay를 갖는 퍼커시브 공통 구조와 잘 맞는다고 함. 반면 eaQHM은 frequency curve 자체에 basis를 적응시키므로, pitch/frequency drop이 있는 smooth curved track를 정련하는 데 더 적합. 
기본적으로 eaQHM은 parameter-refinement mechanism이라 초기 instantaneous component 추정에 의존 + small window에서는 LS conditioning 문제. 반대로 EDSM은 작은 window에서 잘 버티지만, window 안에서는 frequency stationarity를 가정. 
따라서 곡선형 이동을 강하게 보이는 드럼 모달에는 detector로 쓰기엔 한계가 있음. 

synchrosqueezed CWT with generalized Morse wavelets를 기본 time-frequency front-end로 둔 다음 그 위에서 전체 샘플 구간에 대해 ridge candidate를 만들고, Viterbi류 또는 연속 최적화 기반의 path objective로 64개 내외의 track를 선택. 
objective는 최소한 적분 에너지, 최소 지속시간, slope/curvature penalty, crossing penalty, HF 가중 threshold 포함해야 한다. 
여기에 dense/high-frequency region에서만 FRI-TLS 혹은 FRI-SST를 써서 ridge frequency를 grid에서 떼내기
마지막으로 각 track마다 짧은 window EDSM으로 damping과 local sinusoid parameter를 정련하기. 
smoother가 더 필요하면 probabilistic chirp smoother 같은 per-track refiner를 마지막 단계에 붙일 수 있음

실제 우선순위를 하나로 정리하면 이렇다.
최우선은 SSQ-CWT + continuity-regularized ridge tracking이다.
두 번째는 FRI를 HF/dense band의 grid-free local refiner로 추가하는 것이다.
세 번째는 EDSM을 track-level parametric refiner로 붙이는 것이다.
eaQHM은 좋은 방법이지만, current problem인 “검출 단계의 가짜 ridge”를 해결하는 1순위가 아니라서 뒤로 미룬다. eaQHM은 ridge가 이미 어느 정도 맞게 잡힌 뒤, 저중역의 더 smooth한 curved track를 예쁘게 다듬는 데 더 어울린다. 
