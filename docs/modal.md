# Modal analysis and neural control

Modal 분석, feature 추출, 결과 확인, neural control의 실행 방법을 이 문서에서 관리한다.
분석기는 현재 최적화된 Hybrid이며, neural control의 reconstruction/control 품질은 별도로 검증해야 한다.
프로젝트 전체의 두 연구 목표와 비교 기준은 [research implementation goals](research_implementation_goals.md)를 따른다.
여기의 기존 `joint` 학습 stage는 새 목표 A의 통합 parameter predictor나 목표 B의 full-waveform neural decoder를 구현했다는 뜻이 아니다.

- [Analysis](#analysis): 알고리즘, 기본값, feature 규약
- [Extraction](#extraction): 실행, 병렬 처리, resume, metadata
- [Review](#review): 저장 결과 확인과 개별 sample 재분석
- [Neural control](#neural-control): 기존 모델, 편집 규칙, 확장 범위
- [Training and audition](#training-and-audition): 학습과 청취 명령
- [Maintenance](#maintenance): 유지하는 코드와 테스트
- [Related work](#related-work): 설계 시 참고한 연구와 적용 범위

## Analysis

구현은 [modal_analysis_new.py](../drumblender/utils/modal_analysis_new.py)의 `CQTModalAnalysis`다.
[modal_analysis.py](../drumblender/utils/modal_analysis.py)는 실제 peak 관측을 구분하는 legacy/ampzero baseline이다.
Legacy가 원 논문의 수정 없는 tracker와 같다는 뜻은 아니다.
[modal_analysis_original.py](../drumblender/utils/modal_analysis_original.py)는 로컬 Git의
`1a717e525ff70ae9dcb9cee7e3afd1bc71afb7bb`에서 변경 없이 복원한 원본 tracker다.

기본 Hybrid의 signal path:

1. **세 해상도로 관측한다.** 20 Hz부터 약 700 Hz까지 24 bins/octave CQT를 사용한다.
   Anti-aliased decimation과 FFT convolution으로 계산량을 줄인다.
   8192/2048-sample Hann STFT는 전 대역의 frequency 구분과 짧은 envelope 관측을 보완한다.
   48 kHz에서 각각 약 171/43 ms이며 nominal hop은 공통 256 samples다.
2. **원래 complex spectrum에서 peak를 추정한다.** Log-magnitude parabolic interpolation으로
   sub-bin frequency를 구하고 local contrast로 약한 maxima를 거른다.
   Median subtraction으로 합성 amplitude를 변형하지 않는다.
   STFT peak는 최대 네 개의 가까운 성분을 함께 complex Hann-lobe fitting하며,
   ill-conditioned fit은 원래 관측값을 유지한다.
3. **관측을 합친다.** 겹치는 후보는 nominal bandwidth가 좁은 관측을 우선한다.
   넓은 peak 하나로 이미 분리된 두 성분을 평균내지 않는다.
   고립된 HF anchor는 정밀한 frequency를 유지하면서 short STFT로 amplitude/phase를 재측정한다.
   가까운 성분이 모인 영역은 fine-window envelope를 유지한다.
4. **Hz에서 track을 연결한다.** Bounded frequency prediction, Hz/percentage gate,
   soft phase cost, one-to-one greedy matching을 사용한다.
   기본 허용 gap은 두 frame이며, 만료된 track을 긴 무음 뒤에 다시 연결하지 않는다.
5. **실제 관측으로 선별한다.** 최소 네 observed frames, 두 frame의 연속 관측,
   track span 안에서 관측 비율 0.5 이상을 요구한다. Amp=0 placeholder는 관측으로 세지 않는다.
6. **합성 parameter를 내보낸다.** Bounded phase-derived frequency refinement와
   energy/confidence 기반 score를 사용한다. 최고 score보다 40 dB 넘게 낮은 track을 제거한 뒤
   mode ceiling을 적용한다. Harmonic 또는 exponential/monotonic decay prior는 없다.

| 기본값 | 값 / 의미 |
|---|---|
| Sample rate | 48,000 Hz |
| Backend | `hybrid`; 비교용으로 `stft`, `cqt`도 제공 |
| Hop | 256 samples |
| CQT | `fmin=20`, `bins_per_octave=24`, Hybrid 포함 상한 700 Hz |
| STFT | long 8192 / short 2048 samples |
| Mode ceiling | 128; 목표 개수가 아닌 최대 개수 |
| Amplitude floor | -90 dBFS와 입력 waveform peak 대비 -80 dB 중 더 높은 값 |
| Track score floor | 최고 score 대비 -40 dB; 물리 mode/noise 확률이 아님 |

Python API에서 `num_modes=None`은 유효 track을 모두 유지하고,
`min_relative_score_db=None`은 상대 score gate를 끈다.
Per-frame 후보 수는 별도 `max_peaks`로 제한하며 기본값은 `max(256, 2 * num_modes)`다.
192/256 modes도 가능하지만 더 많은 mode가 더 좋은 reconstruction을 보장하지는 않는다.
Legacy와 NEW는 magnitude scaling이 달라 같은 threshold 숫자만으로 sensitivity를 맞출 수 없다.

### Feature and synthesis conventions

- `analyze(mono_numpy)`는 Hz frequency, amplitude, initial phase, `observed`, confidence,
  source, frame times, scores, 후보 수와 처리 시간을 가진 `ModalAnalysisResult`를 반환한다.
- `analyzer(batch)`는 `[B, M, F]`의 frequency/amplitude/phase tuple을 반환한다.
  Torch 입력의 반환값은 CPU float32 tensors다.
- 저장 feature는 `[3, M, F]`, batch는 `[B, 3, M, F]`다.
  **분석 API frequency는 Hz, 저장 frequency는 radians/sample**이다.
  Phase는 fitted initial sine phase를 frame 축에 반복한 값이며 frame별 phase reset이 아니다.
- Missing frame은 amplitude=0, observed=false, confidence=0, source=-1이다.
  Frequency는 마지막 관측값을, 첫 관측 이전에는 첫 관측값을 placeholder로 유지한다.
- Source ID는 0=CQT, 1=long STFT, 2=short STFT,
  3=long frequency + short amplitude/phase다.
- Frame center는 `ModalSynth`의 `align_corners=False` 보간 위치와 맞춘다.
  Phase는 frequency 적분으로 연속적으로 진행한다.
- [modal.py](../drumblender/synths/modal.py)는 active→zero 구간에서 active endpoint의 frequency를,
  zero→active 구간에서는 다음 active endpoint의 frequency를 유지한다.
  Frequency 보간은 두 endpoint 모두 active일 때 적용한다. 오래된 cache의 fmin padding에도 적용된다.
  따라서 **ampzero 관측 필터링과 inactive frequency sweep 수리는 서로 다른 처리**다.

분석은 centered/noncausal이다. 20 Hz CQT의 nominal 관측 길이는 약 1.7초이며
FFT convolution이 이 관측 길이를 줄이지는 않는다.
가까운 stationary LF 성분 분리와 빠른 pitch glide 추적에는 trade-off가 있다.
700 Hz는 구현상 포함 상한이지 이론적인 LF/HF 경계가 아니며,
현재 bandwidth 기반 병합이 움직이는 LF mode의 최적 관측을 보장하지 않는다.
Track crossing, 잡음과 겹친 peak, mode ceiling 때문에 분할·오연결·약한 성분 누락이 생길 수 있다.
Spectrogram만으로 phase 정확성이나 물리 eigenmode를 확인할 수는 없다.

CPU 실행은 STFT block 처리와 peak-local contrast, grouped Hann-response 계산,
bounded duplicate 비교, track 연산의 allocation 감소로 최적화돼 있다.
관측 해상도와 refinement를 줄여 얻는 가속이 아니다.
CUDA 옵션은 CQT/STFT 연산만 옮기며 fitting/tracking은 CPU에 남는다.
Torch synthesizer는 32-mode chunks로, NumPy reference renderer는 mode별로 합성한다.
둘 다 완성된 stateful plugin callback이 아니다. Plugin에서는 sample load 시 분석/cache하고,
재생 중에는 oscillator state와 control smoothing을 유지하는 별도 경로가 필요하다.

## Extraction

레포 최상위에서 Python 3.10–3.12 및 프로젝트에 선언된 PyTorch/torchaudio 환경을 사용한다.
입력은 [preprocessing](../PROCESSING.md)이 끝난 48 kHz mono WAV다.
기본 경로 `../dataset/processed`가 실제 데이터 위치와 다르면 변경한다.
추출기는 sample rate를 자동 변환하지 않으며, 여러 channel이 들어오면 channel 0을 사용한다.

```bash
python -m pip install -e '.[modal,modal_new]'

python -u -m scripts.build_modal_features_new \
  --processed_root ../dataset/processed \
  --out_dir ../dataset/modal_features/processed_modal_new128 \
  --num_modes 128 \
  --compute_device cpu \
  --checkpoint_every 100 \
  --resume
```

단일 worker는 기본 CPU다. `--compute_device cuda`로 spectral transforms를 GPU에서 실행할 수 있다.
먼저 앞의 일부 파일만 확인하려면 `--max_files 20`을 붙이고,
전체 추출 때 해당 옵션을 빼고 같은 명령을 실행한다.

### Parallel workers and resume

```bash
python -u -m scripts.build_modal_features_multi_gpu \
  --gpus 0,1 \
  --processed_root ../dataset/processed \
  --out_dir ../dataset/modal_features/processed_modal_new128 \
  --num_modes 128 \
  --checkpoint_every 100 \
  --resume
```

CPU 병렬 실행은 `--gpus 0,1` 대신 `--cpu_workers 2`를 사용한다.
각 worker가 정렬된 입력 목록의 서로 다른 shard를 처리하고, 완료 후 metadata를 병합한다.
`shard_0/`, `shard_1/` 등의 파일을 root metadata가 참조하므로 shard 폴더를 유지한다.
`--dry_run`은 실행할 child 명령만 표시한다.

Resume은 완료된 WAV/feature 쌍을 건너뛰며 분석 설정이 달라지면 거부한다.
Worker 수나 경로를 변경한 추출은 별도 output을 사용한다.
Checkpoint는 기본 새 성공 sample 100개마다, 정상 종료 시 한 번 더 기록한다.
WAV와 feature는 sample마다 저장하지만 metadata에 아직 기록되지 않은 파일은
비정상 종료 후 resume에서 다시 분석할 수 있다.
`--checkpoint_every 20`은 복구할 분량을 줄이는 대신 metadata 전체 재기록 횟수를 늘린다.
대량 추출에는 기본 100을 출발값으로 사용한다.

### Original CQT baseline

`original`은 jorshi DrumBlender의 nnAudio CQT + 원본 tracker이며,
ampzero 수리가 들어간 `legacy` 및 NEW 구현의 `cqt` backend와 구분한다.
원본 tracker는 그대로 두고 실행 wrapper에서 nnAudio transform/input만 GPU로 옮긴다.
Tracking은 원본처럼 CPU에서 실행한다. 원본의 placeholder/track-selection 동작도 보존한다.

```bash
python -u -m scripts.build_modal_features_multi_gpu \
  --modal_backend original --gpus 0,1 --num_modes 64 \
  --processed_root ../dataset/processed \
  --out_dir ../dataset/modal_features/processed_modal_original64_2gpu \
  --checkpoint_every 100 --resume
```

원본 기본값은 64 modes, hop 256, fmin 20 Hz, 24 bins/octave, 240 bins,
min length 10, threshold -80 dB, diff threshold 5다.
동일 mode ceiling으로 비교하려면 `--num_modes 128`과 별도 output을 사용한다.
이 명령은 현재 processed sample 길이를 사용한다. 원 논문의 2초 고정 dataset protocol을
재현한 것은 아니며, 짧은 파일은 기존 nnAudio reflect-padding fallback이 추가 padding 후 재시도한다.
`analysis_num_samples`에 실제 분석 길이를 기록하므로 길이와 padding을 확인한 뒤 시간을 비교한다.
`analysis_seconds`와 `file_seconds`는 원본에도 기록하지만 NEW의 단계별/source/confidence 진단은 없다.
GPU 실행 연결과 CPU 실행 검증을 GPU 장비에서의 속도 보장으로 해석하지 않는다.

### Outputs and timing

| 출력 | 내용 |
|---|---|
| `audio/<ID>.wav` | 분석 입력 오디오 |
| `features/<ID>.pt` | `[3, num_modes, frames]`; 남는 mode 슬롯은 0 padding |
| `modal_config.json` | 실제 실행 설정 |
| `metadata.json` | 원본 상대 경로, pack, 길이, 실제 mode 수, 후보·관측·confidence 요약, 시간 |
| `progress.json` | 최신 진행 상황, 처리 속도, ETA, 최근 sample |
| `run_stats.json` | 완료 수, mode 분포, 누적 분석/처리/단계별 시간 |
| `failures.jsonl` | 실패 sample과 오류; 실패할 때 생성 |
| `shard_*.log` | 병렬 worker 로그 |

Tqdm은 진행률과 최근 sample의 mode/LF/HF/후보 수, 처리 시간을 갱신한다.
성공 sample마다 별도 영구 console line을 남기는 형태는 아니다.
Metadata의 시간 항목은 sample별로 측정하며 checkpoint 주기에 따라 디스크에 기록한다.

- `analysis_seconds`: analyzer 내부 처리 시간.
- `analysis_stage_seconds`: CQT, long/short STFT, merge, tracking, packing별 시간.
- `file_seconds`: 읽기·분석·저장까지 포함한 sample 처리 시간. Tqdm의 `sec`도 이 값이다.

시간의 평균·최소·최대는 자동 집계하지 않는다. 필요할 때 metadata에서 계산한다.
LF/HF mode 수는 관측 frequency 중앙값을 `cqt_max_frequency`와 비교한 요약이다.
저장 `.pt`에는 전체 observation mask와 frame별 source/confidence가 별도로 들어 있지 않다.
Metadata에는 관측 frame 수와 source별 mode 수 등의 요약만 남는다.

학습 때 `data_dir`은 feature root로, `dataset_kwargs.audio_dir`은 processed WAV root로 지정한다.
`expected_num_modes`, data module의 `num_modes`, `ModalAmpParameters.num_modes` 같은
mode별 학습 parameter도 export ceiling과 맞춘다. 기존 64-mode checkpoint가 자동으로
128-mode checkpoint가 되지는 않는다. Numbered baseline config는 64-mode 기본값을 확인해야 한다.

## Review

### Saved-feature review

완료된 feature를 볼 때는 재추출하지 않는다.

```bash
python -m scripts.review_modal_dataset \
  --new_root ../dataset/modal_features/processed_modal_new128 \
  --render_device cpu \
  --host 0.0.0.0 --port 8765
```

서버는 `http://SERVER_IP:8765/`, 로컬은 `--host`를 생략하고 `http://127.0.0.1:8765/`로 연다.
`--render_device`는 `cpu`, `cuda`, `auto`를 지원하며 **저장 parameter의 합성**에만 적용한다.
Random/ID 선택은 저장된 NEW feature와 원음을 불러온다.
Legacy checkbox를 요청한 경우에만 legacy 분석을 추가로 실행한다.
WAV sample rate/길이와 저장 active-mode 수를 configuration/metadata와 대조한다.

Mel/linear 선택은 표시용 STFT와 frequency 축을 바꾸며 추출 설정을 바꾸지 않는다.
Mel 보기는 8192-sample STFT의 축을 mel로 변환하고, linear 보기는 2048-sample STFT를 사용한다.
관측 overlay는 저장 amplitude가 0이 아닌 frame을 이으며 gap은 잇지 않는다.
Frame별 source가 cache에 없어 선은 단색이고, source 수는 metadata 요약을 사용한다.

### Inspect or reanalyze samples

새 분석의 mask/source까지 확인하려면 inspector를 실행한다. WAV 또는 폴더를 받는다.

```bash
python -m scripts.inspect_modal_analysis path/to/mono_sample.wav \
  --output insights/modal_new_review --num_modes 128

python -m scripts.inspect_modal_analysis path/to/mono_sample.wav \
  --compare_legacy --legacy_modes 128 --output insights/modal_legacy_comparison
```

출력은 `index.html`, sample별 `comparison.png`, `tracks.npz`, `summary.json`, float WAV다.
`tracks.npz`에는 Hz trajectories, observed mask, confidence, source, frame times와 scores가 들어간다.
WAV는 원음/modal/잔차이며 legacy 비교를 요청하면 legacy WAV도 생긴다.
별도 normalization, limiting, clipping을 적용하지 않는다.
`original - modal` 잔차에는 phase/tracking 오차도 있으므로 clean noise target이 아니다.
Legacy 비교는 mode ceiling을 맞춰도 threshold calibration이 같다는 뜻은 아니며,
legacy 합성에도 inactive-frequency 수리가 적용된다.

HTML의 reload는 기존 WAV/그림만 다시 읽는다. 새 분석에는 inspector를 다시 실행한다.
기존 summary로 HTML만 다시 만들 때는 다음을 사용한다.

```bash
python -m scripts.inspect_modal_analysis --refresh_html --output insights/modal_new_review
```

Inspector는 `--backend stft`/`cqt`, builder는 `--modal_backend stft`/`cqt`를 지원한다.
`--no_refine`은 complex-lobe fitting과 phase frequency refinement를 비활성화하는 비교 옵션이다.
Modal-only magnitude convergence만으로 전체 modal/noise reconstruction이나 phase 정확성을 판정하지 않는다.

## Neural control

기존 sample의 modal trajectory에서 frequency/amplitude/initial-phase의 공동 시간 변화를 학습하고,
선택한 learned pattern의 amount를 조절한다. Text, CLAP, instrument/physical annotation,
harmonic alignment, 단조 decay 가정을 요구하지 않는다.
모델·전용 학습·audition 코드는 있지만 학습 품질, held-out control, DDP와 realtime 성능을
검증 완료한 것으로 취급하지 않는다. 기존 `cfg/06_modal.yaml`의 `ModalAmpParameters`는
mode-index별 amplitude multiplier이며 아래 modal pattern encoder와 다른 모델이다.

### Representation and pattern editing

기본 width는 [cfg/modal_control.yaml](../cfg/modal_control.yaml)에서 관리한다.
`B`=batch, `K`=mode 수, `T`=modal frames, `Tc=ceil(T/4)`다.

| 구성 | 기본 shape / 동작 |
|---|---|
| Frequency/amplitude stems | Observed log-frequency 또는 sample-gain-normalized log-amplitude와 mask → shared temporal Conv1d; 각각 `[B,K,Tc,24]` |
| Phase stem | Initial `(sin(phi0),cos(phi0))` → MLP; `[B,K,16]`을 시간축 broadcast |
| Joint backbone | 64-channel projection, temporal residual blocks, 8 inducing tokens를 이용하는 mode interaction 두 개 |
| Decoder | Temporal/mode interaction, upsampling, log-frequency/log-amplitude heads와 circular initial-phase head |
| Temporal SAE | Hidden `[B,K,Tc,64]` → Conv1d `64→256`, kernel 9, frame별 nonnegative TopK 8 |

Mode-index embedding이나 sample 간 frequency-rank alignment를 사용하지 않는다.
Mode token을 유지하고 shared 함수/masked set attention으로 permutation equivariance를 구성한다.
Inducing attention의 score 비용은 대략 `O(Tc*K*8)`이며 projection/convolution 비용은 별도다.
Padding mode/time을 attention/loss에서 제외하고, 학습용 masking과 실제 관측 mask를 구분한다.
Separate stems는 단위/topology 차이에 대한 inductive bias이지 strict disentanglement 보장이 아니다.

Backbone 학습 후 고정한 hidden `H0`에서 SAE를 학습한다.
Feature `m`의 activation은 `s_u[...,m]=(1+u)*s_0[...,m]`, `u>=-1`로 편집하고
`H_u=H0+C(s_u)-C(s_0)`로 SAE reconstruction error가 reference를 덮어쓰지 않게 한다.
Reference TopK support를 고정해 amount 변화 중 feature 교체를 줄인다.
하나의 feature가 여러 mode와 attribute를 함께 바꿀 수 있지만 실제 유용성은 audition해야 한다.
Multiplicative edit는 source에 비활성인 feature를 새로 만들지 못한다.
Kernel 9의 nominal context는 약 `9*4*feature_hop/sample_rate`이며 긴 trajectory 전체가
한 atom에 저장된다는 뜻은 아니다.

편집은 baseline decoder와 edited decoder의 차이를 fitted reference에 적용한다.

```text
log f_u = log f_0 + D_f(H_u) - D_f(H_0)
a_u = a_0 * exp(D_a(H_u) - D_a(H_0))
phi0_u = wrap(phi0_0 + arg(c(H_u) * conj(c(H_0))))
```

`c`는 정규화 circular phase 출력이다. 작은 vector norm에는 fallback을 사용한다.
Zero amount는 fitted reference parameter를 직접 반환한다.
Mode 수, observation support, amp=0 규칙을 유지하며 gain/frequency/Nyquist 경계를 제한한다.
첫 구현은 mode birth/death와 support 밖 tail extension을 제공하지 않는다.
Frequency는 연속 적분하고 initial-phase 변경은 다음 trigger에 적용하는 방향이다.
현재 full-buffer renderer가 이 live plugin 동작까지 구현했다는 뜻은 아니다.

### Noise linkage and neutral

```text
modal cache → modal encoder/decoder → fitted modal → ModalSynth → m
                            │ detached context
original x - detach(m) → DAC-style encoder → conditioned noise bands → NoiseGenerator → n
                                                                       output = m+n
```

통합 모델의 neutral은 **학습된 modal과 그 modal을 기준으로 학습된 noise**다.
Extraction cache 그대로의 합성이라는 뜻은 아니다.
Noise 입력 residual과 modal context를 detach하고 modal은 자신의 renderer와 anchor에서 gradient를 받는다.
Residual에는 modal 오차도 남으므로 clean noise supervision으로 부르지 않는다.
Inference에서는 편집 전에 residual encoding을 한 번 cache한다.
`x-edited_modal`을 다시 noise encoder에 넣거나 edited output을 unchanged source에 맞추지 않는다.

`noise_link=0`은 기준 noise bands 유지이며 기본값이다.
`noise_link=1`은 modal context에 따른 noise 변화량을 적용하며 중간 값은 log-amplitude에서 보간한다.
이는 관측 correlation을 학습한 연동이며 edited-noise 정답을 학습한 counterfactual control은 아니다.
청취 결과에 따라 fixed noise를 유지할 수 있다.

### Extensions and evaluation

VAE의 KL prior만으로 의미 있는 knob가 식별되지는 않는다.
현재 deterministic backbone + SAE는 generative prior나 OOD 안정성을 제공하지 않는다.
VFM은 frozen joint latent trajectory의 variation을 reference proximity와 feature target으로
conditioning하는 후속 가설이며 아직 구현하지 않았다.
예를 들어 `E(H)=lambda*||A_m(H)-target_m||^2+mu*||H-H_reference||^2`를 쓸 수 있지만
perceptual selectivity가 보장되지는 않는다. Hard TopK guidance, permutation-consistent prior/path,
sampling 비용을 별도로 검토해야 한다. Sampling은 audio callback 밖에서 수행하는 방향이다.
CFG는 text 전용이 아니지만 의미 있는 pattern을 자동 발견하지 않으며 첫 모델에 포함하지 않는다.

같은 checkpoint에서 neutral fidelity, zero 복귀, amount sweep, 작은 edit의 연속성,
held-out sample, pattern 조합과 실패 예시를 확인한다.
High activation을 좋은 knob의 증거로 취급하지 않고 실제 decoded audio를 audition한다.
Uniform gain/pitch, PCA/random direction과 비교해 learned pattern의 이득을 확인한다.
Noise seed를 맞춰 stochastic 차이를 control 효과와 구분한다.
기여 후보는 reference에 연결된 coupled temporal pattern의 유용한 편집이며,
SAE/VFM 적용 자체나 Hybrid 사용 자체만으로 novelty를 주장하지 않는다.

## Training and audition

기존 oscillator + filtered-noise 구현 전용 CLI다. [연구 목표 A/B](research_implementation_goals.md)의
새 architecture용 학습 명령으로 해석하지 않는다.

| Stage | 학습 범위 / 목적 |
|---|---|
| `modal` | Modal encoder/decoder; observed parameter anchor, circular initial-phase loss, rendered-modal MR-STFT, masked reconstruction |
| `noise` | Modal 고정; mixture MR-STFT와 약한 detached-residual spectral target |
| `joint` | 두 branch fine-tuning; modal anchor 유지, modal LR은 noise의 0.1배 |
| `patterns` | 최종 backbone 고정; temporal SAE만 sparse hidden reconstruction |

Modal target은 extracted modal을 합성한 waveform이며 raw mixture를 modal 단독 target으로 쓰지 않는다.
Loss는 absolute decoder reconstruction에 건다. Reference를 그대로 통과시키는 neutral edit에만
loss를 걸면 network의 reconstruction 학습 신호가 없다.
Training MR-STFT는 512/1024/2048 중 clip 길이에 맞는 FFT를 쓰며 아주 짧으면 더 작게 한다.
이것은 canonical 평가 metric이나 평가 WAV 길이를 바꾸는 처리가 아니다.
Temporal FFT 보조 loss는 modulation spectrum이며 raw carrier spectrum과 다르다.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MODAL_ROOT=../dataset/modal_features/processed_modal_new128

python -m scripts.train_modal_control --stage modal \
  --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1

python -m scripts.train_modal_control --stage noise \
  --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1 \
  --init_from runs/modal_control/modal/checkpoints/last.ckpt

python -m scripts.train_modal_control --stage joint \
  --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1 \
  --init_from runs/modal_control/noise/checkpoints/last.ckpt

python -m scripts.train_modal_control --stage patterns \
  --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed --devices 0,1 \
  --init_from runs/modal_control/joint/checkpoints/last.ckpt
```

Stage 이동은 weights만 불러오고 optimizer를 새로 만드는 `--init_from`을 사용한다.
명령은 연결하기 쉬운 `last.ckpt` 예시이며 출력된 best checkpoint로 바꿀 수 있다.
같은 stage의 재개는 `--init_from` 대신 `--resume <same-stage-checkpoint>`를 사용한다.
Modal만 확인하려면 `modal → patterns`도 가능하다.
Joint를 다시 학습하면 dictionary 좌표도 달라지므로 patterns를 다시 학습한다.

GPU 한 대는 `--devices 1`, CPU는 `--accelerator cpu --devices 1`이다.
`--devices 0,1`은 CUDA visibility 적용 후 logical IDs다.
기본 batch size 2는 GPU당 값이며 OOM이면 `--batch_size 1`을 사용한다.
긴 sample은 전체 길이를 처리하므로 differentiable modal rendering의 memory를 확인해야 한다.
Width, loss 계수, epochs는 config의 출발값이며 최적값으로 검증된 것은 아니다.
기존 `drumblender fit` 설정과 전용 task/collate/optimizer 경로를 섞지 않는다.

`runs/modal_control/<stage>/`에는 `run_config.json`, `logs/version_*/metrics.csv`,
`checkpoints/last.ckpt`, best validation checkpoint가 기록된다.
Lightning이 DDP와 progress bar를 담당하며 validation noise는 고정 seed를 쓴다.
SAE `feature_fraction`은 batch 내 feature 사용률이지 corpus dead-feature 통계가 아니다.

```bash
python -m scripts.audition_modal_control \
  --checkpoint runs/modal_control/patterns/checkpoints/last.ckpt \
  --feature_root "$MODAL_ROOT" --processed_root ../dataset/processed \
  --amount 0.5 --noise_link 0 --seed 7 --device cuda:0 \
  --output insights/modal_control_fixed_noise
```

같은 sample/feature/seed로 `--noise_link 1`과 다른 output 폴더를 사용하면 연동을 비교할 수 있다.
`--sample_id`와 `--feature`로 대상을 고정한다. 생략하면 seed로 sample을 고르고 가장 활성화된
feature를 선택하지만, 그 feature의 유용성을 보장하지는 않는다.
출력은 `original.wav`, neutral/edited 전체·modal·noise WAV,
`modal_parameters.pt`, feature/amount/seed/peak를 기록한 `report.json`이다.
공통 amplitude 기준의 float WAV와 동일 excitation을 사용하며 개별 peak normalization은 없다.
같은 폴더에는 같은 파일명으로 저장되므로 비교 실행에는 다른 output을 사용한다.

미구현 범위는 VFM, native plugin/stateful streaming, source에 없는 feature의 additive injection,
자동 semantic naming이다. 첫 학습에서는 finite loss/gradient, short/long memory,
checkpoint 전환과 DDP부터 확인하고 이후 neutral/control 청취를 진행한다.

## Maintenance

일회성 synthetic benchmark와 이전 소스 비교 스크립트, 오래된 실행 report는 유지하지 않는다.
실제 데이터 처리·확인 도구와 재발 방지 테스트만 남긴다.

| 파일 | 용도 |
|---|---|
| [build_modal_features_new.py](../scripts/build_modal_features_new.py) | NEW 추출 진입점 |
| [build_modal_features.py](../scripts/build_modal_features.py) | 분석·저장·checkpoint 공통 구현 |
| [build_modal_features_multi_gpu.py](../scripts/build_modal_features_multi_gpu.py), [merge_modal_shards.py](../scripts/merge_modal_shards.py) | CPU/GPU workers와 metadata 병합 |
| [modal_extraction_stats.py](../scripts/modal_extraction_stats.py) | 추출 때 사용하는 metadata 요약; 일회성 실험 도구가 아님 |
| [inspect_modal_analysis.py](../scripts/inspect_modal_analysis.py) | 실제 sample 재분석·시각화 |
| [review_modal_dataset.py](../scripts/review_modal_dataset.py), [HTML](../scripts/modal_dataset_review.html) | 저장 feature의 검토 |
| [test_modal_analysis_new.py](../test/unit/utils/test_modal_analysis_new.py) | 실제 관측, gap, LF boundary, HF envelope, mode 수, padding/phase 회귀 |
| [test_modal_analysis.py](../test/unit/utils/test_modal_analysis.py), [test_modes.py](../test/unit/synths/test_modes.py) | Legacy 분석과 modal 합성 회귀 |

관측 필터링, inactive-frequency 수리, quiet LF tail, HF burst envelope,
CQT fmin guard, frame alignment와 gradient 보존은 서로 다른 회귀 항목이다.
분석 코드 수정 후 필요한 검증은 기존 테스트를 사용한다.

```bash
python -m pytest \
  test/unit/utils/test_modal_analysis_new.py \
  test/unit/utils/test_modal_analysis.py \
  test/unit/synths/test_modes.py \
  --confcutdir=test/unit -o addopts= -p no:cacheprovider -q
```

이 명령은 modal 범위의 회귀 검사이며 전체 학습, GPU throughput,
청취 품질이나 plugin callback deadline의 검증이 아니다.
검토용 생성 파일은 `insights/` 아래에 두며 input WAV와 cache가 없으면 재생성해야 한다.

## Related work

아래는 기존 설계에서 참고한 연구와 적용 범위다. 이 논문의 성능이 현재 percussion 모델에서
재현됐다는 뜻은 아니며, 확장 후보는 위의 구현 상태와 구분한다.

| 연구 | 적용 범위 |
|---|---|
| [Interactive Neural Resonators, NIME 2023](https://nime.org/proceedings/2023/nime2023_81.pdf) | NN parameter prediction + DSP rendering. 물리 annotation 기반 control은 도입하지 않음 |
| [PLAUD, preprint](https://arxiv.org/html/2608.13724) | Playable latent와 reconstruction trade-off; modal-specific 방법으로 간주하지 않음 |
| [TSPulse, preprint](https://arxiv.org/html/2505.13033v1) | Masked temporal/attribute reconstruction 참고. Raw temporal FFT와 modal frequency 채널을 구분 |
| [Object-Centric Latent Action Learning, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/39423/43384) | Mode identity/관계 유지 참고. 자연 감쇠를 agent action으로 해석하거나 action-label probe를 도입하지 않음 |
| [Words in Motion, ICLR 2025](https://arxiv.org/html/2406.11624) | Hidden intervention 참고. 큰 PCA variance를 곧바로 creative knob로 해석하지 않음 |
| [Discovering and Steering Interpretable Concepts in Large Generative Music Models, ICLR 2026](https://arxiv.org/html/2505.18186) | SAE discovery/steering 참고. Temporal modal dictionary는 응용 가설이며 language labeling은 제외 |
| [SAEs Are Good for Steering — If You Select the Right Features, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.519/) | Activation 크기와 실제 output steering 효과를 구분하고 audition으로 선별 |
| [Controlled Generation with Equivariant Variational Flow Matching, ICML 2025](https://proceedings.mlr.press/v267/eijkelboom25a.html) | Reference/feature target 기반 후속 variation 가설. Molecular symmetries나 원문의 성능 주장은 이식하지 않음 |
