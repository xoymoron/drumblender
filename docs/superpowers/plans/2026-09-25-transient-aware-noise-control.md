# Transient-aware Noise Control Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task when implementation is requested. Steps use checkbox syntax. This document is a proposed implementation plan, not an execution record.

**Goal:** 기존 percussion sample의 phase-sensitive reconstruction과 발견 가능한 noise latent controls를 두 audible branches 안에서 구현한다.

**Architecture:** Frozen reference modal과 독립적인 continuous texture autoencoder를 합성한다. Texture decoder는 positive spectral envelope와 learned complex excitation을 곱해 iSTFT로 합성하며, reconstruction 학습 후 SAE delta steering을 추가한다. 원본 분석 결과를 cache하여 modal/noise edits가 서로를 재분석으로 상쇄하지 않게 한다.

**Tech Stack:** Python >=3.10,<3.13, torch/torchaudio 2.7.1, pytorch-lightning 1.9.5, 기존 jsonargparse/pytest. Prototype에 새 필수 library를 추가하지 않는다.

**Spec:** [2026-09-25-transient-aware-noise-control-design.md](../specs/2026-09-25-transient-aware-noise-control-design.md). 수식, tensor shapes, initial hyperparameters, 문헌의 적용 범위는 이 spec을 따른다.

## Global Constraints

- Python >=3.10,<3.13; torch/torchaudio 2.7.1; pytorch-lightning 1.9.5를 유지한다.
- Prototype은 48 kHz mono, offline sample analysis와 cached control rendering이다.
- 기존 checkpoint/config/평가 protocol을 보존하고 새 architecture에는 새 class와 config를 사용한다.
- pretrained codec, VQ, text/CLAP supervision, 세 번째 audible transient branch는 초기 구현의 의존성이 아니다.
- 새로운 reconstruction 경로에는 raw waveform/complex-spectrum decoder skip을 두지 않는다.
- valid length를 보존하고 평가 대상 audio를 조용히 zero-pad하거나 tail-trim하지 않는다.
- 기존 작업 중인 파일과 사용자 변경을 덮어쓰거나 연구 baseline을 삭제하지 않는다.

## Review Focus

- 1-sample/짧은 silence/첫·마지막 sample impulse: STFT boundary와 exact-length reconstruction. Task 2.
- 두 파일의 mode order/metadata가 다르거나 cache가 부분 저장됨: 잘못된 confidence 연결을 조용히 허용하지 않음. Task 1.
- batch padding이 다르거나 sample에 긴 tail이 있음: 같은 sample의 유효 구간 결과가 batch 구성에 따라 변하지 않음. Tasks 3–4.
- neutral/역방향 knob 및 modal+noise 조합: 원본 latent 보존과 residual 재분석 방지. Tasks 6–7.
- 기존 checkpoint와 새 checkpoint를 같은 export 경로에서 읽음: strict loading 및 기존 waveform interface 유지. Task 5.

## File map

| File | 책임 |
|---|---|
| `drumblender/data/texture.py` | Metadata-aware dataset, collate, TextureDataModule |
| `scripts/build_modal_features.py` | 기존 tensor 옆 metadata sidecar와 resume 검증 |
| `drumblender/synths/spectral_noise.py` | STFT geometry, ERB envelope projection, complex synthesis |
| `drumblender/models/texture.py` | Local/global encoder와 두 output heads |
| `drumblender/tasks/texture.py` | analyze/render cache, losses, Lightning training |
| `drumblender/loss.py` | Length-aware complex/envelope/flux reconstruction objectives |
| `drumblender/control/sparse.py` | TopK SAE와 source-anchored delta control |
| `drumblender/evaluation/texture_control.py` | Intervention sweeps/diagnostics/HTML audition |
| `scripts/train_texture_controls.py` | Frozen reconstruction latent 수집과 SAE 학습 |
| `scripts/evaluate_texture_controls.py` | 저장된 candidate bank를 held-out samples에 평가 |
| `scripts/diagnose_texture.py` | DSP oracle와 32-sample overfit manifest 생성/진단 |
| `cfg/06_texture.yaml`, `cfg/data/texture.yaml` | 새 reconstruction task와 dataset |
| `cfg/texture_controls.yaml` | SAE/calibration/knob candidate 설정 |
| `drumblender/utils/model.py`, `scripts/export_recon_wavs.py` | 새 task loading/export와 기존 경로 회귀 보존 |

새 `control`/`evaluation` packages에 `__init__.py`를 추가한다. 기존 `models`/`tasks` package exports는 필요한 class만 노출한다. `pyproject.toml`의 explicit top-level-only package list를 `drumblender*` package discovery로 바꾸고 wheel contents를 확인한다. 기능과 관계없는 전체 repository rename/deletion은 하지 않는다.

## Task 1: Modal metadata를 학습 데이터까지 연결

**Files:** modify `scripts/build_modal_features.py`; create `drumblender/data/texture.py`, `test/unit/data/test_texture.py`.

**Interface:** Dataset item은 `(waveform:[1,L], modal:dict[str,Tensor], length:int)`. Modal keys는 `params:[3,M,Fm]`, `observed:bool[M,Fm]`, `confidence:[M,Fm]`, `frame_times:[Fm]`, `available:bool[]`. Batch에는 `frame_lengths:long[B]`를 추가한다. Params frequency units는 기존 radians/sample, phase는 initial sine phase, frame_times는 seconds다.

- [ ] metadata sidecar roundtrip/misalignment/missing/resume 테스트를 먼저 작성한다. Fixture는 서로 다른 2 modes, 3 frames로 구성하고 한 mode만 observed로 둔다. 저장→load 후 각 params slot과 confidence slot이 동일해야 한다. Feature hash가 다르면 `ValueError`여야 한다.
- [ ] Feature export의 NEW backend에서 detailed analysis result를 한 번 얻어 params와 sidecar를 같은 mode ordering으로 저장한다. Sidecar 이름은 `<key>.modal_meta.pt`; metadata JSON에 `modal_metadata_file` key를 추가한다. Payload는 위 keys 외 `schema_version=1`, `sample_rate`, `num_samples`, `feature_sha256`, `analysis_config_sha256`, `source_ids`를 포함한다. 원래 `[3,M,Fm]` feature는 유지한다.
- [ ] 모드 padding에는 observed=False/confidence=0/source_id=-1을 적용한다. Sidecar와 feature 저장 성공 후 metadata entry를 갱신한다. Resume 시 파일 존재만 보지 말고 hashes/schema/shape를 검증한다. 기존 feature만 있는 항목은 sidecar를 만들기 위해 재분석한다.
- [ ] `TextureDataset(AudioWithParametersDataset)`는 기존 split 선택을 재사용하되 `num_samples=None`, `normalize=False`, mono/48k를 강제한다. 자동 truncate 또는 waveform-only normalization을 거부한다. `require_modal_metadata=True`가 기본값이다. legacy fallback을 명시하면 available=False로 반환한다.
- [ ] `texture_collate`는 audio, modes, frames를 별도로 pad하고 actual lengths를 유지한다. frame_times의 padding 값은 0이며 frame_lengths 밖은 항상 mask한다. TextureDataModule은 AudioDataModule을 상속하고 세 dataloaders에 이 collate를 적용한다.

검증할 핵심 assertion:

```python
x, modal, lengths = texture_collate([
    (torch.ones(1, 11), first_modal, 11),
    (torch.ones(1, 7), second_modal, 7),
])
assert x.shape == (2, 1, 11)
assert lengths.tolist() == [11, 7]
assert modal['frame_lengths'].tolist() == [3, 2]
assert not modal['observed'][1, :, 2:].any()
```

Run: `python -m pytest test/unit/data/test_texture.py --confcutdir=test/unit -o addopts= -q`.

## Task 2: Spectral DSP primitive와 oracle closure

**Files:** create `drumblender/synths/spectral_noise.py`, `test/unit/synths/test_spectral_noise.py`.

**Interface:** `SpectralNoiseSynth(n_fft=512, hop_length=64, sample_rate=48000, bands=32)`; `analysis(x:[B,L])->complex[B,F,T]`; `factorize(R)->(log_A,U)`; `synthesis(log_A,U,length:int)->[B,L]`. `factorize`가 반환하는 log_A는 full frequency grid다. Neural q:[B,32,T]는 `expand_envelope(q)`로 동일 grid를 얻는다.

- [ ] 아래 oracle tests를 먼저 작성하고 import failure를 확인한다. Test L은 1,17,511,512,513,48000; silence/first impulse/last impulse/seeded random input이다.

```python
synth = SpectralNoiseSynth()
x = torch.randn(1, 513) * 0.1
R = synth.analysis(x)
log_A, U = synth.factorize(R)
y = synth.synthesis(log_A, U, length=x.shape[-1])
assert y.shape == x.shape
torch.testing.assert_close(y, x, atol=2e-6, rtol=2e-5)
```

- [ ] Spec의 constant-padding STFT/iSTFT와 ERB projection을 구현한다. Window, Bf, ridge projection은 buffers로 등록한다. DC/Nyquist imag를 zero로 만드는 연산은 autograd를 유지한다. FFT 연산에는 autocast를 끈다.
- [ ] Silence는 finite zero 출력, nonempty short input은 exact length, zero-length input은 명시적 `ValueError`로 처리한다. Factorization target에 denominator epsilon을 추가해서 oracle equality를 깨지 않도록 `U=R/A`와 positive floor를 사용한다.
- [ ] synthesis backward에서 log_A/U의 gradients가 finite인지 확인한다. Real output과 STFT roundtrip을 모두 확인한다. Oracle closure를 neural model의 학습 성공으로 보고하지 않는다.

Run: `python -m pytest test/unit/synths/test_spectral_noise.py --confcutdir=test/unit -o addopts= -q`.

## Task 3: Codec와 독립적인 texture autoencoder

**Files:** create `drumblender/models/texture.py`, `test/unit/models/test_texture.py`.

**Interfaces:** `TextureEncoder.forward(features:[B,8,257,T])->z:[B,64,T]`; `TextureDecoder.forward(z)->(q:[B,32,T], U:complex[B,257,T])`. Decoder는 x/R/modal waveform을 인자로 받지 않는다. `build_features(x,m_ref,modal,scale)`은 Task 4가 호출하는 analysis helper이며 single-item true lengths를 사용한다.

- [ ] odd T=3,17,751에서 shape/finite gradients tests를 작성한다. `inspect.signature(TextureDecoder.forward)`의 입력이 latent뿐인지 검사하기보다 실제 decoder 호출에 x/R를 전달하는 경로를 만들지 않는다. z를 바꾸면 output이 바뀌고 같은 z에는 같은 output이 나오는지를 검사한다.
- [ ] Spec의 frequency stride conv + 2D blocks를 구현한다. LayerNorm은 spatial padding statistics를 섞지 않는 channel normalization을 사용한다. Temporal downsampling은 global context 경로에만 있다.
- [ ] global pooling은 마지막 불완전한 16-frame group을 실제 frame 수로 나눈다. Positional encoding은 physical frame time을 사용한다. Interpolation은 원래 frame-center coordinates에 맞춘다. Local/global representation 합산 후 Linear128→64를 적용한다.
- [ ] decoder는 frequency축만 보간하고 시간축 해상도는 유지한다. q/U output heads를 분리한다. `factorized=False`이면 q를 사용하지 않고 log_A=0으로 처리하는 A=1 ablation을 지원한다.
- [ ] confidence projection은 observed mask를 먼저 적용하고 modal frame_times를 이용한다. missing endpoints를 넘어 confidence를 만들어내지 않는다. metadata-available map을 별도로 제공한다.

Core test:

```python
features = torch.randn(1, 8, 257, 17)
z = TextureEncoder()(features)
q, U = TextureDecoder()(z)
assert z.shape == (1, 64, 17)
assert q.shape == (1, 32, 17)
assert U.shape == (1, 257, 17) and U.is_complex()
(q.square().mean() + U.abs().mean()).backward()
```

Run: `python -m pytest test/unit/models/test_texture.py --confcutdir=test/unit -o addopts= -q`.

## Task 4: Two-branch task, cache와 reconstruction training

**Files:** create `drumblender/tasks/texture.py`, `cfg/06_texture.yaml`, `cfg/data/texture.yaml`, `test/unit/test_texture_task.py`; modify `drumblender/loss.py`; create `test/unit/test_texture_loss.py`.

**Interfaces:** `PercussionTextureTask(config:dict)`는 LightningModule이며 config를 `save_hyperparameters`로 저장한다. Public `forward(x,params,lengths=None)->padded_waveform`와 `loss_fn(y_hat,x,lengths)`를 제공한다. `analyze(x,params,lengths)->list[TextureState]`, `render(states,noise_controls=None,modal_override=None)->padded_waveform`을 추가한다.

`TextureState`는 `z:[1,64,T]`, `modal_ref:[1,L]`, `scale:[1,1]`, `length:int`를 가진다. Original waveform/R/U target은 training auxiliary에만 있고 render state에는 없다. Public cache는 detach하고, training의 internal analysis는 gradient를 유지한다. Controls는 Task 6의 SAE adapter가 state.z에 적용한다. `modal_override`는 cached modal waveform을 대체할 `[B,1,Lmax]`이며 재분석하지 않는다.

- [ ] unequal-length batch와 개별 forward의 유효 samples가 같은지, neutral cached render와 forward가 같은지 tests를 작성한다. modal_override만 변경하면 texture component가 동일한지 확인한다.
- [ ] reference 구현은 batch item마다 true length로 analysis/encoder/decoder를 호출하고 마지막에 waveform을 pad한다. Params도 frame_lengths로 slice한다. 이렇게 padding에 따른 global context/conv contamination을 먼저 제거한다. 이후 vectorization은 동일성 test 아래에서만 진행한다.
- [ ] reference modal은 기존 ModalSynth를 사용하고 freeze한다. modal cache에서 쓰는 radians/sample/initial-phase convention을 유지한다. Config의 num_modes와 cache metadata가 다르면 명시적으로 reject한다.
- [ ] training helper `forward_with_aux`는 prediction과 R/U*/log_A*/predicted q/U를 반환한다. Public forward는 waveform만 반환한다. Loss는 spec의 waveform/complexMR/magnitudeMR/envelope/flux/A/U 정의를 구현한다. Metrics 전용 기존 MR-STFT 이름과 수치는 변경하지 않는다.
- [ ] `drumblender.loss.complex_mr_loss(prediction:Tensor,target:Tensor,lengths:Tensor)->Tensor`를 독립 함수로 제공한다. Public `loss_fn`은 spec의 waveform/complexMR/magnitudeMR/envelope/flux 항만 사용하고 A/U auxiliary는 `forward_with_aux`의 training objective에 더한다.
- [ ] loss length mask 테스트에서 padded region을 큰 값으로 바꿔도 loss가 변하지 않아야 한다. Complex loss는 동일 magnitude/다른 phase를 구분해야 한다.

```python
x = torch.randn(1, 1, 2048) * 0.1
assert complex_mr_loss(x, x, lengths=torch.tensor([2048])) < 1e-7
assert complex_mr_loss(-x, x, lengths=torch.tensor([2048])) > 0.1
```

- [ ] `cfg/06_texture.yaml`의 class_path는 `drumblender.tasks.texture.PercussionTextureTask`. config에는 `sample_rate:48000`, `n_fft:512`, `hop_length:64`, `latent_dim:64`, `bands:32`, `num_modes:128`, `factorized:true`, `global_context:true`, spec loss weights를 명시한다. Optimizer는 AdamW2e-4/betas[.9,.99]/weight_decay1e-4, trainer gradient_clip_val1.0이다. Data config는 TextureDataModule과 Task 1 dataset, num_samples:null, normalize:false, metadata 필수다.
- [ ] GPU memory 최적화 전 full-length reference를 유지한다. Long clip 실패는 clip을 조용히 잘라 해결하지 않는다. 작은 batch 또는 gradient accumulation을 사용하고 로그에 실패 길이를 남긴다.

Run: `python -m pytest test/unit/test_texture_task.py test/unit/test_texture_loss.py --confcutdir=test/unit -o addopts= -q`.

## Task 5: Loader/export/packaging과 baseline 비교 경로

**Files:** modify `drumblender/utils/model.py`, `scripts/export_recon_wavs.py`, `pyproject.toml`; create `test/integration/test_texture_roundtrip.py`, `scripts/diagnose_texture.py`.

- [ ] old DrumBlender와 새 PercussionTextureTask 각각 tiny checkpoint를 저장한 후 config+checkpoint reload한 waveform이 같은지 integration test를 작성한다. missing/unexpected state_dict key는 strict load에서 실패해야 한다.
- [ ] loader parser의 허용 base를 LightningModule로 넓힌다. 새 task에는 `type(init.model).load_from_checkpoint(ckpt,config=init.model.config,strict=True,map_location='cpu')`를 사용한다. 기존 DrumBlender constructor 경로는 유지한다. 지원하지 않는 class는 명확하게 reject한다. 모든 task를 기존 base constructor로 introspect하지 않는다.
- [ ] export는 model config를 기준으로 dataset 경로를 선택하고 structured params를 recursive unsqueeze/device move한다. 기존 tensor input도 유지한다. Loss export는 public waveform-only loss_fn을 사용한다. 신규 training auxiliary losses를 기존 metric 결과로 섞지 않는다.
- [ ] export config override 중 encoder/transient 전용 옵션을 새 task에 지정하면 invalid combination을 알린다. 데이터와 모델 config를 모두 manifest에 보존한다. 현재 script의 data config 해석/직접 dataset 생성 부분도 새 class를 따르게 한다.
- [ ] setuptools package discovery는 다음으로 바꾼다. Wheel을 임시 directory에 만들고 zip member에 `drumblender/models/texture.py`, `drumblender/control/sparse.py`가 포함되는지 Task 6 이후 확인한다.

```toml
[tool.setuptools.packages.find]
include = ["drumblender*"]
exclude = ["drumblender_OG*"]
```

- [ ] `diagnose_texture.py`는 manifest JSON을 입력받아 DSP closure, oracle/free-coefficient diagnostic, 32-sample overfit용 subset과 metrics JSON을 만든다. Manifest에는 sample id/path/hash/type/split/valid length/modal hash/model config hash/seed를 기록한다. Synthetic impulse/flam/HF-burst cases도 저장한다.
- [ ] baseline configs를 복사해 별도 run directory에 pin한다. 현재 `05_all_parallel.yaml` 자체의 encoder는 SoundStream 설정이므로 DAC+LSTM 비교에는 해당 upgrade config override를 명시적으로 적용하고 resolved config를 저장한다. 같은 modal cache로 R0/R1을 비교한다.

Run: `python -m pytest test/integration/test_texture_roundtrip.py -o addopts= -q`.

Diagnostic CLI contract to implement:

```text
python -m scripts.diagnose_texture --build-manifest --data-config cfg/data/texture.yaml --manifest analysis/texture/manifest.json --seed 0
python -m scripts.diagnose_texture --manifest analysis/texture/manifest.json --output analysis/texture/diagnostic --seed 0
drumblender fit --config cfg/06_texture.yaml --trainer.max_steps 20000
```

`--build-manifest`는 data config의 실제 dataset index에서 kick/tom/snare/cymbal을 각 최대 8개 선택하고 sample/hash/length를 검증한다. 부족한 class는 다른 class로 몰래 대체하지 않고 available count를 보고한다. 작은 overfit은 train split에서 ≤2초의 32개를 선택하며 evaluation manifest는 별도의 held-out split으로 만든다. 경로가 없으면 사용법과 필요한 schema를 보여주며 dataset을 추정하지 않는다. 작은 overfit 결과가 성공하기 전 장기 training을 시작하지 않는다.

## Task 6: SAE delta control 학습과 calibration

**Files:** create `drumblender/control/__init__.py`, `drumblender/control/sparse.py`, `scripts/train_texture_controls.py`, `cfg/texture_controls.yaml`, `test/unit/control/test_sparse.py`.

**Interface:** `SparseTextureControls(latent_dim=64, dictionary_size=512, topk=8)`; `encode(z)->a`; `decode(a)->z_approx`; `steer(z,feature:int,alpha:float,schedule:str)->z_edited`. Normalization mean/std, q95, usage counts와 candidate validity는 buffers/checkpoint metadata로 저장한다.

- [ ] zero control identity와 feature index/range validation tests를 작성한다. Input의 z를 in-place 수정하면 실패해야 한다. q95=0인 inactive feature는 request 시 오류로 알려야 한다.

```python
before = z.clone()
edited = controls.steer(z, feature=active_feature, alpha=0.0, schedule='constant')
torch.testing.assert_close(edited, before, atol=0, rtol=0)
torch.testing.assert_close(z, before, atol=0, rtol=0)
```

- [ ] ReLU→TopK8 encoder와 unit-norm linear decoder를 구현한다. Decoder columns는 optimizer step 후 renormalize한다. std floor1e-5, 사용 가능한 frames만 train statistics에 포함한다. α=0에는 원래 z를 반환하는 exact identity 경로를 둔다.
- [ ] training은 reconstruction task eval/frozen으로 train split z를 수집하고 SAE MSE를 최적화한다. AdamW lr1e-3, weight_decay0, batch4096 frames, max100k updates, validation patience10 checks를 초기값으로 둔다. 1000 updates마다 validation을 측정하고 best checkpoint를 보존한다. feature usage는 zero/nonzero counts와 positive q95를 기록한다.
- [ ] calibration은 train positive activation의 q95를 사용한다. schedule은 constant 또는 activation-gated다. Alpha grid와 feature마다 허용 범위를 config에 저장한다. 그 범위는 empirical calibration이며 수학적 artifact-free 보장이 아니다.
- [ ] SAE decoder difference만 원본 latent에 더한다. Re-Bottleneck이나 SAE approximate reconstruction으로 원본 z를 대체하지 않는다. control checkpoint에는 reconstruction checkpoint SHA256와 latent/schema version을 포함하고 mismatch를 reject한다.

Run: `python -m pytest test/unit/control/test_sparse.py --confcutdir=test/unit -o addopts= -q`.

CLI contract:

```text
python -m scripts.train_texture_controls --model-config cfg/06_texture.yaml --checkpoint checkpoints/texture.ckpt --config cfg/texture_controls.yaml --output analysis/texture/controls
```

## Task 7: Output intervention 평가와 artist audition

**Files:** create `drumblender/evaluation/__init__.py`, `drumblender/evaluation/texture_control.py`, `scripts/evaluate_texture_controls.py`, `test/unit/evaluation/test_texture_control.py`.

**Interface:** `evaluate_sweep(model,states,controls,feature,alphas)->list[dict]`. 각 record는 sample/feature/alpha/schedule, raw loudness/peak, pitch estimate와 confidence, onset/envelope/HF/flux, delta-waveform distance, original/neutral/edited WAV paths를 포함한다. pitch가 없는 소리에 강제로 pitch drift를 부여하지 않는다.

- [ ] deterministic model fixture로 α0 waveform identity, 같은 α 재실행 일치, ascending/descending sweep 동일성, modal_override가 texture cache를 갱신하지 않는지 tests를 작성한다.
- [ ] 모든 output을 exact valid length로 저장한다. raw audio와 audition용 loudness-matched 버전을 구분한다. clipping 여부를 기록하고 자동 limiter로 artifact를 감추지 않는다.
- [ ] PCA/random directions는 같은 latent와 held-out samples에 적용한다. Validation에서 median output-change magnitude를 맞춘 뒤 test 비교한다. 변화가 큰 knob를 단순히 더 좋다고 순위 매기지 않는다.
- [ ] 후보 filtering은 nonfinite/clipping/과도한 pre-echo 등 failure flags를 먼저 적용하고 audible distance와 pairwise output-effect redundancy를 보고한다. 단일 합산 점수로 creative quality를 자동 확정하지 않는다. JSON에 모든 raw descriptors를 남긴다.
- [ ] HTML audition은 sample별 Original / Neutral reconstruction / Edited를 명시하고 α sweep과 feature 선택을 제공한다. Top activation exemplars는 참고로만 표시한다. 기존 자료를 reload하는 동작과 모델로 regeneration하는 동작을 구분한다. 8–16개 최종 knob는 청취 후 selection JSON으로 저장한다.
- [ ] modal adapter는 외부 modal task가 만든 waveform override를 받는다. Source-anchored delta 원리는 공유하되 새 modal network 자체를 이 noise 작업에서 재구현하지 않는다. Combined macro는 각 branch의 validated control을 조합하며 cached neutral reference를 유지한다.

Run: `python -m pytest test/unit/evaluation/test_texture_control.py --confcutdir=test/unit -o addopts= -q`.

## Task 8: 실험 실행, 조건부 확장과 정리

**Files:** add `docs/texture_experiments.md`; update 새 configs와 관련 module comments만 정리한다.

- [ ] DSP closure와 tiny overfit 결과를 먼저 기록한다. Failed overfit은 frame mapping/scale/gradient/bottleneck 순서로 진단한다. 학습 전후 loss와 개별 clip audio를 저장한다.
- [ ] Spec R0/R1/R2/C0 순서로 실행한다. training budgets, parameter counts, wall time, 같은 modal/splits를 기록한다. R3/C1/GAN/consistency는 관측된 오류에 대응하는 후속 실험으로만 추가한다.
- [ ] 각 결과에 mean/median/type별/worst5%를 보고하고 stochastic baseline은 fixed seed와 반복 변동을 함께 남긴다. Undefined silence metrics를 0으로 바꾸지 않는다. control 성능은 reconstruction 표와 분리한다.
- [ ] 긴 clip memory가 실제 문제가 되면 full global-context pass와 local chunk pass를 분리한다. Local conv의 receptive-field guard를 layer별 계산하고 chunk outputs에서 guard를 버린 뒤 global frame grid에 재조립한다. 테스트는 3.3초/12.8초 clip, chunk 경계의 impulse, full-vs-chunk complex/waveform equality다. 허용오차2e-5를 넘으면 full path를 기본으로 유지한다.
- [ ] `docs/texture_experiments.md`에 model input/output, neutral semantics, metadata schema, run command, 검증한 길이/환경을 기록한다. 새 코드의 주석은 비자명한 units/masks/phase 처리에만 간결한 English로 작성한다. Existing baseline file을 삭제하거나 옛 결과를 새 protocol 점수로 덮어쓰지 않는다.
- [ ] 변경한 경로의 회귀 tests와 wheel import/load/export를 실행한다. 모델 성능, artifact 유무, realtime에 관한 주장은 실제 측정한 범위로 제한한다.

## 실행 완료의 의미

이 계획을 구현했다는 것은 새 architecture, neutral-preserving intervention, 재현 가능한 비교 경로가 동작한다는 뜻이다. 우수한 reconstruction과 창의적인 knob를 확보했다는 결론은 held-out reconstruction 및 audition 결과가 있어야 한다. 현재 문서는 그 실험을 수행하기 위한 구체적인 제안이며 학습 결과를 포함하지 않는다.
