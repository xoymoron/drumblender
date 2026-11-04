# 개발 환경과 두 GPU 학습 환경

Python 3.10–3.12를 사용한다. 이 로컬에서 검증한 버전은 Python 3.10.11이다.
NumPy 1.26.4, Lightning 1.9.5, torch/torchaudio 2.7.1을 두 환경에서 공통으로
사용한다. NumPy 2에서는 Lightning 1.9의 `np.Inf` 참조가 실패한다.
`setuptools==80.9.0`은 Lightning이 사용하는 `pkg_resources`를 제공한다.
공통 dependency는 `pyproject.toml`, CPU/CUDA build 선택은 `requirements/`가 관리한다.

## 로컬 Windows

전역 패키지를 읽는 기존 `.venv-modal` 대신, 독립된 `.venv`를 사용한다.
다른 채팅의 실행을 방해하지 않도록 기존 환경은 삭제하지 않았다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements/local.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=""
```

`.venv`가 이미 설치되어 있으면 생성·설치를 반복할 필요 없다.
CPU에서 실제 데이터로 점검할 때는 학습 명령에 `--accelerator cpu --devices 1`을 추가한다.
Stage 전환과 checkpoint 사용법은 [modal.md](modal.md)의 학습 절차를 따른다.

## Linux 서버: RTX 3090 두 대

각 checkout의 root에서 별도 환경을 만든다. Main과 baseline은 둘 다 package 이름이
`drumblender`이므로 같은 환경에 두 checkout을 editable install하지 않는다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/server.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 2; print([(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory // 2**20) for i in range(2)])"
```

서버 profile은 Linux x86_64, glibc 2.28 이상용 공식 CUDA 12.6 build를 선택한다.
서버 driver의 CUDA 12.6 지원은
서버에서 확인해야 한다. CUDA toolkit을 따로 설치하는 것으로 driver 문제가 해결되지는 않는다.

Main 모델의 기본 config는 `gpu`, `devices: 2`이고 entrypoint가 표준 `ddp`를 선택한다.
Lightning이 일반 DataLoader에 DistributedSampler를 적용한다. Validation loss는
`sync_dist=True`로 동기화되어 scheduler와 early stopping이 같은 값을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.train_modal_control \
  --stage modal --feature_root "$MODAL_ROOT" --processed_root "$PROCESSED_ROOT"
```

기존 DrumBlender CLI 실험에는 두 번째 config로 server 설정을 적용한다.
이 data module은 직접 sampler를 분할하므로 `replace_sampler_ddp: false`를 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 drumblender fit \
  --config cfg/05_all_parallel.yaml --config cfg/trainer/dual_gpu.yaml
```

Jorshi baseline은 해당 checkout의 `cfg/05_all_parallel_variable.yaml`에 같은 두 GPU
설정이 반영되어 있다. 두 모델 모두 per-GPU batch 4 × GPU 2 × accumulation 8로
기존 effective batch 64를 유지한다. LR와 scheduler, early stopping은 그대로다.
마지막 partial batch는 Lightning의 표준 accumulation 동작을 따른다.

DDP에서는 모델과 optimizer가 각 GPU에 복제된다. 두 장의 24 GiB가 하나의 48 GiB
메모리가 되지는 않는다. GPU 실물에서 20초 입력의 peak VRAM은 아직 측정하지 않았다.
OOM이 발생하면 두 모델 모두 per-GPU batch 1, accumulation 32로 바꾸면 nominal
effective batch 64를 유지할 수 있다. 실제 GPU 검증 전에는 24 GiB 내 동작을 보장하지 않는다.

## 검증 범위

CPU 두 process의 실제 backward와 optimizer step, rank 간 parameter 일치 검사는
`test/integration/test_dual_device_training.py`에 있다. 이는 CPU Gloo 검사이며 서버의
CUDA/NCCL 및 GPU 메모리 검사를 대체하지 않는다. 테스트는 축소된 폭과 짧은 합성
입력을 사용하며 reconstruction quality를 측정하지 않는다.

2026-11-04 검증: 두 checkout의 독립 환경에서 `pip check` 통과. Main 전체 테스트
286 passed / 5 skipped, baseline 전체 테스트 134 passed. 두 process DDP 테스트는
Main 네 stage와 baseline에서 실제 backward·optimizer step 및 parameter 동기화를 확인했다.
Main evaluation extra, Linux wheel dependency 및 Linux 전용 CUDA/NCCL/Triton dependency도
resolver 검사를 통과했다. 서버 GPU에서 실행한 결과는 아니다.

참고: [PyTorch 공식 build 목록](https://pytorch.org/get-started/previous-versions/#v271),
[NumPy 2.0의 alias 제거](https://numpy.org/devdocs/release/2.0.0-notes.html).
