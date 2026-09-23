#!/usr/bin/env python3
"""Export checkpoint reconstructions and run the test-set evaluation.

One unfiltered export produces whole-test and top-level-pack reports via
scripts/compile_results.py. See scripts/evaluation.md for the command and
output layout. Use --sample-pack-key only to restrict the selected packs.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
import yaml
from tqdm import tqdm

from drumblender.data.audio import AudioWithParametersDataset
from drumblender.metrics import audio_pair_fingerprint, load_evaluation_metrics, score_reconstruction
from drumblender.utils.model import load_model


NOISE_ENCODER_BACKBONE_CHOICES = [
    "soundstream",
    "dac",
    "dac_lstm",
    "dac_len",
    "dac_lstm_len",
    "dac_len_lstm_len",
    "wavtokenizer",
    "bscodec",
    "apcodec",
    "spectrostream",
]
TRANSIENT_ENCODER_BACKBONE_CHOICES = [
    "soundstream",
    "dac",
    "wavtokenizer",
    "bscodec",
    "apcodec",
    "spectrostream",
]


def _safe_relpath(path_str: str) -> Path:
    """
    Normalize a metadata path into a safe relative path (no absolute/root traversal).
    """
    rel = Path(path_str.replace("\\", "/"))
    if rel.is_absolute():
        # keep filename only for absolute paths
        return Path(rel.name)

    parts = []
    for part in rel.parts:
        if part in ("", ".", ".."):
            continue
        parts.append(part)
    if len(parts) == 0:
        return Path("unknown.wav")
    return Path(*parts)


def _export_rel_from_meta(meta: Dict[str, Any]) -> Path:
    """
    Prefer original pre-modal path for human-readable filenames.
    Fallback to processed filename if original path is unavailable.
    """
    # ### HIGHLIGHT: build_modal_features.py stores original sample path as `orig_relpath`.
    if isinstance(meta.get("orig_relpath"), str) and len(meta["orig_relpath"]) > 0:
        return _safe_relpath(meta["orig_relpath"])
    if isinstance(meta.get("filename"), str) and len(meta["filename"]) > 0:
        return _safe_relpath(meta["filename"])
    return Path("unknown.wav")


def _optional_int(value: str) -> Optional[int]:
    v = value.strip().lower()
    if v in {"none", "null", ""}:
        return None
    return int(value)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def _resolve_encoder_cfg(
    cfg_dir: Path,
    kind: str,
    backbone: str,
    explicit_cfg: Optional[str],
) -> Optional[str]:
    """
    Resolve encoder config file path from either an explicit path or a backbone name.
    """
    if explicit_cfg is not None and explicit_cfg.strip() != "":
        p = Path(explicit_cfg)
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{kind} encoder config not found: {p}")
        return str(p)

    if backbone == "soundstream":
        return None

    rel = Path("upgrades") / "encoders" / f"{kind}_{backbone}_style.yaml"
    p = (cfg_dir / rel).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"{kind} encoder backbone '{backbone}' requested but config is missing: {p}"
        )
    return str(p)


def _resolve_optional_cfg(cfg_dir: Path, cfg_value: Optional[str]) -> Optional[str]:
    if cfg_value is None or cfg_value.strip() == "":
        return None
    p = Path(cfg_value)
    if not p.is_absolute():
        p = (cfg_dir / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config path not found: {p}")
    return str(p)


def _build_export_model_config(args: argparse.Namespace) -> tuple[Path, bool]:
    """
    Optionally write a temporary config that mirrors training-time encoder/synth overrides.

    Returns:
      (config_path_to_use, is_temp)
    """
    base_cfg_path = Path(args.config).resolve()
    cfg_dir = base_cfg_path.parent

    # If no overrides are requested, reuse the original config as-is.
    if (
        args.loss_cfg is None
        and args.transient_synth_cfg is None
        and args.noise_encoder_cfg is None
        and args.transient_encoder_cfg is None
        and args.noise_encoder_backbone == "soundstream"
        and args.transient_encoder_backbone == "soundstream"
    ):
        return base_cfg_path, False

    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg_obj = yaml.safe_load(f)

    model = cfg_obj.setdefault("model", {})
    init_args = model.setdefault("init_args", {})

    loss_cfg = _resolve_optional_cfg(cfg_dir, args.loss_cfg)
    if loss_cfg is not None:
        init_args["loss_fn"] = loss_cfg

    transient_synth_cfg = _resolve_optional_cfg(cfg_dir, args.transient_synth_cfg)
    if transient_synth_cfg is not None:
        init_args["transient_synth"] = transient_synth_cfg

    noise_encoder_cfg = _resolve_encoder_cfg(
        cfg_dir=cfg_dir,
        kind="noise",
        backbone=args.noise_encoder_backbone,
        explicit_cfg=args.noise_encoder_cfg,
    )
    if noise_encoder_cfg is not None:
        init_args["noise_autoencoder"] = noise_encoder_cfg
        init_args["noise_autoencoder_accepts_audio"] = True

    transient_encoder_cfg = _resolve_encoder_cfg(
        cfg_dir=cfg_dir,
        kind="transient",
        backbone=args.transient_encoder_backbone,
        explicit_cfg=args.transient_encoder_cfg,
    )
    if transient_encoder_cfg is not None:
        init_args["transient_autoencoder"] = transient_encoder_cfg
        init_args["transient_autoencoder_accepts_audio"] = True

    tmp_name = (
        f".export_resolved_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        f"_{os.getpid()}.yaml"
    )
    tmp_path = cfg_dir / tmp_name
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_obj, f, sort_keys=False)
    return tmp_path, True


def _copy_training_context(ckpt_path: Path, package_root: Path) -> None:
    """
    Copy nearby training context files from the run directory.
    """
    # ### HIGHLIGHT: Infer run directory from ".../checkpoints/<file>.ckpt".
    run_dir: Optional[Path] = None
    if ckpt_path.parent.name == "checkpoints":
        run_dir = ckpt_path.parent.parent

    ctx_dir = package_root / "training_context"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Repository-level scripts/configs often needed to reproduce the run.
    for repo_file in [
        Path("run.sh"),
        Path("run_vessl.sh"),
        Path("cfg/05_all_parallel.yaml"),
        Path("cfg/metrics/drumblender_metrics.yaml"),
    ]:
        if repo_file.exists():
            _copy_if_exists(repo_file, ctx_dir / "repo" / repo_file)

    # LightningCLI transient config (if present in current working directory).
    _copy_if_exists(Path("config.yaml"), ctx_dir / "repo" / "config.yaml")

    if run_dir is None or not run_dir.exists():
        return

    # Copy useful run-local logs without pulling full checkpoints directory.
    wanted_patterns = [
        "metrics.csv",
        "hparams.yaml",
        "model-config.yaml",
        "config.yaml",
        "*.log",
        "*.json",
        "*.jsonl",
        "events.out.tfevents*",
    ]
    for pattern in wanted_patterns:
        for src in run_dir.rglob(pattern):
            if not src.is_file():
                continue
            # skip checkpoint files; selected ckpt is copied separately
            if src.suffix == ".ckpt":
                continue
            rel = src.relative_to(run_dir)
            _copy_if_exists(src, ctx_dir / "run_dir" / rel)


def _collect_git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
        info["commit"] = head
    except Exception:
        info["commit"] = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--short"], stderr=subprocess.STDOUT, text=True
        )
        info["status_short"] = status.splitlines()
    except Exception:
        info["status_short"] = []

    return info


def _copy_run_context_bundle(
    ckpt_path: Path,
    package_root: Path,
    explicit_run_context_json: Optional[str] = None,
    extra_paths: Optional[list[str]] = None,
) -> None:
    """
    Copy run-context and nearby logs/settings for ckpts saved under ./ckpt.

    This complements `_copy_training_context` (which is best when ckpt lives under
    .../checkpoints/ in a Lightning run dir).
    """
    ctx_root = package_root / "training_context"
    run_ctx_dir = ctx_root / "run_context"
    refs_dir = ctx_root / "referenced_configs"
    logs_dir = ctx_root / "logs"
    extra_dir = ctx_root / "extra"
    run_ctx_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    extra_dir.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []

    # Explicit run-context JSON (if provided)
    if explicit_run_context_json is not None and explicit_run_context_json.strip() != "":
        p = Path(explicit_run_context_json)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.is_file():
            dst = run_ctx_dir / "run-context.explicit.json"
            _copy_if_exists(p, dst)
            selected.append(dst)

    # Auto-pick nearby run-context files under ckpt directory.
    ckpt_parent = ckpt_path.parent
    candidates = sorted(ckpt_parent.glob("run-context-*.json"))
    if len(candidates) > 0:
        ckpt_mtime = ckpt_path.stat().st_mtime if ckpt_path.is_file() else 0.0
        nearest = min(candidates, key=lambda p: abs(p.stat().st_mtime - ckpt_mtime))
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        auto_pick = [nearest]
        if latest != nearest:
            auto_pick.append(latest)

        for p in auto_pick:
            dst = run_ctx_dir / p.name
            _copy_if_exists(p, dst)
            selected.append(dst)

        with (run_ctx_dir / "auto_candidates.txt").open("w", encoding="utf-8") as f:
            for p in candidates:
                f.write(str(p) + "\n")

    # Parse selected context files and copy referenced config files.
    referenced_keys = [
        "cfg",
        "loss_cfg",
        "noise_encoder_cfg",
        "transient_encoder_cfg",
        "transient_synth_cfg",
    ]
    for ctx_path in selected:
        try:
            with ctx_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        launch_cmd = obj.get("launch_cmd")
        if isinstance(launch_cmd, str) and len(launch_cmd) > 0:
            txt_path = run_ctx_dir / f"{ctx_path.stem}.launch_cmd.txt"
            txt_path.write_text(launch_cmd + "\n", encoding="utf-8")

        script_name = obj.get("script")
        if isinstance(script_name, str) and len(script_name) > 0:
            sp = Path(script_name)
            if not sp.is_absolute():
                sp = Path.cwd() / sp
            if sp.is_file():
                _copy_if_exists(sp, refs_dir / sp.name)

        for k in referenced_keys:
            v = obj.get(k)
            if not isinstance(v, str) or len(v.strip()) == 0:
                continue
            p = Path(v)
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.is_file():
                _copy_if_exists(p, refs_dir / p.name)

    # Copy nearest .log file in ./logs by modification time.
    repo_logs = Path("logs")
    if repo_logs.exists():
        log_files = [p for p in repo_logs.rglob("*.log") if p.is_file()]
        if len(log_files) > 0:
            ckpt_mtime = ckpt_path.stat().st_mtime if ckpt_path.is_file() else 0.0
            nearest_log = min(log_files, key=lambda p: abs(p.stat().st_mtime - ckpt_mtime))
            _copy_if_exists(nearest_log, logs_dir / nearest_log.name)

    # User-provided extra files/dirs.
    if extra_paths is not None:
        for item in extra_paths:
            if item is None:
                continue
            s = str(item).strip()
            if s == "":
                continue
            p = Path(s)
            if not p.is_absolute():
                p = Path.cwd() / p
            if not p.exists():
                continue
            _copy_path(p, extra_dir / p.name)


def _load_data_config_args(data_config_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_path = Path(data_config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_obj = yaml.safe_load(f)

    if not isinstance(cfg_obj, dict):
        raise RuntimeError(f"Invalid data config: {data_config_path}")

    init_args = cfg_obj.get("init_args", {})
    if not isinstance(init_args, dict):
        raise RuntimeError(f"Invalid data config init_args: {data_config_path}")

    dataset_kwargs = init_args.get("dataset_kwargs", {})
    if dataset_kwargs is None:
        dataset_kwargs = {}
    if not isinstance(dataset_kwargs, dict):
        raise RuntimeError(f"Invalid data config dataset_kwargs: {data_config_path}")

    return init_args, dataset_kwargs


def _resolve_data_args(args: argparse.Namespace) -> Dict[str, Any]:
    init_args: Dict[str, Any] = {}
    dataset_kwargs: Dict[str, Any] = {}
    if args.data_config is not None:
        init_args, dataset_kwargs = _load_data_config_args(args.data_config)

    data_dir = args.data_dir if args.data_dir is not None else init_args.get("data_dir")
    audio_dir = (
        args.audio_dir
        if args.audio_dir is not None
        else dataset_kwargs.get("audio_dir")
    )
    meta_file = args.meta_file if args.meta_file is not None else init_args.get("meta_file", "metadata.json")
    sample_rate = (
        args.sample_rate if args.sample_rate is not None else init_args.get("sample_rate")
    )
    num_samples = (
        args.num_samples if args.num_samples is not None else init_args.get("num_samples")
    )
    split_strategy = (
        args.split_strategy
        if args.split_strategy is not None
        else dataset_kwargs.get("split_strategy", "sample_pack")
    )
    parameter_key = (
        args.parameter_key
        if args.parameter_key is not None
        else dataset_kwargs.get("parameter_key", "feature_file")
    )
    expected_num_modes = (
        args.expected_num_modes
        if args.expected_num_modes is not None
        else dataset_kwargs.get("expected_num_modes", init_args.get("num_modes"))
    )
    seed = (
        args.seed
        if args.seed is not None
        else dataset_kwargs.get("seed", init_args.get("seed", 42))
    )
    sample_pack_keys = (
        args.sample_pack_keys
        if args.sample_pack_keys is not None
        else dataset_kwargs.get("sample_pack_keys")
    )

    if data_dir is None:
        raise RuntimeError("data_dir is required. Pass --data-dir or --data-config.")
    if sample_rate is None:
        raise RuntimeError("sample_rate is required. Pass --sample-rate or --data-config.")

    return {
        "data_dir": data_dir,
        "audio_dir": audio_dir,
        "meta_file": meta_file,
        "sample_rate": int(sample_rate),
        "num_samples": num_samples,
        "split_strategy": split_strategy,
        "parameter_key": parameter_key,
        "expected_num_modes": expected_num_modes,
        "seed": int(seed),
        "sample_pack_keys": sample_pack_keys,
        "split_train_ratio": dataset_kwargs.get("split_train_ratio", 0.8),
        "split_val_ratio": dataset_kwargs.get("split_val_ratio", 0.1),
        "normalize": dataset_kwargs.get("normalize", False),
        "sample_types": dataset_kwargs.get("sample_types"),
        "instruments": dataset_kwargs.get("instruments"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Optional data config YAML (e.g. cfg/data/custom_all.yaml).",
    )
    parser.add_argument(
        "--loss-cfg",
        type=str,
        default=None,
        help="Optional loss config override (same semantics as run scripts).",
    )
    parser.add_argument(
        "--transient-synth-cfg",
        type=str,
        default=None,
        help="Optional transient synth config override (e.g., masked residual TCN).",
    )
    parser.add_argument(
        "--noise-encoder-backbone",
        type=str,
        default="soundstream",
        choices=NOISE_ENCODER_BACKBONE_CHOICES,
        help="Noise encoder backbone override.",
    )
    parser.add_argument(
        "--transient-encoder-backbone",
        type=str,
        default="soundstream",
        choices=TRANSIENT_ENCODER_BACKBONE_CHOICES,
        help="Transient encoder backbone override.",
    )
    parser.add_argument(
        "--noise-encoder-cfg",
        type=str,
        default=None,
        help="Explicit noise encoder config path override.",
    )
    parser.add_argument(
        "--transient-encoder-cfg",
        type=str,
        default=None,
        help="Explicit transient encoder config path override.",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Modal feature dataset root")
    parser.add_argument("--audio-dir", type=str, default=None, help="Original wav root")
    parser.add_argument("--meta-file", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-manifest", type=str, default=None,
                        help="Reuse meta_key membership/order from a prior export manifest CSV.")
    parser.add_argument(
        "--split-strategy", type=str, default=None, choices=["sample_pack", "random"]
    )
    parser.add_argument("--parameter-key", type=str, default=None)
    parser.add_argument("--expected-num-modes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthesis-seed", type=int, default=20020420,
                        help="Per-sample synthesis randomness, independent of split seed/order.")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument(
        "--num-samples",
        type=_optional_int,
        default=None,
        help="Fixed-length mode if set, else variable-length (use none/null)",
    )
    parser.add_argument(
        "--sample-pack-key",
        dest="sample_pack_keys",
        action="append",
        default=None,
        help="Optional sample pack filter (repeatable). Overrides data-config pack filter.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recon_bundle",
        help="Directory to save packaged artifacts",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--save-target", dest="save_target", action="store_true", default=True,
        help="Save ground-truth WAVs for reproducible evaluation (default).",
    )
    target_group.add_argument(
        "--no-save-target", dest="save_target", action="store_false",
        help="Skip target WAVs and the evaluation report.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap for quick checks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--metrics-config",
        type=str,
        default="cfg/metrics/drumblender_metrics.yaml",
        help="Metric config YAML used for evaluation summary",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation summary computation",
    )
    parser.add_argument(
        "--make-tar",
        action="store_true",
        help="Create <output-dir>.tar.gz after export",
    )
    parser.add_argument(
        "--run-context-json",
        type=str,
        default=None,
        help="Optional explicit run-context JSON to include in the package.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Extra file/dir path to include under training_context/extra (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_args = _resolve_data_args(args)

    output_dir = Path(args.output_dir)
    recon_root = output_dir / "recon"
    target_root = output_dir / "target"
    eval_root = output_dir / "evaluation"
    ckpt_root = output_dir / "checkpoints"
    config_root = output_dir / "configs"
    recon_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    if args.save_target:
        target_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    model_cfg_to_load, is_temp_cfg = _build_export_model_config(args)
    model, _ = load_model(str(model_cfg_to_load), args.ckpt, include_data=False)
    model = model.to(device)
    model.eval()

    # ### HIGHLIGHT: Keep a copy of configs used for this export.
    _copy_if_exists(Path(args.config), config_root / Path(args.config).name)
    if args.data_config is not None:
        _copy_if_exists(Path(args.data_config), config_root / Path(args.data_config).name)
    if is_temp_cfg:
        _copy_if_exists(model_cfg_to_load, config_root / "resolved_export_config.yaml")
    _copy_if_exists(Path(args.metrics_config), config_root / Path(args.metrics_config).name)
    _copy_if_exists(Path(args.metrics_config), config_root / "evaluation_metrics.yaml")

    ckpt_path = Path(args.ckpt)
    _copy_if_exists(ckpt_path, ckpt_root / ckpt_path.name)
    _copy_training_context(ckpt_path, output_dir)
    _copy_run_context_bundle(
        ckpt_path=ckpt_path,
        package_root=output_dir,
        explicit_run_context_json=args.run_context_json,
        extra_paths=args.extra_path,
    )

    dataset = AudioWithParametersDataset(
        data_dir=data_args["data_dir"],
        meta_file=data_args["meta_file"],
        sample_rate=data_args["sample_rate"],
        num_samples=data_args["num_samples"],
        split=None if args.split_manifest else args.split,
        split_strategy=data_args["split_strategy"],
        parameter_key=data_args["parameter_key"],
        expected_num_modes=data_args["expected_num_modes"],
        audio_dir=data_args["audio_dir"],
        seed=data_args["seed"],
        sample_pack_keys=data_args["sample_pack_keys"],
        split_train_ratio=data_args["split_train_ratio"],
        split_val_ratio=data_args["split_val_ratio"],
        normalize=data_args["normalize"],
        sample_types=data_args["sample_types"],
        instruments=data_args["instruments"],
    )
    if args.split_manifest:
        with Path(args.split_manifest).open(newline="", encoding="utf-8") as f:
            keys = [row["meta_key"] for row in csv.DictReader(f)]
        if len(set(keys)) != len(keys) or not keys:
            raise ValueError("Split manifest must contain unique, nonempty meta_key IDs.")
        missing = set(keys) - set(dataset.metadata)
        if missing:
            raise ValueError(f"Split manifest IDs are absent from metadata: {sorted(missing)[:5]}")
        lengths_by_key = dict(zip(dataset.file_list, dataset.lengths))
        dataset.file_list = [key for key in keys if key in lengths_by_key]
        dataset.lengths = [lengths_by_key[key] for key in dataset.file_list]
        _copy_if_exists(Path(args.split_manifest), config_root / "input_split_manifest.csv")

    # Instantiate evaluation metrics from YAML.
    metric_protocol = None
    if not args.no_eval:
        metric_modules, metric_protocol = load_evaluation_metrics(args.metrics_config)
        metric_modules.to(device)

    limit = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)
    if limit <= 0:
        raise ValueError("The selected split is empty or max-items is not positive.")
    manifest_path = output_dir / "manifest.csv"
    per_file_loss_path = eval_root / "per_file_loss.csv"

    losses = []
    metric_rows = []

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file, per_file_loss_path.open(
        "w", newline="", encoding="utf-8"
    ) as loss_file:
        manifest_writer = csv.writer(manifest_file)
        loss_writer = csv.writer(loss_file)
        manifest_writer.writerow(
            [
                "index",
                "meta_key",
                "source_filename",
                "length",
                "recon_path",
                "target_path",
            ]
        )
        loss_writer.writerow(["index", "meta_key", "source_filename", "length", "test/loss"])

        for idx in tqdm(range(limit), desc="export"):
            meta_key = dataset.file_list[idx]
            meta = dataset.metadata[meta_key]
            src_rel = _export_rel_from_meta(meta)

            waveform, params, length = dataset[idx]
            length_i = int(length)

            with torch.no_grad():
                x = waveform.unsqueeze(0).to(device)
                p = params.unsqueeze(0).to(device)
                lengths = torch.tensor([length_i], dtype=torch.long, device=device)
                sample_seed = int.from_bytes(hashlib.sha256(
                    f"{args.synthesis_seed}:{meta_key}".encode()).digest()[:8], "big") % (2**63)
                torch.manual_seed(sample_seed)
                y_hat = model(x, p, lengths=lengths)
                y_hat, x = y_hat[..., :length_i], x[..., :length_i]
                loss_kwargs = {"lengths": lengths} if getattr(model, "_loss_accepts_lengths", False) else {}
                loss_value = float(model.loss_fn(y_hat, x, **loss_kwargs).detach().cpu())

            losses.append(loss_value)

            # Evaluate additional metrics.
            if not args.no_eval:
                metric_rows.append({
                    "index": idx, "meta_key": meta_key, "source_filename": str(src_rel),
                    "length": length_i, "sample_pack_key": dataset.top_level_pack(meta),
                    "test/objective": loss_value, "test/loss": loss_value,
                    "metric_version": metric_protocol["version"],
                    "metric_config_sha256": metric_protocol["config_sha256"],
                    **score_reconstruction(metric_modules, y_hat, x,
                                           sample_rate=data_args["sample_rate"]),
                })

            # Save the same valid-length float32 samples used for evaluation.
            recon = y_hat.squeeze(0).detach().cpu()
            target = x.squeeze(0).detach().cpu()

            recon_path = recon_root / src_rel
            recon_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(recon_path), recon, data_args["sample_rate"], encoding="PCM_F", bits_per_sample=32)

            target_path_str = ""
            if args.save_target:
                target_path = target_root / src_rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(target_path), target, data_args["sample_rate"], encoding="PCM_F", bits_per_sample=32)
                target_path_str = str(target_path)

            if not args.no_eval:
                metric_rows[-1]["audio_sha256"] = audio_pair_fingerprint(output_dir, src_rel)

            manifest_writer.writerow(
                [
                    idx,
                    meta_key,
                    str(src_rel),
                    length_i,
                    str(recon_path),
                    target_path_str,
                ]
            )
            loss_writer.writerow([idx, meta_key, str(src_rel), length_i, loss_value])

    # Build summary metrics payload.
    summary = {
        "export_time_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "data_config": args.data_config,
        "metrics_config": args.metrics_config,
        "ckpt": args.ckpt,
        "data_dir": data_args["data_dir"],
        "audio_dir": data_args["audio_dir"],
        "meta_file": data_args["meta_file"],
        "split": args.split,
        "split_manifest": args.split_manifest,
        "split_strategy": data_args["split_strategy"],
        "seed": data_args["seed"],
        "synthesis_seed": args.synthesis_seed,
        "split_train_ratio": data_args["split_train_ratio"],
        "split_val_ratio": data_args["split_val_ratio"],
        "split_policy": "explicit_manifest" if args.split_manifest else "within_top_level_pack_before_filter_v2",
        "metadata_sha256": hashlib.sha256((Path(data_args["data_dir"]) / data_args["meta_file"]).read_bytes()).hexdigest(),
        "selected_ids_sha256": hashlib.sha256(json.dumps(dataset.file_list[:limit]).encode()).hexdigest(),
        "metric_protocol": metric_protocol,
        "resolved_data": data_args,
        "sample_rate": data_args["sample_rate"],
        "num_samples": data_args["num_samples"],
        "sample_pack_keys": data_args["sample_pack_keys"],
        "num_items": limit,
        "git": _collect_git_info(),
        "metrics": {},
    }

    if len(losses) > 0:
        summary["metrics"]["test/loss"] = float(sum(losses) / len(losses))
        loss_std = 0.0
        if len(losses) > 1:
            # ### HIGHLIGHT: Report population std over per-file test/loss values.
            loss_std = float(statistics.pstdev(losses))
        summary["loss_stats"] = {
            "mean": float(statistics.fmean(losses)),
            "std": loss_std,
            "min": float(min(losses)),
            "max": float(max(losses)),
            "median": float(statistics.median(losses)),
            "p95": float(sorted(losses)[max(0, int(0.95 * len(losses)) - 1)]),
            "count": len(losses),
        }

    if not args.no_eval:
        for key in metric_rows[0]:
            if key.startswith("test/"):
                summary["metrics"][key] = statistics.fmean(row[key] for row in metric_rows)
        with (eval_root / "per_file_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)

    summary_path = eval_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also write a CSV one-liner for easy spreadsheet loading.
    summary_csv_path = eval_root / "summary_metrics.csv"
    metric_keys = sorted(summary["metrics"].keys())
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(metric_keys)
        w.writerow([summary["metrics"][k] for k in metric_keys])

    if not args.no_eval and args.save_target:
        from compile_results import main as compile_results

        compile_results([str(output_dir)])

    if args.make_tar:
        tar_path = output_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)
        print(f"[OK] archive: {tar_path}")

    if is_temp_cfg:
        try:
            Path(model_cfg_to_load).unlink(missing_ok=True)
        except Exception:
            pass

    print(f"[OK] exported {limit} items to {output_dir}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] eval summary: {summary_path}")


if __name__ == "__main__":
    main()
