"""Train modal representation, conditioned noise, joint reconstruction, then SAE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from drumblender.data.modal_control import ModalControlDataModule
from drumblender.tasks.modal_control import ModalControlTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "cfg/modal_control.yaml")
    parser.add_argument("--stage", choices=("modal", "noise", "joint", "patterns"), required=True)
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--processed_root", type=Path, default=Path("../dataset/processed"))
    parser.add_argument("--output", type=Path, default=Path("runs/modal_control"))
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--devices", default="1", help="Device count, or GPU IDs such as 0,1.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--init_from", type=Path, help="Previous stage checkpoint; resets optimizer.")
    checkpoint.add_argument("--resume", type=Path, help="Same stage checkpoint; restores optimizer/epoch.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pl.seed_everything(config["seed"], workers=True)
    torch.set_float32_matmul_precision("high")
    devices = [int(x) for x in args.devices.split(",")] if "," in args.devices else int(args.devices)
    if ((isinstance(devices, int) and devices < 1)
            or (isinstance(devices, list) and (min(devices) < 0 or len(set(devices)) != len(devices)))):
        parser.error("Specify a positive device count or distinct nonnegative GPU IDs.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.accelerator == "cpu" and isinstance(devices, list):
        parser.error("CPU devices must be a process count, not a list of GPU IDs.")
    if args.stage != "modal" and not (args.init_from or args.resume):
        parser.error("noise/joint/patterns require --init_from or --resume; do not use a random backbone.")
    model = ModalControlTask(stage=args.stage, **config["model"])
    path = args.init_from or args.resume
    if path:
        # Only load checkpoints produced by this project and trusted by the user.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        previous_stage = saved.get("modal_control_stage")
        allowed = {"modal": {"modal"}, "noise": {"modal", "noise"},
                   "joint": {"noise", "joint"}, "patterns": {"modal", "noise", "joint", "patterns"}}
        if previous_stage not in allowed[args.stage]:
            parser.error(f"Unsupported stage transition: {previous_stage} -> {args.stage}")
        if args.resume and previous_stage != args.stage:
            parser.error("--resume must keep the same stage; use --init_from for a new stage.")
        # Hyperparameters affecting parameter meaning must agree, not just shapes.
        for key in ("sample_rate", "channels", "context_tokens", "features", "top_k"):
            if saved["hyper_parameters"][key] != config["model"][key]:
                parser.error(f"Checkpoint/config mismatch for {key}.")
        model.load_state_dict(saved["state_dict"], strict=True)
        if args.stage in ("modal", "joint"):
            model.pattern_updates.zero_()
    data_config = dict(config["data"])
    for key in ("batch_size", "num_workers"):
        if getattr(args, key) is not None:
            data_config[key] = getattr(args, key)
    data = ModalControlDataModule(
        str(args.feature_root), str(args.processed_root),
        sample_rate=config["model"]["sample_rate"], seed=config["seed"], **data_config,
    )
    output = args.output / args.stage
    callback = ModelCheckpoint(
        dirpath=str(output / "checkpoints"), filename="epoch{epoch:03d}",
        auto_insert_metric_name=False, monitor="validation/loss", mode="min",
        save_top_k=1, save_last=True,
    )
    count = len(devices) if isinstance(devices, list) else devices
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=devices,
        strategy="ddp" if count > 1 else "auto",
        max_epochs=args.epochs or config["epochs"][args.stage],
        default_root_dir=str(output), logger=CSVLogger(str(output), name="logs"),
        callbacks=[callback], gradient_clip_val=1.0, precision=32,
        log_every_n_steps=10, enable_progress_bar=True,
    )
    if trainer.is_global_zero:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"config": config, "arguments": {k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()},
                    "resolved_data": data_config}
        (output / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    trainer.fit(model, datamodule=data, ckpt_path=str(args.resume) if args.resume else None)
    if trainer.is_global_zero:
        print(f"Best checkpoint: {callback.best_model_path}")
        print(f"Resume checkpoint: {callback.last_model_path}")


if __name__ == "__main__":
    main()
