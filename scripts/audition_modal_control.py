"""Export paired neural modal edits with identical stochastic excitation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import soundfile as sf
import torch

from drumblender.data.audio import AudioWithParametersDataset
from drumblender.tasks.modal_control import ModalControlTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--processed_root", type=Path, default=Path("../dataset/processed"))
    parser.add_argument("--sample_id", help="Metadata key; omitted selects a reproducible random sample.")
    parser.add_argument("--feature", type=int, help="SAE feature ID; omitted uses the strongest active feature.")
    parser.add_argument("--amount", type=float, default=0.5)
    parser.add_argument("--noise_link", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("insights/modal_control_audition"))
    args = parser.parse_args()
    model = ModalControlTask.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.to(args.device).eval()
    dataset = AudioWithParametersDataset(
        data_dir=args.feature_root, meta_file="metadata.json", num_samples=None,
        audio_dir=args.processed_root, sample_rate=model.hparams.sample_rate,
        expected_num_modes=None,
    )
    if not len(dataset):
        parser.error("Dataset is empty.")
    if args.sample_id is not None and args.sample_id not in dataset.file_list:
        parser.error(f"Unknown metadata sample ID: {args.sample_id}")
    index = (dataset.file_list.index(args.sample_id) if args.sample_id is not None
             else random.Random(args.seed).randrange(len(dataset)))
    audio, params, _ = dataset[index]
    audio, params = audio[None].to(args.device), params[None].to(args.device)
    with torch.no_grad():
        reference = model.encode_reference(audio, params)
        sparse = model.patterns.encode(reference.encoding.hidden, reference.encoding.mask)
        strength = sparse.sum(dim=(0, 1, 2))
        feature = int(strength.argmax()) if args.feature is None else args.feature
        neutral = model.edit_reference(reference, feature, 0, args.noise_link, args.seed)
        edited = model.edit_reference(reference, feature, args.amount, args.noise_link, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {"original": audio}
    for name, result in (("neutral", neutral), ("edited", edited)):
        for branch in ("audio", "modal_audio", "noise_audio"):
            outputs[f"{name}_{branch}"] = result[branch]
    for name, value in outputs.items():
        sf.write(str(args.output / f"{name}.wav"), value[0, 0].cpu().numpy(),
                 model.hparams.sample_rate, subtype="FLOAT")
    torch.save({"extracted": params.cpu(), "neutral": neutral["params"].cpu(),
                "edited": edited["params"].cpu()}, args.output / "modal_parameters.pt")
    values, indices = strength.topk(min(16, strength.numel()))
    report = {
        "sample_id": dataset.file_list[index], "metadata": dataset.metadata[dataset.file_list[index]],
        "checkpoint": str(args.checkpoint.resolve()), "feature": feature,
        "amount": args.amount, "noise_link": args.noise_link, "noise_seed": args.seed,
        "noise_trained": bool(model.noise_updates.item()),
        "patterns_trained": bool(model.pattern_updates.item()),
        "num_modes": int((params[0, 1] > 0).any(-1).sum()),
        "top_features": [{"id": int(i), "activation_sum": float(v)} for i, v in zip(indices, values)],
        "output_peak": {name: float(value.abs().max()) for name, value in outputs.items()},
        "normalization": "None; float WAVs preserve a common amplitude reference.",
        "note": "High activation is not proof of useful steering. Noise linking is experimental.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sample {report['sample_id']}; feature {feature}; output {args.output.resolve()}")


if __name__ == "__main__":
    main()
