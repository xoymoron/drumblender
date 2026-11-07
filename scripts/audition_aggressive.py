"""Exploratory joint/strike gain sweep; no training or reconstruction changes."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from drumblender.intervention import BETA, InterventionSAE, edit_direction, strike_code
from scripts.audition_intervention import (
    as_wave,
    difference_score,
    save_waves,
    setup_folder,
)
from scripts.control_intervention import RunData
from scripts.control_joint import sha256, write_html, write_json

GAINS = (-64, -16, -4, -1, 0, 1, 4, 16, 64)
PAGES = {
    "index": "Joint + strike",
    "modal-only": "Modal edits with neutral noise and transient",
    "modal": "Modal solo",
    "noise": "Noise solo",
    "strike": "Strike-only and fixed-strike A/B",
    "baseline": "Comparison SAE: joint + strike",
    "baseline-modal": "Comparison modal solo",
    "baseline-noise": "Comparison noise solo",
}


def gain_from_knob(position):
    """Continuous signed exponential map; the HTML auditions sampled values."""
    return float(np.sign(position) * np.expm1(abs(position) * np.log(65)))


def write_page(path, cards, title, *, no_strike=False):
    write_html(path, cards)
    text = path.read_text(encoding="utf-8")
    text = text.replace("One-shot knob audition", title)
    text = text.replace("active validation clips", "audition examples")
    names = dict(PAGES)
    if no_strike:
        names.pop("strike")
        names.update(index="Joint S/N control", baseline="Matched comparison SAE")
    nav = " · ".join(
        f'<a href="{name}.html">{label}</a>' for name, label in names.items()
    )
    description = (
        "Frozen reconstruction; newly trained post-hoc SAE and matched comparison. "
        "Joint editing updates S and conditioned N; the original T follows S with neutral strike code. "
        if no_strike
        else "Same frozen reconstruction and SAE; no new training. Joint + strike updates S, N and s(h). "
    )
    text = text.replace(
        "Compare each hit at −1, 0, and +1. Stronger edits are marked +2 and +4. Choose up to two atoms you would use in a rhythm.",
        description + "The buttons amplify the latent edit, "
        "not the output volume. Try 0 / +16x / -16x, then 64x. "
        "Every page has RMS-matched listening copies; original levels remain in raw/. "
        "This is an exploratory range sweep, not a real-time VST or a claim of convergence. "
        "The noise carrier/offset is fixed. Modal and noise solos are diagnostic. "
        f"<br>{nav}",
    )
    text = text.replace("button{background", "a{color:#c7e76d}button{background")
    path.write_text(text, encoding="utf-8")


def delta_db(value, neutral, reference):
    ratio = (value - neutral).norm() / reference.norm().clamp_min(1e-12)
    return float(20 * ratio.clamp_min(1e-15).log10())


@torch.no_grad()
def render(args):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    run = RunData(args)
    payload = torch.load(
        args.run / "final.pt", map_location=args.device, weights_only=True
    )
    if payload["identity"] != run.identity or payload["cache_sha256"] != sha256(
        args.cache / "cache.json"
    ):
        raise ValueError("Checkpoint or cache identity mismatch.")
    sae = InterventionSAE().to(args.device).eval()
    sae.load_state_dict(payload["state_dict"])
    baseline = copy.deepcopy(sae)
    baseline.load_state_dict(payload["baseline_state_dict"])
    selection_path = args.selection or args.run / "screening.json"
    shortlist = json.loads(selection_path.read_text())
    selected = shortlist["intervention_top"]
    comparison_selected = shortlist.get("baseline_top", selected)
    assert 8 <= len(selected) <= 12
    examples = {row["atom"]: row["examples"] for row in shortlist["candidates"]}
    args.output.mkdir(parents=True, exist_ok=False)
    setup_folder(args.output)
    titles = dict(PAGES)
    if args.no_strike:
        titles.pop("strike")
        titles.update(index="Joint S/N control", baseline="Matched comparison SAE")
    pages = {key: [] for key in titles}
    metrics = []
    for atom in tqdm(
        list(dict.fromkeys(selected + comparison_selected)), desc="Joint gain sweep"
    ):
        cards = {
            key: {
                "atom": atom,
                "active": 3,
                "packs": len(
                    {run.caches["val"].rows[i]["pack"] for i in examples[atom]}
                ),
                "clips": [],
            }
            for key in titles
        }
        for index in examples[atom]:
            ref = run.reference("val", index)
            neutral = run.model.decode(ref, checkpoint_activations=False)
            directions = [
                edit_direction(net, run.codes(net, ref), atom, payload["floors"][atom])[
                    0
                ]
                for net in (sae, baseline)
            ]
            waves = {key: {} for key in titles}
            for alpha in GAINS:
                label = "0" if alpha == 0 else f"{alpha:+d}x"
                h = ref.h_ref + run.std * BETA * alpha * directions[0]
                if alpha == 0:
                    result = neutral
                    rb = neutral
                    transient = only = neutral.transient_waveform
                    comparison = neutral.waveform
                else:
                    result = run.model.decode(ref, h, checkpoint_activations=False)
                    if args.no_strike:
                        transient, only = (
                            result.transient_waveform,
                            neutral.transient_waveform,
                        )
                    else:
                        film = run.model.transient.film_coefficients(
                            strike_code(run.model, h, ref.rho)
                        )
                        transient = run.model._transient(
                            result.modal_waveform, film, checkpoint_activations=False
                        )
                        only = run.model._transient(
                            neutral.modal_waveform, film, checkpoint_activations=False
                        )
                    hb = ref.h_ref + run.std * BETA * alpha * directions[1]
                    rb = run.model.decode(ref, hb, checkpoint_activations=False)
                    if args.no_strike:
                        tb = rb.transient_waveform
                    else:
                        fb = run.model.transient.film_coefficients(
                            strike_code(run.model, hb, ref.rho)
                        )
                        tb = run.model._transient(
                            rb.modal_waveform, fb, checkpoint_activations=False
                        )
                    comparison = rb.modal_waveform + rb.noise_waveform + tb
                mix = result.modal_waveform + result.noise_waveform + transient
                waves["index"][label] = as_wave(mix, ref)
                waves["modal-only"][label] = as_wave(
                    result.modal_waveform
                    + neutral.noise_waveform
                    + neutral.transient_waveform,
                    ref,
                )
                waves["modal"][label] = as_wave(result.modal_waveform, ref)
                waves["noise"][label] = as_wave(result.noise_waveform, ref)
                waves["baseline"][label] = as_wave(comparison, ref)
                waves["baseline-modal"][label] = as_wave(rb.modal_waveform, ref)
                waves["baseline-noise"][label] = as_wave(rb.noise_waveform, ref)
                if not args.no_strike:
                    waves["strike"][label] = as_wave(
                        neutral.modal_waveform + neutral.noise_waveform + only, ref
                    )
                if abs(alpha) == 64 and not args.no_strike:
                    waves["strike"][f"Joint s fixed {label}"] = as_wave(
                        result.waveform, ref
                    )
                    waves["strike"][f"Joint + strike {label}"] = as_wave(mix, ref)
                if alpha:
                    mask = ref.observed
                    correction = result.log_corrections - neutral.log_corrections
                    metrics.append(
                        {
                            "atom": atom,
                            "sample_id": run.caches["val"].rows[index]["id"],
                            "alpha": alpha,
                            "knob_position": float(
                                np.sign(alpha) * np.log1p(abs(alpha)) / np.log(65)
                            ),
                            "h_max_frame_rms": float(
                                ((h - ref.h_ref) / run.std)
                                .square()
                                .mean(-1)
                                .sqrt()
                                .max()
                            ),
                            "mix_spectral_change": difference_score(
                                as_wave(mix, ref), as_wave(neutral.waveform, ref)
                            ),
                            "modal_spectral_change": difference_score(
                                as_wave(result.modal_waveform, ref),
                                as_wave(neutral.modal_waveform, ref),
                            ),
                            "noise_spectral_change": difference_score(
                                as_wave(result.noise_waveform, ref),
                                as_wave(neutral.noise_waveform, ref),
                            ),
                            "baseline_mix_spectral_change": difference_score(
                                as_wave(comparison, ref), as_wave(neutral.waveform, ref)
                            ),
                            "baseline_modal_spectral_change": difference_score(
                                as_wave(rb.modal_waveform, ref),
                                as_wave(neutral.modal_waveform, ref),
                            ),
                            "baseline_noise_spectral_change": difference_score(
                                as_wave(rb.noise_waveform, ref),
                                as_wave(neutral.noise_waveform, ref),
                            ),
                            "modal_delta_mix_db": delta_db(
                                result.modal_waveform,
                                neutral.modal_waveform,
                                neutral.waveform,
                            ),
                            "noise_delta_mix_db": delta_db(
                                result.noise_waveform,
                                neutral.noise_waveform,
                                neutral.waveform,
                            ),
                            "strike_only_delta_mix_db": delta_db(
                                only, neutral.transient_waveform, neutral.waveform
                            ),
                            "strike_increment_mix_db": delta_db(
                                transient, result.transient_waveform, neutral.waveform
                            ),
                            "modal_cents_abs_mean": float(
                                (
                                    correction[..., 0][mask].abs() * 1200 / np.log(2)
                                ).mean()
                            )
                            if mask.any()
                            else 0,
                            "modal_amp_db_abs_mean": float(
                                (
                                    correction[..., 1][mask].abs() * 20 / np.log(10)
                                ).mean()
                            )
                            if mask.any()
                            else 0,
                        }
                    )
            name = (
                run.datasets["val"]
                .metadata[run.caches["val"].rows[index]["id"]]
                .get("orig_relpath", str(index))
            )
            for key in titles:
                links = save_waves(args.output, f"{key}-{atom}-{index}", waves[key])
                cards[key]["clips"].append({"name": name, "audio": links})
        for key in titles:
            pages[key].append(cards[key])
    for key, title in titles.items():
        ranked = comparison_selected if key.startswith("baseline") else selected
        ordered = [
            next(card for card in pages[key] if card["atom"] == atom) for atom in ranked
        ]
        write_page(
            args.output / f"{key}.html", ordered, title, no_strike=args.no_strike
        )
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "cards.json", pages)
    write_json(
        args.output / "protocol.json",
        {
            "identity": run.identity,
            "sae_sha256": sha256(args.run / "final.pt"),
            "atoms": selected,
            "baseline_atoms": comparison_selected,
            "strike_code_updated": not args.no_strike,
            "seed": run.seed,
            "gains": GAINS,
            "knob_map": "sign(k)*(65**abs(k)-1), -1<=k<=1; sampled WAVs only",
            "h_edit": "h + std*0.25*alpha*normalized_atom_delta",
            "training_seconds": 0,
            "render_seconds": time.monotonic() - start,
            "scope": "same validation examples, joint S/N edits; no reconstruction changes",
        },
    )
    if sha256(args.checkpoint) != run.identity["checkpoint"]:
        raise RuntimeError("Reconstruction checkpoint changed.")
    print("COMPLETE", args.output, time.monotonic() - start, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "noise-bank", "data-dir", "cache", "run", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--no-strike", action="store_true")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
