"""Full-clip comparisons and Float32 listening artifacts for intervention SAE."""

import copy
import json
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft
from tqdm import tqdm

from drumblender.control import FEATURES
from drumblender.intervention import BETA, edit_direction, modal_prefix, strike_code
from scripts.control_joint import (
    clip_activity,
    matched_waves,
    shortlist,
    write_html,
    write_json,
)


def as_wave(value, ref):
    return (value.detach() * ref.gain).flatten().cpu().numpy()


def difference_score(wave, neutral):
    normalized = matched_waves({"edit": wave, "neutral": neutral})
    scores = []
    for size in (512, 2048, 4096):
        size = min(size, len(wave))
        left = np.abs(stft(normalized["edit"], nperseg=size, noverlap=size * 3 // 4)[2])
        right = np.abs(
            stft(normalized["neutral"], nperseg=size, noverlap=size * 3 // 4)[2]
        )
        scores.append(
            float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-12))
        )
    return float(np.mean(scores))


def save_waves(folder, prefix, waves):
    """Keep original floats and separate, level-matched Float32 listening copies."""
    links = {}
    for i, (key, value) in enumerate(matched_waves(waves).items()):
        if not np.isfinite(value).all() or not np.isfinite(waves[key]).all():
            raise FloatingPointError(f"Invalid export: {prefix}/{key}")
        filename = f"{prefix}-{i}.wav"
        sf.write(folder / "raw" / filename, waves[key], 48000, subtype="FLOAT")
        sf.write(folder / "audio" / filename, value, 48000, subtype="FLOAT")
        links[key] = f"audio/{filename}"
    with (folder / "signal_diagnostics.jsonl").open("a", encoding="utf-8") as file:
        for key, value in waves.items():
            rms = float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))
            peak = float(np.max(np.abs(value)))
            file.write(
                json.dumps(
                    dict(
                        prefix=prefix,
                        label=key,
                        rms=rms,
                        peak=peak,
                        dc=float(np.mean(value)),
                        crest=peak / max(rms, 1e-12),
                        first=float(value[0]),
                        last=float(value[-1]),
                        max_sample_step=float(np.max(np.abs(np.diff(value))))
                        if len(value) > 1
                        else 0,
                    )
                )
                + "\n"
            )
    return links


def setup_folder(folder):
    (folder / "audio").mkdir(parents=True, exist_ok=True)
    (folder / "raw").mkdir(exist_ok=True)


def page(path, cards, title, description):
    write_html(path, cards)
    text = path.read_text(encoding="utf-8")
    text = text.replace("One-shot knob audition", title)
    text = text.replace(
        "Compare each hit at −1, 0, and +1. Stronger edits are marked +2 and +4. Choose up to two atoms you would use in a rhythm.",
        description
        + '<p><a style="color:#c7e76d" href="index.html">Intervention SAE</a> · '
        '<a style="color:#c7e76d" href="baseline.html">Comparison SAE</a> · '
        '<a style="color:#c7e76d" href="strike.html">Strike A/B</a> · '
        '<a style="color:#c7e76d" href="diagnostic.html">Transient feasibility probe</a></p>',
    )
    path.write_text(text, encoding="utf-8")


def transient_diagnostic(run, sae, payload, output, deadline):
    start = time.monotonic()
    folder = output / "audition"
    setup_folder(folder)
    rng = np.random.default_rng(run.seed + 907)
    rows, cards = [], []
    for index in rng.choice(len(run.caches["val"].rows), 3, replace=False):
        if time.monotonic() + 30 >= deadline:
            break
        with torch.no_grad():
            ref = run.reference("val", int(index))
            neutral = run.model.decode(ref, checkpoint_activations=False)
            codes = run.codes(sae, ref)
            active = codes[..., payload["cohort"]].sum((0, 1))
            atom = int(payload["cohort"][active.argmax()])
            direction, _ = edit_direction(sae, codes, atom, payload["floors"][atom])
        delta = torch.nn.Parameter(0.1 * BETA * direction.detach())
        optimizer = torch.optim.Adam([delta], lr=0.05)
        best_ratio, best_wave, best_delta = -1.0, None, None
        losses = []
        for step in range(21):
            if time.monotonic() + 15 >= deadline:
                break
            h = ref.h_ref + run.std * delta
            strike = strike_code(run.model, h, ref.rho)
            transient = run.model._transient(
                neutral.modal_waveform,
                run.model.transient.film_coefficients(strike),
                checkpoint_activations=True,
            )
            diff = transient - neutral.transient_waveform
            ratio = (
                diff.square().mean() + 1e-30
            ).sqrt() / neutral.waveform.square().mean().sqrt().clamp_min(1e-8)
            value = float(ratio.detach())
            losses.append(value)
            if value > best_ratio:
                best_ratio = value
                best_wave = transient.detach().clone()
                best_delta = delta.detach().clone()
            if step == 20:
                break
            optimizer.zero_grad(set_to_none=True)
            (-ratio).backward()
            if delta.grad is None or not delta.grad.isfinite().all():
                break
            optimizer.step()
            with torch.no_grad():
                radius = delta.norm(dim=-1, keepdim=True) / 16
                delta.mul_((BETA / radius.clamp_min(1e-8)).clamp_max(1))
        if best_wave is None:
            continue
        waves = {
            "Neutral": as_wave(neutral.waveform, ref),
            "Input-only probe": as_wave(
                neutral.modal_waveform + neutral.noise_waveform + best_wave, ref
            ),
        }
        row = {
            "sample_id": run.caches["val"].rows[index]["id"],
            "iterations": len(losses),
            "delta_over_mix_db": float(20 * np.log10(max(best_ratio, 1e-15))),
            "max_standardized_frame_rms": float(best_delta.norm(dim=-1).amax() / 16),
            "ratios": losses,
        }
        rows.append(row)
        cards.append(
            {
                "atom": "probe",
                "active": 1,
                "packs": 1,
                "clips": [
                    {
                        "name": row["sample_id"],
                        "audio": save_waves(folder, f"probe-{index}", waves),
                    }
                ],
            }
        )
        print(
            "TRANSIENT_PROBE",
            json.dumps({k: v for k, v in row.items() if k != "ratios"}),
            flush=True,
        )
    write_json(
        output / "transient_diagnostic.json",
        {"rows": rows, "seconds": time.monotonic() - start},
    )
    page(
        folder / "diagnostic.html",
        cards,
        "Input-only transient feasibility probe",
        "This is a per-clip diagnostic, NOT a reusable SAE knob. Original S/N and all reconstruction weights are fixed. Float32 level-matched listening copies; raw files are in raw/.",
    )


@torch.no_grad()
def audition(run, sae, baseline, payload, output):
    start = time.monotonic()
    folder = output / "audition"
    setup_folder(folder)
    initial = copy.deepcopy(sae)
    initial.load_state_dict(
        torch.load(run.args.sae, map_location=run.args.device, weights_only=True)[
            "state_dict"
        ]
    )
    cache, val = run.caches["val"], run.datasets["val"]
    initial_activity = clip_activity(initial, cache, run.mean, run.std)
    activities = [
        clip_activity(net, cache, run.mean, run.std) for net in (sae, baseline)
    ]
    write_json(
        output / "validation_activity.json",
        {
            label: {
                "never_active_fraction": float((activity.max(0) == 0).mean()),
                "clip_support": (activity > 0).sum(0).tolist(),
            }
            for label, activity in zip(("intervention", "baseline"), activities)
        },
    )
    eligible = set(payload["cohort"].tolist())
    candidates = [
        row
        for row in shortlist(initial_activity, cache.rows, FEATURES)
        if row[3] in eligible
    ]
    jobs = {}
    metadata = {}
    for active, packs, _strength, atom, examples in candidates:
        metadata[atom] = {
            "atom": atom,
            "initial_active": active,
            "initial_packs": packs,
            "examples": examples,
            "scores": [[], []],
        }
        for index in examples:
            jobs.setdefault(index, []).append(atom)
    if not candidates:
        raise ValueError("No supported validation candidates.")
    for index, atoms in tqdm(jobs.items(), desc="Paired full-clip screening"):
        ref = run.reference("val", index)
        neutral = run.model.decode(ref, checkpoint_activations=False)
        center = as_wave(neutral.waveform, ref)
        for net_index, net in enumerate((sae, baseline)):
            codes = run.codes(net, ref)
            for atom in atoms:
                direction, _ = edit_direction(net, codes, atom, payload["floors"][atom])
                scores = []
                solo_scores = []
                for alpha in (-1, 1):
                    h = ref.h_ref + run.std * BETA * alpha * direction
                    s = modal_prefix(run.model, ref, h, ref.length).reshape_as(
                        neutral.modal_waveform
                    )
                    mix = as_wave(
                        s + neutral.noise_waveform + neutral.transient_waveform, ref
                    )
                    scores.append(difference_score(mix, center))
                    solo_scores.append(
                        difference_score(
                            as_wave(s, ref), as_wave(neutral.modal_waveform, ref)
                        )
                    )
                metadata[atom]["scores"][net_index].append(
                    {"index": index, "mix": scores, "solo": solo_scores}
                )
    rankings = []
    for net_index in (0, 1):
        rankings.append(
            sorted(
                metadata,
                key=lambda atom: (
                    -float(
                        np.median(
                            [max(r["mix"]) for r in metadata[atom]["scores"][net_index]]
                        )
                    ),
                    atom,
                ),
            )[:10]
        )
    write_json(
        folder / "screening.json",
        {
            "selection": "Same three initialization-selected validation examples for both models; no test data",
            "initialization": run.args.sae.name,
            "candidates": list(metadata.values()),
            "intervention_top": rankings[0],
            "baseline_top": rankings[1],
        },
    )
    cards = {}
    random_rows, strike_rows, strike_cards = [], [], []
    strike_done = set()
    for atom in tqdm(
        sorted(set(rankings[0] + rankings[1])), desc="Render listening comparisons"
    ):
        clips = []
        for index in metadata[atom]["examples"]:
            ref = run.reference("val", index)
            neutral = run.model.decode(ref, checkpoint_activations=False)
            codes = [run.codes(net, ref) for net in (sae, baseline)]
            directions = [
                edit_direction(net, code, atom, payload["floors"][atom])[0]
                for net, code in zip((sae, baseline), codes)
            ]
            waves = {"0": as_wave(neutral.waveform, ref)}
            solos = {"S 0": as_wave(neutral.modal_waveform, ref)}
            for n, label in enumerate(("IA", "Baseline")):
                for alpha in (-1, 1):
                    h = ref.h_ref + run.std * BETA * alpha * directions[n]
                    s = modal_prefix(run.model, ref, h, ref.length).reshape_as(
                        neutral.modal_waveform
                    )
                    waves[f"{label} {alpha:+d}"] = as_wave(
                        s + neutral.noise_waveform + neutral.transient_waveform, ref
                    )
                    solos[f"S {label} {alpha:+d}"] = as_wave(s, ref)
            for random_seed in range(3):
                permutation = np.random.default_rng(
                    run.seed + atom * 11 + random_seed
                ).permutation(256)
                direction = directions[0][
                    ..., torch.tensor(permutation, device=run.args.device)
                ]
                for alpha in (-1, 1):
                    h = ref.h_ref + run.std * BETA * alpha * direction
                    s = modal_prefix(run.model, ref, h, ref.length).reshape_as(
                        neutral.modal_waveform
                    )
                    mix = as_wave(
                        s + neutral.noise_waveform + neutral.transient_waveform, ref
                    )
                    random_rows.append(
                        {
                            "atom": atom,
                            "index": index,
                            "seed": random_seed,
                            "alpha": alpha,
                            "score": difference_score(mix, waves["0"]),
                        }
                    )
                    if random_seed == 0:
                        waves[f"Random {alpha:+d}"] = mix
            links = save_waves(folder, f"mix-{atom}-{index}", waves)
            links.update(save_waves(folder, f"solo-{atom}-{index}", solos))
            key = cache.rows[index]["id"]
            name = val.metadata[key].get("orig_relpath", key)
            clips.append({"id": key, "name": name, "audio": links})
            if atom in rankings[0][:2] and index not in strike_done:
                strike_done.add(index)
                h = ref.h_ref + run.std * BETA * directions[0]
                joint = run.model.decode(ref, h, checkpoint_activations=False)
                code = strike_code(run.model, h, ref.rho)
                film = run.model.transient.film_coefficients(code)
                changed = run.model._transient(
                    joint.modal_waveform, film, checkpoint_activations=False
                )
                only = run.model._transient(
                    neutral.modal_waveform, film, checkpoint_activations=False
                )
                strike_waves = {
                    "Neutral": as_wave(neutral.waveform, ref),
                    "Joint, s fixed": as_wave(joint.waveform, ref),
                    "Joint + strike": as_wave(
                        joint.modal_waveform + joint.noise_waveform + changed, ref
                    ),
                    "Strike only": as_wave(
                        neutral.modal_waveform + neutral.noise_waveform + only, ref
                    ),
                }
                for scope, difference in (
                    ("joint_strike_increment", changed - joint.transient_waveform),
                    ("strike_only", only - neutral.transient_waveform),
                ):
                    ratio = difference.norm() / neutral.waveform.norm().clamp_min(1e-12)
                    strike_rows.append(
                        {
                            "atom": atom,
                            "sample_id": key,
                            "scope": scope,
                            "delta_over_mix_db": float(
                                20 * ratio.clamp_min(1e-15).log10()
                            ),
                        }
                    )
                strike_cards.append(
                    {
                        "atom": atom,
                        "active": 1,
                        "packs": 1,
                        "clips": [
                            {
                                "name": name,
                                "audio": save_waves(
                                    folder, f"strike-{atom}-{index}", strike_waves
                                ),
                            }
                        ],
                    }
                )
        cards[atom] = {"atom": atom, "clips": clips}
    for n, filename in enumerate(("index.html", "baseline.html")):
        chosen = []
        for atom in rankings[n]:
            active = np.flatnonzero(activities[n][:, atom] > 0)
            chosen.append(
                {
                    **cards[atom],
                    "active": len(active),
                    "packs": len({cache.rows[i]["pack"] for i in active}),
                }
            )
        page(
            folder / filename,
            chosen,
            "Intervention SAE" if n == 0 else "Matched-update comparison SAE",
            "IA and Baseline use the same latent edit budget (alpha ±1 means maximum frame RMS 0.25 in standardized h). Noise and transient are neutral. S buttons are modal solos. Random is a matched-budget diagnostic. All listening WAVs are Float32 and level matched; original levels are preserved in raw/. Choose two atoms after listening.",
        )
    page(
        folder / "strike.html",
        strike_cards,
        "Frozen strike-code A/B",
        "Joint + strike differs from Joint, s fixed only through the existing strike code. Strike only keeps modal and noise neutral. No transient amplification.",
    )
    write_json(folder / "random_comparison.json", random_rows)
    write_json(folder / "strike.json", strike_rows)
    write_json(
        folder / "candidates.json",
        {"intervention": rankings[0], "baseline": rankings[1], "cards": cards},
    )
    write_json(output / "audition_timing.json", {"seconds": time.monotonic() - start})
    print(
        "AUDITION",
        json.dumps(
            {
                "intervention": rankings[0],
                "baseline": rankings[1],
                "seconds": time.monotonic() - start,
            }
        ),
        flush=True,
    )
