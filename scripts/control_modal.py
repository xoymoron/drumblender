"""Fit a modal-aware SAE and audition frozen reconstruction branches."""

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import replace
from math import ceil, log
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft
from tqdm import tqdm

from drumblender.control import (
    BATCH,
    CHANNELS,
    CORE,
    CORE_SLICE,
    FEATURES,
    ControlSAE,
    FrameCache,
    latent_delta,
    training_loss,
    weighted_error,
)
from drumblender.models.joint_resynthesis import HOP, MODE_CHUNK
from scripts.control_joint import (
    SAMPLE_RATE,
    clip_activity,
    dataset_for,
    matched_waves,
    normalization,
    open_cache,
    open_joint,
    prepare_batch,
    reference,
    save_best,
    sha256,
    shortlist,
    write_html,
    write_json,
)

PROBES = 12
MODAL_WEIGHT = 0.1


class ModalAwareSAE(ControlSAE):
    """Legacy modal-only experiment; no feature indices are reserved."""


def strike_code(model, h, rho):
    features = model.strike_norm(h)
    weights = (model.strike_score(features).squeeze(-1) + rho.log()).softmax(-1)
    return model.strike_out((weights[..., None] * features).sum(dim=1))


def probe_head(model, h, position, u):
    """Evaluate the existing head at indexed native mode/frame observations."""
    left = position.long()
    right = (left + 1).clamp_max(h.shape[1] - 1)
    batch = torch.arange(h.shape[0], device=h.device)[:, None]
    samples = torch.lerp(h[batch, left], h[batch, right], (position - left)[..., None])
    observed = torch.ones_like(position[:, None], dtype=torch.bool)
    delta = model._correction(u[:, None], model.modal_h(samples), observed)[:, 0]
    return delta * delta.new_tensor([0.03, log(4)])


@torch.no_grad()
def make_probes(model, dataset, cache, count, rng, device):
    """Store a fixed, small sample of the teacher's mode/frame pairs per crop."""
    clips = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    offsets = np.searchsorted(cache.clip_indices, np.arange(len(dataset) + 1))
    crop_ids, positions, tokens, weights = [], [], [], []
    for i in tqdm(np.sort(clips), desc="Modal teacher probes", unit="clip"):
        item = dataset[int(i)]
        params = item["params"][None].to(device)
        if not params.shape[2]:
            continue
        extent = item["analysis_num_samples"]
        frames = params.shape[-1]
        contributing = max(
            1, min(frames, ceil((item["length"] - 0.5) * frames / extent - 0.5) + 1)
        )
        halo = min(frames, contributing + 8)
        gain = item["audio"].abs().amax().clamp_min(1e-4).to(device)
        params[:, 1] /= gain
        observed = params[:, 1, :, :halo] > 0
        u = (
            torch.cat(
                [
                    model._modal_tokens(p, m, True)
                    for p, m in zip(
                        params[..., :halo].split(MODE_CHUNK, 2),
                        observed.split(MODE_CHUNK, 1),
                    )
                ],
                dim=1,
            )[0, :, :contributing]
            .cpu()
            .numpy()
        )
        amp = params[0, 1, :, :contributing].cpu().numpy()
        pos = np.clip(
            ((np.arange(contributing) + 0.5) * extent / frames - 64) / HOP,
            0,
            cache.frames[i] - 1,
        )
        for crop_id in range(offsets[i], offsets[i + 1]):
            start = cache.starts[crop_id]
            end = min(start + CORE, cache.frames[i])
            valid_frame = (np.floor(pos) >= start) & (np.ceil(pos) < end)
            mode, frame = np.nonzero((amp > 0) & valid_frame[None])
            if not len(mode):
                continue
            # Sample observed pairs with a blend of uniform and energy weighting.
            w = amp[mode, frame].astype(np.float64) ** 2
            prob = 0.5 / len(w) + 0.5 * w / w.sum()
            selected = rng.choice(len(w), size=PROBES, p=prob)
            crop_ids.append(crop_id)
            positions.append(pos[frame[selected]] - start)
            tokens.append(u[mode[selected], frame[selected]])
            weights.append(np.ones(PROBES, dtype=np.float32))
    if not crop_ids:
        raise ValueError("No observed modal teacher probes.")
    return (
        np.asarray(crop_ids),
        torch.tensor(np.stack(positions), dtype=torch.float32),
        torch.from_numpy(np.stack(tokens)),
        torch.from_numpy(np.stack(weights)),
    )


def probe_batches(data, order, cache, mean, std, device):
    ids, pos, u, w = data
    for start in range(0, len(order), BATCH):
        selected = order[start : start + BATCH]
        hidden, rho = prepare_batch(cache.batch(ids[selected]), mean, std)
        yield (
            hidden,
            rho,
            pos[selected].to(device),
            u[selected].to(device),
            w[selected].to(device),
        )


@torch.no_grad()
def targets(model, data, cache, mean, std, device):
    return torch.cat(
        [
            probe_head(model, h[:, CORE_SLICE] * std + mean, p, u).cpu()
            for h, _, p, u, _ in probe_batches(
                data, np.arange(len(data[0])), cache, mean, std, device
            )
        ]
    )


@torch.no_grad()
def validate_modal(sae, model, data, target, cache, mean, std, scale, device):
    latent_numerator = latent_denominator = modal_sum = modal_count = 0.0
    for i, (h, rho, p, u, w) in enumerate(
        probe_batches(data, np.arange(len(data[0])), cache, mean, std, device)
    ):
        decoded, _ = sae.reconstruct(h, rho > 0)
        core = decoded[:, CORE_SLICE]
        pred = probe_head(model, core * std + mean, p, u)
        expected = target[i * BATCH : i * BATCH + len(h)].to(device)
        latent_numerator += float(
            weighted_error(core - h[:, CORE_SLICE], rho[:, CORE_SLICE])
        )
        latent_denominator += float(rho[:, CORE_SLICE].sum()) * CHANNELS
        modal_sum += float((((pred - expected) / scale).square() * w[..., None]).sum())
        modal_count += float(w.sum()) * 2
    return latent_numerator / latent_denominator, modal_sum / modal_count


def fit(args, model, data, caches, manifest, artifact, mean, std):
    rng = np.random.default_rng(manifest["seed"])
    device = args.device
    train = make_probes(
        model, data["train"], caches["train"], args.train_clips, rng, device
    )
    val = make_probes(model, data["val"], caches["val"], args.val_clips, rng, device)
    train_target = targets(model, train, caches["train"], mean, std, device)
    val_target = targets(model, val, caches["val"], mean, std, device)
    # A robust train-only scale prevents amplitude from overwhelming frequency.
    scale = (
        train_target.flatten(0, 1)
        .abs()
        .quantile(0.95, dim=0)
        .clamp_min(1e-5)
        .to(device)
    )
    sae = ModalAwareSAE().to(device)
    sae.load_state_dict(artifact["state_dict"])
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4, foreach=True)
    last_seen = torch.full((FEATURES,), -1, dtype=torch.long, device=device)
    frames = step = 0
    history = []
    best = float("inf")
    baseline = validate_modal(
        sae, model, val, val_target, caches["val"], mean, std, scale, device
    )
    print(
        f"Initial validation latent/modal MSE {baseline}; probe crops {len(train[0])}/{len(val[0])}",
        flush=True,
    )
    start = time.monotonic()
    order = rng.permutation(len(train[0]))
    while step < args.steps and time.monotonic() - start < args.minutes * 60:
        for offset, (h, rho, p, u, w) in enumerate(
            probe_batches(train, order, caches["train"], mean, std, device)
        ):
            if step >= args.steps or time.monotonic() - start >= args.minutes * 60:
                break
            n = int(rho[:, CORE_SLICE].count_nonzero())
            opt.zero_grad(set_to_none=True)
            latent, core = training_loss(
                sae, h, rho, last_seen, n, frames, return_core=True
            )
            prediction = probe_head(model, core * std + mean, p, u)
            selected = order[offset * BATCH : offset * BATCH + len(h)]
            error = (prediction - train_target[selected].to(device)) / scale
            loss = latent + MODAL_WEIGHT * (error.square() * w[..., None]).sum() / (
                2 * w.sum()
            )
            loss.backward()
            opt.step()
            frames += n
            step += 1
            if step % 250 == 0 or step == args.steps:
                stats = validate_modal(
                    sae, model, val, val_target, caches["val"], mean, std, scale, device
                )
                history.append(
                    {
                        "step": step,
                        "latent_mse": stats[0],
                        "modal_mse": stats[1],
                        "seconds": time.monotonic() - start,
                    }
                )
                print(json.dumps(history[-1]), flush=True)
                objective = stats[0] + MODAL_WEIGHT * stats[1]
                if objective < best:
                    best = objective
                    save_best(
                        args.output / "best.pt",
                        {
                            "state_dict": sae.state_dict(),
                            "cache_sha256": artifact["cache_sha256"],
                            "modal_scale": scale.cpu(),
                            **history[-1],
                        },
                    )
        order = rng.permutation(len(train[0]))
    if not history or history[-1]["step"] != step:
        stats = validate_modal(
            sae, model, val, val_target, caches["val"], mean, std, scale, device
        )
        history.append(
            {
                "step": step,
                "latent_mse": stats[0],
                "modal_mse": stats[1],
                "seconds": time.monotonic() - start,
            }
        )
        if stats[0] + MODAL_WEIGHT * stats[1] < best:
            save_best(
                args.output / "best.pt",
                {
                    "state_dict": sae.state_dict(),
                    "cache_sha256": artifact["cache_sha256"],
                    "modal_scale": scale.cpu(),
                    **history[-1],
                },
            )
    write_json(
        args.output / "training.json",
        {
            "baseline": baseline,
            "history": history,
            "probe_crops": [len(train[0]), len(val[0])],
            "weight": MODAL_WEIGHT,
        },
    )
    sae.load_state_dict(
        torch.load(args.output / "best.pt", map_location=device, weights_only=True)[
            "state_dict"
        ]
    )
    return sae.eval()


def use_cached_reference(model, item, seed, device, cache, index):
    ref = reference(model, item, seed, device)
    stored = cache.values[cache.offsets[index] : cache.offsets[index + 1]]
    diff = ref.h_ref[0].cpu().numpy() - stored
    if not (np.abs(diff).max() <= 0.003 and np.sqrt(np.mean(diff**2)) <= 0.0002):
        raise ValueError(f"Unexpected encoder/cache mismatch for {item['sample_id']}.")
    h = torch.from_numpy(stored.copy()).unsqueeze(0).to(device)
    s = strike_code(model, h, ref.rho)
    return replace(ref, h_ref=h, s_ref=s, film_ref=model.transient.film_coefficients(s))


def spectrum(wave):
    n = min(512, len(wave))
    return np.abs(stft(wave, fs=SAMPLE_RATE, nperseg=n, noverlap=n * 3 // 4)[2])


def wave(result, gain):
    return (result * gain).flatten().cpu().numpy()


def write_set(folder, stem, waves):
    links = {}
    for key, y in matched_waves(waves).items():
        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite render for {stem}.")
        rel = f"audio/{stem}-{len(links)}.wav"
        sf.write(folder / rel, y, SAMPLE_RATE, subtype="PCM_24")
        links[key] = rel
    return links


@torch.inference_mode()
def audition(args, model, sae, datasets, caches, manifest, mean, std):
    val = datasets["val"]
    cache = caches["val"]
    activity = clip_activity(sae, cache, mean, std)
    pool = shortlist(activity, cache.rows, FEATURES)
    strong = sorted(
        (r for r in pool[16:] if r[0] >= len(val) * 0.10), key=lambda r: -r[2]
    )[:32]
    candidates = pool[:16] + strong
    jobs = defaultdict(list)
    cards = {}
    for active, packs, _, atom, examples in candidates:
        cards[atom] = {"atom": atom, "active": active, "packs": packs, "clips": []}
        for i in examples:
            jobs[i].append(atom)
    out = args.output / "audition"
    (out / "audio").mkdir(parents=True, exist_ok=False)
    # Rank candidates by their actual neutral-noise, level-matched mix response.
    scratch = out / "screen"
    (scratch / "audio").mkdir(parents=True, exist_ok=False)
    for i, atoms in tqdm(jobs.items(), desc="Screen modal auditions", unit="clip"):
        item = val[i]
        ref = use_cached_reference(model, item, manifest["seed"], args.device, cache, i)
        neutral = model.decode(ref, checkpoint_activations=False)
        h = ref.h_ref
        codes = sae.reconstruct((h - mean) / std, ref.rho > 0)[1]
        for atom in atoms:
            delta = latent_delta(sae, codes, atom, std)
            mixes = {"0": wave(neutral.waveform, ref.gain)}
            solos = {"0": wave(neutral.modal_waveform, ref.gain)}
            for alpha in (-1, 1, 4):
                params, _ = model._refine(ref, h + alpha * delta, False)
                modal = model.modal_synth(params, ref.length, ref.analysis_num_samples)
                solos[f"{alpha:+d}"] = wave(modal, ref.gain)
                mixes[f"{alpha:+d}"] = wave(
                    modal + neutral.noise_waveform + neutral.transient_waveform,
                    ref.gain,
                )
            compared = matched_waves(mixes)
            center = spectrum(compared["0"])
            scores = [
                float(
                    np.linalg.norm(spectrum(compared[a]) - center)
                    / max(np.linalg.norm(center), 1e-10)
                )
                for a in ("-1", "+1", "+4")
            ]
            links = write_set(scratch, f"mix-{atom}-{i}", mixes)
            solo_links = write_set(scratch, f"solo-{atom}-{i}", solos)
            cards[atom]["clips"].append(
                {
                    "id": item["sample_id"],
                    "name": val.metadata[item["sample_id"]].get(
                        "orig_relpath", item["sample_id"]
                    ),
                    "audio": {**links, **{"S " + a: f for a, f in solo_links.items()}},
                    "scores": scores,
                }
            )
    for card in cards.values():
        card["in_range"] = float(
            np.median([max(x["scores"][:2]) for x in card["clips"]])
        )
        card["plus4"] = float(np.median([x["scores"][2] for x in card["clips"]]))
    ordered = sorted(cards.values(), key=lambda c: (-c["in_range"], c["atom"]))
    selected = (
        ordered[:8] + sorted(ordered[8:], key=lambda c: (-c["plus4"], c["atom"]))[:2]
    )
    for card in selected:
        for clip in card["clips"]:
            for rel in clip["audio"].values():
                (scratch / rel).replace(out / rel)
    for p in (scratch / "audio").glob("*.wav"):
        p.unlink()
    (scratch / "audio").rmdir()
    scratch.rmdir()
    write_json(
        out / "candidates.json",
        {"screened": len(cards), "candidates": selected},
    )
    write_html(out / "index.html", selected)
    page = out / "index.html"
    page.write_text(
        page.read_text(encoding="utf-8")
        .replace(
            "Stronger edits are marked +2 and +4.",
            "Noise and transient are held neutral. S buttons solo the modal branch. +4 is exploratory.",
        )
        .replace(
            '<div class="deck">',
            '<p><a href="strike.html" style="color:inherit">Frozen strike-code check</a></p><div class="deck">',
        ),
        encoding="utf-8",
    )
    strike_check(args, model, datasets, caches, manifest, out, selected[:2])
    print(
        "Modal shortlist",
        [(c["atom"], round(c["in_range"], 3), round(c["plus4"], 3)) for c in selected],
        flush=True,
    )


@torch.inference_mode()
def strike_check(args, model, datasets, caches, manifest, out, cards):
    # Compare only real strike codes, keeping all model weights and S/N fixed.
    train = caches["train"]
    rng = np.random.default_rng(manifest["seed"])
    donor_ids = rng.choice(
        len(train.rows), size=min(64, len(train.rows)), replace=False
    )
    codes = []
    for i in donor_ids:
        h = torch.from_numpy(
            train.values[train.offsets[i] : train.offsets[i + 1]].copy()
        )[None].to(args.device)
        rho = (
            (
                train.samples[i] - torch.arange(h.shape[1], device=args.device) * HOP
            ).clamp(0, HOP)
            / HOP
        )[None]
        codes.append(strike_code(model, h, rho)[0])
    codes = torch.stack(codes)
    spread = codes.std(0, unbiased=False).clamp_min(1e-6)
    val = datasets["val"]
    index = {k: i for i, k in enumerate(val.file_list)}
    diagnostics, clips = [], []
    for key in dict.fromkeys(clip["id"] for card in cards for clip in card["clips"]):
        i = index[key]
        ref = use_cached_reference(
            model, val[i], manifest["seed"], args.device, caches["val"], i
        )
        neutral = model.decode(ref, checkpoint_activations=False)
        nearest = ((codes - ref.s_ref) / spread).square().mean(1).sort().indices
        donor = nearest[int(0.90 * (len(nearest) - 1))]
        waves = {"Neutral": wave(neutral.waveform, ref.gain)}
        for amount in (0.5, 1.0):
            s = torch.lerp(ref.s_ref, codes[donor][None], amount)
            t = model._transient(
                neutral.modal_waveform,
                model.transient.film_coefficients(s),
                checkpoint_activations=False,
            )
            diff = (
                t - neutral.transient_waveform
            ).norm() / neutral.waveform.norm().clamp_min(1e-12)
            waves[f"Swap {amount:g}"] = wave(
                neutral.modal_waveform + neutral.noise_waveform + t, ref.gain
            )
            diagnostics.append(
                {
                    "sample_id": key,
                    "donor": train.rows[donor_ids[int(donor)]]["id"],
                    "amount": amount,
                    "delta_over_mix_db": float(20 * diff.clamp_min(1e-12).log10()),
                }
            )
        clips.append(
            {
                "name": val.metadata[key].get("orig_relpath", key),
                "audio": write_set(out, f"strike-{i}", waves),
            }
        )
    with (out / "strike.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    write_html(
        out / "strike.html",
        [{"atom": 0, "active": len(clips), "packs": 1, "clips": clips}],
    )
    page = out / "strike.html"
    page.write_text(
        page.read_text(encoding="utf-8")
        .replace(
            "Compare each hit at −1, 0, and +1. Stronger edits are marked +2 and +4. Choose up to two atoms you would use in a rhythm.",
            "Only the existing strike code changes. Swap 0.5 interpolates halfway to a real train code. Swap 1 uses that code directly.",
        )
        .replace("Atom 0</label>", "Frozen strike code</label>"),
        encoding="utf-8",
    )
    print(
        "Strike delta range dB",
        min(r["delta_over_mix_db"] for r in diagnostics),
        max(r["delta_over_mix_db"] for r in diagnostics),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "noise-bank", "data-dir", "cache", "sae", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--train-clips", type=int, default=2048)
    parser.add_argument("--val-clips", type=int, default=384)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--minutes", type=float, default=25)
    args = parser.parse_args()
    if not (
        args.train_clips > 0
        and args.val_clips > 0
        and args.steps > 0
        and 0 < args.minutes <= 180
    ):
        parser.error("Counts must be positive and time must be at most 180 minutes.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    manifest = open_cache(args.cache)
    model, seed, identity = open_joint(args, manifest["identity"])
    artifact = torch.load(args.sae, map_location="cpu", weights_only=True)
    if artifact["cache_sha256"] != sha256(args.cache / "cache.json"):
        raise ValueError("SAE and h cache identity differ.")
    data = {split: dataset_for(args, seed, split) for split in ("train", "val")}
    caches = {}
    for split, dataset in data.items():
        rows = manifest["splits"][split]
        if dataset.file_list != [r["id"] for r in rows] or dataset.lengths != [
            r["samples"] for r in rows
        ]:
            raise ValueError(f"Cache membership or lengths differ for {split}.")
        caches[split] = FrameCache(args.cache, split, rows)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        {
            "identity": identity,
            "warm_start": sha256(args.sae),
            "split": "train/validation only",
            "seed": seed,
            "training_caps": {"steps": args.steps, "minutes": args.minutes},
            "teacher_probes_per_crop": PROBES,
            "train_clip_cap": args.train_clips,
            "val_clip_cap": args.val_clips,
        },
    )
    mean, std = normalization(manifest, args.device)
    sae = fit(args, model, data, caches, manifest, artifact, mean, std)
    audition(args, model, sae, data, caches, manifest, mean, std)


if __name__ == "__main__":
    main()
