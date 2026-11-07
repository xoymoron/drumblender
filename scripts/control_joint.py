"""Cache frozen joint latents, train one temporal SAE, and export an audition."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drumblender.control import (
    BATCH,
    CHANNELS,
    CORE_SLICE,
    FEATURES,
    ControlSAE,
    FrameCache,
    latent_delta,
    training_loss,
    weighted_error,
)
from drumblender.data import JointResynthesisDataset
from drumblender.models.joint_resynthesis import HOP
from drumblender.synths.noise_band import SAMPLE_RATE, load_noise_bank
from drumblender.tasks.joint_resynthesis import JointResynthesisTask, clip_offsets


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def open_joint(args, expected=None):
    bank, checksum = load_noise_bank(args.noise_bank)
    identity = {
        "checkpoint": sha256(args.checkpoint),
        "bank": checksum,
        "manifest": sha256(args.data_dir / args.meta_file),
        "modal_config": sha256(args.data_dir / "modal_config.json"),
    }
    if expected is not None and identity != expected:
        raise ValueError("The checkpoint, bank, or dataset differs from the h cache.")
    task = JointResynthesisTask.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", bank=bank, bank_sha256=checksum
    )
    return (
        task.model.eval().requires_grad_(False).to(args.device),
        task.hparams.seed,
        identity,
    )


def dataset_for(args, seed, split):
    dataset = JointResynthesisDataset(
        args.data_dir,
        args.meta_file,
        SAMPLE_RATE,
        None,
        split=split,
        audio_dir=args.audio_dir,
        seed=seed,
    )
    if not len(dataset):
        raise ValueError(f"Empty {split} split.")
    return dataset


def reference(model, item, seed, device):
    return model.encode_reference(
        item["audio"][None].to(device, non_blocking=True),
        item["params"][None].to(device, non_blocking=True),
        item["analysis_num_samples"],
        clip_offsets(item["sample_id"], seed).to(device),
        checkpoint_activations=False,
    )


@torch.inference_mode()
def cache_latents(args):
    model, seed, identity = open_joint(args)
    datasets = {s: dataset_for(args, seed, s) for s in ("train", "val")}
    args.cache.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "identity": identity, "seed": seed, "splits": {}}
    count = 0.0
    total = np.zeros(CHANNELS, dtype=np.float64)
    squares = total.copy()
    for split, data in datasets.items():
        rows = [
            {
                "id": key,
                "samples": size,
                "pack": data.sample_pack_key(data.metadata[key]),
            }
            for key, size in zip(data.file_list, data.lengths)
        ]
        nframes = sum((n + HOP - 1) // HOP for n in data.lengths)
        cache = np.lib.format.open_memmap(
            args.cache / f"{split}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(nframes, CHANNELS),
        )
        loader = DataLoader(
            data,
            batch_size=None,
            num_workers=args.workers,
            pin_memory=args.device.type == "cuda",
        )
        cursor = 0
        for row, item in tqdm(
            zip(rows, loader), total=len(rows), desc=f"Cache {split}"
        ):
            if item["sample_id"] != row["id"]:
                raise ValueError("DataLoader order differs from the cache manifest.")
            h = reference(model, item, seed, args.device).h_ref[0].cpu().numpy()
            frames = (row["samples"] + HOP - 1) // HOP
            if h.shape != (frames, CHANNELS) or not np.isfinite(h).all():
                raise ValueError(f"Invalid latent for {row['id']}.")
            cache[cursor : cursor + frames] = h
            cursor += frames
            if split == "train":
                rho = np.ones(frames, dtype=np.float64)
                rho[-1] = (row["samples"] - (frames - 1) * HOP) / HOP
                count += rho.sum()
                total += np.einsum("tc,t->c", h, rho, dtype=np.float64)
                squares += np.einsum("tc,tc,t->c", h, h, rho, dtype=np.float64)
        cache.flush()
        del cache
        manifest["splits"][split] = rows
        print(f"{split}: {nframes:,} frames, {nframes * CHANNELS * 4 / 2**30:.2f} GiB")
    mean = total / count
    manifest["mean"] = mean.tolist()
    manifest["std"] = np.sqrt(np.maximum(squares / count - mean**2, 1e-12)).tolist()
    write_json(args.cache / "cache.json", manifest)


def open_cache(path):
    manifest = read_json(path / "cache.json")
    if manifest.get("version") != 1:
        raise ValueError("Unsupported h cache format.")
    return manifest


def batches(cache, order, device):
    groups = (order[start : start + BATCH] for start in range(0, len(order), BATCH))
    if device.type != "cuda":
        for ids in groups:
            yield ids, cache.batch(ids)
        return

    def pinned(ids):
        return tuple(tensor.pin_memory() for tensor in cache.batch(ids))

    with ThreadPoolExecutor(max_workers=1) as pool:
        ids = next(groups, None)
        if ids is None:
            return
        pending = pool.submit(pinned, ids)
        for following in groups:
            batch = pending.result()
            pending = pool.submit(pinned, following)
            yield ids, batch
            ids = following
        yield ids, pending.result()


def normalization(manifest, device):
    return tuple(
        torch.tensor(manifest[k], dtype=torch.float32, device=device)
        for k in ("mean", "std")
    )


def prepare_batch(batch, mean, std):
    hidden, weights = batch
    hidden = hidden.to(mean.device, non_blocking=True)
    weights = weights.to(mean.device, non_blocking=True)
    return (hidden - mean) / std, weights


@torch.inference_mode()
def validate(sae, cache, mean, std, deadline):
    summary = torch.zeros(2, device=mean.device, dtype=torch.float64)
    live = torch.zeros(sae.features, device=mean.device, dtype=torch.bool)
    for _, batch in batches(cache, np.arange(len(cache.starts)), mean.device):
        if time.monotonic() >= deadline:
            return None
        hidden, weights = prepare_batch(batch, mean, std)
        reconstruction, codes = sae.reconstruct(hidden, weights > 0)
        core = weights[:, CORE_SLICE]
        error = reconstruction[:, CORE_SLICE] - hidden[:, CORE_SLICE]
        summary += torch.stack((weighted_error(error, core), core.sum() * CHANNELS))
        live |= (codes[:, CORE_SLICE] > 0).any(dim=(0, 1))
    return float(summary[0] / summary[1]), int((~live).sum())


def save_best(path, state):
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def train_sae(args):
    from drumblender.control_protocol import WARMUP_KIND

    manifest = open_cache(args.cache)
    train, val = (
        FrameCache(args.cache, s, manifest["splits"][s]) for s in ("train", "val")
    )
    mean, std = normalization(manifest, args.device)
    torch.manual_seed(manifest["seed"])
    rng = np.random.default_rng(manifest["seed"])
    sae = ControlSAE().to(args.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4, foreach=True)
    last_seen = torch.full((FEATURES,), -1, dtype=torch.int64, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    ckpt = args.output / "best.pt"
    identity = sha256(args.cache / "cache.json")
    frame_total = step = 0
    best = float("inf")
    start = time.monotonic()
    target = start + args.minutes * 60
    deadline = start + args.max_minutes * 60
    epoch = 0
    history = []
    progress = tqdm(
        total=args.minutes * 60,
        desc="SAE warm-up",
        unit="s",
        dynamic_ncols=True,
        mininterval=1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}{postfix}]",
    )
    while time.monotonic() < target or step == 0:
        epoch += 1
        for _, batch in batches(train, rng.permutation(len(train.starts)), args.device):
            if step and time.monotonic() >= target:
                break
            nframes = int(batch[1][:, CORE_SLICE].count_nonzero())
            hidden, weights = prepare_batch(batch, mean, std)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(sae, hidden, weights, last_seen, nframes, frame_total)
            loss.backward()
            optimizer.step()
            step += 1
            frame_total += nframes
            progress.update(min(time.monotonic() - start, progress.total) - progress.n)
            if step == 20 or step % 100 == 0:
                current_loss = float(loss.detach())
                elapsed = time.monotonic() - start
                progress.set_postfix(
                    step=step, loss=f"{current_loss:.5f}", refresh=False
                )
                tqdm.write(
                    f"{step:,} steps | {frame_total / elapsed:,.0f} valid frames/s (run average) | MSE+AuxK {current_loss:.5f}",
                )

        progress.set_description("Warm-up validation")
        measured = validate(sae, val, mean, std, deadline)
        progress.update(min(time.monotonic() - start, progress.total) - progress.n)
        progress.set_description("SAE warm-up")
        if measured is None:
            break
        score, dead = measured
        entry = {
            "epoch": epoch,
            "step": step,
            "frames": frame_total,
            "elapsed_s": time.monotonic() - start,
            "val_mse": score,
            "val_dead": dead,
        }
        history.append(entry)
        write_json(args.output / "training.json", history)
        tqdm.write(json.dumps(entry))
        if score < best:
            best = score
            save_best(
                ckpt,
                {
                    "state_dict": sae.state_dict(),
                    "cache_sha256": identity,
                    "identity": manifest["identity"],
                    "seed": manifest["seed"],
                    "kind": WARMUP_KIND,
                    **entry,
                },
            )
    progress.close()
    if not ckpt.is_file():
        raise RuntimeError("The time cap expired before a complete validation pass.")
    # Budget the entire invocation, not the time of the best validation epoch.
    artifact = torch.load(ckpt, map_location="cpu", weights_only=True)
    artifact.update(
        warmup_seconds=time.monotonic() - start,
        warmup_total_steps=step,
        loss="L_h + L_AuxK / 32",
        initialization="random",
        optimizer="Adam",
        learning_rate=3e-4,
    )
    save_best(ckpt, artifact)
    write_json(
        args.output / "timing.json",
        {
            "warmup_seconds": artifact["warmup_seconds"],
            "target_minutes": args.minutes,
            "max_minutes": args.max_minutes,
            "selected_step": artifact["step"],
            "total_steps": step,
        },
    )
    print(f"Best checkpoint: {ckpt}")


@torch.inference_mode()
def clip_activity(sae, cache, mean, std):
    activity = np.zeros((len(cache.rows), FEATURES), dtype=np.float64)
    for ids, batch in batches(cache, np.arange(len(cache.starts)), mean.device):
        h, w = prepare_batch(batch, mean, std)
        mask = w > 0
        codes, _, _ = sae.select(sae.preactivations(h, mask), mask, sae.top_k)
        sums = (codes[:, CORE_SLICE].square() * w[:, CORE_SLICE, None]).sum(dim=1)
        np.add.at(activity, cache.clip_indices[ids], sums.cpu().numpy())
    return np.sqrt(activity / (cache.samples[:, None] / HOP))


def shortlist(activity, rows, limit):
    packs = np.asarray([row["pack"] for row in rows])
    pack_total = len(set(packs))
    ranked = []
    for feature, strength in enumerate(activity.T):
        active = np.flatnonzero(strength > 0)
        n_packs = len(set(packs[active]))
        if len(active) < 3 or n_packs < min(2, pack_total):
            continue
        order = active[np.argsort(-strength[active], kind="stable")]
        distinct, seen = [], set()
        for i in order:
            if packs[i] not in seen:
                distinct.append(int(i))
                seen.add(packs[i])
        examples = (distinct + [int(i) for i in order if int(i) not in distinct])[:3]
        ranked.append(
            (
                len(active),
                n_packs,
                float(np.median(strength[active])),
                feature,
                examples,
            )
        )
    ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    return ranked[:limit]


def matched_waves(waves, target_db=-20):
    """Equal RMS copies with common headroom, leaving render metrics untouched."""
    rms = {k: np.sqrt(np.mean(v.astype(np.float64) ** 2)) for k, v in waves.items()}
    target = 10 ** (target_db / 20)
    ratios = [np.max(np.abs(v)) / rms[k] for k, v in waves.items() if rms[k] > 1e-8]
    if ratios:
        target = min(target, 0.95 / max(ratios))
    return {k: v * (target / rms[k]) if rms[k] > 1e-8 else v for k, v in waves.items()}


def write_html(path, cards):
    blocks = []
    for card in cards:
        atom = card["atom"]
        clips = []
        for clip in card["clips"]:
            buttons = " ".join(
                f'<button type="button" data-audio="{escape(wav, quote=True)}">{escape(amount)}</button>'
                for amount, wav in clip["audio"].items()
            )
            clips.append(
                f'<div class="clip"><span>{escape(clip["name"])}</span><div>{buttons}</div></div>'
            )
        blocks.append(
            f'<section><h2><label><input type="checkbox" value="{atom}"> Atom {atom}</label></h2>'
            f"<p>{card['active']} active validation clips · {card['packs']} packs</p>"
            + "".join(clips)
            + "</section>"
        )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>One-shot knob audition</title><style>
*{box-sizing:border-box}body{margin:0;background:#141820;color:#eef2f6;font:16px system-ui}
main{max-width:960px;margin:auto;padding:24px}h1{margin:.4em 0}h2{margin:0;font-size:1.15rem}
.deck{position:sticky;top:0;background:#141820;padding:12px 0;z-index:1}audio{max-width:100%;width:480px}
section{background:#202631;border:1px solid #404c60;border-radius:12px;margin:14px 0;padding:16px}
section p{color:#bfcde0;margin:6px 0 12px}.clip{display:flex;flex-wrap:wrap;align-items:center;
justify-content:space-between;gap:8px;padding:8px 0;border-top:1px solid #404c60}.clip span{overflow-wrap:anywhere}
button{background:#303b4d;color:inherit;border:1px solid #60708a;border-radius:6px;padding:6px 12px;cursor:pointer}
button[aria-pressed=true]{background:#c7e76d;color:#142002}output{display:block;color:#c7e76d;padding:8px 0}
</style><main><h1>One-shot knob audition</h1>
<p>Compare each hit at −1, 0, and +1. Stronger edits are marked +2 and +4. Choose up to two atoms you would use in a rhythm.</p>
<div class="deck"><audio controls></audio><output>Selected atoms: —</output></div>
"""
        + "\n".join(blocks)
        + """</main><script>
const player = document.querySelector('audio');
const output = document.querySelector('output');
document.addEventListener('click', e => {
  const button = e.target.closest('[data-audio]');
  if (!button) return;
  document.querySelector('[aria-pressed=true]')?.setAttribute('aria-pressed', 'false');
  button.setAttribute('aria-pressed', 'true');
  player.src = button.dataset.audio;
  player.play().catch(() => {});
});
document.addEventListener('change', e => {
  if (!e.target.matches('input[type=checkbox]')) return;
  const selected = Array.from(document.querySelectorAll('input:checked'));
  if (selected.length > 2) { e.target.checked = false; return; }
  output.textContent = 'Selected atoms: ' + (selected.map(x => x.value).join(', ') || '—');
});
</script></html>"""
    )
    path.write_text(page, encoding="utf-8")


@torch.inference_mode()
def audition(args):
    manifest = open_cache(args.cache)
    model, seed, _ = open_joint(args, manifest["identity"])
    data = dataset_for(args, seed, "val")
    rows = manifest["splits"]["val"]
    if data.file_list != [row["id"] for row in rows]:
        raise ValueError("Validation membership differs from the h cache.")
    if data.lengths != [row["samples"] for row in rows]:
        raise ValueError("Validation clip lengths differ from the h cache.")
    cache = FrameCache(args.cache, "val", rows)
    artifact = torch.load(args.sae, map_location="cpu", weights_only=True)
    if artifact["cache_sha256"] != sha256(args.cache / "cache.json"):
        raise ValueError("The SAE was fitted to another h cache.")
    sae = ControlSAE().eval().to(args.device)
    sae.load_state_dict(artifact["state_dict"])
    mean, std = normalization(manifest, args.device)
    activity = clip_activity(sae, cache, mean, std)
    choices = shortlist(activity, rows, args.candidates)
    if not choices:
        raise ValueError(
            "No feature activated on three validation clips with sufficient pack coverage."
        )
    if len(choices) < args.candidates:
        print(f"Only {len(choices)} features meet the validation coverage rule.")
    by_clip = defaultdict(list)
    cards = []
    for n_active, n_packs, _, feature, examples in choices:
        clips = []
        for i in examples:
            row = rows[i]
            name = data.metadata[row["id"]].get("orig_relpath") or row["id"]
            clip = {"id": row["id"], "name": name, "audio": {}}
            clips.append(clip)
            by_clip[i].append((feature, clip))
        cards.append(
            {"atom": feature, "active": n_active, "packs": n_packs, "clips": clips}
        )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    for i, jobs in tqdm(by_clip.items(), desc="Render validation", unit="clip"):
        ref = reference(model, data[i], seed, args.device)
        mask = ref.rho > 0
        h = ref.h_ref
        cached = cache.values[cache.offsets[i] : cache.offsets[i + 1]]
        if not np.allclose(h[0].cpu().numpy(), cached, rtol=1e-4, atol=1e-5):
            raise ValueError(
                f"Cached h differs for {rows[i]['id']}. Check the input WAV, "
                "modal parameters, and device used to build the cache."
            )
        z = (h - mean) / std
        codes, _, _ = sae.select(sae.preactivations(z, mask), mask, sae.top_k)
        neutral = (
            (model.decode(ref, checkpoint_activations=False).waveform * ref.gain)
            .flatten()
            .cpu()
            .numpy()
        )
        for feature, clip in jobs:
            delta = latent_delta(sae, codes, feature, std)
            waves = {}
            for alpha in (-1, 0, 1, *([2, 4] if args.strong else [])):
                if alpha == 0:
                    waves[alpha] = neutral
                else:
                    result = model.decode(
                        ref, h + alpha * delta, checkpoint_activations=False
                    )
                    waves[alpha] = (result.waveform * ref.gain).flatten().cpu().numpy()
            if not all(np.isfinite(w).all() for w in waves.values()):
                raise ValueError(
                    f"Non-finite render for atom {feature}, clip {clip['id']}."
                )
            for alpha, wave in matched_waves(waves).items():
                filename = f"audio/atom{feature:03d}-clip{i:05d}-{alpha:+d}.wav"
                sf.write(args.output / filename, wave, SAMPLE_RATE, subtype="PCM_24")
                clip["audio"][f"{alpha:+d}" if alpha else "0"] = filename
    write_json(
        args.output / "candidates.json",
        {
            "cache_sha256": artifact["cache_sha256"],
            "sae_sha256": sha256(args.sae),
            "candidates": cards,
        },
    )
    write_html(args.output / "index.html", cards)
    print(f"{len(cards)} candidates: {args.output / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    cache = stages.add_parser("cache", help="Encode the train and validation splits.")
    train = stages.add_parser(
        "train", help="Train one SAE and keep the best validation checkpoint."
    )
    render = stages.add_parser(
        "audition", help="Write validation WAVs and a local HTML player."
    )
    for sub in (cache, train, render):
        sub.add_argument("--cache", type=Path, required=True)
        sub.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    for sub in (cache, render):
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--noise-bank", type=Path, required=True)
        sub.add_argument("--data-dir", type=Path, required=True)
        sub.add_argument("--audio-dir", type=Path)
        sub.add_argument("--meta-file", default="metadata.json")
    cache.add_argument("--workers", type=int, default=2)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--minutes", type=float, default=25)
    train.add_argument("--max-minutes", type=float, default=30)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--sae", type=Path, required=True)
    render.add_argument("--candidates", type=int, choices=range(8, 13), default=10)
    render.add_argument(
        "--strong", action="store_true", help="Also audition +2 and +4."
    )
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment.")
    if args.stage == "train" and not 0 < args.minutes < args.max_minutes <= 180:
        parser.error("Require 0 < --minutes < --max-minutes <= 180.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    {"cache": cache_latents, "train": train_sae, "audition": audition}[args.stage](args)


if __name__ == "__main__":
    main()
