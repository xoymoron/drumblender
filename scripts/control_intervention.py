"""One bounded post-hoc run: intervention SAE, paired baseline, and auditions."""

import argparse
import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from drumblender.control import BATCH, CORE_SLICE, FEATURES, FrameCache, training_loss
from drumblender.control_protocol import select_cohort
from drumblender.intervention import (
    BETA,
    DIMENSION,
    EXCERPT,
    AcousticView,
    BoundedReader,
    InterventionSAE,
    atom_delta,
    cached_reference,
    edit_direction,
    modal_prefix,
)
from scripts.control_joint import (
    clip_activity,
    dataset_for,
    normalization,
    open_cache,
    open_joint,
    prepare_batch,
    save_best,
    sha256,
    write_json,
)


class RunData:
    def __init__(self, args):
        self.args = args
        self.manifest = open_cache(args.cache)
        self.model, self.seed, self.identity = open_joint(
            args, self.manifest["identity"]
        )
        self.mean, self.std = normalization(self.manifest, args.device)
        self.datasets, self.caches = {}, {}
        for split in ("train", "val"):
            dataset = dataset_for(args, self.seed, split)
            rows = self.manifest["splits"][split]
            if dataset.file_list != [r["id"] for r in rows] or dataset.lengths != [
                r["samples"] for r in rows
            ]:
                raise ValueError(f"Cache membership mismatch: {split}")
            self.datasets[split] = dataset
            self.caches[split] = FrameCache(args.cache, split, rows)
        self.references = OrderedDict()

    def hidden(self, split, index):
        cache = self.caches[split]
        values = cache.values[cache.offsets[index] : cache.offsets[index + 1]].copy()
        return torch.from_numpy(values)[None].to(self.args.device)

    def reference(self, split, index):
        key = (split, int(index))
        if key not in self.references:
            item = self.datasets[split][int(index)]
            self.references[key] = cached_reference(
                self.model, item, self.hidden(split, index), self.seed
            )
            if len(self.references) > 4:
                self.references.popitem(last=False)
        self.references.move_to_end(key)
        return self.references[key]

    def codes(self, sae, ref):
        q = (ref.h_ref - self.mean) / self.std
        return sae.select(sae.preactivations(q, ref.rho > 0), ref.rho > 0, sae.top_k)[0]


def crop_bounds(ref, frame):
    center = frame * 128 + 64
    left = max(0, min(center - EXCERPT // 2, ref.length - EXCERPT))
    return left, min(ref.length, left + EXCERPT)


@torch.no_grad()
def prepare(run, sae, output):
    start = time.monotonic()
    path = output / "calibration.pt"
    if path.exists():
        return torch.load(path, map_location=run.args.device, weights_only=True)
    cache = run.caches["train"]
    activity = clip_activity(sae, cache, run.mean, run.std)
    np.save(output / "train_activity.npy", activity)
    cohort = select_cohort(activity, cache.rows)
    if len(cohort) < 4:
        raise ValueError("Fewer than four supported intervention features.")
    rng = np.random.default_rng(run.seed + 901)
    radii = {i: [] for i in cohort}
    waves, clip_ids = [], []
    for index in tqdm(
        rng.permutation(len(cache.rows)), desc="Calibrate train-only units"
    ):
        ref = run.reference("train", int(index))
        codes = run.codes(sae, ref)
        for i in cohort:
            if len(radii[i]) < 128 and codes[..., i].any():
                delta = atom_delta(sae, codes, i)
                radius = float(delta.norm(dim=-1).amax() / 16)
                if radius > 0:
                    radii[i].append(radius)
        if len(waves) < 32:
            left, end = crop_bounds(ref, int(rng.integers(ref.h_ref.shape[1])))
            wave = modal_prefix(run.model, ref, ref.h_ref, end)[:, left:end]
            if float(wave.square().mean().sqrt()) >= 1e-5:
                waves.append(wave)
        clip_ids.append(cache.rows[index]["id"])
        if (
            len(clip_ids) >= 128
            and min(map(len, radii.values())) >= 16
            and len(waves) >= 32
        ):
            break
        if len(clip_ids) >= 2048:
            break
    cohort = [i for i in cohort if len(radii[i]) >= 16]
    if len(cohort) < 4 or not waves:
        raise ValueError("Insufficient positive calibration support.")
    floors = torch.full((FEATURES,), 1e-6, device=run.args.device)
    for i in cohort:
        floors[i] = max(float(np.quantile(radii[i], 0.1)), 1e-6)
    observer = AcousticView().to(run.args.device)
    observer.calibrate(waves)
    payload = {
        "cohort": torch.tensor(cohort, device=run.args.device),
        "floors": floors,
        "observer": observer.state_dict(),
        "clip_ids": clip_ids,
        "support_counts": {str(i): len(radii[i]) for i in cohort},
        "cache_sha256": sha256(run.args.cache / "cache.json"),
        "seconds": time.monotonic() - start,
    }
    save_best(path, payload)
    write_json(
        output / "calibration.json",
        {
            "cohort": cohort,
            "clip_ids": clip_ids,
            "floors": {str(i): float(floors[i]) for i in cohort},
            "support_counts": payload["support_counts"],
            "seconds": payload["seconds"],
        },
    )
    return payload


def choose_group(codes, cohort, counts, rng):
    for _ in range(16):
        frame = int(rng.integers(codes.shape[1]))
        eligible = (
            torch.nonzero(codes[0, frame, cohort].detach() > 0).flatten().cpu().numpy()
        )
        if len(eligible) >= 4:
            weights = 1 / (1 + counts[eligible])
            picked = rng.choice(eligible, 4, replace=False, p=weights / weights.sum())
            return frame, torch.tensor(picked, device=codes.device)
    return None


def observation_views(observer, neutral, rng):
    views = []
    for _ in range(2):
        jitter = int(rng.integers(128))
        scale = 10 ** (-float(rng.uniform(30, 40)) / 20)
        first, second = (
            torch.randn_like(neutral) * scale,
            torch.randn_like(neutral) * scale,
        )
        with torch.no_grad():
            origin = observer(neutral, jitter, first)
            zero = (observer(neutral, jitter, second) - origin) / np.sqrt(DIMENSION)
        views.append((jitter, second, origin, zero))
    return views


def effects(observer, wave, views):
    return torch.cat(
        [
            (observer(wave, jitter, noise) - origin) / np.sqrt(DIMENSION)
            for jitter, noise, origin, _ in views
        ]
    )


@torch.no_grad()
def validate(run, sae, baseline, reader, observer, cohort, floors, end_time):
    rng = np.random.default_rng(run.seed + 903)
    cache = run.caches["val"]
    errors = [0.0, 0.0]
    weight = 0.0
    for ids in np.array_split(
        rng.choice(len(cache.starts), min(256, len(cache.starts)), replace=False), 16
    ):
        if time.monotonic() > end_time:
            break
        q, rho = prepare_batch(cache.batch(ids), run.mean, run.std)
        w = rho[:, CORE_SLICE]
        for i, net in enumerate((sae, baseline)):
            pred, _ = net.reconstruct(q, rho > 0)
            errors[i] += float(
                ((pred[:, CORE_SLICE] - q[:, CORE_SLICE]).square() * w[..., None]).sum()
            )
        weight += float(w.sum()) * 256
    rows = []
    for index in rng.choice(len(cache.rows), min(8, len(cache.rows)), replace=False):
        if time.monotonic() > end_time:
            break
        ref = run.reference("val", int(index))
        codes = run.codes(sae, ref)
        chosen = choose_group(codes, cohort, np.zeros(len(cohort)), rng)
        if chosen is None:
            continue
        frame, group = chosen
        left, end = crop_bounds(ref, frame)
        neutral = modal_prefix(run.model, ref, ref.h_ref, end)[:, left:end]
        if float(neutral.square().mean().sqrt()) < 1e-5:
            continue
        views = observation_views(observer, neutral, rng)
        for position, ci in enumerate(group):
            atom = int(cohort[ci])
            alpha = (-1 if position % 2 else 1) * 0.75
            direction, _ = edit_direction(sae, codes, atom, floors[atom])
            h = ref.h_ref + run.std * BETA * alpha * direction
            wave = modal_prefix(run.model, ref, h, end)[:, left:end]
            effect = effects(observer, wave, views)
            loss, measured = reader.loss(effect, group, position, alpha)
            rows.append(
                {"loss": float(loss), **{k: float(v) for k, v in measured.items()}}
            )
    return {
        "latent_mse": [e / max(weight, 1) for e in errors],
        "reader": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        if rows
        else {},
        "reader_examples": len(rows),
    }


def fit(run, sae, calibration, output):
    cohort, floors = calibration["cohort"], calibration["floors"]
    observer = AcousticView().to(run.args.device)
    observer.load_state_dict(calibration["observer"])
    reader = BoundedReader(len(cohort)).to(run.args.device)
    baseline = copy.deepcopy(sae)
    optim = torch.optim.Adam(sae.parameters(), lr=1e-4, foreach=True)
    shadow_optim = torch.optim.Adam(baseline.parameters(), lr=1e-4, foreach=True)
    read_optim = torch.optim.Adam(reader.parameters(), lr=3e-4, foreach=True)
    seen = [
        torch.full((FEATURES,), -1, dtype=torch.long, device=run.args.device)
        for _ in range(2)
    ]
    rng = np.random.default_rng(run.seed + 904)
    counts = np.zeros(len(cohort), dtype=np.int64)
    cache = run.caches["train"]
    frames = step = groups = skipped = 0
    start = time.monotonic()
    hard_end = start + run.args.max_minutes * 60
    update_end = min(start + run.args.minutes * 60 - 600, hard_end - 900)
    history, rolling = [], []
    batch_ids = []
    initial = validate(
        run, sae, baseline, reader, observer, cohort, floors, hard_end - 600
    )
    write_json(output / "initial_validation.json", initial)
    print("INITIAL", json.dumps(initial), flush=True)
    last_group_seconds = 1.0

    def snapshot(name):
        payload = {
            "state_dict": sae.state_dict(),
            "baseline_state_dict": baseline.state_dict(),
            "reader_state_dict": reader.state_dict(),
            "observer": observer.state_dict(),
            "cohort": cohort,
            "floors": floors,
            "identity": run.identity,
            "cache_sha256": calibration["cache_sha256"],
            "seed": run.seed,
            "steps": step,
            "groups": groups,
            "seconds": time.monotonic() - start,
            "feature_exposure": counts.tolist(),
            "beta": BETA,
        }
        save_best(output / name, payload)
        return payload

    while (
        groups < run.args.groups
        and time.monotonic() + max(last_group_seconds * 2, 10) < update_end
    ):
        step_start = time.monotonic()
        step += 1
        ids = rng.integers(len(cache.starts), size=BATCH)
        q, rho = prepare_batch(cache.batch(ids), run.mean, run.std)
        nframes = int(rho[:, CORE_SLICE].count_nonzero())
        optim.zero_grad(set_to_none=True)
        shadow_optim.zero_grad(set_to_none=True)
        read_optim.zero_grad(set_to_none=True)
        loss_h = training_loss(sae, q, rho, seen[0], nframes, frames)
        loss_h.backward()
        reconstruction_gradient = float(
            torch.stack(
                [
                    p.grad.detach().square().sum()
                    for p in sae.parameters()
                    if p.grad is not None
                ]
            )
            .sum()
            .sqrt()
        )
        shadow_loss = training_loss(baseline, q, rho, seen[1], nframes, frames)
        shadow_loss.backward()
        frames += nframes
        index = int(rng.integers(len(cache.rows)))
        trace = {
            "step": step,
            "train_index": index,
            "crop_ids": ids.tolist(),
            "sample_id": cache.rows[index]["id"],
            "pack": cache.rows[index]["pack"],
            "intervention": False,
        }
        batch_ids.append(trace)
        ref = run.reference("train", index)
        codes = run.codes(sae, ref)
        chosen = choose_group(codes, cohort, counts, rng)
        metrics = []
        if chosen is not None:
            frame, group = chosen
            left, end = crop_bounds(ref, frame)
            with torch.no_grad():
                neutral = modal_prefix(run.model, ref, ref.h_ref, end)[:, left:end]
            if float(neutral.square().mean().sqrt()) >= 1e-5:
                alpha = float(rng.uniform(0.25, 1) * rng.choice([-1, 1]))
                trace.update(
                    intervention=True,
                    frame=frame,
                    crop=[left, end],
                    alpha=alpha,
                    atoms=cohort[group].tolist(),
                    eligible=cohort[codes[0, frame, cohort].detach() > 0].tolist(),
                )
                views = observation_views(observer, neutral, rng)
                for position, ci in enumerate(group):
                    atom = int(cohort[ci])
                    direction, radius = edit_direction(sae, codes, atom, floors[atom])
                    edited = ref.h_ref + run.std * BETA * alpha * direction
                    leaf = edited.detach().requires_grad_(True)
                    wave = modal_prefix(run.model, ref, leaf, end)[:, left:end]
                    effect = effects(observer, wave, views)
                    learn, measured = reader.loss(
                        effect, group, position, alpha, frozen=True
                    )
                    if not torch.isfinite(learn):
                        raise FloatingPointError("Nonfinite intervention loss.")
                    gradient = torch.autograd.grad(0.1 * learn / 4, leaf)[0]
                    edited.backward(gradient, retain_graph=position < 3)
                    read_loss, _ = reader.loss(effect.detach(), group, position, alpha)
                    (read_loss / 4).backward()
                    counts[int(ci)] += 1
                    metrics.append(
                        {
                            "intervention": float(learn.detach()),
                            **{k: float(v) for k, v in measured.items()},
                            "h_gradient": float(gradient.norm()),
                            "floor_hit": float(radius.detach() < floors[atom]),
                        }
                    )
                zero = torch.cat([view[3] for view in views])
                _, zero_amount = reader(zero)
                (
                    0.25
                    * F.smooth_l1_loss(
                        zero_amount[:, group], torch.zeros_like(zero_amount[:, group])
                    )
                ).backward()
                groups += 1
            else:
                skipped += 1
                trace["skip_reason"] = "silent_modal_excerpt"
        else:
            skipped += 1
            trace["skip_reason"] = "fewer_than_four_cohort_features"
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sae.parameters(), 1.0, error_if_nonfinite=True
        )
        torch.nn.utils.clip_grad_norm_(
            baseline.parameters(), 1.0, error_if_nonfinite=True
        )
        optim.step()
        shadow_optim.step()
        if metrics:
            read_optim.step()
        last_group_seconds = time.monotonic() - step_start
        rolling.append(
            {
                "latent_loss": float(loss_h.detach()),
                "baseline_loss": float(shadow_loss.detach()),
                "gradient_norm": float(grad_norm),
                "reconstruction_gradient": reconstruction_gradient,
                "clipped": float(grad_norm > 1),
                "step_seconds": last_group_seconds,
                **(
                    {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
                    if metrics
                    else {}
                ),
            }
        )
        if step == 1 or step % 20 == 0:
            keys = sorted({k for row in rolling for k in row})
            entry = {
                "step": step,
                "groups": groups,
                "skipped": skipped,
                "seconds": time.monotonic() - start,
                **{k: float(np.mean([r[k] for r in rolling if k in r])) for k in keys},
            }
            history.append(entry)
            write_json(output / "training.json", history)
            print("TRAIN", json.dumps(entry), flush=True)
            rolling = []
        if step % 200 == 0:
            snapshot("latest.pt")
            with (output / "sampling.jsonl").open("a", encoding="utf-8") as file:
                file.write("".join(json.dumps(row) + "\n" for row in batch_ids))
            batch_ids = []
        if step % 500 == 0 and time.monotonic() + 180 < update_end:
            result = validate(
                run, sae, baseline, reader, observer, cohort, floors, update_end - 60
            )
            result.update(step=step, groups=groups)
            with (output / "validation.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(result) + "\n")
            print("VALIDATION", json.dumps(result), flush=True)
        del codes, ref, loss_h, shadow_loss
    with (output / "sampling.jsonl").open("a", encoding="utf-8") as file:
        file.write("".join(json.dumps(row) + "\n" for row in batch_ids))
    final = snapshot("final.pt")
    result = validate(
        run, sae, baseline, reader, observer, cohort, floors, hard_end - 600
    )
    write_json(output / "final_validation.json", result)
    print("FINAL_VALIDATION", json.dumps(result), flush=True)
    return final, sae, baseline, start, hard_end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "noise-bank", "data-dir", "cache", "sae", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--max-minutes", type=float, default=180)
    parser.add_argument("--groups", type=int, default=3000)
    parser.add_argument("--audition-only", action="store_true")
    args = parser.parse_args()
    if not 10 < args.minutes <= args.max_minutes <= 180 or args.groups < 1:
        parser.error("Require 10 < minutes <= max-minutes <= 180 and positive groups.")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    run = RunData(args)
    torch.manual_seed(run.seed)
    np.random.seed(run.seed)
    sae = InterventionSAE().to(args.device)
    artifact = torch.load(args.sae, map_location=args.device, weights_only=True)
    if artifact["cache_sha256"] != sha256(args.cache / "cache.json"):
        raise ValueError("Warm start and h cache identity mismatch.")
    sae.load_state_dict(artifact["state_dict"])
    from scripts.audition_intervention import audition, transient_diagnostic

    if args.audition_only:
        payload = torch.load(
            args.output / "final.pt", map_location=args.device, weights_only=True
        )
        sae.load_state_dict(payload["state_dict"])
        baseline = copy.deepcopy(sae)
        baseline.load_state_dict(payload["baseline_state_dict"])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(
            args.output / "protocol.json",
            {
                "identity": run.identity,
                "warm_start": sha256(args.sae),
                "seed": run.seed,
                "cache_sha256": artifact["cache_sha256"],
                "groups_target": args.groups,
                "minutes_target": args.minutes,
                "minutes_hard_cap": args.max_minutes,
                "split": "train/val only",
            },
        )
        calibration = prepare(run, sae, args.output)
        payload, sae, baseline, start, hard_end = fit(
            run, sae, calibration, args.output
        )
        transient_diagnostic(
            run, sae, payload, args.output, min(hard_end - 30, time.monotonic() + 600)
        )
        write_json(
            args.output / "timing.json",
            {
                "preparation_seconds": calibration["seconds"],
                "training_validation_diagnostic_seconds": time.monotonic() - start,
                "hard_cap_seconds": args.max_minutes * 60,
            },
        )
    audition(run, sae, baseline, payload, args.output)
    if sha256(args.checkpoint) != run.identity["checkpoint"]:
        raise RuntimeError("Reconstruction checkpoint changed during the run.")
    print("COMPLETE", str(args.output), flush=True)


if __name__ == "__main__":
    main()
