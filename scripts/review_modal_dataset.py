"""Review saved NEW and FAST modal features without rerunning extraction.

Original audio and both feature tensors are loaded from completed extraction
directories. Legacy analysis runs only when explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from drumblender.synths.modal import modal_synth
from scripts.inspect_modal_analysis import legacy_render


def _safe_asset(root: Path, relative: str) -> Path:
    """Resolve only paths inside the extraction directory."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset outside extraction directory: {relative}")
    return path


def _hz_to_mel(hz):
    return 2595 * np.log10(1 + np.asarray(hz) / 700)


def _mel_to_hz(mel):
    return 700 * (10 ** (np.asarray(mel) / 2595) - 1)


def _plot_spectrogram(path, wave, original, sample_rate, scale, title):
    """Use a fixed plotting rectangle so transparent track layers align."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = 8192 if scale == "mel" else 2048
    nperseg = min(window, len(wave))
    hop = min(nperseg, max(1, nperseg // 4, len(wave) // 900))
    kwargs = dict(
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg - hop,
        boundary="zeros",
    )
    frequencies, times, spectrum = signal.stft(wave, **kwargs)
    _, _, reference_spectrum = signal.stft(original, **kwargs)
    reference = max(float(np.abs(reference_spectrum).max()), 1e-12)
    db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12) / reference)

    figure = plt.figure(figsize=(16, 5), dpi=110)
    axis = figure.add_axes((0.07, 0.13, 0.91, 0.75))
    axis.pcolormesh(
        times, frequencies, db, shading="auto", cmap="magma", vmin=-80, vmax=0,
        rasterized=True,
    )
    axis.set_xlim(0, len(wave) / sample_rate)
    axis.set_ylim(20, sample_rate / 2)
    if scale == "mel":
        axis.set_yscale("function", functions=(_hz_to_mel, _mel_to_hz))
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(title, loc="left")
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _plot_observed_layer(path, tracks, duration, sample_rate, scale, color):
    """Draw only consecutive observed points; missing frames never join lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 5), dpi=110)
    axis = figure.add_axes((0.07, 0.13, 0.91, 0.75))
    axis.set_xlim(0, duration)
    axis.set_ylim(20, sample_rate / 2)
    if scale == "mel":
        axis.set_yscale("function", functions=(_hz_to_mel, _mel_to_hz))
    for frequencies, observed, times in tracks:
        axis.plot(
            times, np.where(observed, frequencies, np.nan), color=color,
            linewidth=0.65, alpha=0.8,
        )
    axis.set_axis_off()
    axis.patch.set_alpha(0)
    figure.patch.set_alpha(0)
    figure.savefig(path, dpi=110, transparent=True)
    plt.close(figure)


def _stored_tracks(feature, sample_count, sample_rate):
    """A saved tensor retains observed amplitudes, but not each track's source."""
    frames = feature.shape[-1]
    times = (np.arange(frames) + 0.5) * sample_count / (frames * sample_rate)
    frequencies = feature[0].numpy() * sample_rate / (2 * np.pi)
    observed = feature[1].numpy() != 0
    return [(f, a, times) for f, a in zip(frequencies, observed)]


def _load_stored(feature_path, sample_count, device):
    """Render a saved feature tensor with the production modal synthesizer."""
    try:
        feature = torch.load(feature_path, map_location="cpu", weights_only=True)
    except TypeError:
        feature = torch.load(feature_path, map_location="cpu")
    if feature.ndim != 3 or feature.shape[0] != 3:
        raise ValueError(f"Expected saved feature [3, modes, frames]: {feature_path}")
    feature = feature.float()
    if not torch.isfinite(feature).all():
        raise ValueError(f"Nonfinite modal feature: {feature_path}")
    with torch.no_grad():
        if feature.shape[1] == 0:
            wave = np.zeros(sample_count, dtype=np.float32)
        else:
            wave = modal_synth(
                feature[0][None].to(device), feature[1][None].to(device),
                sample_count, feature[2][None].to(device),
            )[0].cpu().numpy()
    return wave, feature


def _read_extraction(root: Path, expected_fast: bool):
    """Read only metadata at startup; load per-sample tensors on demand."""
    root = root.resolve()
    config_path = root / "modal_config.json"
    metadata_path = root / "metadata.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected modal_config.json and metadata.json under {root}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if bool(config.get("fast")) != expected_fast:
        kind = "FAST" if expected_fast else "full NEW"
        raise ValueError(f"{root} is not a completed {kind} extraction")
    items = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats_path = root / "run_stats.json"
    run = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    return root, config, items, run


def _summary(items, run, mode_ceiling):
    """Summarize the actual metadata, with optional measured wall time."""
    values = list(items.values())
    modes = sorted(int(item.get("active_modes", 0)) for item in values)
    count = len(modes)
    source_names = ("cqt", "long_stft", "short_stft", "long_stft_short_envelope")
    sources = {
        name: sum(item.get("modal_source_modes", {}).get(name, 0) for item in values)
        for name in source_names
    }
    return {
        "files": count,
        "mode_ceiling": mode_ceiling,
        "mean_modes": round(sum(modes) / count, 2) if count else None,
        "median_modes": modes[count // 2] if count else None,
        "capped_files": sum(mode == mode_ceiling for mode in modes),
        "lf_modes": sum(item.get("modal_lf_modes", 0) for item in values),
        "hf_modes": sum(item.get("modal_hf_modes", 0) for item in values),
        "source_modes": sources,
        "wall_seconds": run.get("wall_seconds"),
    }


def render_sample(sample_id, new_item, fast_item, new_root, fast_root,
                  output, new_config, fast_config, render_device,
                  include_legacy=False):
    """Build one comparison from saved tensors; legacy runs only by opt-in."""
    source = _safe_asset(new_root, new_item["filename"])
    fast_source = _safe_asset(fast_root, fast_item["filename"])
    new_path = _safe_asset(new_root, new_item["feature_file"])
    fast_path = _safe_asset(fast_root, fast_item["feature_file"])
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono preprocessed WAV: {source}")
    audio = audio[:, 0]
    if len(audio) == 0:
        raise ValueError(f"Empty preprocessed WAV: {source}")
    fast_audio, fast_rate = sf.read(fast_source, dtype="float32", always_2d=True)
    if (fast_rate != sample_rate or fast_audio.shape != (len(audio), 1)
            or not np.allclose(fast_audio[:, 0], audio, rtol=0, atol=1e-6)):
        raise ValueError(f"NEW/FAST source WAV differs for {sample_id}; comparison would be invalid")
    if sample_rate != new_config["sample_rate"] or sample_rate != fast_config["sample_rate"]:
        raise ValueError(f"Sample rate differs between audio and extraction configs: {source}")
    if new_item.get("orig_relpath") != fast_item.get("orig_relpath"):
        raise ValueError(f"NEW/FAST sample identity mismatch for {sample_id}")
    if int(new_item.get("num_samples", len(audio))) != len(audio):
        raise ValueError(f"NEW metadata/audio length mismatch for {sample_id}")
    if int(fast_item.get("num_samples", len(audio))) != len(audio):
        raise ValueError(f"FAST metadata/audio length mismatch for {sample_id}")

    new_wave, new_feature = _load_stored(new_path, len(audio), render_device)
    fast_wave, fast_feature = _load_stored(fast_path, len(audio), render_device)
    for label, feature, item in (
        ("NEW", new_feature, new_item), ("FAST", fast_feature, fast_item)
    ):
        stored_count = int((feature[1] != 0).any(dim=-1).sum())
        if item.get("active_modes") is not None and stored_count != item["active_modes"]:
            raise ValueError(f"{label} active-mode metadata differs from {sample_id}.pt")
    # Legacy is the current ampzero-aware modal_analysis.py, never _OLD.py.
    legacy = legacy_info = None
    if include_legacy:
        legacy, legacy_info = legacy_render(audio, sample_rate, new_config["num_modes"])

    # Publish a complete generation atomically at the directory level. Keeping
    # older generations allows a browser tab to finish loading during refresh.
    safe_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    generation = f"{safe_id}_{random.getrandbits(48):012x}"
    folder = output / generation
    folder.mkdir(parents=True)
    waves = {
        "original": audio,
        "new": new_wave,
        "fast": fast_wave,
    }
    if legacy is not None:
        waves["legacy"] = legacy
    for name, wave in waves.items():
        sf.write(folder / f"{name}.wav", wave, sample_rate, subtype="FLOAT")
        for scale in ("mel", "linear"):
            _plot_spectrogram(
                folder / f"{name}_{scale}.png", wave, audio, sample_rate,
                scale, f"{name.upper()} · {scale} frequency axis",
            )
    for scale in ("mel", "linear"):
        _plot_observed_layer(
            folder / f"observed_new_{scale}.png",
            _stored_tracks(new_feature, len(audio), sample_rate),
            len(audio) / sample_rate, sample_rate, scale, "cyan",
        )
        _plot_observed_layer(
            folder / f"observed_fast_{scale}.png",
            _stored_tracks(fast_feature, len(audio), sample_rate),
            len(audio) / sample_rate, sample_rate, scale, "lime",
        )

    return {
        "id": sample_id,
        "name": new_item["orig_relpath"],
        "pack": new_item["sample_pack_key"],
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "lf_hf_boundary_hz": new_config["cqt_max_frequency"],
        "fast_modes": int((fast_feature[1] != 0).any(dim=-1).sum()),
        "fast_lf_modes": fast_item.get("modal_lf_modes"),
        "fast_hf_modes": fast_item.get("modal_hf_modes"),
        "legacy_modes": legacy_info["modes"] if legacy_info else None,
        "new_modes": int((new_feature[1] != 0).any(dim=-1).sum()),
        "new_lf_modes": new_item.get("modal_lf_modes"),
        "new_hf_modes": new_item.get("modal_hf_modes"),
        "new_sources": new_item.get("modal_source_modes", {}),
        "fast_sources": fast_item.get("modal_source_modes", {}),
        "legacy_seconds": round(legacy_info["analysis_seconds"], 3) if legacy_info else None,
        "new_saved_seconds": new_item.get("analysis_seconds"),
        "fast_saved_seconds": fast_item.get("analysis_seconds"),
        "new_candidates": new_item.get("modal_candidates_pre_limit"),
        "fast_candidates": fast_item.get("modal_candidates_pre_limit"),
        "new_limit_rejected": new_item.get("modal_limit_rejected"),
        "fast_limit_rejected": fast_item.get("modal_limit_rejected"),
        "new_observed_frames": new_item.get("modal_observed_frames"),
        "fast_observed_frames": fast_item.get("modal_observed_frames"),
        "new_mean_confidence": new_item.get("modal_mean_confidence"),
        "fast_mean_confidence": fast_item.get("modal_mean_confidence"),
        "legacy_included": include_legacy,
        "media": f"/media/{generation}/",
    }


class ReviewHandler(BaseHTTPRequestHandler):
    def _json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, mime):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        server = self.server
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            self._file(server.page, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/info":
            self._json({
                "count": len(server.items),
                "packs": sorted({x[0]["sample_pack_key"] for x in server.items.values()}),
                "new_summary": server.new_summary,
                "fast_summary": server.fast_summary,
                "new_boundary_hz": server.new_config["cqt_max_frequency"],
                "fast_boundary_hz": server.fast_config["cqt_max_frequency"],
            })
            return
        if parsed.path in ("/api/random", "/api/sample"):
            query = parse_qs(parsed.query)
            if parsed.path == "/api/sample":
                sample_id = query.get("id", [""])[0]
                candidates = [sample_id] if sample_id in server.items else []
            else:
                pack = query.get("pack", [""])[0]
                exclude = query.get("exclude", [""])[0]
                candidates = [
                    key for key, (item, _) in server.items.items()
                    if key != exclude and (not pack or item["sample_pack_key"] == pack)
                ]
                random.shuffle(candidates)
            if not candidates:
                self._json({"error": "No matching sample."}, 404)
                return
            include_legacy = query.get("legacy", ["0"])[0] == "1"
            with server.analysis_lock:
                for sample_id in candidates:
                    try:
                        new_item, fast_item = server.items[sample_id]
                        report = render_sample(
                            sample_id, new_item, fast_item,
                            server.new_root, server.fast_root,
                            server.output, server.new_config, server.fast_config,
                            server.render_device, include_legacy,
                        )
                        self._json(report)
                        return
                    except Exception as error:
                        if parsed.path == "/api/sample":
                            self._json({"error": str(error)}, 500)
                            return
                        print(f"Skipping {sample_id}: {error}", flush=True)
            self._json({"error": "No usable sample in this selection."}, 500)
            return
        if parsed.path.startswith("/media/"):
            relative = parsed.path.removeprefix("/media/")
            path = _safe_asset(server.output, relative)
            if path.is_file() and path.suffix in (".png", ".wav"):
                mime = "image/png" if path.suffix == ".png" else "audio/wav"
                self._file(path, mime)
                return
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new_root", type=Path,
        default=Path("../dataset/modal_features/processed_modal_new128"),
    )
    parser.add_argument(
        "--fast_root", type=Path,
        default=Path("../dataset/modal_features/processed_modal_fast128"),
    )
    parser.add_argument("--output", type=Path, default=Path("analysis/modal_dataset_review"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--render_device", "--compute_device", dest="render_device",
        choices=("auto", "cpu", "cuda"), default="auto",
        help="Device for saved-feature synthesis; no spectral analysis is run by default.",
    )
    args = parser.parse_args()
    try:
        new_root, new_config, new_items, new_run = _read_extraction(args.new_root, False)
        fast_root, fast_config, fast_items, fast_run = _read_extraction(args.fast_root, True)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if new_config["sample_rate"] != fast_config["sample_rate"]:
        parser.error("NEW and FAST sample rates differ.")
    if new_config["cqt_max_frequency"] != fast_config["cqt_max_frequency"]:
        parser.error("NEW and FAST LF/HF boundaries differ; counts are not comparable.")
    fast_by_path = {
        item.get("orig_relpath"): item for item in fast_items.values()
        if item.get("orig_relpath")
    }
    items = {
        key: (item, fast_by_path[item["orig_relpath"]])
        for key, item in new_items.items()
        if item.get("orig_relpath") in fast_by_path
    }
    if not items:
        parser.error("No matching samples by orig_relpath between NEW and FAST.")
    if args.render_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA rendering requested, but PyTorch CUDA is unavailable.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.render_device == "auto" else args.render_device
    args.output.mkdir(parents=True, exist_ok=True)
    page = args.output / "index.html"
    shutil.copyfile(Path(__file__).with_name("modal_dataset_review.html"), page)
    server = HTTPServer((args.host, args.port), ReviewHandler)
    server.new_root = new_root
    server.fast_root = fast_root
    server.output = args.output.resolve()
    server.page = page
    server.items = items
    server.new_config = new_config
    server.fast_config = fast_config
    server.new_summary = _summary(new_items, new_run, int(new_config["num_modes"]))
    server.fast_summary = _summary(fast_items, fast_run, int(fast_config["num_modes"]))
    server.render_device = device
    server.analysis_lock = threading.Lock()
    print(
        f"Review {len(items)} paired samples at http://{args.host}:{args.port}/ "
        f"(saved NEW + FAST, render on {device}; legacy only on request)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
