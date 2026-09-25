"""Serve a live modal review page for a completed FAST extraction.

The page chooses a random preprocessed sample, recomputes the current legacy
and full NEW analyzers, and loads the saved FAST feature tensor. Run this on
the machine that holds the complete extraction directory.
"""

from __future__ import annotations

import argparse
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
from drumblender.utils.modal_analysis_new import CQTModalAnalysis
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

    figure = plt.figure(figsize=(12, 3), dpi=110)
    axis = figure.add_axes((0.09, 0.18, 0.87, 0.68))
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

    figure = plt.figure(figsize=(12, 3), dpi=110)
    axis = figure.add_axes((0.09, 0.18, 0.87, 0.68))
    axis.set_xlim(0, duration)
    axis.set_ylim(20, sample_rate / 2)
    if scale == "mel":
        axis.set_yscale("function", functions=(_hz_to_mel, _mel_to_hz))
    for frequencies, observed, times in tracks:
        axis.plot(
            times, np.where(observed, frequencies, np.nan), color=color,
            linewidth=0.55, alpha=0.85,
        )
    axis.set_axis_off()
    axis.patch.set_alpha(0)
    figure.patch.set_alpha(0)
    figure.savefig(path, dpi=110, transparent=True)
    plt.close(figure)


def _new_tracks(result):
    return [
        (frequencies, observed, result.frame_times)
        for frequencies, observed in zip(result.frequencies, result.observed)
    ]


def _fast_tracks(feature, sample_count, sample_rate):
    frames = feature.shape[-1]
    times = (np.arange(frames) + 0.5) * sample_count / (frames * sample_rate)
    frequencies = feature[0].numpy() * sample_rate / (2 * np.pi)
    observed = feature[1].numpy() != 0
    return [(f, a, times) for f, a in zip(frequencies, observed)]


def _load_fast(feature_path, sample_count):
    """Render the stored tensor without rerunning FAST analysis."""
    try:
        feature = torch.load(feature_path, map_location="cpu", weights_only=True)
    except TypeError:
        feature = torch.load(feature_path, map_location="cpu")
    if feature.ndim != 3 or feature.shape[0] != 3:
        raise ValueError(f"Expected FAST feature [3, modes, frames]: {feature_path}")
    feature = feature.float()
    with torch.no_grad():
        if feature.shape[1] == 0:
            wave = np.zeros(sample_count, dtype=np.float32)
        else:
            wave = modal_synth(
                feature[0][None], feature[1][None], sample_count,
                feature[2][None],
            )[0].numpy()
    return wave, feature


def render_sample(sample_id, item, root, output, config, device):
    """Create audio and plots for one dataset item and return UI metadata."""
    source = _safe_asset(root, item["filename"])
    fast_path = _safe_asset(root, item["feature_file"])
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono preprocessed WAV: {source}")
    audio = audio[:, 0]
    if sample_rate != config["sample_rate"]:
        raise ValueError(f"Sample rate differs from FAST configuration: {source}")

    # Legacy means the current ampzero-aware modal_analysis.py, not _OLD.py.
    legacy, legacy_info = legacy_render(audio, sample_rate, config["num_modes"])
    analyzer = CQTModalAnalysis(
        sample_rate,
        hop_length=256,
        fmin=config["fmin"],
        n_bins=config["n_bins"],
        bins_per_octave=config["bins_per_octave"],
        min_length=config["min_length"],
        num_modes=config["num_modes"],
        threshold=config["threshold_db"],
        diff_threshold=config["diff_threshold"],
        backend=config["modal_backend"],
        cqt_max_frequency=config["cqt_max_frequency"],
        short_window=config["short_window"],
        long_window=config["long_window"],
        relative_threshold_db=config["relative_threshold_db"],
        min_relative_score_db=config["min_relative_score_db"],
        min_prominence_db=config["min_prominence_db"],
        max_gap=2,
        max_deviation_hz=config["max_deviation_hz"],
        refine=True,
        fast=False,
        compute_device=device,
    )
    new_result = analyzer.analyze(audio)
    # Use the same production synthesizer for all three modal comparisons.
    with torch.no_grad():
        if len(new_result.frequencies) == 0:
            new_wave = np.zeros(len(audio), dtype=np.float32)
        else:
            frequency, amplitude, phase = (
                torch.from_numpy(value)[None].float()
                for value in new_result.parameters()
            )
            new_wave = modal_synth(
                frequency * (2 * torch.pi / sample_rate),
                amplitude,
                len(audio),
                phase,
            )[0].numpy()
    fast_wave, fast_feature = _load_fast(fast_path, len(audio))

    # Publish a complete generation atomically at the directory level. Keeping
    # older generations allows a browser tab to finish loading during refresh.
    generation = f"{sample_id}_{random.getrandbits(48):012x}"
    folder = output / generation
    folder.mkdir(parents=True)
    waves = {
        "original": audio,
        "legacy": legacy,
        "new": new_wave,
        "fast": fast_wave,
    }
    for name, wave in waves.items():
        sf.write(folder / f"{name}.wav", wave, sample_rate, subtype="FLOAT")
        for scale in ("mel", "linear"):
            _plot_spectrogram(
                folder / f"{name}_{scale}.png", wave, audio, sample_rate,
                scale, f"{name.upper()} · {scale} frequency axis",
            )
    for scale in ("mel", "linear"):
        _plot_observed_layer(
            folder / f"observed_new_{scale}.png", _new_tracks(new_result),
            len(audio) / sample_rate, sample_rate, scale, "cyan",
        )
        _plot_observed_layer(
            folder / f"observed_fast_{scale}.png",
            _fast_tracks(fast_feature, len(audio), sample_rate),
            len(audio) / sample_rate, sample_rate, scale, "lime",
        )

    return {
        "id": sample_id,
        "name": item["orig_relpath"],
        "pack": item["sample_pack_key"],
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "fast_modes": int((fast_feature[1] != 0).any(dim=-1).sum()),
        "fast_lf_modes": item.get("modal_lf_modes"),
        "fast_hf_modes": item.get("modal_hf_modes"),
        "legacy_modes": legacy_info["modes"],
        "new_modes": len(new_result.frequencies),
        "legacy_seconds": round(legacy_info["analysis_seconds"], 3),
        "new_seconds": round(new_result.analysis_seconds, 3),
        "fast_saved_seconds": item.get("analysis_seconds"),
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
                "packs": sorted({x["sample_pack_key"] for x in server.items.values()}),
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
                    key for key, item in server.items.items()
                    if key != exclude and (not pack or item["sample_pack_key"] == pack)
                ]
                random.shuffle(candidates)
            if not candidates:
                self._json({"error": "No matching sample."}, 404)
                return
            with server.analysis_lock:
                for sample_id in candidates:
                    try:
                        report = render_sample(
                            sample_id, server.items[sample_id], server.root,
                            server.output, server.config, server.device,
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
        "--fast_root", type=Path,
        default=Path("../dataset/modal_features/processed_modal_fast128"),
    )
    parser.add_argument("--output", type=Path, default=Path("analysis/modal_dataset_review"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--compute_device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    root = args.fast_root.resolve()
    config = json.loads((root / "modal_config.json").read_text(encoding="utf-8"))
    items = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not config.get("fast"):
        parser.error("--fast_root must contain a FAST extraction")
    args.output.mkdir(parents=True, exist_ok=True)
    page = args.output / "index.html"
    shutil.copyfile(Path(__file__).with_name("modal_dataset_review.html"), page)
    server = HTTPServer((args.host, args.port), ReviewHandler)
    server.root = root
    server.output = args.output.resolve()
    server.page = page
    server.items = items
    server.config = config
    server.device = args.compute_device
    server.analysis_lock = threading.Lock()
    print(f"Review {len(items)} samples at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
