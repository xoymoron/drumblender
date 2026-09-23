#!/usr/bin/env python3
"""Compile reconstruction evaluation from one exported bundle or a run directory.

The main table reports MR-STFT, LSD, and temporal SF. Per-file and per-pack
statistics, supplementary metrics, and distribution plots are saved alongside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import html
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from drumblender.metrics import (
    audio_pair_fingerprint, evaluation_score_keys,
    load_evaluation_metrics, score_reconstruction,
)

try:
    import torchaudio  # type: ignore
except Exception:
    torchaudio = None

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None


TOP_K = 3
RANK_METRICS = [
    ("MR-STFT", "test/mr_stft"),
    ("LSD", "test/lsd"),
    ("SF", "test/flux_onset"),
]
SUMMARY_HEADERS = [
    "bundle",
    "pack",
    "items",
    "mr_stft_mean",
    "mr_stft_std",
    "lsd_mean",
    "lsd_std",
    "sf_mean",
    "sf_std",
]


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    xs = [x for x in values if _is_finite(x)]
    if len(xs) == 0:
        return float("nan"), float("nan")
    mean = float(statistics.fmean(xs))
    std = float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0
    return mean, std


def _fmt_float(value: float, digits: int = 6) -> str:
    if not _is_finite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _escape_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", r"\|")


def _markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    header_line = "| " + " | ".join(_escape_cell(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _bundle_dirs(root: Path) -> List[Path]:
    direct_summary = root / "evaluation" / "summary.json"
    if direct_summary.is_file():
        return [root]

    bundles = sorted(
        {
            summary_path.parents[1]
            for summary_path in root.rglob("summary.json")
            if summary_path.is_file() and summary_path.parent.name == "evaluation"
        },
        key=lambda p: p.as_posix(),
    )
    if len(bundles) == 0:
        raise RuntimeError(f"No evaluation bundles found under: {root}")
    return bundles


def _bundle_relpath(root: Path, bundle_dir: Path) -> str:
    try:
        rel = bundle_dir.relative_to(root)
        return bundle_dir.name if str(rel) == "." else rel.as_posix()
    except Exception:
        return bundle_dir.name


def _safe_report_name(relpath: str) -> str:
    if relpath in ("", "."):
        return "root"
    return relpath.replace("/", "__").replace("\\", "__")


def _load_audio_mono(path: Path) -> Tuple[torch.Tensor, int]:
    if torchaudio is not None:
        try:
            waveform, sr = torchaudio.load(str(path))
            return waveform[:1, :].to(torch.float32), int(sr)
        except Exception:
            pass

    if sf is not None:
        try:
            data, sr = sf.read(str(path), always_2d=True)
            mono = torch.from_numpy(data[:, :1].T).to(torch.float32)
            return mono, int(sr)
        except Exception:
            pass

    raise RuntimeError(f"Failed to read audio: {path}")


def _sample_name(row: Dict[str, object]) -> str:
    for key in ("source_filename", "orig_relpath", "filename", "meta_key", "index"):
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return str(row[key])
    return f"row_{row.get('__row_idx__', 'unknown')}"


def _metric_config(bundle_dir: Path):
    for name in ("evaluation_metrics.yaml", "drumblender_metrics.yaml"):
        path = bundle_dir / "configs" / name
        if path.is_file():
            return path
    return None


def _compute_metric_rows(bundle_dir, loss_rows, metrics=None, protocol=None):
    if metrics is None:
        metrics, protocol = load_evaluation_metrics(_metric_config(bundle_dir))
    metric_rows = []
    for i, source in enumerate(loss_rows):
        src_rel = str(source.get("source_filename", "")).replace("\\", "/")
        fingerprint = audio_pair_fingerprint(bundle_dir, src_rel)
        recon_path = bundle_dir / "recon" / src_rel
        target_path = bundle_dir / "target" / src_rel
        if not recon_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"Cannot rescore {src_rel}: export both recon and target (--save-target).")
        recon, sr_r = _load_audio_mono(recon_path)
        target, sr_t = _load_audio_mono(target_path)
        if sr_r != sr_t or recon.shape != target.shape:
            raise ValueError(f"Mismatched sample rates or shapes: {src_rel}")
        length = int(source.get("length") or target.shape[-1])
        if not 0 < length <= target.shape[-1]:
            raise ValueError(f"Invalid manifest length: {src_rel}")
        # A legacy bundle may contain right padding. Exclude it identically to export.
        scores = score_reconstruction(metrics, recon[None, :, :length],
                                      target[None, :, :length], sample_rate=sr_t)
        row = dict(source)
        row.update(scores)
        row.update({"index": source.get("index", i), "sample_pack_key": _row_pack(source),
                    "test/objective": source.get("test/loss", ""),
                    "metric_version": protocol["version"],
                    "metric_config_sha256": protocol["config_sha256"],
                    "audio_sha256": fingerprint})
        metric_rows.append(row)
    if not metric_rows:
        raise ValueError(f"Empty evaluation: {bundle_dir}")
    _write_csv(bundle_dir / "evaluation" / "per_file_metrics.csv", metric_rows, list(metric_rows[0]))
    return metric_rows


def _row_pack(row):
    source = str(row.get("source_filename", "")).replace("\\", "/")
    if "/" in source:
        return source.split("/", 1)[0]
    return str(row.get("sample_pack_key") or "__root__").replace("\\", "/").split("/", 1)[0]


def _load_or_compute_metric_rows(bundle_dir: Path, recompute=False):
    loss_csv = bundle_dir / "evaluation" / "per_file_loss.csv"
    if not loss_csv.is_file():
        raise FileNotFoundError(f"Missing per_file_loss.csv: {loss_csv}")
    loss_rows = _read_csv(loss_csv)
    metrics, protocol = load_evaluation_metrics(_metric_config(bundle_dir))
    cache = bundle_dir / "evaluation" / "per_file_metrics.csv"
    if cache.is_file() and not recompute:
        rows = _read_csv(cache)
        required = evaluation_score_keys(metrics)
        valid = bool(rows) and len(rows) == len(loss_rows)
        for row, source in zip(rows, loss_rows):
            valid = valid and all(str(row.get(k)) == str(source.get(k))
                                  for k in ("meta_key", "source_filename", "length"))
            valid = valid and row.get("metric_version") == protocol["version"]
            valid = valid and row.get("metric_config_sha256") == protocol["config_sha256"]
            valid = valid and all(_is_finite(_to_float(row.get(k))) for k in required)
            valid = valid and row.get("audio_sha256") == audio_pair_fingerprint(bundle_dir, source["source_filename"])
            if not valid:
                break
        if valid:
            return rows
    return _compute_metric_rows(bundle_dir, loss_rows, metrics, protocol)


def _pack_label(summary: Dict[str, Any]) -> str:
    keys = summary.get("sample_pack_keys")
    if isinstance(keys, list) and len(keys) > 0:
        return ",".join(str(k) for k in keys)
    return "all"


def _summary_row(
    root: Path,
    bundle_dir: Path,
    summary: Dict[str, Any],
    rows: List[Dict[str, object]],
    pack_override: Optional[str] = None,
) -> Dict[str, object]:
    mr_stft_mean, mr_stft_std = _mean_std(_to_float(r.get("test/mr_stft")) for r in rows)
    lsd_mean, lsd_std = _mean_std(_to_float(r.get("test/lsd")) for r in rows)
    sf_mean, sf_std = _mean_std(_to_float(r.get("test/flux_onset")) for r in rows)
    return {
        "bundle": _bundle_relpath(root, bundle_dir),
        "pack": pack_override if pack_override is not None else _pack_label(summary),
        "items": len(rows),
        "mr_stft_mean": mr_stft_mean,
        "mr_stft_std": mr_stft_std,
        "lsd_mean": lsd_mean,
        "lsd_std": lsd_std,
        "sf_mean": sf_mean,
        "sf_std": sf_std,
    }


def _summary_markdown(summary_rows: List[Dict[str, object]]) -> str:
    rows = [
        [
            row["bundle"],
            row["pack"],
            row["items"],
            _fmt_float(_to_float(row["mr_stft_mean"])),
            _fmt_float(_to_float(row["mr_stft_std"])),
            _fmt_float(_to_float(row["lsd_mean"])),
            _fmt_float(_to_float(row["lsd_std"])),
            _fmt_float(_to_float(row["sf_mean"])),
            _fmt_float(_to_float(row["sf_std"])),
        ]
        for row in summary_rows
    ]
    return _markdown_table(SUMMARY_HEADERS, rows)


def _summary_latex(summary_rows: List[Dict[str, object]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Test-set reconstruction errors by top-level sample pack. Values are per-file mean $\pm$ standard deviation; lower is better. The all row is sample-weighted.}",
        r"\label{tab:reconstruction-evaluation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrccc}",
        r"\hline",
        r"Run & Pack & $N$ & MR-STFT $\downarrow$ & LSD $\downarrow$ & SF $\downarrow$ \\",
        r"\hline",
    ]
    previous_bundle = None
    for row in summary_rows:
        if previous_bundle is not None and row["bundle"] != previous_bundle:
            lines.append(r"\hline")
        bundle = _latex_escape(row["bundle"]) if row["bundle"] != previous_bundle else ""
        pack = _latex_escape(row["pack"])
        n_items = _latex_escape(row["items"])
        def cell(prefix):
            mean = _fmt_float(_to_float(row[f"{prefix}_mean"]), 3)
            std = _fmt_float(_to_float(row[f"{prefix}_std"]), 3)
            return f"${mean} \\pm {std}$"
        lines.append(
            f"{bundle} & {pack} & {n_items} & {cell('mr_stft')} & "
            f"{cell('lsd')} & {cell('sf')} \\\\"
        )
        previous_bundle = row["bundle"]
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _extreme_table_rows(selected: List[Tuple[Dict[str, object], float]]) -> List[List[object]]:
    rows: List[List[object]] = []
    for rank, (row, _) in enumerate(selected, start=1):
        rows.append(
            [
                rank,
                _sample_name(row),
                _fmt_float(_to_float(row.get("test/mr_stft"))),
                _fmt_float(_to_float(row.get("test/lsd"))),
                _fmt_float(_to_float(row.get("test/flux_onset"))),
            ]
        )
    return rows


def _write_bundle_extremes(
    report_dir: Path,
    root: Path,
    bundle_dir: Path,
    summary: Dict[str, Any],
    rows: List[Dict[str, object]],
) -> str:
    relpath = _bundle_relpath(root, bundle_dir)
    safe_name = _safe_report_name(relpath)
    out_path = report_dir / f"extremes_{safe_name}.txt"

    lines = [
        f"bundle: {relpath}",
        f"pack: {_pack_label(summary)}",
        f"items: {len(rows)}",
        "",
    ]

    headers = ["rank", "sample", "MR-STFT", "LSD", "SF"]
    for label, column in RANK_METRICS:
        values: List[Tuple[Dict[str, object], float]] = []
        for row in rows:
            value = _to_float(row.get(column))
            if _is_finite(value):
                values.append((row, value))

        lines.append(f"## {label} worst {TOP_K}")
        if len(values) == 0:
            lines.append("No finite values available.")
            lines.append("")
            continue

        descending = sorted(values, key=lambda item: item[1], reverse=True)[:TOP_K]
        lines.append(_markdown_table(headers, _extreme_table_rows(descending)))
        lines.append("")

        lines.append(f"## {label} best {TOP_K}")
        ascending = sorted(values, key=lambda item: item[1])[:TOP_K]
        lines.append(_markdown_table(headers, _extreme_table_rows(ascending)))
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path.name


def _percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _group_rows(rows):
    groups = {"all samples": rows}
    for pack in sorted({_row_pack(row) for row in rows}):
        groups[f"pack: {pack}"] = [row for row in rows if _row_pack(row) == pack]
    return groups


def _distribution_stats(bundle, rows):
    result = []
    primary = {column for _, column in RANK_METRICS}
    extra = sorted(key for key in rows[0]
                   if key.startswith("test/") and key not in primary
                   and key not in ("test/loss", "test/objective"))
    reported_metrics = RANK_METRICS + [(key.removeprefix("test/"), key) for key in extra]
    for group, examples in _group_rows(rows).items():
        for label, column in reported_metrics:
            values = [_to_float(row[column]) for row in examples]
            if not values or not all(map(math.isfinite, values)):
                raise ValueError(f"Invalid {label} scores in {bundle}/{group}")
            mean, std = _mean_std(values)
            result.append({"bundle": bundle, "group": group, "metric": label,
                           "n": len(values), "mean": mean, "std": std,
                           "sem": std / math.sqrt(len(values)),
                           "p05": _percentile(values, .05),
                           "p25": _percentile(values, .25),
                           "median": _percentile(values, .5),
                           "p75": _percentile(values, .75),
                           "p95": _percentile(values, .95),
                           "p99": _percentile(values, .99),
                           "min": min(values), "max": max(values)})
    return result


def _write_dashboard(path, bundles):
    """Standalone SVG/HTML distribution plots, with no plotting dependency."""
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<title>Reconstruction evaluation</title><style>',
             'body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#172337}',
             'svg{width:100%;height:auto;background:#f5f7fb;border-radius:12px;margin-bottom:24px}',
             'text{font:12px system-ui;fill:#172337}h2{margin-top:48px}</style>',
             '<h1>Reconstruction evaluation</h1><p>Lower is better. Each file has equal weight. ',
             'Boxes: P25–P75; line: median; whiskers: P5–P95; dot: mean. ',
             'SF uses a log(1 + value) axis. Full-range extrema are in distribution_stats.csv. ',
             'Pack panels are subsets of the same evaluation, not independent repetitions.</p>']
    for bundle, rows in bundles:
        parts.append(f'<h2>{html.escape(bundle)}</h2>')
        groups = _group_rows(rows)
        primary = {column for _, column in RANK_METRICS}
        secondary = sorted(key for key in rows[0]
                           if key.startswith("test/") and key not in primary
                           and key not in ("test/loss", "test/objective"))
        plots = RANK_METRICS + [(key.removeprefix("test/").replace("_", " ").title(), key)
                                for key in secondary]
        for label, column in plots:
            is_secondary = column not in primary
            if is_secondary:
                parts.append(f'<details><summary>{html.escape(label)}</summary>')
            arrays = [[float(row[column]) for row in examples] for examples in groups.values()]
            transform = math.log1p if label == "SF" else float
            upper = max(transform(max(values)) for values in arrays) or 1.0
            scale = lambda value: 260 + 600 * transform(value) / upper
            height = 60 + len(groups) * 44
            parts.append(f'<h3>{label}</h3><svg role="img" aria-label="{label} distributions" viewBox="0 0 920 {height}">')
            for tick in range(5):
                value = upper * tick / 4
                raw = math.expm1(value) if label == "SF" else value
                x = 260 + 150 * tick
                parts.append(f'<path d="M{x} 28 V{height-24}" stroke="#dce2eb"/><text x="{x}" y="18" text-anchor="middle">{raw:.3g}</text>')
            for i, ((group, examples), values) in enumerate(zip(groups.items(), arrays)):
                y = 48 + i * 44
                p5, p25, p50, p75, p95 = [scale(_percentile(values, q)) for q in (.05, .25, .5, .75, .95)]
                mean = scale(statistics.fmean(values))
                parts.append(f'<text x="12" y="{y+4}">{html.escape(group)} (n={len(examples)})</text>')
                parts.append(f'<path d="M{p5} {y} H{p95} M{p5} {y-6} V{y+6} M{p95} {y-6} V{y+6}" stroke="#426b9a"/>')
                parts.append(f'<rect x="{p25}" y="{y-10}" width="{max(.5,p75-p25)}" height="20" fill="#b5d4f3" stroke="#426b9a"/>')
                parts.append(f'<path d="M{p50} {y-10} V{y+10}" stroke="#172337"/><circle cx="{mean}" cy="{y}" r="3" fill="#b24728"/>')
            parts.append('</svg>')
            if is_secondary:
                parts.append('</details>')
    parts.append('</html>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=str, help="Run directory or one exported bundle directory")
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for generated reports. Defaults to <root>/reports or <bundle>/evaluation.",
    )
    parser.add_argument(
        "--tex-out",
        type=str,
        default=None,
        help="Optional explicit path for the summary TeX table.",
    )
    parser.add_argument("--recompute", action="store_true", help="Ignore cached metrics and rescore WAV pairs.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    bundle_dirs = _bundle_dirs(root)
    bundle_dirs = sorted(bundle_dirs, key=lambda p: _bundle_relpath(root, p))
    protocols = {
        load_evaluation_metrics(_metric_config(bundle))[1]["config_sha256"]
        for bundle in bundle_dirs
    }
    if len(protocols) != 1:
        raise ValueError("Cannot combine runs using different evaluation metric configurations")

    if args.report_dir is not None:
        report_dir = Path(args.report_dir).resolve()
    elif len(bundle_dirs) == 1 and bundle_dirs[0] == root:
        report_dir = root / "evaluation"
    else:
        report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    extremes_index: List[Dict[str, object]] = []
    distributions = []
    plot_bundles = []
    pack_macro = []

    for bundle_dir in bundle_dirs:
        summary_path = bundle_dir / "evaluation" / "summary.json"
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        metric_rows = _load_or_compute_metric_rows(bundle_dir, recompute=args.recompute)
        if (summary.get("metric_protocol") or {}).get("version") != metric_rows[0]["metric_version"]:
            summary.setdefault("legacy_metrics", summary.get("metrics", {}).copy())
            for old_key in ("test/mss", "test/mss_sc", "test/mss_log"):
                summary.setdefault("metrics", {}).pop(old_key, None)
        summary["metric_protocol"] = {"version": metric_rows[0]["metric_version"],
                                      "config_sha256": metric_rows[0]["metric_config_sha256"]}
        current_metrics = summary.setdefault("metrics", {})
        for column in metric_rows[0]:
            if column.startswith("test/") and column not in ("test/loss", "test/objective"):
                current_metrics[column] = statistics.fmean(float(row[column]) for row in metric_rows)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _write_csv(bundle_dir / "evaluation" / "summary_metrics.csv", [current_metrics], sorted(current_metrics))
        relpath = _bundle_relpath(root, bundle_dir)
        stats = _distribution_stats(relpath, metric_rows)
        _write_csv(bundle_dir / "evaluation" / "statistics.csv", stats, list(stats[0]))
        (bundle_dir / "evaluation" / "statistics.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        distributions.extend(stats)
        plot_bundles.append((relpath, metric_rows))
        for label in sorted({r["metric"] for r in stats}):
            pack_stats = [r for r in stats if r["metric"] == label and r["group"].startswith("pack:")]
            pack_macro.append({"bundle": relpath, "metric": label, "packs": len(pack_stats),
                               "mean_of_pack_means": statistics.fmean(r["mean"] for r in pack_stats)})
        summary_rows.append(_summary_row(root, bundle_dir, summary, metric_rows))
        pack_names = sorted({_row_pack(row) for row in metric_rows})
        if len(pack_names) > 1:
            for pack in pack_names:
                subset = [row for row in metric_rows if _row_pack(row) == pack]
                summary_rows.append(_summary_row(root, bundle_dir, summary, subset, pack))
        report_name = _write_bundle_extremes(report_dir, root, bundle_dir, summary, metric_rows)
        extremes_index.append(
            {
                "bundle": _bundle_relpath(root, bundle_dir),
                "pack": _pack_label(summary),
                "extremes_report": report_name,
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: str(row["bundle"]))
    summary_table_txt = report_dir / "summary_table.txt"
    summary_table_txt.write_text(_summary_markdown(summary_rows) + "\n", encoding="utf-8")
    _write_csv(report_dir / "summary_table.csv", summary_rows, SUMMARY_HEADERS)
    tex_out = Path(args.tex_out).resolve() if args.tex_out is not None else report_dir / "summary_table.tex"
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text(_summary_latex(summary_rows), encoding="utf-8")
    _write_csv(report_dir / "extremes_index.csv", extremes_index, ["bundle", "pack", "extremes_report"])
    _write_csv(report_dir / "distribution_stats.csv", distributions, list(distributions[0]))
    _write_csv(report_dir / "pack_macro.csv", pack_macro, list(pack_macro[0]))
    _write_dashboard(report_dir / "dashboard.html", plot_bundles)

    print(f"[compile_results] bundles: {len(summary_rows)}")
    print(f"[compile_results] summary: {summary_table_txt}")
    print(f"[compile_results] tex: {tex_out}")
    print(f"[compile_results] reports: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
