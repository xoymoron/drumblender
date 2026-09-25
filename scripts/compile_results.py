#!/usr/bin/env python3
"""Compile reconstruction evaluation from one exported bundle or a run directory.

The main table reports MR-STFT, LSD, and temporal SF. Per-file and per-pack
statistics, supplementary metrics, and distribution plots are saved alongside it.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
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


# Number of examples shown in each best/worst report section.
TOP_K = 3
# Shared SVG dimensions keep all metric distribution panels aligned.
PLOT_VIEW_WIDTH = 920
PLOT_LEFT = 260
PLOT_SPAN = 600
PLOT_ROW_HEIGHT = 44
PLOT_FIRST_ROW = 48
PLOT_BASE_HEIGHT = 60
PLOT_GRID_INTERVALS = 4
RANK_METRICS = [
    # Keep the paper table and ranking focused on these three primary metrics.
    ("MR-STFT", "test/mr_stft"),
    ("LSD", "test/lsd"),
    # Retain the CSV key for compatibility; it now stores temporal spectral flux.
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
    """Convert CSV values consistently; invalid cells become non-finite."""
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    """Return the per-file mean and population standard deviation."""
    xs = [x for x in values if _is_finite(x)]
    if len(xs) == 0:
        return float("nan"), float("nan")
    mean = float(statistics.fmean(xs))
    std = float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0
    return mean, std


def _fmt_float(value: float, digits: int = 6) -> str:
    """Format finite values and make missing values explicit in reports."""
    if not _is_finite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _escape_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", r"\|")


def _markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    """Render the small text tables used by summaries and extremes reports."""
    header_line = "| " + " | ".join(_escape_cell(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def _latex_escape(value: object) -> str:
    """Escape path and pack names before inserting them into TeX cells."""
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
    """Accept one bundle directly or find bundles below a run directory."""
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
    """Read the first channel, preferring torchaudio and falling back to SoundFile."""
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


def _compute_metric_rows(bundle_dir, loss_rows, metrics, config_sha256):
    """Rescore paired WAVs and refresh the per-file cache for this bundle."""
    metric_rows = []
    for source in loss_rows:
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
        # Older bundles may contain right padding; score only the recorded valid length.
        scores = score_reconstruction(metrics, recon[None, :, :length],
                                      target[None, :, :length], sample_rate=sr_t)
        row = dict(source)
        row.update(scores)
        row.update({"sample_pack_key": _row_pack(source),
                    "metric_config_sha256": config_sha256,
                    "audio_sha256": fingerprint})
        metric_rows.append(row)
    if not metric_rows:
        raise ValueError(f"Empty evaluation: {bundle_dir}")
    _write_csv(bundle_dir / "evaluation" / "per_file_metrics.csv", metric_rows, list(metric_rows[0]))
    return metric_rows


def _row_pack(row):
    """Group by the first path component, matching the dataset's top-level packs."""
    source = str(row.get("source_filename", "")).replace("\\", "/")
    if "/" in source:
        return source.split("/", 1)[0]
    return str(row.get("sample_pack_key") or "__root__").replace("\\", "/").split("/", 1)[0]


def _cache_matches(bundle_dir, rows, loss_rows, metrics, config_sha256):
    """Reuse scores only when sample order, metric setup, and both WAVs still match."""
    if not rows or len(rows) != len(loss_rows):
        return False

    required_scores = evaluation_score_keys(metrics)
    expected_columns = set(loss_rows[0]) | set(required_scores) | {
        "sample_pack_key", "metric_config_sha256", "audio_sha256"
    }
    if set(rows[0]) != expected_columns:
        return False
    for row, source in zip(rows, loss_rows):
        # Reject changes in sample order, metric setup, scores, or paired WAV content.
        if any(row.get(key) != source.get(key)
               for key in ("meta_key", "source_filename", "length")):
            return False
        if row.get("metric_config_sha256") != config_sha256:
            return False
        if any(not _is_finite(_to_float(row.get(key))) for key in required_scores):
            return False
        if row.get("audio_sha256") != audio_pair_fingerprint(
            bundle_dir, source["source_filename"]
        ):
            return False
    return True


def _load_or_compute_metric_rows(bundle_dir: Path, metrics, config_sha256, recompute=False):
    """Load a valid cache or rescore the paired audio when it is stale or forced."""
    loss_csv = bundle_dir / "evaluation" / "per_file_loss.csv"
    if not loss_csv.is_file():
        raise FileNotFoundError(f"Missing per_file_loss.csv: {loss_csv}")
    loss_rows = _read_csv(loss_csv)
    cache = bundle_dir / "evaluation" / "per_file_metrics.csv"
    if cache.is_file() and not recompute:
        rows = _read_csv(cache)
        if _cache_matches(bundle_dir, rows, loss_rows, metrics, config_sha256):
            return rows
    return _compute_metric_rows(bundle_dir, loss_rows, metrics, config_sha256)


def _pack_label(summary: Dict[str, Any]) -> str:
    """Label filtered exports by their requested packs, or mark them as all packs."""
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
    """Summarize the three primary metrics for one bundle or pack subset."""
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
    """Render the primary per-bundle scores as a Markdown table."""
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
    """Build the compact paper table; supplementary metrics stay in the reports."""
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
    """Format ranked samples with their pack, filename, and metric values."""
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
    """List the best and worst examples for each primary metric."""
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


def _percentiles(values, probabilities):
    """Interpolate quantiles on one sorted copy of the values."""
    ordered = sorted(values)
    result = []
    for probability in probabilities:
        position = (len(ordered) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        result.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    return result


def _group_rows(rows):
    """Create an overall group plus overlapping subsets for each top-level pack."""
    by_pack = {}
    for row in rows:
        by_pack.setdefault(_row_pack(row), []).append(row)
    return {"all samples": rows, **{
        f"pack: {pack}": by_pack[pack] for pack in sorted(by_pack)
    }}


def _distribution_stats(bundle, rows):
    """Report distribution summaries for every saved score, overall and per pack."""
    result = []
    primary = {column for _, column in RANK_METRICS}
    extra = sorted(key for key in rows[0]
                   if key.startswith("test/") and key not in primary
                   and key != "test/loss")
    reported_metrics = RANK_METRICS + [(key.removeprefix("test/"), key) for key in extra]
    for group, examples in _group_rows(rows).items():
        for label, column in reported_metrics:
            values = [_to_float(row[column]) for row in examples]
            if not values or not all(map(math.isfinite, values)):
                raise ValueError(f"Invalid {label} scores in {bundle}/{group}")
            # SEM and quantiles complement the per-file mean and population spread.
            mean, std = _mean_std(values)
            p05, p25, median, p75, p95, p99 = _percentiles(
                values, (.05, .25, .5, .75, .95, .99)
            )
            result.append({"bundle": bundle, "group": group, "metric": label,
                           "n": len(values), "mean": mean, "std": std,
                           "sem": std / math.sqrt(len(values)),
                           "p05": p05, "p25": p25, "median": median,
                           "p75": p75, "p95": p95, "p99": p99,
                           "min": min(values), "max": max(values)})
    return result


def _write_dashboard(path, bundles):
    """Write self-contained SVG plots; supplemental metrics are collapsed by default."""
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
                           and key != "test/loss")
        plots = RANK_METRICS + [(key.removeprefix("test/").replace("_", " ").title(), key)
                                for key in secondary]
        for label, column in plots:
            is_secondary = column not in primary
            if is_secondary:
                parts.append(f'<details><summary>{html.escape(label)}</summary>')
            arrays = [[float(row[column]) for row in examples] for examples in groups.values()]
            # SF is long-tailed, so log1p keeps the rest of its distribution visible.
            transform = math.log1p if label == "SF" else float
            upper = max(transform(max(values)) for values in arrays) or 1.0
            scale = lambda value: PLOT_LEFT + PLOT_SPAN * transform(value) / upper
            height = PLOT_BASE_HEIGHT + len(groups) * PLOT_ROW_HEIGHT
            parts.append(f'<h3>{label}</h3><svg role="img" aria-label="{label} distributions" viewBox="0 0 {PLOT_VIEW_WIDTH} {height}">')
            for tick in range(PLOT_GRID_INTERVALS + 1):
                value = upper * tick / PLOT_GRID_INTERVALS
                raw = math.expm1(value) if label == "SF" else value
                x = PLOT_LEFT + (PLOT_SPAN // PLOT_GRID_INTERVALS) * tick
                parts.append(f'<path d="M{x} 28 V{height-24}" stroke="#dce2eb"/><text x="{x}" y="18" text-anchor="middle">{raw:.3g}</text>')
            for i, ((group, examples), values) in enumerate(zip(groups.items(), arrays)):
                y = PLOT_FIRST_ROW + i * PLOT_ROW_HEIGHT
                p5, p25, p50, p75, p95 = [
                    scale(value) for value in _percentiles(values, (.05, .25, .5, .75, .95))
                ]
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


def _update_bundle_summary(bundle_dir, summary, metric_rows):
    """Replace stored aggregate scores with the current per-file evaluation."""
    summary.pop("metric_protocol", None)
    summary.pop("legacy_metrics", None)
    summary["metric_config_sha256"] = metric_rows[0]["metric_config_sha256"]
    current_metrics = {}
    for key in metric_rows[0]:
        if key.startswith("test/"):
            current_metrics[key] = statistics.fmean(float(row[key]) for row in metric_rows)
    summary["metrics"] = current_metrics

    eval_dir = bundle_dir / "evaluation"
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(eval_dir / "summary_metrics.csv", [current_metrics], sorted(current_metrics))


def _write_bundle_statistics(bundle_dir, bundle_name, metric_rows):
    """Save the full per-metric distributions beside the per-file scores."""
    stats = _distribution_stats(bundle_name, metric_rows)
    eval_dir = bundle_dir / "evaluation"
    _write_csv(eval_dir / "statistics.csv", stats, list(stats[0]))
    (eval_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    """Compile per-bundle caches into statistics, tables, extremes, and plots."""
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
    configured_bundles = [
        (bundle, *load_evaluation_metrics(bundle / "configs" / "evaluation_metrics.yaml"))
        for bundle in bundle_dirs
    ]
    # Comparisons are valid only when every bundle uses the same metric configuration.
    if len({config_sha256 for _, _, config_sha256 in configured_bundles}) != 1:
        raise ValueError("Cannot combine runs using different evaluation metric configurations")

    # Keep a single bundle's reports beside it; combine run reports at the root.
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

    # Validate caches and summarize each bundle before writing combined reports.
    for bundle_dir, metrics, config_sha256 in configured_bundles:
        summary_path = bundle_dir / "evaluation" / "summary.json"
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        metric_rows = _load_or_compute_metric_rows(
            bundle_dir, metrics, config_sha256, recompute=args.recompute
        )
        _update_bundle_summary(bundle_dir, summary, metric_rows)
        relpath = _bundle_relpath(root, bundle_dir)
        stats = _write_bundle_statistics(bundle_dir, relpath, metric_rows)
        distributions.extend(stats)
        plot_bundles.append((relpath, metric_rows))
        for label in sorted({r["metric"] for r in stats}):
            pack_stats = [r for r in stats if r["metric"] == label and r["group"].startswith("pack:")]
            pack_macro.append({"bundle": relpath, "metric": label, "packs": len(pack_stats),
                               "mean_of_pack_means": statistics.fmean(r["mean"] for r in pack_stats)})
        summary_rows.append(_summary_row(root, bundle_dir, summary, metric_rows))
        # Pack rows are subsets of the overall row; both views appear in the table.
        grouped_rows = _group_rows(metric_rows)
        if len(grouped_rows) > 2:
            for group, subset in list(grouped_rows.items())[1:]:
                summary_rows.append(_summary_row(
                    root, bundle_dir, summary, subset, group.removeprefix("pack: ")
                ))
        report_name = _write_bundle_extremes(report_dir, root, bundle_dir, summary, metric_rows)
        extremes_index.append(
            {
                "bundle": _bundle_relpath(root, bundle_dir),
                "pack": _pack_label(summary),
                "extremes_report": report_name,
            }
        )

    # Stable sorting makes report diffs independent of filesystem traversal order.
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
    # The HTML dashboard reads the same per-file scores as the numeric reports.
    _write_dashboard(report_dir / "dashboard.html", plot_bundles)

    print(f"[compile_results] bundles: {len(summary_rows)}")
    print(f"[compile_results] summary: {summary_table_txt}")
    print(f"[compile_results] tex: {tex_out}")
    print(f"[compile_results] reports: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
