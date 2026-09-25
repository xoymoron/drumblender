"""Export modal-only audio, diagnostics, and comparable LF/HF spectrograms.

Run from the repository root with ``python -m scripts.inspect_modal_analysis``.
No training, instrument annotations, loudness matching, or peak normalization
is applied. The waveform residual is a diagnostic, not a trained noise branch.
"""

import argparse
import html
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from scipy import signal

from drumblender.utils.modal_analysis_new import CQTModalAnalysis, render_modal


def legacy_render(audio, sample_rate, num_modes):
    """Compare with the current ampzero-aware legacy analyzer and fixed synth."""
    import torch

    from drumblender.synths.modal import modal_synth
    from drumblender.utils.modal_analysis import CQTModalAnalysis as LegacyAnalysis

    analyzer = LegacyAnalysis(
        sample_rate,
        hop_length=256,
        fmin=20,
        n_bins=240,
        bins_per_octave=24,
        min_length=10,
        num_modes=num_modes,
        diff_threshold=5,
        verbose=False,
    )
    waveform = torch.from_numpy(audio[None])
    # nnAudio reflect padding requires enough context even for a short hit.
    minimum = analyzer.cqt.kernel_width // 2 + 1
    if waveform.shape[-1] < minimum:
        waveform = torch.nn.functional.pad(waveform, (0, minimum - waveform.shape[-1]))
    with torch.no_grad():
        started = perf_counter()
        frequency, amplitude, phase = analyzer(waveform)
        analyzed = perf_counter() - started
        output = modal_synth(
            frequency * (2 * torch.pi / sample_rate),
            amplitude,
            waveform.shape[-1],
            phase,
        )[0, : len(audio)].numpy()
    return output, dict(analysis_seconds=analyzed, modes=frequency.shape[1])


def write_html(output, summaries):
    """Create a single-sample viewer that works directly from a file URL."""
    options, cards = [], []
    for index, summary in enumerate(summaries, 1):
        name = Path(summary["source"]).name
        directory = f"{index:02d}_{Path(summary['source']).stem}"
        safe_name = html.escape(name)
        safe_directory = html.escape(directory, quote=True)
        options.append(f'<option value="{safe_directory}">{safe_name}</option>')
        players = []
        for label, filename, description in [
            ("① Original", "original", "분석기에 넣은 원본 WAV"),
            ("② Legacy modal", "legacy", "기존 CQT 분석기로 얻은 modal branch만 합성"),
            ("③ NEW modal", "modal", "새 hybrid 분석기로 얻은 modal branch만 합성"),
            (
                "④ Residual",
                "residual",
                "Original − NEW modal: 현재 waveform에서 뺀 잔차",
            ),
        ]:
            if filename == "legacy" and "legacy" not in summary:
                continue
            relative = f"{safe_directory}/{filename}.wav"
            players.append(
                f'<div class="audio-card"><strong>{label}</strong><span>{description}</span>'
                f'<audio controls preload="none" data-source="{relative}" src="{relative}"></audio></div>'
            )
        old = summary.get("legacy")
        old_info = (
            f' · Legacy {old["modes"]}개 / {old["analysis_seconds"]:.2f}초'
            if old is not None
            else ""
        )
        plot = f"{safe_directory}/comparison.png"
        cards.append(
            f'<section class="sample" data-sample="{safe_directory}" hidden>'
            f"<h2>{safe_name}</h2>"
            f'<p class="meta">오디오 {summary["duration_seconds"]:.2f}초 · '
            f'NEW {summary["modes"]}개 / {summary["analysis_seconds"]:.2f}초{old_info}</p>'
            f'<div class="audio-grid">{"".join(players)}</div>'
            f'<img class="plot" data-source="{plot}" src="{plot}" '
            'alt="원본, 기존 modal, NEW modal, 잔차의 LF 및 full-band spectrogram">'
            "</section>"
        )

    head = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Modal analysis review</title>
<style>
  :root {color-scheme:dark;font-family:system-ui,sans-serif}
  body {max-width:1460px;margin:0 auto;padding:2rem;background:#17191d;color:#eef1f4;line-height:1.55}
  h1,h2,h3 {line-height:1.25} h1 {margin-top:0} .muted {color:#bac4cc}
  .toolbar {display:flex;gap:.75rem;flex-wrap:wrap;align-items:center;padding:1rem;
    border:1px solid #536172;border-radius:.8rem;background:#222831}
  select,button {font:inherit;color:inherit;background:#303c49;border:1px solid #788a9b;
    border-radius:.5rem;padding:.5rem .7rem} button {cursor:pointer}
  button:hover {background:#435669} button:focus-visible,select:focus-visible {outline:3px solid #70d7e3}
  .explain {margin:1.5rem 0;padding:1.25rem;border-radius:.8rem;background:#242b34}
  .explain ul {padding-left:1.5rem} .explain li {margin:.4rem 0}
  .audio-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:.8rem}
  .audio-card {display:flex;flex-direction:column;gap:.35rem;padding:1rem;background:#252e39;
    border-radius:.7rem} .audio-card strong {color:#96ecf3} .audio-card span {min-height:3em}
  audio {width:100%} .plot {display:block;max-width:100%;margin-top:1.2rem;border-radius:.4rem}
  .sample[hidden] {display:none} .meta {color:#ccd5dc} code {white-space:normal;overflow-wrap:anywhere}
</style>
<h1>Modal analysis: sample 비교</h1>
<div class="toolbar"><label for="sample-select">Sample</label><select id="sample-select">
"""
    middle = """</select>
<button id="reload-sample" type="button">선택 sample 다시 읽기</button>
<button id="reload-report" type="button">목록 새로고침</button>
<span id="refresh-status" role="status" class="muted"></span></div>
<div class="explain"><h2>네 개의 소리는 무엇인가?</h2><ul>
<li><b>① Original</b>: 입력 WAV 그대로. modal과 noise를 모두 포함한 기준 소리.</li>
<li><b>② Legacy modal</b>: 기존 CQT modal analysis로 추출한 mode만 합성. 현재 수정된 synthesizer로 재생한다.</li>
<li><b>③ NEW modal</b>: 새 LF CQT + 두 STFT 분석기로 추출한 mode만 합성. noise branch는 포함하지 않는다.</li>
<li><b>④ Residual</b>: Original − NEW modal의 <em>sample별 waveform 차이</em>. 남은 tonal 성분과 phase 오차도 포함하므로 noise branch의 복원 결과가 아니다.</li>
</ul><p>아래 그림의 <b>네 행</b>도 위 순서와 같다. 왼쪽 열은 20–2000 Hz를 긴 STFT로, 오른쪽 열은 전체 대역을 짧은 STFT로 <em>표시</em>한다. 두 열은 합성 branch가 아니라 시각화 방법이다. 모든 행은 해당 열의 Original과 같은 dB 기준으로 표시된다. 맨 위의 청록색 선은 NEW에서 실제 peak가 관측된 frame에만 그린 frequency track이다.</p>
<p><b>선택 sample 다시 읽기</b>는 화면의 WAV와 그림을 파일에서 다시 불러온다. <b>목록 새로고침</b>은 재생성된 index.html을 다시 읽는다. 새 WAV 분석이나 분석 결과 갱신은 터미널에서 아래 명령으로 실행한 뒤 목록을 새로고침해야 한다.</p>
<code>python -m scripts.inspect_modal_analysis "WAV_경로" --compare_legacy --legacy_modes 128 --output analysis/modal_new_review</code>
</div>
"""
    tail = """<script>
const selector = document.querySelector('#sample-select');
const status = document.querySelector('#refresh-status');
const samples = [...document.querySelectorAll('.sample')];
function showSample(value) {
  samples.forEach(sample => sample.hidden = sample.dataset.sample !== value);
  selector.value = value;
  location.hash = encodeURIComponent(value);
}
const requested = decodeURIComponent(location.hash.slice(1));
showSample(samples.some(sample => sample.dataset.sample === requested)
  ? requested : samples[0].dataset.sample);
selector.addEventListener('change', () => showSample(selector.value));
document.querySelector('#reload-sample').addEventListener('click', () => {
  const current = samples.find(sample => !sample.hidden);
  for (const audio of current.querySelectorAll('audio')) {
    audio.pause();
    const url = new URL(audio.dataset.source, document.baseURI);
    url.searchParams.set('refresh', Date.now());
    audio.src = url.href;
    audio.load();
  }
  const plot = current.querySelector('.plot');
  const url = new URL(plot.dataset.source, document.baseURI);
  url.searchParams.set('refresh', Date.now());
  plot.src = url.href;
  status.textContent = '선택 sample의 WAV와 그림을 다시 읽었다.';
});
document.querySelector('#reload-report').addEventListener('click', () => location.reload());
</script></html>"""
    (output / "index.html").write_text(
        head + "".join(options) + middle + "".join(cards) + tail,
        encoding="utf-8",
    )


def plot_comparison(path, audio, reconstructed, result, sample_rate, legacy=None):
    """Use the same dB reference and limits for every row in each view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("Original + observed NEW tracks", audio)]
    if legacy is not None:
        rows.append(("Legacy modal (padding repair applied)", legacy))
    rows.extend(
        [
            ("NEW modal", reconstructed),
            ("Waveform residual: original - NEW", audio - reconstructed),
        ]
    )
    figure, axes = plt.subplots(
        len(rows), 2, figsize=(15, 3.1 * len(rows)), squeeze=False
    )
    for column, (window, maximum, view_name) in enumerate(
        [
            (8192, 2000, "LF / long STFT"),
            (2048, sample_rate / 2, "Full band / short STFT"),
        ]
    ):
        nperseg = min(window, len(audio))
        hop = max(1, nperseg // 4, len(audio) // 1200)
        hop = min(hop, nperseg)
        spectra = [
            signal.stft(
                wave,
                sample_rate,
                nperseg=nperseg,
                noverlap=nperseg - hop,
                boundary="zeros",
            )
            for _, wave in rows
        ]
        reference = max(float(np.abs(spectra[0][2]).max()), 1e-12)
        for row, ((label, _), (frequency, times, spectrum)) in enumerate(
            zip(rows, spectra)
        ):
            axis = axes[row, column]
            image = axis.pcolormesh(
                times,
                frequency,
                20 * np.log10(np.maximum(np.abs(spectrum), 1e-12) / reference),
                shading="auto",
                cmap="magma",
                vmin=-80,
                vmax=0,
                rasterized=True,
            )
            axis.set(
                xlim=(0, len(audio) / sample_rate),
                ylim=(20, min(maximum, sample_rate / 2)),
                xlabel="Time (s)",
                ylabel="Frequency (Hz)",
                title=f"{label} | {view_name}",
            )
            if column == 0:
                axis.set_yscale("log")
            if row == 0:
                for values, observed in zip(result.frequencies, result.observed):
                    # NaNs break the line at gaps: the plot cannot visually join
                    # separate observations or draw silent endpoint padding.
                    axis.plot(
                        result.frame_times,
                        np.where(observed, values, np.nan),
                        color="cyan",
                        linewidth=0.45,
                        alpha=0.55,
                    )
            figure.colorbar(image, ax=axis, label="dB relative to original peak")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def inspect_file(source, output, args):
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"Expected preprocessed mono audio: {source}")
    audio = audio[:, 0]
    analyzer = CQTModalAnalysis(
        sample_rate,
        num_modes=args.num_modes,
        backend=args.backend,
        min_length=args.min_length,
        min_relative_score_db=args.min_relative_score_db,
        refine=not args.no_refine,
    )
    result = analyzer.analyze(audio)
    started = perf_counter()
    reconstructed = render_modal(result, len(audio), sample_rate)
    render_seconds = perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    # FLOAT WAV preserves gain and possible overload for honest comparison.
    for name, wave in [
        ("original", audio),
        ("modal", reconstructed),
        ("residual", audio - reconstructed),
    ]:
        sf.write(output / f"{name}.wav", wave, sample_rate, subtype="FLOAT")
    np.savez_compressed(
        output / "tracks.npz",
        **{
            key: getattr(result, key)
            for key in (
                "frequencies",
                "amplitudes",
                "phases",
                "observed",
                "confidence",
                "source",
                "frame_times",
                "scores",
            )
        },
    )
    duration = len(audio) / sample_rate
    summary = dict(
        source=str(source.resolve()),
        sample_rate=sample_rate,
        duration_seconds=duration,
        backend=args.backend,
        mode_limit=args.num_modes,
        modes=len(result.frequencies),
        candidates_before_limit=result.candidates_before_limit,
        candidates_before_score_gate=result.candidates_before_score_gate,
        analysis_seconds=result.analysis_seconds,
        analysis_rtf=result.analysis_seconds / duration,
        numpy_render_seconds=render_seconds,
        numpy_render_rtf=render_seconds / duration,
        original_rms=float(np.sqrt(np.mean(audio**2))),
        modal_rms=float(np.sqrt(np.mean(reconstructed**2))),
    )
    legacy = None
    if args.compare_legacy:
        legacy, summary["legacy"] = legacy_render(audio, sample_rate, args.legacy_modes)
        sf.write(output / "legacy.wav", legacy, sample_rate, subtype="FLOAT")
    plot_comparison(
        output / "comparison.png", audio, reconstructed, result, sample_rate, legacy
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="*", type=Path, help="Mono WAV files or directories."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/modal_new_review")
    )
    parser.add_argument(
        "--backend", choices=("hybrid", "stft", "cqt"), default="hybrid"
    )
    parser.add_argument("--num_modes", type=int, default=128)
    parser.add_argument("--min_length", type=int, default=4)
    parser.add_argument("--min_relative_score_db", type=float, default=-40.0)
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--compare_legacy", action="store_true")
    parser.add_argument("--legacy_modes", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=8)
    parser.add_argument(
        "--refresh_html",
        action="store_true",
        help="Rebuild HTML from existing summary.json without reanalyzing audio.",
    )
    args = parser.parse_args()
    if args.refresh_html:
        summaries = json.loads(
            (args.output / "summary.json").read_text(encoding="utf-8")
        )
        write_html(args.output, summaries)
        print(args.output / "index.html")
        return
    sources = []
    for source in args.inputs:
        sources.extend(sorted(source.rglob("*.wav")) if source.is_dir() else [source])
    sources = list(dict.fromkeys(sources))
    if args.max_files > 0:
        sources = sources[: args.max_files]
    if not sources:
        parser.error("No WAV files found")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, source in enumerate(sources):
        directory = f"{index + 1:02d}_{source.stem}"
        summary = inspect_file(source, args.output / directory, args)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    write_html(args.output, summaries)


if __name__ == "__main__":
    main()
