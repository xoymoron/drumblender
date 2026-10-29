# Completed NEW versus FAST modal review

The [viewer script](../scripts/review_modal_dataset.py) uses the completed `processed_modal_new128` and `processed_modal_fast128` directories. NEW/FAST feature tensors and the NEW folder's preprocessed WAV are loaded on demand. The normal Random and ID buttons do **not** call a modal analyzer. Legacy comparison is an explicit checkbox and is the only path that recomputes modal analysis.

Run on the server containing both complete extraction directories:

```bash
conda activate drumblender
python -m scripts.review_modal_dataset \
  --new_root ../dataset/modal_features/processed_modal_new128 \
  --fast_root ../dataset/modal_features/processed_modal_fast128 \
  --render_device cuda \
  --host 0.0.0.0 --port 8765
```

Open `http://SERVER_IP:8765/` from your browser. For a local-only server, omit `--host` and use `http://127.0.0.1:8765/`. The CLI also accepts `--render_device cpu` or `auto`. CUDA here accelerates **synthesis of saved features**; no CQT/STFT peak extraction runs for NEW or FAST.

The viewer pairs files by the original `orig_relpath`, even when extraction metadata IDs differ. It checks the NEW and FAST source WAVs for the same sample rate, length and audio samples, then checks both metadata records before rendering. The plot's mel/linear setting changes the **display STFT and vertical frequency axis**. It does not change how the saved modes were extracted.

The completed `.pt` tensors contain frequency, amplitude, and phase. An amplitude above zero marks observed frames for the overlay. They do **not** contain each mode's CQT/long STFT/short STFT source label, so NEW and FAST overlays each have one color. Source counts in the cards come from saved metadata.

The attached files at `C:/Users/mindi/Downloads` provide NEW metadata/configuration/run statistics. They do not contain WAV or feature `.pt` files, so local sample spectrograms cannot be rendered from those three files alone. The server paths above provide the actual media. The previous FAST metadata/run statistics are also not present in those attached paths; the viewer reads them directly from its FAST root and compares sample and corpus statistics when running on the server.

## Attached NEW run: metadata-derived observations

| Statistic | Value |
|---|---:|
| Completed files | 17,173 |
| Failed files | 0 |
| Wall time | 10,359.835 s = 2 h 52 m 39.835 s |
| Saved active modes | 1,478,546, mean 86.097 per file |
| Median / 90th percentile | 128 / 128 modes |
| At 128-mode ceiling | 8,789 files, 51.18% |
| LF / HF | 256,985 / 1,221,561 modes = 17.38% / 82.62% |
| Mode source CQT | 65,985 = 4.46% |
| Mode source long STFT | 1,118,940 = 75.68% |
| Mode source short STFT | 40,700 = 2.75% |
| Mode source long STFT anchor + short envelope | 252,921 = 17.11% |
| Rejected at score gate / mode ceiling | 5,586,756 / 3,135,685 candidates |
| Median / 90th percentile sample analysis time | 0.715 s / 2.198 s |

These are extraction counts, not listening or reconstruction quality scores. In particular, 51.18% at the ceiling means the 128-mode limit frequently decides which candidates survive. It does not prove all candidates are real modes or that a larger limit would sound better.

The NEW run configured LF/HF at **700 Hz**, long STFT window 8192, short STFT window 2048, hop 256, max gap 2, prominence 2 dB, no `fast` shortcuts and refinement enabled. The heavy analysis stages accumulated 2.157 h for long STFT and 1.928 h for short STFT across both shards; CQT accumulated 0.288 h. The approximately 27-minute FAST wall time reported earlier gives an approximate **6.4×** NEW/FAST wall-time ratio. Per-file mode/source differences require the FAST metadata and are calculated by the updated viewer from its root.
