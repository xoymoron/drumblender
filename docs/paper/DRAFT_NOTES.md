# EAIM 2027 draft notes

Collaboration notes kept outside the manuscript. Nothing in this file goes into `eaim.tex`.

## Files

The local files mirror the Overleaf layout, which was split into per-section files on 2026-11-05.

- `eaim.tex`: main file (preamble, author block, `\input{sections/...}` lines, bibliography, appendix).
- `sections/abstract.tex` (abstract and keywords), `introduction.tex`, `related_work.tex`, `method.tex`, `experiments.tex`, `results.tex`, `conclusion.tex`, `appendix.tex`.
- `eaim.bib`: entries appended to Overleaf `eaim.bib` after the template's two sample entries (52 here, 54 in Overleaf).
- Overleaf is the source of truth. The local copies matched Overleaf byte for byte (SHA-256) on 2026-11-05 right after the split; later edits made in Overleaf are not mirrored automatically.
- Split check: the `\input` expansion equals the pre-split `eaim.tex` exactly, and both versions compiled to 11 pages with 0 errors, 0 warnings, and 0 info messages.
- The results table (as edited by the user on 2026-11-05) now shows a single DrumBlender row with the per-pack rows commented out.

## Introduction: open placeholders

| Location | Placeholder | Depends on |
|---|---|---|
| Contribution 4 | control evaluation criteria | control experiments actually run |

Do not fill these with results or properties that have not been implemented and measured.

Filled on 2026-11-05 from the implementation spec (`docs/superpowers/specs/2026-11-05-joint-modal-noise-transient-design.md`): Intro P4 (joint training, bounded modal corrections, noise bands conditioned on corrected tracks, restricted TCN, hybrid statement, post-hoc SAE with zero-edit identity), Contributions 2 and 3, Related Work B1 and B3 contrast sentences. These describe the design, not measured results; recheck them if the spec changes.
- "including cymbals" was removed from the P4 goal sentence (user, 2026-11-05): no per-class results support it.
- P2 now cites the DrumBlender decay artifacts (Shier et al. 2023, sections 1, 4.1, 5: "artifacts, similar to comb filtering, in the decay portion", from informal listening) and Simionato & Fasciani 2025 (separately trained sub-models; the paper's phase/gradient claim was not verified and is not used).
- B1 cites DDSP-Piano: the JAES article states that listening-test ratings "confirm that the second training phase does not improve the perceived quality"; that phase fine-tunes the detuner, inharmonicity model, and Z-encoder modifiers.
- Modulation Discovery (Mitcheltree et al. 2025) is not cited (user decision). Set Transformer is reserved for Method.

## Related Work: open placeholders

None. Both TBDs were filled on 2026-11-05 (see above).

## Remaining sections: skeleton status

- Every subsection has comments with its content plan and one visible `\textbf{[TBD: ...]}` marker. No prose has been written for these sections yet.
- Comments marked "Decided" cite their source: hybrid analyzer (`docs/modal.md`), preprocessing (`PROCESSING.md`), baseline protocol (`jorshidrumblender/docs/baseline_protocol.md`), control evaluation (`docs/research_implementation_goals.md` section 6).
- Comments marked "Direction" record the 2026-11-05 leanings (three DSP branches, NoiseBandNet-style noise, encoded modal refinement). The control method is no longer a direction: post-hoc SAE on h, no β-VAE. Direction comments are not final and must not become prose until decided.
- The results table added on 2026-11-05 lists two checkpoints (epoch036-step00015910, epoch047-step00020640) without naming the system. The Reconstruction subsection asks which system it is and which checkpoint is reported.
- Resolved (2026-11-05): spec section 12 now matches spectral + envelope only; modal phase, anchor, and transient-energy penalties are removed. Option A fixes the analyzed initial sine phase, removes dphi/phase pooling, and integrates refined frequency for the later phase trajectory. The planned model has 3,066,484 parameters; the transient path stays at 20,930.
- Method was restructured on 2026-11-05 to five subsections that follow the implementation spec: Modal Analysis and Reference (`sec:method-analysis`), Shared Representation (`sec:method-encoder`), Synthesis Components (`sec:method-synthesis`, written as Modal / Noise / Transient paragraphs), Training Objective (`sec:method-training`), Label-Free Controls (`sec:method-control`). The old Modal, Noise, and Transient subsections and their labels were merged; nothing referenced them. Every comment cites the spec section it comes from. Pushed to Overleaf and mirrored locally (SHA-256 match).
- Method prose written on 2026-11-05 (user: "assume the implementation is done"). It describes the spec as implemented; nothing in it is a measured result. Remaining TBD: control (feature) selection. A system-diagram figure is still missing (TODO comment at the top).
- Facts in the Method prose that rest on the spec rather than on code that exists today: encoder strides/channels, set pooling, GRU trunk, correction bounds, 256-band bank, TCN layout, loss sizes, SAE edit rule. The 20 s phase-accuracy sentence rests on the spec's CPU check of modal_cycles.py (constant tones, max error below 1e-4 rad). Recheck all of these against the final code.
- New citations, all verified on the publisher pages: `lee2019set` (PMLR 97), `cho2014learning` (ACL Anthology D14-1179), `perez2018film` (AAAI OJS; no page numbers listed; `number` dropped because the class's BibTeX style rejects volume plus number). Simionato's MR-STFT + RMS-energy MAE loss was confirmed in the article text.
- Page budget: 16 PDF pages after the first Method draft; 14 pages after the abstraction pass of 2026-11-05, with Results, Conclusion, and References starting on page 9.
- Method citations used: Set Transformer (set pooling); Hayes 2023 and DrumBlender (bounded frequency correction); NoiseBandNet (band layout); DrumBlender (TCN); GRU; FiLM; DDSP; Simionato (loss).
- Citation review applied on 2026-11-05 (each claim checked against the source text):
  - Added `waissbluth2022synthesis` to Related Work B2 (AIMC 2022; VAE that reconstructs snare samples as sparse mixtures of a small fixed library, latent moves swap samples in and out). The B2 contrast sentence now says the parameters come from an analysis of the recording rather than a fixed library. Venue string is the PDF footer; no DOI found.
  - Added `smith1987parshl` to the Method peak-interpolation sentence (PARSHL fits a parabola to three dB-magnitude bins; read from the CCRMA version; no page numbers found).
  - The spectral-loss attribution was corrected: the two terms are Parallel WaveGAN's spectral convergence + log-magnitude L1 (`yamamoto2020parallel`), which DrumBlender also uses; DDSP is cited only for averaging over several STFT sizes (DDSP's own loss is linear-magnitude L1 + log-magnitude L1, FFT 64–2048).
  - `higgins2017beta` was deleted from eaim.bib (it was uncited). Brown's CQT paper was not added. DAC stays; it is the first thing to drop with the B3 codec sentence if space is needed.
- Superseded note — Method still to cite when written: Set Transformer (set pooling); Hayes 2023 and DDSP-Piano (bounded frequency correction); NoiseBandNet (band layout); DrumBlender (TCN). The parameter budget in the spec is arithmetic and must be checked against the implemented model before it is reported.
- The keywords environment still reads "Keywords to be determined"; candidates are in a comment above it.
- The user commented out `\appendix` and `\input{sections/appendix}` in Overleaf on 2026-11-05; the appendix file is unchanged.
- Title changed by the user on 2026-11-05 to "Sparse Timbre Control of Percussive Audio with Joint Modal, Noise, and Transient Synthesis" The full title overflowed the running head by 45.5 pt, so the short running title "Sparse Timbre Control of Percussive Audio" was added (user approved, 2026-11-05); the compile is back to 0 errors, 0 warnings, 0 info.

## Experiments: sources, decisions, and open checks

Sources: `rawdata/DATASET_STATS.md` (raw count, sample rates); `PROCESSING.md` and `scripts/preprocess_datasets_pure.py` (preprocessing); `jorshidrumblender/cfg/05_all_parallel_variable.yaml`, `drumblender/tasks/drumblender.py`, `drumblender/models/tcn.py`, `docs/baseline_protocol.md` (baseline); `cfg/metrics/drumblender_metrics.yaml`, `drumblender/metrics.py`, `scripts/compile_results.py` (evaluation).

Decisions (user, 2026-11-05):
- Every sample pack stays undisclosed. No pack names, no dataset citations (Prockup et al. 2013 and the Dynamic Percussion Dataset were removed from the text and the bib), no URL footnotes, and no "nine sources".
- No instrument-label counts (kick, snare, tom, cymbal) in the paper.
- The Dataset text states (abstraction pass, 2026-11-05): samples collected from commercial and freely available packs, with no raw count; highly diverse, including tonal percussion (bells, marimbas, xylophones); selected as one-shots with a single strike and a single sound event; sample rate of at least 44.1 kHz; up to 20 s so that long-tailed instruments are covered.
- Preprocessing text keeps only: larger-RMS channel kept as mono for stereo, resampling to 48 kHz, leading-silence trim at -65 dB, trailing silence kept, no normalization. Decode, near-zero, and silent-file rejection are not described.
- The split is written as an 80/10/10 split within each source pack, applied to about 17,000 samples after preprocessing (user, 2026-11-05). The test-set count and the sentence on stratified versus held-out packs were removed; the stratified split remains a candidate for Limitations.
- The results table's pack names were replaced with Pack A–I in table order: A DynamicPercussion, B IndieDrums, C MDLib2, D Splice, E TidalCycles808, F TriSamples, G TriSamples808, H drumthrasher, I musicradar-drum-samples. Keep this mapping out of the manuscript. `scripts/compile_results.py` writes the real names, so a regenerated table must be relabelled again.
- The baseline equation is written as x̂ = S + T(S) + N, matching `transient_parallel: true` and `transient_takes_noise: false`.
- Training is one subsection for both systems (`sec:exp-train`; `sec:exp-impl` was removed): same Adam schedule; nominal batch 32 for both systems (proposed: 1 per GPU × 2 GPUs × 16 accumulation, `cfg/joint_resynthesis.yaml` and spec section 15; corrected from 64 on 2026-11-05), plus gradient clipping 1.0 for the proposed model. Seed and parameter counts were removed (user edit in Overleaf).
- Adam, Parallel WaveGAN (MR-STFT), and auraloss are cited. Parallel WaveGAN is co-authored by Eunwoo Song and is cited in the third person.

Open checks:
- Internal record only: the raw collection had 17,346 files and the test set 1,728; the paper now says "about 17,000" after preprocessing.
- `rawdata/CITATION.md` (not used in the paper) names Vesa Välimäki as the Dynamic Percussion Dataset author; the Zenodo record lists only Elias Veikko Juhani Nieminen.
- "All systems are scored by the same evaluation code from exported reconstruction and target waveforms": confirm that the baseline numbers come from `drumblender/scripts/compile_results.py` rescoring of exported WAVs.
- SF in this evaluation differentiates magnitude along time. The released DrumBlender code (`jorshidrumblender/drumblender/metrics.py`) differentiates along frequency, so these SF values are not comparable with the published DrumBlender table.
- LSD uses n_fft = 8092 (not 8192), inherited from the DrumBlender implementation; the text states 8092.
- No per-instrument-class results (user decision, 2026-11-05). The promise was removed from Introduction contribution 4 and the TBD from Evaluation; results are reported overall and per pack only.
- GPU: two NVIDIA RTX 3090 (user, 2026-11-05). TBD in the text: control evaluation criteria.

## Related Work: decisions taken in the draft

- Three run-in `\paragraph{}` blocks in the Introduction's order (modeling → generative control → label-free control); each ends with an "In contrast, this work …" sentence.
- No comparison table (user decision, 2026-11-05).
- The KSMI paper is not cited; its reference list supplied Karplus–Strong, Kirby & Sandler, Nagata & Saito, Drysdale et al. 2020, CRASH, StyleWaveGAN, Ramires et al. 2022, Drysdale & Hockman 2024, SoundStream, and DAC.
- Papers already introduced in the Introduction (DrumGAN, Ramires 2020, RAVE, network bending, Aouameur) are not described again; B2 points back to the Introduction instead.
- Ramires et al. 2022 is used as evidence for unsupervised controls, with its own caveat that the preference was not statistically significant.
- Esling et al. 2018 is cited as label-dependent (instrument labels plus human dissimilarity ratings), which is the contrast with label-free control.
- Removed on 2026-11-05 after the implementation spec dropped the β-VAE: the B3 sentence on the β-VAE (Higgins et al.) and PLAUD "coupling such a latent space". PLAUD is still described in the Introduction (P3). `higgins2017beta` stays in eaim.bib but is no longer cited.
- Shier et al. 2025's finding that temporal morphology was poorly mapped is stated as reported; it is not yet tied to this paper's decay argument.
- Length: about 700 words, about 1.6 pages in the single-column PMLR layout (the plan's 0.8-page estimate assumed a denser layout). Kept at this length by user decision (2026-11-05). If the page limit becomes tight later, cut in this order: merge Kirby and Nagata into one sentence, drop the B3 codec sentence, drop the Drysdale & Hockman 2024 sentence, shorten the Karplus–Strong sentence.

## Introduction: decisions taken in the draft

- Paragraph order: general analysis-synthesis and DDSP → percussion structure and the cymbal problem → timbre control → this work. The drum-machine and descriptor-conditioned examples were moved from the percussion-structure paragraph to the control paragraph.
- Register follows the KSMI paper (formal, "In particular" / "Nevertheless" style connectives, hedged claims), but no sentence is copied from it. The CFP warns that self-plagiarism can lead to desk rejection.
- "We" is not used; the draft uses "this paper". The KSMI paper used "we", so switch if preferred.
- The KSMI paper (Kim & Keum, 2026) is non-archival and is not cited anywhere in the paper (decided 2026-11-05). Its reference list is reused as a source of related work.
- "DrumBlender" is the name of the code release (github.com/jorshi/drumblender), not a name used in the paper title; the draft introduces it as "\citet{shier2023differentiable} ... with DrumBlender".
- "NeuroDrum" is not used as a name. The cited paper is Ramires et al. (ICASSP 2020).
- The transient-branch argument is framed with the original paper's own result: the TCN helped membranophone onsets, but cymbals had the largest errors and idiophones were best without the TCN. The cymbal difficulty is therefore placed in the decay representation, not in the onset.
- "A sinusoidal component with a limited number of tracks can therefore explain only part of a cymbal's energy" is an inference from Skare & Abel and Shier et al., not a sentence any source states directly.

## Citations to double-check before submission

- `fletcher1998physics`: confirm the chapter and page that states that percussion modes are generally inharmonic (P2, sentence 2).
- `skare2019realtime`, `liu2024ddspsfx`, `aouameur2019neural`, `nistal2022drumganvst`, `levine1998sines`: page numbers or DOIs were not available and are omitted.
- `caillon2021rave` is arXiv-only. `kotowski2026plaud` and `paek2025learning` are workshop or conference papers with arXiv versions.
- `hayes2024review`: Frontiers lists it as volume 3 (2023), published 11 January 2024; the bib uses year 2024 to match the Frontiers citation format.
- `higgins2017beta` (not cited since 2026-11-05; check only if it returns): author first names come from ML Anthology because OpenReview blocked automated access; check "Loïc" and the middle initials against the PDF.
- `simionato2025sines`: Frontiers lists volume 4 (2024) but the article was published 29 January 2025; year 2025 follows the hayes2024review convention (publication year). Confirm against the Frontiers "cite" box.
- `nagata2025sound`: the drum type is not stated in the abstract (a precursor paper used a snare), so the text says only "drums". Published online in 2024; the print issue is 2025.
- `drysdale2020adversarial`: the paper prints "Maciek Tomczak"; other sources use "Maciej".
- `gao2025scaling`, `singh2026discovering`: pages come from the ICLR proceedings BibTeX export.
- `zeghidour2022soundstream`: the DOI contains 2021 (early access), but the official volume year is 2022.

## Submission checklist (from the CFP and the Overleaf template)

- Main text 8–11 pages excluding references.
- Remove `\appendix` from the main PDF; appendices go in a separate supplementary ZIP.
- `\bibliography{eaim}` replaced the temporary `\section*{References}` heading (done 2026-11-04).
- Add the GenAI-use disclosure (acknowledgements or methodology).
- Keep the camera-ready author block out of the submitted source unless allowed.

## Polish pass (2026-11-05) — exemplar reading and fact fixes

Exemplars read (three parallel readers): DDSP, NoiseBandNet, DDX7, Hayes 2023, the DDSP review, Diaz 2023; DrumBlender, DrumGAN, CRASH, PLAUD, Shier 2024 (Drysdale & Hockman 2024 abstract only); Paek 2025, Singh 2026, Gao 2025, RAVE, Esling 2018, network bending.
Patterns applied: each Intro paragraph ends on the limitation the next one resolves; contributions point to sections; Related Work paragraphs end with a contrast; Method opens with notation and a one-line roadmap of its subsections.

Fact fixes applied (each from the source text):
- DDSP is trained end to end from features of the input audio (f0, loudness), not raw audio; its multi-scale loss sums (not averages) over FFT sizes, so Method now says "combined".
- Hayes review: the difficulty with unpitched/non-harmonic instruments comes from pitch tracking and predominantly harmonic spectra.
- DrumBlender: cymbals had the largest multi-resolution spectral (MSS) errors, not the largest on every metric; its TCN produces the transient from the sinusoidal signal.
- RAVE: compact latent plus faster-than-real-time synthesis; the paper does not claim real-time latent manipulation.
- Paek 2025: SAEs on the latent spaces of audio autoencoders (their SAE is ReLU+L1, not top-k).
- Diaz 2023: predicts differentiable resonators (IIR biquads) from shape, material, and strike position.
- Drysdale & Hockman 2024: only the abstract was readable; the sentence now says latent arithmetic, dimensionality reduction, and visualization. Read the full paper before submission if the sentence stays.
- Gao 2025: SAEs reconstruct activations through a wide layer of sparse features (dictionary framing removed).
- Shier 2024: "without parameter supervision" instead of "without labels".
Not changed: Esling (accurate; ratings were reused from earlier studies), CRASH, PLAUD, DrumGAN, Singh, network bending.
Intro P4 was reordered (task and audience first, then the three components, then the controls); the eigenmode sentence now lives only in Method.
Compile after the pass: 0 errors, 0 warnings, 1 info (Underfull \vbox at a page break inside Method, badness 1910; cosmetic).
