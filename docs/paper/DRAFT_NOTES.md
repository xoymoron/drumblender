# EAIM 2027 draft notes

Collaboration notes kept outside the manuscript. Nothing in this file goes into `eaim.tex`.

## Files

- `intro.tex`: the `\section{Introduction}` block of Overleaf `eaim.tex`.
- `eaim.bib`: entries appended to Overleaf `eaim.bib` after the template's two sample entries.
- Applied to Overleaf on 2026-11-04 and compiled with 0 errors and 0 warnings. Overleaf is now the source of truth; edits made there are not mirrored back to these files.

## Introduction: open placeholders

| Location | Placeholder | Depends on |
|---|---|---|
| P4, sentence 4 | how neural components complement the signal-processing components | final synthesis architecture |
| Contribution 2 | synthesis architecture | final synthesis architecture |
| Contribution 3 | control mechanism learned without timbre labels | final control method |
| Contribution 4 | control evaluation criteria | control experiments actually run |

Do not fill these with results or properties that have not been implemented and measured.

## Introduction: decisions taken in the draft

- Paragraph order: general analysis-synthesis and DDSP → percussion structure and the cymbal problem → timbre control → this work. The drum-machine and descriptor-conditioned examples were moved from the percussion-structure paragraph to the control paragraph.
- Register follows the KSMI paper (formal, "In particular" / "Nevertheless" style connectives, hedged claims), but no sentence is copied from it. The CFP warns that self-plagiarism can lead to desk rejection.
- "We" is not used; the draft uses "this paper". The KSMI paper used "we", so switch if preferred.
- The KSMI paper (Kim & Keum, 2026) is not cited in the Introduction. Cite it in Related Work in the third person.
- "DrumBlender" is the name of the code release (github.com/jorshi/drumblender), not a name used in the paper title; the draft introduces it as "\citet{shier2023differentiable} ... with DrumBlender".
- "NeuroDrum" is not used as a name. The cited paper is Ramires et al. (ICASSP 2020).
- The transient-branch argument is framed with the original paper's own result: the TCN helped membranophone onsets, but cymbals had the largest errors and idiophones were best without the TCN. The cymbal difficulty is therefore placed in the decay representation, not in the onset.
- "A sinusoidal component with a limited number of tracks can therefore explain only part of a cymbal's energy" is an inference from Skare & Abel and Shier et al., not a sentence any source states directly.

## Citations to double-check before submission

- `fletcher1998physics`: confirm the chapter and page that states that percussion modes are generally inharmonic (P2, sentence 2).
- `skare2019realtime`, `liu2024ddspsfx`, `aouameur2019neural`, `nistal2022drumganvst`, `levine1998sines`: page numbers or DOIs were not available and are omitted.
- `caillon2021rave` is arXiv-only. `kotowski2026plaud` and `paek2025learning` are workshop or conference papers with arXiv versions.
- `hayes2024review`: Frontiers lists it as volume 3 (2023), published 11 January 2024; the bib uses year 2024 to match the Frontiers citation format.

## Submission checklist (from the CFP and the Overleaf template)

- Main text 8–11 pages excluding references.
- Remove `\appendix` from the main PDF; appendices go in a separate supplementary ZIP.
- `\bibliography{eaim}` replaced the temporary `\section*{References}` heading (done 2026-11-04).
- Add the GenAI-use disclosure (acknowledgements or methodology).
- Keep the camera-ready author block out of the submitted source unless allowed.
