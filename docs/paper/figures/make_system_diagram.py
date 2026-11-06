"""Figure 1: system diagram of the joint modal, noise, and transient model.

Run: python make_system_diagram.py  ->  figure1.png (600 dpi) / figure1.pdf
"""
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 7,
    "pdf.fonttype": 42,
})

INK = "#262626"
LINE = "#3d3d3d"
DASH = (0, (3, 2))
STYLE = {
    "input": ("#f1f3f6", "#6c7a89"),
    "shared": ("#e8e1f2", "#6a5a8c"),
    "modal": ("#d9e8f5", "#3f6f9e"),
    "noise": ("#dcefdc", "#4f8a55"),
    "transient": ("#fbe3d3", "#c06a3a"),
    "sae": ("#ffffff", "#6a5a8c"),
}

fig, ax = plt.subplots(figsize=(7.0, 2.75))
ax.set_xlim(0, 21.1)
ax.set_ylim(1.0, 8.0)
ax.axis("off")


def box(x, y, w, h, title, sub=None, kind="input", dashed=False, bold=False):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        fc=fill, ec=edge, lw=1.2 if bold else 0.85, ls=DASH if dashed else "-"))
    cx, cy = x + w / 2, y + h / 2
    if sub is None:
        ax.text(cx, cy, title, ha="center", va="center", color=INK, fontsize=8.5 if bold else 6.6)
    else:
        ty = y + h - 0.36 if h > 1.5 else y + h * 0.67
        sy = y + h * 0.45 if h > 1.5 else y + h * 0.30
        ax.text(cx, ty, title, ha="center", va="center", color=INK, fontsize=6.6)
        ax.text(cx, sy, sub, ha="center", va="center", color=INK, fontsize=5.6, linespacing=1.4)
    return SimpleNamespace(l=x, r=x + w, b=y, t=y + h, cx=cx, cy=cy)


def path(points, arrow=True, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=LINE, lw=0.75, ls=DASH if dashed else "-", solid_capstyle="butt")
    if arrow:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        norm = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-9)
        tail = (x1 - 0.2 * (x1 - x0) / norm, y1 - 0.2 * (y1 - y0) / norm)
        ax.annotate("", xy=points[-1], xytext=tail,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16",
                                    color=LINE, lw=0.75, shrinkA=0, shrinkB=0))


def label(x, y, s, ha="center", va="bottom", size=6.2):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=INK)


# ---------------------------------------------------------------- analysis and encoders
ana = box(0.95, 5.75, 3.1, 1.1, "Modal analysis", "CQT + long/short STFT")
menc = box(4.75, 5.75, 3.2, 1.1, "Modal encoder", r"tokens $u_k$ + set pooling")
wenc = box(0.95, 3.5, 3.1, 1.1, "Waveform encoder", r"input $[x,\ r_0]$, hop 128")

ax.text(0.2, 5.18, r"$x$", ha="center", va="center", fontsize=9, color=INK)
path([(0.38, 5.18), (0.6, 5.18), (0.6, ana.cy), (ana.l, ana.cy)])
path([(0.6, 5.18), (0.6, wenc.cy), (wenc.l, wenc.cy)])

path([(ana.r, ana.cy), (menc.l, ana.cy)])
label((ana.r + menc.l) / 2, ana.cy + 0.07, r"$P_0$")
path([(ana.cx, ana.b), (ana.cx, wenc.t)])
label(ana.cx + 0.12, (ana.b + wenc.t) / 2, r"$r_0 = x - m_0$", ha="left", va="center")

# ---------------------------------------------------------------- shared representation
trunk = box(8.6, 3.5, 1.6, 3.35, "Fusion", "GRU\n" + r"2$\times$256" + "\n+ FFN", kind="shared")
path([(menc.r, menc.cy), (trunk.l, menc.cy)])
path([(wenc.r, wenc.cy), (trunk.l, wenc.cy)])

hb = box(10.5, 4.1, 0.85, 0.9, r"$h$", kind="shared", bold=True)
path([(trunk.r, hb.cy), (hb.l, hb.cy)])

sae = box(4.4, 1.25, 6.7, 1.2, r"Post-hoc temporal SAE on standardized $h$",
          "$h$′" + r"$= h + \boldsymbol{\sigma} \odot [\mathbf{d}_i * (\alpha\, c_i)]$",
          kind="sae", dashed=True)
path([(hb.cx, sae.t), (hb.cx, hb.b)], dashed=True)
label(hb.cx - 0.1, 3.3, "edit", ha="right", va="center", size=5.8)

# ---------------------------------------------------------------- modal branch
mhead = box(12.0, 6.25, 3.0, 1.1, "Modal correction", r"bounded $\Delta\log\nu,\ \Delta\log A$",
            kind="modal")
msyn = box(15.55, 6.25, 2.8, 1.1, "Sinusoidal synth", r"analyzed $\phi_k$ kept", kind="modal")
path([(mhead.r, mhead.cy), (msyn.l, mhead.cy)])
label((mhead.r + msyn.l) / 2, mhead.cy + 0.07, r"$\hat P$")
path([(7.3, menc.t), (7.3, 7.7), (12.6, 7.7), (12.6, mhead.t)])
label(10.0, 7.73, r"$u_k$, $P_0$")

# ---------------------------------------------------------------- noise branch
nhead = box(12.0, 4.0, 3.0, 1.1, "Noise amplitudes", r"from $h$, re-pooled $\hat P$", kind="noise")
nsyn = box(15.55, 4.0, 2.8, 1.1, "256 noise bands", "NoiseBandNet-style", kind="noise")
# h feeds the three heads symmetrically: straight to noise, through one bus to modal and pooling
BUS = 11.7
path([(hb.r, hb.cy), (nhead.l, nhead.cy)])
path([(BUS, hb.cy), (BUS, mhead.cy), (mhead.l, mhead.cy)])
ax.add_patch(Circle((BUS, hb.cy), 0.05, fc=LINE, ec=LINE))
path([(14.4, mhead.b), (14.4, nhead.t)])
label(14.48, (mhead.b + nhead.t) / 2, r"$\hat P$", ha="left", va="center")
path([(nhead.r, nhead.cy), (nsyn.l, nhead.cy)])
label((nhead.r + nsyn.l) / 2, nhead.cy + 0.07, r"$a_b$")

# ---------------------------------------------------------------- transient branch
pool = box(12.0, 1.75, 3.0, 1.1, "Attention pooling", r"$\mathbf{s}$ ($\mathbf{s}_{\mathrm{ref}}$ during edits)",
           kind="transient")
tcn = box(15.55, 1.75, 2.8, 1.1, "Causal TCN", r"10 blocks, FiLM$(\mathbf{s})$", kind="transient")
path([(BUS, hb.cy), (BUS, pool.cy), (pool.l, pool.cy)])
path([(pool.r, pool.cy), (tcn.l, pool.cy)])

# ---------------------------------------------------------------- outputs
plus = (19.85, nsyn.cy)
ax.add_patch(Circle(plus, 0.22, fc="white", ec=LINE, lw=0.8))
ax.text(*plus, "+", ha="center", va="center", fontsize=9, color=INK)
path([(msyn.r, msyn.cy), (plus[0], msyn.cy), (plus[0], plus[1] + 0.22)])
path([(nsyn.r, nsyn.cy), (plus[0] - 0.22, nsyn.cy)])
path([(tcn.r, tcn.cy), (plus[0], tcn.cy), (plus[0], plus[1] - 0.22)])
for y, s in [(msyn.cy, "S"), (nsyn.cy, "N"), (tcn.cy, "T")]:
    ax.text(19.3, y + 0.08, rf"${s}$", fontsize=8, color=INK, ha="center", va="bottom")

# stop-gradient copy of S into the TCN, kept left of the S/N/T labels
path([(18.75, msyn.cy), (18.75, 3.42), (17.55, 3.42), (17.55, tcn.t)], dashed=True)
label(17.63, 3.13, r"$\mathrm{sg}(S)$", ha="left", va="center")

path([(plus[0] + 0.22, plus[1]), (20.95, plus[1])])
ax.text(20.65, plus[1] + 0.28, r"$\hat x$", fontsize=9, color=INK, ha="center")
ax.text(20.6, plus[1] - 0.45, r"$\times g$", fontsize=6.2, color=INK, ha="center")

out = Path(__file__).with_name("figure1")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
