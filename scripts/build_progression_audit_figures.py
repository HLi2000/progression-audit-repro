#!/usr/bin/env python
"""Submission figures for the progression-audit TAE paper (papers/progression-audit-tae/figures/).

Reads ONLY the committed locked panel (reports/progression_audit_panel.json) -- no re-scoring. Okabe-Ito
colourblind-safe palette, light grid, distinct markers per series, serif type. Every figure is authored at
canvas width 5.5 in (the single-column width) so \\includegraphics[width=\\linewidth] renders it 1:1: the font
sizes below are the final print sizes.

Colour semantics (Okabe-Ito, one meaning per colour):
  ORANGE = camera floor / device / shortcut      GREY  = chance / neutral / secondary bars
  BLACK  = reported / published (open markers)    BLUE  = RETFound-LP / VF statistics
  PURPLE = RETFound-FT                             GREEN = GPPM / OCT
  VERMIL = reverts to / at the floor
Endpoints keep order PLR2->PLR3->MD everywhere (markers o / ^ / s, colours blue / vermil / green).

Run:  .venv/bin/python scripts/build_progression_audit_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "papers" / "progression-audit-tae" / "figures"
PANEL = ROOT / "reports" / "progression_audit_panel.json"

# ---- Okabe-Ito palette --------------------------------------------------------------------------------
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermil="#D55E00", purple="#CC79A7", grey="#8a8a8a")
FLOOR = OI["orange"]     # camera floor / device / shortcut
CHANCE = OI["grey"]
PUB = OI["black"]
EPS = ["PLR2", "PLR3", "MD"]
EP_COL = {"PLR2": OI["blue"], "PLR3": OI["vermil"], "MD": OI["green"]}
EP_MK = {"PLR2": "o", "PLR3": "^", "MD": "s"}
WHITELIST = ["fig1_audit", "fig2_device", "fig3_revert", "fig4_circular"]

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300, "figure.dpi": 150,
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 6.5, "axes.titlesize": 7.5, "axes.labelsize": 7.0, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.7, "axes.edgecolor": "#444", "xtick.color": "#333", "ytick.color": "#333",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e9e9e9", "grid.linewidth": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def _panel():
    return json.loads(PANEL.read_text())


def _save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  · {stem}")


def _hdr(ax, letter, msg, pad=5):
    """Shared message-header: bold single letter (mathtext) + short normal-weight message, top-left.
    set_title reserves space ABOVE the axes frame, so it never clashes with the top tick."""
    s = rf"$\bf{{{letter}}}$   {msg}" if letter else msg
    ax.set_title(s, loc="left", pad=pad, fontsize=7.5, color=PUB)


# ============================================================ FIG 1 -- the audit + reported->floor teaser
def fig1_audit(P):
    fig = plt.figure(figsize=(5.5, 2.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.6, 1.0], hspace=0.55)

    # (a) the schematic: three numbered checks on a line, then the prospective forecast set apart
    axa = fig.add_subplot(gs[0]); axa.axis("off"); axa.set_xlim(0, 100); axa.set_ylim(0, 20); axa.grid(False)
    stages = [("1", "Device floor", "reads the camera?"),
              ("2", "Fair split", "fellow eyes together?"),
              ("3", "Circularity", "input in the label?")]
    xs = [12.0, 40.0, 68.0]; xland = 89.0; yline = 13.5
    axa.plot([xs[0] - 4, xs[-1] + 4], [yline, yline], color=OI["grey"], lw=1.0, zorder=1)
    for i, (num, name, q) in enumerate(stages):
        x = xs[i]
        axa.plot(x, yline, "o", ms=16, color=OI["blue"], zorder=2, markeredgecolor="white", markeredgewidth=1.2)
        axa.text(x, yline, num, color="white", fontsize=8.0, fontweight="bold", ha="center", va="center", zorder=3)
        axa.text(x, yline - 4.4, name, color=PUB, fontsize=7.0, fontweight="bold", ha="center", va="top")
        axa.text(x, yline - 8.6, q, color="#666", fontsize=6.0, ha="center", va="top", style="italic")
    axa.plot([xs[-1] + 6, xland - 6], [yline, yline], color=OI["grey"], lw=1.0, ls=(0, (2, 2)), zorder=1)
    axa.plot(xland, yline, "o", ms=16, color="white", markeredgecolor=OI["green"], markeredgewidth=2.0, zorder=2)
    axa.plot(xland, yline, "*", ms=8.0, color=OI["green"], zorder=3)
    axa.text(xland, yline - 4.4, "Prospective\nforecast", color=OI["green"], fontsize=7.0, fontweight="bold",
             ha="center", va="top", linespacing=0.95)
    _hdr(axa, "a", "the audit: three checks, then a forecast")

    # (b) teaser: every reported number reverts from its open-black cluster to the orange floor
    axb = fig.add_subplot(gs[1])
    SA = P["serial_ablation"]["models"]                               # -all = device+split+circularity removed together
    _rev = lambda name: [SA[name]["all"][e][0] for e in EPS]
    rows = [("Huang", [0.71, 0.80, 0.73], _rev("Huang")), ("Fouda", [0.86], [_rev("Fouda")[1]]),
            ("TSDF", [0.87, 0.96, 0.92], _rev("TSDF")), ("DiffSight", [0.79, 0.88, 0.80], _rev("DiffSight")),
            ("GPPM", [0.85, 0.82, 0.69], _rev("GPPM"))]
    axb.axhspan(0.56, 0.60, color=FLOOR, alpha=0.16, zorder=0)
    axb.axhline(0.5, color=CHANCE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for i, (name, rep, rev) in enumerate(rows):
        axb.plot([i, i], [float(np.mean(rev)), float(np.mean(rep))], color="#d8d8d8", lw=0.9, zorder=1)
        for a in rep:
            axb.plot(i, a, "o", color="white", markeredgecolor=PUB, markeredgewidth=1.1, ms=6.2, zorder=4)
        for h in rev:
            axb.plot(i, h, "o", color=OI["vermil"], ms=6.2, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    axb.set_xticks(range(len(rows))); axb.set_xticklabels([r[0] for r in rows])
    axb.set_xlim(-0.6, len(rows) - 0.4); axb.set_ylim(0.25, 1.0)
    axb.set_yticks([0.4, 0.6, 0.8, 1.0])
    axb.set_ylabel("progression AUC", labelpad=2); axb.grid(axis="x", visible=False)
    axb.text(2.5, 0.615, "camera floor", fontsize=6.0, color="#8a5a00", ha="center", va="bottom")
    axb.text(0.5, 0.487, "chance", fontsize=6.0, color=CHANCE, ha="center", va="top")
    handles = [Line2D([], [], marker="o", color="w", markeredgecolor=PUB, markerfacecolor="white",
                      markeredgewidth=1.1, ms=6.4, label="reported"),
               Line2D([], [], marker="o", color="w", markerfacecolor=OI["vermil"], markeredgecolor="white",
                      markeredgewidth=0.6, ms=6.4, label="reverts to the floor")]
    axb.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=2,
               labelspacing=0.25, columnspacing=1.8, handlelength=1.3, handletextpad=0.4)
    _hdr(axb, "b", "every reported number reverts to the floor")
    _save(fig, "fig1_audit")


# ======================================== FIG 2 -- the device spine: camera confound + triangulation floor
def fig2_device(P):
    tri, floor, dev = P["triangulation"], P["floor"], P["device"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(5.5, 2.05),
                                   gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.42})

    # (a) the camera is a usable shortcut: per-camera progression rate + fold callouts
    x = np.arange(3); bw = 0.36
    rt = [dev["rate_topcon"][e] for e in EPS]; rc = [dev["rate_canon"][e] for e in EPS]
    axa.bar(x - bw / 2, rt, width=bw, color=FLOOR, edgecolor="none", label="TOPCON", zorder=3)
    axa.bar(x + bw / 2, rc, width=bw, color=OI["grey"], edgecolor="none", label="Canon", zorder=3)
    for xi, e, t, c in zip(x, EPS, rt, rc):
        if e == "PLR3":            # 14 events -> the ratio is ~1 eye of difference, not meaningfully estimable
            continue
        axa.text(xi, max(t, c) + 0.008, f"{t/c:.1f}$\\times$", ha="center", va="bottom", fontsize=6.5,
                 color="#8a5a00", fontweight="bold")
    axa.set_xticks(x); axa.set_xticklabels(EPS); axa.set_ylim(0, 0.235)
    axa.set_yticks([0.0, 0.1, 0.2]); axa.set_ylabel("progression rate", labelpad=2)
    axa.set_xlim(-0.6, 2.6); axa.grid(axis="x", visible=False)
    axa.legend(loc="upper right", ncol=1, handlelength=1.0, handletextpad=0.5,
               labelspacing=0.35, borderaxespad=0.2)
    _hdr(axa, "a", "the camera is a shortcut")

    # (b) triangulation forest: every baseline-fundus estimate sits at its endpoint's floor
    est = [("floor", FLOOR, "D", None),
           ("retfound_lp", OI["blue"], "o", "ci"),
           ("gppm_cfp", OI["green"], "s", None)]
    offs = [0.28, 0.10, -0.10, -0.28]
    gc = {e: 2 - i for i, e in enumerate(EPS)}
    for e in EPS:
        c = gc[e]; f = floor[e]
        axb.plot([f, f], [c - 0.40, c + 0.40], color=FLOOR, lw=1.4, ls=(0, (3, 2)), zorder=2)
        for (key, col, mk, kind), dy in zip(est, offs):
            y = c + dy
            if kind == "ci":
                auc, lo, hi = tri[e][key]
                axb.plot([lo, hi], [y, y], color=col, lw=1.0, zorder=3, solid_capstyle="round")
                axb.plot(auc, y, mk, color=col, ms=5.6, zorder=4, markeredgecolor="white", markeredgewidth=0.6)
            else:
                v = tri[e][key]
                auc = v if not isinstance(v, list) else v[0]
                axb.plot(auc, y, mk, color=col, ms=6.0, zorder=4, markeredgecolor="white", markeredgewidth=0.6)
    axb.axvline(0.5, color=CHANCE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axb.set_yticks(list(gc.values())); axb.set_yticklabels(EPS)
    for t, e in zip(axb.get_yticklabels(), EPS):
        t.set_color(EP_COL[e]); t.set_fontweight("bold")
    axb.set_ylim(-0.65, 2.65); axb.set_xlim(0.35, 0.80)
    axb.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    axb.set_xlabel("baseline-fundus AUC  (covariate-adjusted, 95% CI)", labelpad=2)
    axb.grid(axis="y", visible=False)
    axb.text(0.5, -0.60, "chance", color=CHANCE, fontsize=6.0, ha="center", va="center")
    handles = [Line2D([], [], marker="D", color="w", markerfacecolor=FLOOR, ms=6, label="camera floor"),
               Line2D([], [], marker="o", color="w", markerfacecolor=OI["blue"], ms=6, label="RETFound-CFP probe"),
               Line2D([], [], marker="s", color="w", markerfacecolor=OI["green"], ms=6, label="GPPM CFP-only")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.14),
               columnspacing=1.2, handletextpad=0.35, frameon=False)
    _hdr(axb, "b", "nothing beats the camera floor")
    _save(fig, "fig2_device")


# ============ FIG 3 -- reproduce, then revert: one row per model, claimed vs each shortcut controlled
def fig3_revert(P):
    sa = P["serial_ablation"]; order = sa["order"]; flval = P["floor"]
    STAGES = ["published", "reproduced", "device", "fair_split", "circular", "all"]
    SLAB = ["published", "reproduced", "device", "fair split", "circularity", "all three"]
    OFF = {"PLR2": -0.20, "PLR3": 0.0, "MD": 0.20}

    def cval(m, st, ep):
        if ep not in m.get("published", {}):
            return None                                     # endpoint not reported (Fouda: PLR3 only)
        c = m.get(st)
        if c is None:
            return "na"                                     # circularity undefined (no serial window)
        if st == "published":
            v = m["published"].get(ep)
            return None if v is None else (v, v, v)
        v = c.get(ep)
        return None if v is None else (tuple(v) if isinstance(v, list) else (v, v, v))

    fig, axes = plt.subplots(len(order), 1, sharex=True, figsize=(5.5, 3.55))
    for ax, name in zip(axes, order):
        m = sa["models"][name]
        ax.axhspan(0.56, 0.60, color=FLOOR, alpha=0.16, zorder=0)
        ax.axhline(0.5, color=CHANCE, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.axvline(1.5, color="#c4c4c4", lw=1.0, zorder=1)                 # claimed | audited gutter
        for xi, st in enumerate(STAGES):
            na = False
            for ep in EPS:
                r = cval(m, st, ep)
                if r is None:
                    continue
                if r == "na":
                    na = True; continue
                a, lo, hi = r
                x = xi + OFF[ep]; has_ci = hi > lo
                clears = (lo > flval[ep]) if has_ci else (a > flval[ep])
                col = OI["blue"] if clears else OI["vermil"]
                if has_ci:
                    ax.plot([x, x], [lo, hi], color=col, lw=0.9, alpha=0.45, zorder=2, solid_capstyle="round")
                face = "white" if (has_ci and clears) else col
                ax.plot(x, a, EP_MK[ep], ms=4.4, zorder=5, markeredgewidth=1.0, color=face, markeredgecolor=col)
            if na:
                ax.text(xi, 0.82, "n/a", fontsize=6.0, color="#a0a0a0", ha="center", va="center", style="italic")
        ax.set_ylim(0.22, 1.05); ax.set_xlim(-0.5, len(STAGES) - 0.5)
        ax.set_yticks([0.3, 0.6, 0.9]); ax.tick_params(labelsize=6.0)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("AUC", fontsize=6.5, labelpad=2)
    axes[-1].set_xticks(range(len(STAGES)))
    axes[-1].set_xticklabels(SLAB, rotation=28, ha="right", rotation_mode="anchor")

    handles = [Line2D([], [], marker=EP_MK[e], color="#555", ls="none", ms=4.6, label=e) for e in EPS] + [
        Line2D([], [], marker="o", color="w", markerfacecolor=OI["blue"], markeredgecolor=OI["blue"],
               ms=5.4, label="beats the camera"),
        Line2D([], [], marker="o", color="w", markerfacecolor=OI["vermil"], markeredgecolor=OI["vermil"],
               ms=5.4, label="no better than the camera")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5,
               columnspacing=1.3, handletextpad=0.4)
    fig.subplots_adjust(hspace=0.30, left=0.15, right=0.985, top=0.87, bottom=0.17)

    # top strip (figure coords): CLAIMED / AUDITED column-group labels, well clear of the 0.9 tick
    fig.text(0.29, 0.915, "CLAIMED", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#555")
    fig.text(0.70, 0.915, "AUDITED  (each shortcut controlled, one at a time)", ha="center", va="center",
             fontsize=6.5, fontweight="bold", color="#555")
    # model names in the left margin, vertically centred on each row (never over the dots)
    fig.canvas.draw()
    for ax, name in zip(axes, order):
        pos = ax.get_position()
        fig.text(0.018, (pos.y0 + pos.y1) / 2, name, ha="left", va="center", fontweight="bold", fontsize=7.0)
    _save(fig, "fig3_revert")


# ============================================================ FIG 4 -- circularity dial + span + ablation
def fig4_circular(P):
    circ = P["circularity"]; dial = circ["dial"]; tv = P["two_vft"]
    pb_ref = P["prospective_baseline"]["dial_ref"]   # well-powered baseline prospective (~0.59, at the floor)
    korder = [1, 2, 3, 4, 5, 6, "full"]; xp = list(range(len(korder)))
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(5.5, 2.2),
                                        gridspec_kw={"width_ratios": [1.3, 1.2, 0.92], "wspace": 0.58})
    ep_handles = [Line2D([], [], color=EP_COL[e], marker=EP_MK[e], ms=3.6, lw=1.5, label=e) for e in EPS]

    # (a) the dial: AUC vs input window; each endpoint's label recovers from its own VF statistics
    curves = {e: [next(r for r in dial[e] if r["k"] == k)["auc"] for k in korder] for e in EPS}
    # the circularity gap: shade between the honest prospective floor and the circular curves, once the
    # label-defining slope becomes computable (k>=3, x>=2)
    top = [max(curves[e][i] for e in EPS) for i in range(len(korder))]
    xg = [xp[i] for i in range(len(xp)) if xp[i] >= 2]; tg = [top[i] for i in range(len(xp)) if xp[i] >= 2]
    axa.fill_between(xg, pb_ref, tg, color=OI["sky"], alpha=0.18, zorder=1, linewidth=0)
    for e in EPS:
        axa.plot(xp, curves[e], "-", color=EP_COL[e], lw=1.5, marker=EP_MK[e], ms=3.8, zorder=3)
    axa.axhline(pb_ref, color=OI["grey"], lw=1.1, ls=(0, (5, 2)), zorder=2)
    axa.axhline(0.5, color=CHANCE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axa.axvline(2, color=OI["grey"], lw=0.8, ls=":", zorder=1)
    axa.text(4.5, 0.705, "circularity\ngap", fontsize=6.0, color="#2b6b8a", ha="center", va="center",
             fontweight="bold", linespacing=0.95)
    axa.text(2.65, 0.945, "slope + $p$ computable", fontsize=6.0, color="#555", va="top", ha="left")
    axa.text(6.32, pb_ref - 0.014, "prospective floor 0.59", fontsize=6.0, color="#555", va="top", ha="right")
    axa.text(6.32, 0.487, "chance", fontsize=6.0, color=CHANCE, va="top", ha="right")
    axa.set_xticks(xp); axa.set_xticklabels([str(k) for k in korder])
    axa.set_xlim(-0.35, 6.4); axa.set_ylim(0.43, 0.97)
    axa.set_xlabel("input visits (VF statistics)", labelpad=2)
    axa.set_ylabel("AUC vs the fixed label", labelpad=2)
    axa.legend(handles=ep_handles, loc="upper left", handlelength=1.4, handletextpad=0.5,
               labelspacing=0.25, borderaxespad=0.3)
    _hdr(axa, "a", "the label from its statistics")

    # (b) the two-field model: a VF pair recovers the label only as the 2nd test comes later
    follows = tv["follows"]; spans = [tv["reproduce"][k]["mean_span"] for k in follows]
    for e in EPS:
        ys = [tv["reproduce"][k][e][0] for k in follows]
        los = [tv["reproduce"][k][e][1] for k in follows]; his = [tv["reproduce"][k][e][2] for k in follows]
        dx = {"PLR2": -0.03, "PLR3": 0.0, "MD": 0.03}[e]
        axb.errorbar([s + dx for s in spans], ys, yerr=[np.subtract(ys, los), np.subtract(his, ys)],
                     fmt="none", ecolor=EP_COL[e], elinewidth=0.7, capsize=1.3, alpha=0.5, zorder=2)
        axb.plot(spans, ys, "-", color=EP_COL[e], lw=1.5, marker=EP_MK[e], ms=3.8, zorder=3)
    axb.axhline(pb_ref, color=OI["grey"], lw=1.1, ls=(0, (5, 2)), zorder=2)
    axb.axhline(0.5, color=CHANCE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axb.text(3.05, pb_ref - 0.012, "prospective floor 0.59", fontsize=6.0, color="#555", va="top", ha="right")
    axb.text(3.05, 0.487, "chance", fontsize=6.0, color=CHANCE, va="top", ha="right")
    axb.set_xticks([1, 2, 3]); axb.set_xlim(0.5, 3.1); axb.set_ylim(0.36, 0.94)
    axb.set_xlabel("span between the two fields (yr)", labelpad=2)
    axb.set_ylabel("AUC vs the fixed label", labelpad=2)
    axb.legend(handles=ep_handles, loc="upper left", handlelength=1.4, handletextpad=0.5,
               labelspacing=0.3, borderaxespad=0.4)
    _hdr(axb, "b", "span between the two fields")

    # (c) statistic ablation: the label-defining slope + its p-value carry the 0.9
    # ablation lists are ordered [PLR2, PLR3, MD]; the dial narrative is MD (index 2), so read MD to match the 0.90 peak
    ab = circ["ablation"]; items = list(ab.items())
    labels = [k for k, _ in items]; vals = [v[2] for _, v in items]
    yb = np.arange(len(labels))[::-1]
    defining = {"VF slope", "slope p-value"}
    cols = [OI["blue"] if l in defining else OI["grey"] for l in labels]
    axc.barh(yb, vals, height=0.60, color=cols, edgecolor="none", zorder=3)
    for yi, v in zip(yb, vals):
        axc.text(v + 0.015, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.0, color=PUB)
    axc.axvline(0.5, color=CHANCE, lw=0.9, ls=(0, (4, 3)), zorder=2)
    axc.set_yticks(yb); axc.set_yticklabels(labels, fontsize=6.0); axc.grid(axis="y", visible=False)
    axc.set_xlim(0.45, 1.05); axc.set_xticks([0.5, 0.7, 0.9]); axc.set_ylim(-0.7, len(labels) - 0.3)
    axc.set_xlabel(r"AUC $\rightarrow$ MD label", labelpad=2)
    _hdr(axc, "c", "which statistic drives it")
    _save(fig, "fig4_circular")


def _clean_stale():
    if not FIGDIR.exists():
        return
    keep = set()
    for s in WHITELIST:
        keep |= {f"{s}.pdf", f"{s}.png", f"{s}-crop.pdf"}
    for f in FIGDIR.iterdir():
        if f.suffix in (".pdf", ".png") and f.name not in keep:
            f.unlink(); print(f"  (removed stale {f.name})")


def main():
    P = _panel()
    print("building progression-audit figures ->", FIGDIR)
    fig1_audit(P); fig2_device(P); fig3_revert(P); fig4_circular(P)
    _clean_stale()
    print("done.")


if __name__ == "__main__":
    main()
