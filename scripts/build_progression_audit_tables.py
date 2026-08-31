#!/usr/bin/env python
"""Auto-generate the appendix tables for the progression-audit TAE paper from the locked data panel.

Reads reports/progression_audit_panel.json and writes papers/progression-audit-tae/appendix_tables.tex
(booktabs + the \\ci{lo}{hi} macro defined in main.tex). No re-scoring. Mirrors the sibling OCT paper's
table builder. table1.tex (the prior-claims audit map) is hand-curated, not generated here.

Run:  .venv/bin/python scripts/build_progression_audit_tables.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "reports" / "progression_audit_panel.json"
OUT = ROOT / "papers" / "progression-audit-tae" / "appendix_tables.tex"
EPS = ["PLR2", "PLR3", "MD"]


def ci(v):
    return f" \\ci{{{v[1]:.2f}}}{{{v[2]:.2f}}}"


def tbl_collapse(P):
    col = P["collapse"]; models = col["models"]; floor = col["floor"]; pub = col["published"]
    L = [r"\begin{table}[h]", r"\caption{\textbf{Reproduce-then-revert (Check~2).} Fine-tuning the architectures the "
         r"GRAPE/Fouda papers use, with their imbalance recipe, under the published eye-level protocol reaches the "
         r"published range (the seed distribution's upper tail brackets the single published numbers), then reverts "
         r"to the training-free camera-only floor under a patient-grouped split. Eye-level entries are median (max) "
         r"over 5 seeds; grouped entries are patient-grouped five-fold pooled OOF. All values are AUC.}",
         r"\label{tab:collapse}", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l" + "cc" * 3 + "}", r"\toprule",
         r" & \multicolumn{2}{c}{PLR2 (40 ev)} & \multicolumn{2}{c}{PLR3 (14 ev)} & \multicolumn{2}{c}{MD (27 ev)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Model & eye-level & grouped & eye-level & grouped & eye-level & grouped \\", r"\midrule"]
    for m in models:
        cells = []
        for e in EPS:
            med, mx, hon = models[m][e]
            cells.append(f"{med:.2f} ({mx:.2f})")
            cells.append(f"{hon:.2f}")
        L.append(f"{m} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\textit{published (lit.)} & \multicolumn{2}{c}{" + f"{pub['PLR2']:.2f}" +
             r"} & \multicolumn{2}{c}{" + f"{pub['PLR3']:.2f}" + r" $\cdot$ " + f"{col['fouda_plr3']:.2f} ens" +
             r"} & \multicolumn{2}{c}{" + f"{pub['MD']:.2f}" + r"} \\")
    L.append(r"\textit{camera-only floor} & \multicolumn{2}{c}{" + f"{floor['PLR2']:.2f}" +
             r"} & \multicolumn{2}{c}{" + f"{floor['PLR3']:.2f}" +
             r"} & \multicolumn{2}{c}{" + f"{floor['MD']:.2f}" + r"} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def tbl_ladder(P):
    lad = P["honest_ladder"]; reps = lad["reps"]; fl = lad["floors"]
    L = [r"\begin{table}[h]", r"\caption{\textbf{Representation ladder (Check~1).} "
         r"Covariate-adjusted, patient-grouped repeated-OOF AUC [95\% CI] of frozen linear probes over eight "
         r"representation/crop combinations, beside the training-free camera-only floor. Every representation's interval "
         r"includes or sits below the floor on all three endpoints, except the lone ConvNeXt-B MD point whose lower "
         r"bound clears it; that crossing does not replicate on ConvNeXt-L or under fine-tuning (\S\ref{sec:findings}).}",
         r"\label{tab:ladder}", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Representation (frozen LP, covariate-adjusted) & PLR2 & PLR3 & MD \\", r"\midrule"]
    for rep, d in reps.items():
        L.append(f"{rep} & " + " & ".join(f"{d[e]['aaroc'][0]:.3f}" + ci(d[e]['aaroc']) for e in EPS) + r" \\")
    L.append(r"\midrule")
    L.append(r"\textit{camera-only floor} & " + " & ".join(f"{fl[e]:.3f}" for e in EPS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def tbl_audit_grid(P):
    """Per-model audited AUCs [95% CI] behind Fig. fig:revert (the serial_ablation grid)."""
    sa = P["serial_ablation"]; order = sa["order"]; models = sa["models"]
    stages = [("device", "device"), ("fair_split", "fair split"), ("circular", "circularity"), ("all", "all three")]

    def cell(m, st, ep):
        if ep not in m.get("published", {}):
            return "--"                                     # endpoint the method does not report (Fouda: PLR3 only)
        c = m.get(st)
        if c is None:
            return r"\textit{n/a}"                          # control does not apply (no serial window)
        v = c.get(ep)
        if v is None:
            return "--"
        return (f"{v[0]:.2f}" + ci(v)) if isinstance(v, list) else f"{v:.2f}"

    L = [r"\begin{table}[h]", r"\caption{\textbf{Per-model audited AUCs [95\% CI] behind Fig.~\ref{fig:revert}.} "
         r"Each reported number, reproduced by its own architecture, once each shortcut is controlled one at a time: "
         r"the device (a covariate-adjusted AUC), a patient-grouped split, a non-circular input, and all three together. "
         r"\emph{n/a}: the control does not apply, because a single- or two-image input has no serial window to leak; "
         r"Fouda reports PLR3 only. Compare against the camera floor (\num{0.561}/\num{0.600}/\num{0.580}): no audited "
         r"read clears it. GPPM's device and split reads are the highest of the five, but their intervals include the "
         r"floor.}",
         r"\label{tab:auditgrid}", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{ll ccc}", r"\toprule",
         r"Model & Control & PLR2 & PLR3 & MD \\", r"\midrule"]
    for i, name in enumerate(order):
        m = models[name]
        for j, (st, lab) in enumerate(stages):
            head = name if j == 0 else ""
            L.append(f"{head} & {lab} & " + " & ".join(cell(m, st, e) for e in EPS) + r" \\")
        if i < len(order) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def tbl_dial(P):
    """The circularity dial behind Fig. fig:circular: AUC vs input window, and the per-statistic ablation."""
    circ = P["circularity"]; dial = circ["dial"]; ab = circ["ablation"]
    korder = [1, 2, 3, 4, 5, 6, "full"]

    def dval(e, k):
        r = next((r for r in dial[e] if r["k"] == k), None)
        return f"{r['auc']:.2f}" if r else "--"

    L = [r"\begin{table}[h]", r"\caption{\textbf{The circularity dial behind Fig.~\ref{fig:circular}.} "
         r"\emph{Top:} out-of-fold AUC for predicting the fixed label from the visual-field statistics of the first "
         r"$k$ visits, climbing to \num{0.90} with the jump at $k{=}3$, when a slope and its $p$-value first become "
         r"computable. \emph{Bottom:} full-series AUC from each single statistic against the MD label; the "
         r"label-defining slope and its $p$-value carry the \num{0.90}.}",
         r"\label{tab:dial}", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{l" + "c" * len(korder) + "}", r"\toprule",
         r"$k$ input visits & " + " & ".join(str(k) for k in korder) + r" \\", r"\midrule"]
    for e in EPS:
        L.append(f"{e} & " + " & ".join(dval(e, k) for k in korder) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\vspace{4pt}", r"\centering",
          r"\begin{tabular}{l" + "c" * len(ab) + "}", r"\toprule"]
    keys = list(ab.keys())
    L.append(r"Single statistic & " + " & ".join(k.replace("#", r"\#").replace("p-value", r"$p$-value") for k in keys) + r" \\")
    L.append(r"\midrule")
    # ablation lists are ordered [PLR2, PLR3, MD]; the row is the MD label, so read MD (index 2)
    L.append(r"MD-label AUC & " + " & ".join(f"{ab[k][2]:.2f}" for k in keys) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def tbl_endpoint_sens(P):
    es = P["endpoint_sensitivity"]; floor = P["floor"]
    mods = [("VF", "VF"), ("OCT", "OCT-RNFL"), ("fundus_morph", "fundus (morph.)"),
            ("fundus_FM", "fundus (FM)"), ("IOP", "IOP")]
    L = [r"\begin{table}[h]", r"\caption{\textbf{Endpoint-sensitivity of the modality ranking (Finding~5).} "
         r"Single-modality camera-residualized repeated-OOF AUC [95\% CI] per endpoint. The leading modality "
         r"(\textbf{bold}) changes with the label: fundus for PLR2, VF for PLR3, OCT for MD. Cross-label rank "
         r"agreement is near zero (mean Spearman $\rho\approx0.10$), and the three labels themselves agree on only "
         r"2.5\% of eyes ($\kappa$ 0.12--0.52). Fundus~(FM) is a frozen RETFound-CFP disc-crop embedding; "
         r"fundus~(morph.) is guideline disc morphometry.}",
         r"\label{tab:endpoint}", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Modality & PLR2 (40 ev) & PLR3 (14 ev) & MD (27 ev) \\", r"\midrule"]
    for mk, mlab in mods:
        cells = []
        for e in EPS:
            v = es[e][mk]
            s = f"{v[0]:.3f}" + ci(v)
            if es[e]["leader"] == mk:
                s = r"\textbf{" + f"{v[0]:.3f}" + "}" + ci(v)
            cells.append(s)
        L.append(f"{mlab} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\textit{camera-only floor} & " + " & ".join(f"{floor[e]:.3f}" for e in EPS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    P = json.loads(PANEL.read_text())
    parts = ["% auto-generated by scripts/build_progression_audit_tables.py -- do not edit by hand",
             tbl_collapse(P), "", tbl_ladder(P), "", tbl_audit_grid(P), "", tbl_dial(P), ""]
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)  — 4 tables (collapse, ladder, audit-grid, dial)")


if __name__ == "__main__":
    main()
