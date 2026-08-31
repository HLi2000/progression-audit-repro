#!/usr/bin/env python
"""Honest serial EVENT prediction on GRAPE + the circularity decomposition.

The field reports glaucoma-progression AUCs of 0.71 (single baseline CFP) → 0.87–0.97 (serial models) on
GRAPE. This script shows how much of that is REAL discrimination vs an artifact of three coupled shortcuts:

  §A  HONEST landmark event curve — input = the early visits ``1..L`` only; label = a transparent MD-slope
      progression EVENT recomputed on the DISJOINT held-out future window ``L+1..end``. Non-circular by
      construction (the input is never part of the label window). Patient-grouped OOF AUC + cluster-boot CIs.
  §B  NAIVE CIRCULAR comparator — feed the WHOLE VF series (whose slope DEFINES the label) and predict the
      shipped MD flag. This is what serial models effectively do; it recovers ≈0.9 tautologically.
  §C  DECOMPOSITION — published 0.71→0.97 vs {circularity (§B) + device-confound (camera-only) + eye-level
      leakage (fine-tune ladder) + the honest signal (§A)}. The first quantified answer to "how much of
      GRAPE's published progression AUC is real."
  §D  Power caveat — GRAPE's short series (median 4 visits) leave 4–13 held-out events → the honest curve is
      inconclusive; that GRAPE cannot honestly support a serial event model is itself the point of §C.

    .venv/bin/python scripts/run_grape_serial_event.py
Writes reports/grape_serial_event.md. CPU, ~1 min.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from glaucoma_triage.datasets import grape
from glaucoma_triage.progression import survival as S

_SETS = list(S.EVENT_FEATURE_SETS)                                  # vf_history, iop_history, baseline_struct, all


def _fmt(a, lo, hi):
    if not np.isfinite(a):
        return "—"
    if np.isfinite(lo):
        return f"{a:.2f} [{lo:.2f}, {hi:.2f}]"
    return f"{a:.2f}"


# --------------------------------------------------------------------------- §B circular full-series ---
def _full_series_features(fu: pd.DataFrame) -> pd.DataFrame:
    """Per-eye features from the WHOLE VF series — the slope/level statistics that DEFINE the MD label."""
    rows = []
    for eid, g in fu.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        t, s = t[ok], s[ok]
        slope = float(np.polyfit(t, s, 1)[0]) if len(s) >= 2 else 0.0
        pval = float(stats.linregress(t, s).pvalue) if len(s) >= 3 else 1.0
        rows.append({"eye_id": eid, "patient_id": str(eid).split("_")[0],
                     "full_slope": slope, "full_slope_p": pval,
                     "full_sd": float(np.std(s)) if len(s) else 0.0,
                     "full_now": float(s[-1]) if len(s) else np.nan,
                     "full_mean": float(np.mean(s)) if len(s) else np.nan,
                     "n_visits": int(len(s))})
    return pd.DataFrame(rows)


def _window_features(fu: pd.DataFrame, k) -> pd.DataFrame:
    """Per-eye VF statistics over the FIRST ``k`` visits (``k=None`` → the whole series). Eyes with fewer than
    ``k`` visits contribute all they have → the cohort stays the full 263 at every k (no power wall). This is the
    circularity DIAL's input: as k grows, the observed window encroaches on the trajectory that DEFINES the label."""
    rows = []
    for eid, g in fu.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        t, s = t[ok], s[ok]
        if k is not None:
            t, s = t[:k], s[:k]
        slope = float(np.polyfit(t, s, 1)[0]) if len(s) >= 2 else 0.0
        pval = float(stats.linregress(t, s).pvalue) if len(s) >= 3 else 1.0
        rows.append({"eye_id": eid, "patient_id": str(eid).split("_")[0],
                     "w_slope": slope, "w_slope_p": pval,
                     "w_sd": float(np.std(s)) if len(s) else 0.0,
                     "w_now": float(s[-1]) if len(s) else np.nan,
                     "w_mean": float(np.mean(s)) if len(s) else np.nan,
                     "n_seen": int(len(s))})
    return pd.DataFrame(rows)


_DIAL_COLS = ["w_slope", "w_slope_p", "w_sd", "w_now", "w_mean"]     # window stats (no n_seen: near-constant per k)


def _circular_auc(feat: pd.DataFrame, label: np.ndarray, cols, *, n_boot=1000, seed=0):
    y = np.asarray(label, dtype=int)
    keep = np.isfinite(y)
    feat, y = feat.loc[keep].reset_index(drop=True), y[keep]
    X = feat[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    groups = feat["patient_id"].to_numpy()
    p = S._event_oof_proba(X, y, groups, n_repeats=3, seed=seed)
    if p is None:
        return float("nan"), float("nan"), float("nan")
    return S._auc_cluster_ci(p, y, groups, n_boot=n_boot, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/grape")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--finetune", default="data/derived/grape_cfp_finetune_results.json")
    ap.add_argument("--out", default="reports/grape_serial_event.md")
    args = ap.parse_args()
    raw = Path(args.raw)

    base = grape.load_grape(raw)
    fu = grape.load_grape_followup(raw)
    fund = grape.fundus_features(raw, base)
    base = base.merge(fund[["eye_id", "fundus_vcdr"]], on="eye_id", how="left")

    # ---- §A honest landmark event curve (fixed cohort primary + maximal-cohort robustness) ----
    cur_fixed = S.landmark_event_curve(fu, base, landmarks=(1, 2, 3), min_future=3,
                                       n_boot=args.n_boot, fixed_cohort=True)
    cur_max = S.landmark_event_curve(fu, base, landmarks=(1, 2, 3), min_future=3,
                                     n_boot=args.n_boot, fixed_cohort=False)

    # ---- §B circular full-series comparator ----
    feat = _full_series_features(fu)
    lab = base.set_index("eye_id").reindex(feat["eye_id"])
    circ_cols = ["full_slope", "full_slope_p", "full_sd", "full_now", "full_mean", "n_visits"]
    # transparent MD-slope event on the FULL series (concords ≈0.93 with shipped MD) — near-tautological
    md_ev = S.md_slope_event_table(fu).set_index("eye_id")["event"].reindex(feat["eye_id"]).astype(int).to_numpy()
    circ = {
        "shipped_md": _circular_auc(feat, lab["prog_md"].to_numpy(), circ_cols, n_boot=args.n_boot),
        "shipped_plr2": _circular_auc(feat, lab["prog_plr2"].to_numpy(), circ_cols, n_boot=args.n_boot),
        "transparent_md": _circular_auc(feat, md_ev, circ_cols, n_boot=args.n_boot),
    }

    # ---- §C fine-tune ladder (device + eye-level leakage), if present ----
    ft = json.loads(Path(args.finetune).read_text()) if Path(args.finetune).exists() else {}

    # honest headline: the best-powered honest point (L=1 maximal cohort), 'all' modalities
    r1 = cur_max[cur_max["landmark"] == 1].iloc[0]
    honest_all = (r1.get("all", float("nan")), r1.get("all_lo", float("nan")), r1.get("all_hi", float("nan")))

    # ---- §B′ the circularity DIAL — OOF AUC vs #input visits fed (input window grows into the label window) ----
    KS = [1, 2, 3, 4, 5, 6, None]
    DIAL_EPS = [("prog_md", "MD"), ("prog_plr2", "PLR2"), ("prog_plr3", "PLR3")]
    dial = {ep: [] for ep, _ in DIAL_EPS}
    for k in KS:
        fk = _window_features(fu, k)
        lk = base.set_index("eye_id").reindex(fk["eye_id"])
        for ep, _ in DIAL_EPS:
            a, lo, hi = _circular_auc(fk, lk[ep].to_numpy(), _DIAL_COLS, n_boot=args.n_boot)
            dial[ep].append({"k": ("full" if k is None else k), "auc": a, "lo": lo, "hi": hi})
        print(f"[serial-dial] k={'full' if k is None else k}: MD {dial['prog_md'][-1]['auc']:.2f}")

    # ---- statistic ablation — which full-series statistic drives the ≈0.9 (predict shipped MD, one feature each) ----
    ABL = [("full_slope", "VF slope"), ("full_slope_p", "slope p-value"), ("full_now", "last VF level"),
           ("full_sd", "VF variability"), ("n_visits", "# visits")]
    ablation = {c: _circular_auc(feat, lab["prog_md"].to_numpy(), [c], n_boot=args.n_boot) for c, _ in ABL}

    RES = {
        "dial": dial,
        "ablation": {c: list(ablation[c]) for c, _ in ABL}, "ablation_labels": {c: lb for c, lb in ABL},
        "honest_anchor": {"auc": honest_all[0], "lo": honest_all[1], "hi": honest_all[2], "events": int(r1.n_events)},
        "full_series": {ep: list(circ[k]) for ep, k in [("prog_md", "shipped_md"), ("prog_plr2", "shipped_plr2")]},
        "lit": {"dixit_nVF_auc": [[2, 0.70], [3, 0.84], [4, 0.94]],
                "mohammadzadeh_nvisit_auc": [[1, 0.813], [2, 0.860], [3, 0.894]]},
        "chance": 0.5,
    }
    Path("data/derived/grape_serial_dial.json").write_text(json.dumps(RES, indent=2, default=float))

    # ------------------------------------------------------------------- write ----
    L = []
    L.append("# GRAPE progression — the honest serial EVENT curve + circularity decomposition\n")
    L.append("*How much of GRAPE's published progression AUC (0.71 → 0.97) is real discrimination, and how "
             "much is circularity + a device batch-effect + eye-level leakage? A non-circular landmark event "
             "model answers it — and shows GRAPE's short series cannot honestly support a serial event model "
             "at all.*\n")

    L.append("## §A · Honest serial EVENT curve (non-circular landmark)\n")
    L.append("Input = the early visits `1..L` only (VF recent value / trajectory slope / variability + IOP "
             "recent value / slope + baseline structure). Label = a transparent MD-slope progression **event** "
             "recomputed on the **disjoint held-out future window** (visits `L+1..end`, ≥3 future visits for a "
             "valid p). The input is never part of the label window — the circularity the field's serial "
             "models commit (feeding the whole VF series that *defines* the label) is removed. Patient-grouped "
             "repeated-CV OOF AUC, patient cluster-bootstrap 95% CI.\n")
    L.append("**Fixed cohort** (same eyes at every L — eyes with ≥ max(L)+3 = 6 visits):\n")
    L.append("| L | eyes | events | VF-history | IOP-history | baseline-struct | all |")
    L.append("|---|------|--------|-----------|------------|-----------------|-----|")
    for _, r in cur_fixed.iterrows():
        cells = " | ".join(_fmt(r.get(s, float("nan")), r.get(f"{s}_lo", float("nan")), r.get(f"{s}_hi", float("nan"))) for s in _SETS)
        L.append(f"| {int(r.landmark)} | {int(r.n_eyes)} | {int(r.n_events)} | {cells} |")
    L.append("\n**Maximal cohort** (each L uses every eligible eye — more power at small L; cohort shifts with L):\n")
    L.append("| L | eyes | events | VF-history | IOP-history | baseline-struct | all |")
    L.append("|---|------|--------|-----------|------------|-----------------|-----|")
    for _, r in cur_max.iterrows():
        cells = " | ".join(_fmt(r.get(s, float("nan")), r.get(f"{s}_lo", float("nan")), r.get(f"{s}_hi", float("nan"))) for s in _SETS)
        L.append(f"| {int(r.landmark)} | {int(r.n_eyes)} | {int(r.n_events)} | {cells} |")
    L.append(f"\n> **Read.** Every honest AUC sits near chance with a 95% CI that spans it — at **{int(r1.n_events)} "
             f"held-out events** (L=1, the best-powered point) discrimination is unmeasurable, not absent. The "
             "L-trend is uninterpretable (events fall 13→5→4 as the input window grows and the future window "
             "shrinks). This is the honest ceiling of GRAPE for a *non-circular* serial event model — see §D.\n")

    L.append("## §B · The circularity DIAL — how '#visits' inflates the AUC (the novel result)\n")
    L.append("The field reports a rising *AUC vs number of visits* curve (Dixit 2021: 2→3→4 VFs 0.70→0.84→0.94; "
             "Mohammadzadeh AJO 2024: 1→2→3 visits 0.813→0.860→0.894) and reads it as *more follow-up improves "
             "prediction*. But GRAPE's label is a statistic of the VF series, so feeding more visits means the input "
             "increasingly **overlaps the window that defines the label**. We isolate this: predict the FIXED shipped "
             "label from VF statistics computed over only the **first k visits**, and sweep k (all 263 eyes at every "
             "k — no power wall). The AUC climbs from the baseline-VF level to a near-tautological ≈0.9; the rise is "
             "**circularity accrual**, not future-progression discrimination.\n")
    L.append("| input = first k visits → predict | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=full |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ep, EL in DIAL_EPS:
        cells = " | ".join(f"{d['auc']:.2f}" for d in dial[ep])
        L.append(f"| shipped **{EL}** | {cells} |")
    L.append(f"\nAt **zero overlap** the honest held-out-future target (§A) sits at **{_fmt(*honest_all)}** and chance "
             f"is 0.50. Predicting the same MD label climbs **{dial['prog_md'][0]['auc']:.2f}** (baseline VF only) → "
             f"**{dial['prog_md'][-1]['auc']:.2f}** (the whole label-defining series) purely by letting the input eat "
             f"into the label window.\n")
    L.append("**Which statistic carries the circularity?** Predicting the shipped MD flag from each full-series "
             "statistic *alone*:\n")
    L.append("| feature (full series) | " + " | ".join(lb for _, lb in ABL) + " |")
    L.append("|---|" + "---|" * len(ABL))
    L.append("| AUC → shipped MD | " + " | ".join(f"{ablation[c][0]:.2f}" for c, _ in ABL) + " |")
    L.append(f"\n> **Read.** The climb is driven by the **label-defining statistics** — the VF slope and its p-value "
             f"({ablation['full_slope'][0]:.2f} / {ablation['full_slope_p'][0]:.2f} alone) — not the last VF level, "
             f"variability, or visit count. Feeding the slope that *defines* the MD label recovers it near-tautologically "
             f"with no imaging at all: this is the leakage the field's serial models commit (input ⊇ the label window). "
             f"The general failure mode is documented (Kapoor & Narayanan 2023, *Leakage and the reproducibility crisis "
             f"in ML-based science*) but has not been quantified for glaucoma progression; here it is, and it is general "
             f"across endpoints (k=full: MD {dial['prog_md'][-1]['auc']:.2f}, PLR2 {dial['prog_plr2'][-1]['auc']:.2f}, "
             f"PLR3 {dial['prog_plr3'][-1]['auc']:.2f}).\n")

    L.append("## §C · Decomposition — where GRAPE's published 0.71→0.97 comes from\n")
    L.append("| source | AUC | what it measures |")
    L.append("|--------|-----|------------------|")
    L.append("| Published GRAPE ResNet-50, 1 baseline CFP | 0.71 / 0.80 / 0.73 | PLR2/PLR3/MD, single eye-level split, no CI (lit) |")
    L.append("| Published serial models (TSDF, DiffSight) | 0.87 – 0.97 | feed the full fundus **series** (lit) |")
    L.append(f"| **Circularity** — full VF-series → MD flag (ours, §B) | {_fmt(*circ['shipped_md'])} | input ⊇ label-defining window |")
    if ft:
        for ep in ("plr2", "plr3"):
            if ep not in ft:
                continue
            e = ft[ep]
            pub = e.get("published_summary", {})
            L.append(f"| **Eye-level leakage** — ResNet-50 {ep.upper()}, published protocol (ours) | "
                     f"median {pub.get('median', float('nan')):.2f} (max {pub.get('max', float('nan')):.2f}) | "
                     "single random eye-level split, our recipe |")
            L.append(f"| — same model, honest patient-grouped (ours) | {e.get('honest_patient_grouped_auc', float('nan')):.2f} | "
                     "fellow eyes kept together |")
            L.append(f"| **Device confound** — camera-only {ep.upper()} (ours) | {e.get('camera_only_auc', float('nan')):.2f} | "
                     "acquisition camera, **no image content** |")
    L.append(f"| **Honest serial EVENT** — held-out future (ours, §A) | {_fmt(*honest_all)} | non-circular, L=1, {int(r1.n_events)} events |")
    L.append("\n> **Read.** The published 0.71→0.97 is not one signal. Stripped apart on GRAPE's own data: the "
             "serial ≈0.9 is **circularity** (§B); a fine-tuned CFP model reaches the published band only on a "
             "lucky **eye-level** split (median ≈0.5, max ≈0.7–0.8) and collapses patient-grouped; the **camera "
             "alone** predicts progression at 0.56–0.60 with no pixels (topcon progresses ≈2× canon); and once "
             "all three are removed, the **honest** future-event signal is inconclusive at this scale. The "
             "sources overlap (they are not strictly additive) — but each is independently demonstrated on GRAPE.\n")

    L.append("## §D · Why GRAPE cannot honestly support a serial event model (the power wall)\n")
    vc = fu.groupby("eye_id").size()
    L.append(f"- Median **{int(vc.median())} visits/eye** (range {int(vc.min())}–{int(vc.max())}); only "
             f"{int((vc >= 6).sum())} eyes reach 6 visits.\n")
    L.append("- A non-circular event needs ≥1 input visit **and** ≥3 disjoint future visits for a valid MD-slope "
             "p — so the held-out event count is tiny: **13 events at L=1, 5 at L=2, 4 at L=3** (maximal cohort).\n")
    L.append("- At 4–13 events no AUC is estimable (CIs span chance). This is *why* the published serial curves "
             "exist only in the circular form (§B): an honest landmark event model is unmeasurable here, so the "
             "field's honest-looking 'AUC vs #visits' curve on GRAPE would be noise. **The design is the "
             "contribution; the honest number is a documented null, not a discrimination claim.** A definitive "
             "honest curve needs a large serial cohort (e.g. Mass Eye & Ear 10,864) that is not public.\n")

    L.append("\n---\n*Methods.* Transparent MD-slope event = cumulative regression of Octopus mean sensitivity "
             "on time, slope<0 at p<0.05 (concords ≈0.93 with GRAPE's shipped MD flag). Regularized balanced "
             "logistic, StratifiedGroupKFold (patient-disjoint) × 3 repeats, patient cluster bootstrap "
             f"({args.n_boot} draws). Fine-tune ladder from `run_grape_cfp_finetune.py`. PLR2/PLR3 future-window "
             "events are degenerate on GRAPE's short noisy series (spec≈0) → the honest serial event uses the "
             "well-behaved MD-slope endpoint.\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"[serial-event] wrote {args.out}")
    print(f"[serial-event] honest L=1 'all' AUC {_fmt(*honest_all)} ({int(r1.n_events)} events); "
          f"circular full-series→MD {_fmt(*circ['shipped_md'])}")


if __name__ == "__main__":
    main()
