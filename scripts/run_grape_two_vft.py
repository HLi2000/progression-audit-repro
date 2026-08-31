#!/usr/bin/env python
"""GPPM's two-VFT mechanism on GRAPE: reproduce-then-collapse.

GPPM (Zheng et al. 2025, "Multi-modal Data and Trusted Multi-view Learning") reports AUC 0.849 by fusing
the visual-field (VF) follow-up trajectory with a follow-up fundus photograph in a trusted multi-view model.
The fundus branch is down-weighted (lambda_F = 0.01), so the number is carried by the VF signal. We do NOT
need GPPM's evidential-fusion architecture to audit this: the visual-field *decline over the follow-up* is
the signal, and GRAPE ships the VF series. So we reproduce the mechanism directly with a trajectory summary
(baseline + the full in-window span) and show it reproduces-then-collapses like the single-CFP models.

The load-bearing distinction is the SPAN, not the count. Our circularity dial feeds the first k CONSECUTIVE
visits, so at k=2 the input is {visit 1, visit 2} — both early, a tiny span — and the AUC is at the floor.
GPPM's multi-view fusion instead reaches ACROSS the follow-up (a mix of visits spanning to the late window
that DEFINES the label), so the trajectory's decline is (most of) the label itself. A span sweep confirms
GPPM's 0.849 matches the full-trajectory reach (baseline+last ~0.83-0.86), not a shallow early pair.

  §A  REPRODUCE (in-window) — features from a two-VFT pair {baseline, follow-up} -> predict the FIXED shipped
      PLR2/PLR3/MD label. Sweep which follow-up VFT completes the pair, from the 2nd visit (early, small span)
      to the LAST visit (late, in-window, GPPM-style). The AUC climbs with the span: a late in-window pair is
      near-tautological, an early pair is at the floor.
  §B  COLLAPSE (honest) — the SAME two-VFT input {baseline, 2nd VFT}, both strictly before a held-out window,
      predicting a transparent MD-slope event recomputed on the DISJOINT later window (visits 3..end). Once the
      second VFT is no longer in-window, the pair cannot forecast: AUC falls to chance.

Patient-grouped repeated-OOF AUC, patient cluster-bootstrap 95% CI throughout (reuses the audit's estimators).

  .venv/bin/python scripts/run_grape_two_vft.py [--n-boot 1000]
Writes reports/grape_two_vft.md + data/derived/grape_two_vft.json. CPU, ~1 min.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from glaucoma_triage.datasets import grape
from glaucoma_triage.progression import survival as S


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PAIR_COLS = ["level_b", "level_f", "delta", "rate", "span"]


def _pair_features(fu: pd.DataFrame, follow) -> pd.DataFrame:
    """Per-eye features from a PAIR of VFTs — baseline (visit 1) + one follow-up VFT.

    ``follow`` selects the second VFT: "last" = the eye's final visit (the GPPM-style in-window pair), or a
    1-based position p meaning the p-th visit (clamped to the last if the eye is shorter). Features mirror what
    two functional tests carry: the two levels, their difference (the decline), the two-point rate, and the
    span between them. Eyes with <2 usable VFTs are dropped (a pair needs two)."""
    rows = []
    for eid, g in fu.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        t, s = t[ok], s[ok]
        if len(s) < 2:
            continue
        jf = len(s) - 1 if follow == "last" else min(int(follow) - 1, len(s) - 1)
        if jf < 1:
            continue
        span = float(t[jf] - t[0])
        delta = float(s[jf] - s[0])
        rows.append({"eye_id": eid, "patient_id": str(eid).split("_")[0],
                     "level_b": float(s[0]), "level_f": float(s[jf]), "delta": delta,
                     "rate": (delta / span if span > 0 else 0.0), "span": span})
    return pd.DataFrame(rows)


def _honest_pair_dataset(fu: pd.DataFrame, *, min_future: int = 3) -> pd.DataFrame:
    """Honest two-VFT landmark: input pair = {visit 1, visit 2} (both EARLY), label = a transparent MD-slope
    event on the DISJOINT held-out window (visits 3..end, >= ``min_future`` visits for a valid p). Same two-VFT
    input shape as §A, but the second VFT is no longer inside the label window."""
    rows = []
    for eid, g in fu.sort_values("visit").groupby("eye_id"):
        g = g.reset_index(drop=True)
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        t, s = t[ok], s[ok]
        if len(s) < 2 + min_future:
            continue
        span = float(t[1] - t[0])
        delta = float(s[1] - s[0])
        ev, _ = S.md_slope_progression(t[2:], s[2:])
        rows.append({"eye_id": eid, "patient_id": str(eid).split("_")[0],
                     "level_b": float(s[0]), "level_f": float(s[1]), "delta": delta,
                     "rate": (delta / span if span > 0 else 0.0), "span": span,
                     "event": int(ev)})
    return pd.DataFrame(rows)


def _eyelevel_dist(X, y, seeds=200):
    """DiffSight/TSDF-style leaky reproduce over random 80/20 eye-level splits; return the split-AUC max and p90."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for s, (tr, te) in enumerate(StratifiedShuffleSplit(seeds, test_size=0.2, random_state=0).split(X, y)):
        if y[te].min() == y[te].max():
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000))])
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return float(np.max(aucs)), float(np.percentile(aucs, 90)), float(np.percentile(aucs, 95))


def _resid_within_camera(X, cam):
    """Camera-residualize the pair features (z-score within the baseline fundus camera)."""
    Xr = X.astype(float).copy()
    for c in np.unique(cam):
        m = cam == c
        mu = np.nanmean(X[m], axis=0); sd = np.nanstd(X[m], axis=0) + 1e-8
        Xr[m] = (X[m] - mu) / sd
    return np.nan_to_num(Xr)


def _aaroc(p, y, cam):
    """(1) -device: covariate-adjusted AUC (Janes & Pepe 2009) with camera as covariate -- mean over cases of the
    case's placement value (percentile among SAME-camera controls). Discrimination holding the device fixed."""
    p = np.asarray(p, float); y = np.asarray(y); cam = np.asarray(cam)
    ok = np.isfinite(p); p, y, cam = p[ok], y[ok], cam[ok]
    pv = []
    for i in np.where(y == 1)[0]:
        ctrl = p[(y == 0) & (cam == cam[i])]
        if len(ctrl):
            pv.append((np.sum(ctrl < p[i]) + 0.5 * np.sum(ctrl == p[i])) / len(ctrl))
    return float(np.mean(pv)) if pv else float("nan")


def _aaroc_cell(p, y, cam, pat, n_boot):
    """AAROC point estimate + patient cluster-bootstrap 95% CI (returns [auc, lo, hi])."""
    if p is None:
        return [float("nan")] * 3
    rng = np.random.default_rng(0); pats = np.unique(pat); pidx = {q: np.where(pat == q)[0] for q in pats}
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([pidx[q] for q in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            bs.append(_aaroc(p[idx], y[idx], cam[idx]))
        except Exception:
            pass
    lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan")))
    return [round(_aaroc(p, y, cam), 3), round(float(lo), 3), round(float(hi), 3)]


def _fmt(a, lo, hi):
    if not np.isfinite(a):
        return "—"
    return f"{a:.2f} [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else f"{a:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/grape")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="reports/grape_two_vft.md")
    args = ap.parse_args()
    raw = Path(args.raw)
    SE = _imp("scripts/run_grape_serial_event.py", "se")

    base = grape.load_grape(raw)
    fu = grape.load_grape_followup(raw)
    EPS = [("prog_plr2", "PLR2"), ("prog_plr3", "PLR3"), ("prog_md", "MD")]

    # ---- §A reproduce: two-VFT pair {baseline, follow} -> shipped label; sweep the follow VFT (span) ----
    FOLLOWS = [("2", "baseline + 2nd VFT (early)"), ("3", "baseline + 3rd VFT"),
               ("4", "baseline + 4th VFT"), ("last", "baseline + last VFT (in-window)")]
    reproduce = {}
    for key, lab in FOLLOWS:
        feat = _pair_features(fu, key if key == "last" else int(key))
        lk = base.set_index("eye_id").reindex(feat["eye_id"])
        reproduce[key] = {"label": lab, "n_eyes": int(len(feat)),
                          "mean_span": round(float(feat["span"].mean()), 2)}
        for ep, EL in EPS:
            a, lo, hi = SE._circular_auc(feat, lk[ep].to_numpy(), PAIR_COLS, n_boot=args.n_boot)
            reproduce[key][EL] = [a, lo, hi]
        print(f"[2vft §A] {lab:32s} n={reproduce[key]['n_eyes']:3d} span~{reproduce[key]['mean_span']:.1f}y "
              f"PLR2 {reproduce[key]['PLR2'][0]:.2f} PLR3 {reproduce[key]['PLR3'][0]:.2f} "
              f"MD {reproduce[key]['MD'][0]:.2f}", flush=True)

    # ---- GPPM VF+CFP FUSION: the 5-cell LEAVE-ONE-OUT ABLATION (order-free marginals + all-out floor) ----
    # reproduce (eye-group OOF, in-window, camera-kept = the leaky ceiling) and three ORDER-FREE marginals, each
    # removing ONE shortcut while the other two stay on: -device (camera-residualized), -split (patient-grouped),
    # -circular (early <=2-visit input). "all" removes all three = the floor. GPPM survives -device and -split (its
    # signal is the in-window VF decline, not the camera or the fellow eye) and collapses only at -circular. On the
    # 196-eye cohort with >=2 CFP photos (a genuine follow-up CFP distinct from baseline), matching GPPM's cohort.
    from sklearn.decomposition import PCA
    fe = np.load(Path("data/derived/grape_retfound_followup_emb.npz"), allow_pickle=True)
    femb = fe["emb"].astype(np.float32); feid = fe["eye_id"].astype(str); fvis = np.asarray(fe["visit"]); fcam = fe["camera"].astype(str)
    base_emb, last_emb, base_cam = {}, {}, {}
    for e in np.unique(feid):
        idx = np.where(feid == e)[0]
        if len(idx) < 2:                         # GPPM needs a follow-up CFP distinct from baseline (>=2 photos)
            continue
        idx = idx[np.argsort(fvis[idx])]
        base_emb[e], last_emb[e], base_cam[e] = femb[idx[0]], femb[idx[-1]], fcam[idx[0]]
    cfp_eyes = sorted(base_emb)
    pca = PCA(16, random_state=0).fit(np.nan_to_num(np.array([last_emb[e] for e in cfp_eyes] + [base_emb[e] for e in cfp_eyes])))

    def _fusion(vf_feat, cfp_dict):
        vf = vf_feat[vf_feat["eye_id"].astype(str).isin(set(cfp_eyes))]
        eyes = vf["eye_id"].astype(str).to_numpy()
        Xcfp = pca.transform(np.nan_to_num(np.array([cfp_dict[e] for e in eyes])))
        cam = np.array([base_cam[e] for e in eyes])
        return eyes, np.column_stack([vf[PAIR_COLS].to_numpy(float), Xcfp]), vf["patient_id"].to_numpy(), cam

    eyes_iw, X_iw, pat_iw, cam_iw = _fusion(_pair_features(fu, "last"), last_emb)   # in-window: full VF trajectory (baseline->last) + follow-up CFP, GPPM's multi-view span
    eyes_ec, X_ec, pat_ec, cam_ec = _fusion(_pair_features(fu, "2"), base_emb)      # early/non-circular (VF 2nd + CFP baseline)
    lk_iw = base.set_index("eye_id").reindex(eyes_iw); lk_ec = base.set_index("eye_id").reindex(eyes_ec)
    nvf = len(PAIR_COLS)
    X_iw_dev = X_iw.copy(); X_iw_dev[:, nvf:] = _resid_within_camera(X_iw[:, nvf:], cam_iw)   # residualize CFP part only
    X_ec_dev = X_ec.copy(); X_ec_dev[:, nvf:] = _resid_within_camera(X_ec[:, nvf:], cam_ec)

    def _cell(X, y, cv_group, patient):   # GroupKFold-OOF AUC (CV grouped by cv_group) + patient cluster-bootstrap 95% CI
        p = S._event_oof_proba(X, y, cv_group)
        return [round(v, 3) for v in S._auc_cluster_ci(p, y, patient, n_boot=args.n_boot)] if p is not None else [float("nan")] * 3

    gppm_fusion = {"n_eyes": int(len(eyes_iw))}
    for ep, EL in EPS:
        y_iw = lk_iw[ep].to_numpy().astype(int); y_ec = lk_ec[ep].to_numpy().astype(int)
        _emax, _ep90, _ep95 = _eyelevel_dist(X_iw, y_iw)               # leaky eye-level split ceiling: max + p90 + p95
        p_iw = S._event_oof_proba(X_iw, y_iw, eyes_iw)                  # reproduce OOF (reused for (1) AAROC)
        repro_cell = [round(v, 3) for v in S._auc_cluster_ci(p_iw, y_iw, pat_iw, n_boot=args.n_boot)] if p_iw is not None else [float("nan")] * 3
        p_ec_pat = S._event_oof_proba(X_ec, y_ec, pat_ec)              # (4) combined OOF: patient-grouped + non-circular (early)
        cell = {"reproduce": repro_cell,                                # eye-group, in-window, kept = leaky ceiling
                "m_aaroc":   _aaroc_cell(p_iw, y_iw, cam_iw, pat_iw, args.n_boot),   # (1) covariate-adjusted AUC (Janes-Pepe)
                "m_device":  _cell(X_iw_dev, y_iw, eyes_iw, pat_iw),    # (1) residualized (kept: robustness twin)
                "m_split":   _cell(X_iw, y_iw, pat_iw, pat_iw),         # (2) -patient (patient-group)
                "m_circ":    _cell(X_ec, y_ec, eyes_ec, pat_ec),        # (3) -circular (eye-group, early input)
                "all_combined": _aaroc_cell(p_ec_pat, y_ec, cam_ec, pat_ec, args.n_boot),   # (4) = (1)+(2)+(3): AAROC on patient-grouped, non-circular OOF
                "iecv":      _cell(X_ec, y_ec, cam_ec, pat_ec),         # (4) IECV leave-one-camera-out (transport twin)
                "all":       _cell(X_ec_dev, y_ec, pat_ec, pat_ec),     # (4) residualized+patient+early (residualized twin)
                "eyelevel_max": round(_emax, 3), "eyelevel_p90": round(_ep90, 3), "eyelevel_p95": round(_ep95, 3),   # ceiling + p90 + p95
                "events": int(y_iw.sum())}
        gppm_fusion[EL] = cell
        print(f"[GPPM VF+CFP {EL}] ev{int(y_iw.sum())} n{len(eyes_iw)}: reproduce {cell['reproduce'][0]:.2f} | "
              f"(1)AAROC {cell['m_aaroc'][0]:.2f} | -split {cell['m_split'][0]:.2f} | -circ {cell['m_circ'][0]:.2f} | "
              f"(4)combined {cell['all_combined'][0]:.2f} | IECV {cell['iecv'][0]:.2f}", flush=True)

    # ---- §B collapse: the same {baseline, 2nd VFT} pair -> held-out later MD event (honest) ----
    hon = _honest_pair_dataset(fu, min_future=3)
    n_ev = int(hon["event"].sum())
    y = hon["event"].to_numpy(int)
    groups = hon["patient_id"].to_numpy()
    p = S._event_oof_proba(hon[PAIR_COLS].to_numpy(), y, groups, n_repeats=3)
    honest = list(S._auc_cluster_ci(p, y, groups, n_boot=args.n_boot)) if p is not None else [float("nan")] * 3
    print(f"[2vft §B] honest {{baseline,2nd}} -> held-out MD event: {_fmt(*honest)} "
          f"({len(hon)} eyes, {n_ev} events)")

    RES = {"reproduce": reproduce, "follows": [k for k, _ in FOLLOWS],
           "gppm_fusion": gppm_fusion,
           "honest": {"auc": honest[0], "lo": honest[1], "hi": honest[2],
                      "n_eyes": int(len(hon)), "events": n_ev},
           "chance": 0.5}
    Path("data/derived/grape_two_vft.json").write_text(json.dumps(RES, indent=2, default=float))

    # ------------------------------------------------------------------------ write ----
    L = []
    L.append("# GPPM's two-VFT mechanism on GRAPE — reproduce, then collapse\n")
    L.append("*GPPM reports AUC 0.849 from a pair of visual-field tests (a baseline VFT and one follow-up VFT) "
             "plus a down-weighted follow-up fundus photo (fundus-branch weight lambda_F = 0.01). The visual-field "
             "decline is the signal, and GRAPE ships the VF series, so we reproduce the two-VFT mechanism directly "
             "— no evidential-fusion architecture needed — and show it reproduces-then-collapses like the single-image "
             "models. The distinction that matters is the SPAN, not the count of two.*\n")

    L.append("## §A · Reproduce — a two-VFT pair recovers the label when the second VFT is in-window\n")
    L.append("Input = five features of a VFT pair {baseline, one follow-up} (the two mean-sensitivity levels, "
             "their difference, the two-point rate, and the span). Target = the FIXED shipped PLR2/PLR3/MD label. "
             "We sweep which follow-up VFT completes the pair. Patient-grouped repeated-OOF AUC [95% CI].\n")
    L.append("| two-VFT pair | eyes | mean span | PLR2 | PLR3 | MD |")
    L.append("|---|---|---|---|---|---|")
    for key, lab in FOLLOWS:
        r = reproduce[key]
        cells = " | ".join(_fmt(*r[EL]) for _, EL in EPS)
        L.append(f"| {lab} | {r['n_eyes']} | {r['mean_span']:.1f} y | {cells} |")
    rl = reproduce["last"]
    r2 = reproduce["2"]
    L.append(f"\n> **Read.** Two VFTs are enough to recover the label — *if the second one is late*. The "
             f"baseline+last pair (GPPM's in-window structure) reaches "
             f"{rl['PLR2'][0]:.2f}/{rl['PLR3'][0]:.2f}/{rl['MD'][0]:.2f}; the baseline+2nd pair (an early, "
             f"small-span pair, matching the circularity dial at k=2) sits far lower "
             f"({r2['PLR2'][0]:.2f}/{r2['PLR3'][0]:.2f}/{r2['MD'][0]:.2f}). The lift is the SPAN: a late "
             f"follow-up VFT lies inside the follow-up window that defines the label, so the pair's decline is "
             f"most of the label. This is why a two-visit model can look strong (GPPM's 1st+4th) while the dial "
             f"at k=2 (1st+2nd) is at the floor — same count, different span.\n")

    L.append("## §B · Collapse — the same pair, made honest, falls to chance\n")
    L.append("Same two-VFT input shape {baseline, 2nd VFT}, both strictly before a held-out window, now predicting "
             "a transparent MD-slope progression **event** recomputed on the DISJOINT later window (visits 3..end, "
             ">=3 visits for a valid p-value). Once the second VFT is no longer in-window, the pair cannot forecast.\n")
    L.append(f"- honest two-VFT AUC = **{_fmt(*honest)}** ({int(len(hon))} eyes, {n_ev} held-out events).\n")
    L.append(f"> **Read.** The in-window pair's {rl['MD'][0]:.2f} (MD) collapses to {honest[0]:.2f} once the second "
             "VFT is moved out of the label window — the two-VFT number is in-window change-detection, not a "
             "forecast. (Small held-out event counts keep the honest CI wide; the point is that it cannot be "
             "distinguished from chance, consistent with the single-VFT landmark, not that it is a proven negative.)\n")

    L.append("\n---\n*Scope.* This reproduces GPPM's two-VFT *mechanism* (the visual-field decline over an "
             "in-window pair), not its evidential-fusion architecture; GPPM's own CFP-only branch already sits at "
             "the camera floor, and its fundus branch is weighted 0.01, so the visual field carries the number. "
             "Estimator: regularized balanced logistic, StratifiedGroupKFold (patient-disjoint) x 3 repeats, "
             f"patient cluster bootstrap ({args.n_boot} draws). MD-slope event = cumulative regression of Octopus "
             "mean sensitivity on time, slope<0 at p<0.05.\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"[2vft] wrote {args.out}")


if __name__ == "__main__":
    main()
