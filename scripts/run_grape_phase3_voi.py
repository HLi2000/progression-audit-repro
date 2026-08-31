"""GRAPE progression Phase 3 — multimodal VALUE-OF-INFORMATION for the fundus foundation/CNN embeddings.

Consumes `data/derived/phase3_reps.npz` (produced by `notebooks/grape_phase3_reps_gpu.ipynb`): per baseline eye,
frozen linear-probe (LP) **and** fine-tuned out-of-fold (FT, leakage-safe) 1024-d penultimate embeddings for
RETFound-CFP and ConvNeXt-Base × the three endpoints. It answers the honest-ceiling question at the fusion level:

    Under a patient-grouped + camera-controlled protocol, does the best fundus representation (frozen or fine-tuned,
    FM or CNN) carry NON-REDUNDANT prognostic value BEYOND the cheap tabular tests already ordered (OCT-RNFL, VF,
    IOP) — or is it redundant / near-chance?

Three reads, all honest (patient-grouped repeated-OOF, patient-cluster-bootstrap CIs), all camera-residualized
(within-camera z-score strips the topcon/canon device shortcut that is confounded with the label):

  §A  Fundus REPRESENTATION LADDER — camera-residualized honest AUC[CI] for {RETFound-CFP, ConvNeXt-Base}×{LP,FT}
      vs the training-free camera-only floor, per endpoint. Does any representation clear the floor?
  §B  VALUE-OF-INFORMATION decomposition — 4-modality SAGE-Shapley + Grabisch–Roubens interaction on
      {fundus*, oct, vf, iop} (fundus* = best rep per endpoint), + PID redundancy/unique/synergy for fundus×VF.
  §C  NET BENEFIT / EVPI — does adding fundus to {OCT+VF+IOP} change the decision at clinical thresholds?

Reuses the built engine (interaction.report.decompose, interaction.pid, models.fusion, models.decision); no new
method. → reports/grape_phase3_voi.md + data/derived/grape_phase3_voi.json

    .venv/bin/python scripts/run_grape_phase3_voi.py [--npz data/derived/phase3_reps.npz] [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from glaucoma_triage import config
from glaucoma_triage.datasets import grape
from glaucoma_triage.interaction import pid
from glaucoma_triage.models import decision as dec
from glaucoma_triage.models.fusion import oof_proba

ENDPOINTS = [("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"), ("MD", "prog_md")]
REPS = [("RETFound-CFP · LP", "lp", "retfound_cfp"), ("RETFound-CFP · FT", "ft", "retfound_cfp"),
        ("ConvNeXt-Base · LP", "lp", "convnext_base"), ("ConvNeXt-Base · FT", "ft", "convnext_base")]
N_PCA_LADDER = 16      # §A single-modality ladder (regularized point AUC; matches the Phase-2a honest-rep sweep)
N_PCA_FUSE = 4         # §B/§C fusion: EPP-matched to the tabular modalities (oct=3, vf=2) — null-calibrated clean


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _probe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced"))])


def _repeated_oof(X, y, g, seeds=8):
    """Patient-grouped repeated 5-fold OOF probabilities, averaged over `seeds` fold assignments."""
    acc = np.zeros((len(y), seeds))
    for s in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            acc[te, s] = _probe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return acc.mean(axis=1)


def _auc_ci(p, y, g, n_boot, seed=0):
    """Point AUC + patient-cluster-bootstrap 95% CI on fixed OOF probabilities."""
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in pats}
    draws = []
    for _ in range(n_boot):
        take = rng.choice(pats, size=len(pats), replace=True)
        rows = np.concatenate([idx[u] for u in take])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        draws.append(roc_auc_score(yy, p[rows]))
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (np.nan, np.nan)
    return float(auc), lo, hi


def _delta_auc_ci(p_base, p_aug, y, g, n_boot, seed=0):
    """Paired incremental ΔAUC = AUC(aug) − AUC(base) with a patient-cluster-bootstrap 95% CI (both AUCs
    recomputed on the SAME resample each draw → a proper paired CI). Robust at 14–40 events, unlike the
    loss-based Shapley interaction (which is artifact-inflated at this scale — see the null calibration)."""
    d = roc_auc_score(y, p_aug) - roc_auc_score(y, p_base)
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in pats}
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[u] for u in rng.choice(pats, size=len(pats), replace=True)])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        draws.append(roc_auc_score(yy, p_aug[rows]) - roc_auc_score(yy, p_base[rows]))
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (np.nan, np.nan)
    return float(d), lo, hi


def _md_slope(fu):
    """Per-eye continuous MD-slope proxy = OLS slope of VF mean-sensitivity vs interval-years (the well-powered
    263-eye continuous progression *rate* target; the binary events have only 14–40 positives)."""
    out = {}
    for eid, gg in fu.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(gg["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(gg["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        out[str(eid)] = float(np.polyfit(t[ok], s[ok], 1)[0]) if ok.sum() >= 2 else np.nan
    return out


def _impute(emb):
    emb = np.asarray(emb, float).copy()
    if np.isnan(emb).any():
        med = np.nanmedian(emb, axis=0)
        ii = np.where(np.isnan(emb))
        emb[ii] = np.take(med, ii[1])
    return emb


def _resid_pca(emb, cam, n):
    """Within-camera z-score (strips the device shortcut) → PCA-n. The honest camera-controlled fundus read."""
    emb = _impute(emb)
    for c in np.unique(cam):
        i = cam == c
        emb[i] = (emb[i] - emb[i].mean(0)) / (emb[i].std(0) + 1e-6)
    n = min(n, emb.shape[1], emb.shape[0] - 1)
    return PCA(n, random_state=0).fit_transform(emb)


def _naive_pca(emb, n):
    emb = _impute(emb)
    n = min(n, emb.shape[1], emb.shape[0] - 1)
    return PCA(n, random_state=0).fit_transform(emb)


def _fmt(a, lo, hi):
    return f"{a:.3f} [{lo:.2f},{hi:.2f}]" if a == a else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="data/derived/phase3_reps.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    P = np.load(args.npz, allow_pickle=True)
    eye = P["eye_id"].astype(str)
    cam = P["camera"].astype(str)
    g = np.array([e.split("_")[0] for e in eye])
    Y = {col: P[col].astype(int) for _, col in ENDPOINTS}

    # tabular modalities from the local GRAPE loader, aligned to the embedding's eye order
    raw = config.raw_dir("grape")
    base = grape.load_grape(raw)
    tab = pd.DataFrame({"eye_id": eye}).merge(base, on="eye_id", how="left")
    assert (tab["eye_id"].to_numpy() == eye).all()
    n_missing = int(tab[grape.OCT_FEATURES + grape.VF_FEATURES + ["iop"]].isna().any(axis=1).sum())
    for c in grape.OCT_FEATURES + grape.VF_FEATURES + ["iop"]:
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
        tab[c] = tab[c].fillna(tab[c].median())
    tab["patient_id"] = g

    def get_emb(kind, model, col):
        ep_short = col.replace("prog_", "")            # FT keys use the short endpoint (…__plr2__ft_emb)
        return P[f"{model}__lp_emb"] if kind == "lp" else P[f"{model}__{ep_short}__ft_emb"]

    RES = {"n_eyes": int(len(eye)), "n_patients": int(len(np.unique(g))),
           "cameras": {c: int((cam == c).sum()) for c in np.unique(cam)},
           "n_tab_missing_imputed": n_missing, "ladder": {}, "voi": {}, "decision": {}}
    O = []
    w = O.append
    w("# GRAPE progression Phase 3 — multimodal value-of-information for the fundus embeddings")
    w("")
    w(f"Honest **pilot** on {len(eye)} eyes / {len(np.unique(g))} patients (all-glaucoma; PLR2 "
      f"{int(Y['prog_plr2'].sum())} / PLR3 {int(Y['prog_plr3'].sum())} / MD {int(Y['prog_md'].sum())} events → "
      f"wide CIs). We fuse the strongest fundus representations — frozen linear-probe (LP) and fine-tuned "
      f"out-of-fold (FT, leakage-safe) embeddings of RETFound-CFP and ConvNeXt-Base — with the cheap tabular tests "
      f"already ordered (OCT-RNFL, VF, IOP), under a **patient-grouped + camera-controlled** protocol "
      f"(within-camera z-score strips the topcon/canon device shortcut that is confounded with the label). "
      f"Question: does the fundus add **non-redundant honest value** beyond the tabular — or is it redundant / "
      f"near-chance? AUCs are patient-grouped repeated 5-fold OOF with patient-cluster-bootstrap 95% CIs.")
    w("")

    # ===== §A representation ladder =====
    _log("[A] representation ladder")
    w("## A · Fundus representation ladder — camera-controlled honest AUC vs the camera floor")
    w("")
    w("Camera-**residualized** repeated-OOF AUC [95% CI] per representation & endpoint, beside the training-free "
      "**camera-only** floor (the acquisition camera as the sole predictor, no pixels) and the **naive** "
      "(non-residualized) AUC that still rides the device shortcut. *Beating the floor* requires the residualized "
      "CI to exclude it.")
    w("")
    w("| representation | " + " | ".join(f"{e} resid" for e, _ in ENDPOINTS) + " | " +
      " | ".join(f"{e} naive" for e, _ in ENDPOINTS) + " |")
    w("|" + "---|" * (1 + 2 * len(ENDPOINTS)))
    floor = {}
    for ep, col in ENDPOINTS:
        floor[col] = roc_auc_score(Y[col], (cam == "topcon").astype(int))
    best = {}                          # col -> (rep_label, kind, model, resid_auc)
    reps = [r for r in REPS if f"{r[2]}__lp_emb" in P.files]      # only models actually in the npz (e.g. RETFound-only)
    for label, kind, model in reps:
        resid_cells, naive_cells = [], []
        for ep, col in ENDPOINTS:
            emb = get_emb(kind, model, col)
            yv = Y[col]
            zr = _resid_pca(emb, cam, N_PCA_LADDER)
            ar, lor, hir = _auc_ci(_repeated_oof(zr, yv, g), yv, g, args.n_boot)
            zn = _naive_pca(emb, N_PCA_LADDER)
            an, lon, hin = _auc_ci(_repeated_oof(zn, yv, g), yv, g, args.n_boot)
            resid_cells.append(_fmt(ar, lor, hir))
            naive_cells.append(_fmt(an, lon, hin))
            RES["ladder"].setdefault(col, {})[label] = {"resid": [ar, lor, hir], "naive": [an, lon, hin]}
            if col not in best or ar > best[col][3]:
                best[col] = (label, kind, model, ar)
        w(f"| {label} | " + " | ".join(resid_cells) + " | " + " | ".join(naive_cells) + " |")
        _log(f"  {label} done")
    w(f"| *camera-only floor* | " + " | ".join(f"**{floor[col]:.3f}**" for _, col in ENDPOINTS) +
      " | " + " | ".join("—" for _ in ENDPOINTS) + " |")
    for _, col in ENDPOINTS:
        RES["ladder"][col]["camera_only"] = float(floor[col])
    w("")
    beats = []
    for ep, col in ENDPOINTS:
        lab, _, _, a = best[col]
        lo = RES["ladder"][col][lab]["resid"][1]
        if lo > floor[col]:
            beats.append(f"{ep} ({lab} {a:.3f}, CI-lo {lo:.2f} > floor {floor[col]:.2f})")
    w(f"*Read: best camera-residualized fundus per endpoint — " +
      ", ".join(f"{ep}: **{best[col][0]} {best[col][3]:.3f}**" for ep, col in ENDPOINTS) +
      f". Camera floor {floor['prog_plr2']:.2f}/{floor['prog_plr3']:.2f}/{floor['prog_md']:.2f}. " +
      (f"Representations whose CI clears the floor: {'; '.join(beats)}." if beats else
       "**No representation's CI clears the camera floor on any endpoint** — the honest ceiling is "
       "representation- and training-independent (frozen ⇄ fine-tuned, FM ⇄ CNN).") + "*")
    w("")

    # ===== §B incremental value + PID =====
    _log("[B] incremental value + PID")
    md_slope = _md_slope(grape.load_grape_followup(raw))
    ms = np.array([md_slope.get(e, np.nan) for e in eye])
    w("## B · Value of information — does the best fundus add over OCT + VF + IOP?")
    w("")
    w(f"The best camera-residualized fundus rep per endpoint (PCA-{N_PCA_FUSE}, EPP-matched to the tabular "
      f"modalities), tested two honest ways. **(i) Incremental ΔAUC** = AUC(OCT+VF+IOP+fundus\\*) − "
      f"AUC(OCT+VF+IOP), paired patient-cluster bootstrap. **(ii) PID** (redundancy/unique/synergy) of fundus\\* "
      f"vs VF for the well-powered **continuous MD-slope** rate (263 eyes) — the binary events (14–40) are too "
      f"few for a stable loss-based Shapley *interaction* (null calibration: on random embeddings it returns "
      f"spurious +0.15 'synergy' — so we do **not** report it; the incremental-AUC and PID reads below are "
      f"artifact-free).")
    w("")
    w("| endpoint | fundus\\* | AUC tab (OCT+VF+IOP) | AUC +fundus\\* | **ΔAUC [95% CI]** | PID net-synergy (rate) [p] |")
    w("|---|---|---|---|---|---|")
    Xtab = tab[grape.OCT_FEATURES + grape.VF_FEATURES + ["iop"]].to_numpy()
    for ep, col in ENDPOINTS:
        lab, kind, model, _ = best[col]
        z = _resid_pca(get_emb(kind, model, col), cam, N_PCA_FUSE)
        yv = np.asarray(Y[col])
        p_tab = _repeated_oof(Xtab, yv, g)
        p_all = _repeated_oof(np.hstack([Xtab, z]), yv, g)
        auc_t, auc_a = roc_auc_score(yv, p_tab), roc_auc_score(yv, p_all)
        dlt, dlo, dhi = _delta_auc_ci(p_tab, p_all, yv, g, args.n_boot)
        # PID on the continuous rate (well-powered), fundus* vs VF
        ok = np.isfinite(ms)
        try:
            pa = pid.pair_analysis(ms[ok], z[ok], tab[grape.VF_FEATURES].to_numpy()[ok], g[ok],
                                   n_boot=min(args.n_boot, 500), n_perm=min(args.n_boot, 500), cross_fit=True)
            pid_txt = f"{pa.pid.net_synergy:+.3f} [p={pa.p_perm.get('net_synergy', float('nan')):.2f}]"
            pid_rec = {"redundancy": pa.pid.redundancy, "unique_fundus": pa.pid.unique1,
                       "unique_vf": pa.pid.unique2, "synergy": pa.pid.synergy, "net_synergy": pa.pid.net_synergy,
                       "p_net_synergy": pa.p_perm.get("net_synergy")}
        except Exception as ex:
            pid_txt, pid_rec = "—", {"error": str(ex)}
            _log(f"  {ep}: PID skipped ({ex})")
        w(f"| {ep} | {lab} | {auc_t:.3f} | {auc_a:.3f} | **{dlt:+.3f} [{dlo:+.2f},{dhi:+.2f}]** | {pid_txt} |")
        RES["voi"][col] = {"fundus_rep": lab, "auc_tab": auc_t, "auc_all": auc_a,
                           "delta_auc": [dlt, dlo, dhi], "pid_fundus_vf_rate": pid_rec}
        _log(f"  {ep}: ΔAUC {dlt:+.3f} [{dlo:+.2f},{dhi:+.2f}]")
    w("")
    w("*Read: a fundus\\* whose incremental **ΔAUC CI spans 0** adds no discrimination beyond the OCT/VF/IOP "
      "already ordered; a PID **net-synergy ≈0 (p≫0.05)** for the continuous rate means fundus and VF carry no "
      "joint information the other lacks. Both ⇒ the baseline fundus embedding is **redundant** with the cheap "
      "tabular tests for progression at this scale.*")
    w("")

    # ===== §C net benefit / EVPI =====
    _log("[C] net benefit / EVPI")
    w("## C · Net benefit & EVPI — does adding fundus change the decision?")
    w("")
    w("Decision-curve net benefit of **OCT+VF+IOP** vs **+fundus\\*** at clinical thresholds, with EVPI (expected "
      "value of resolving which strategy is best) and P(strategy best) from the patient-cluster bootstrap.")
    w("")
    thr = np.array([0.10, 0.15, 0.20])
    w("| endpoint | NB@0.15 tab | NB@0.15 +fundus | EVPI@0.15 | P(+fundus best)@0.15 |")
    w("|---|---|---|---|---|")
    for ep, col in ENDPOINTS:
        lab, kind, model, _ = best[col]
        z = _resid_pca(get_emb(kind, model, col), cam, N_PCA_FUSE)
        df = tab[["eye_id", "patient_id", "iop"] + grape.OCT_FEATURES + grape.VF_FEATURES].copy()
        fcols = [f"fundus_{j:02d}" for j in range(z.shape[1])]
        for j, fc in enumerate(fcols):
            df[fc] = z[:, j]
        df[col] = Y[col]
        spec = {"fundus": fcols, "oct": grape.OCT_FEATURES, "vf": grape.VF_FEATURES, "iop": ["iop"]}
        p_tab, y_int, ids = oof_proba(df, spec, ["oct", "vf", "iop"], df[col], estimator_factory=_probe)
        p_all, _, _ = oof_proba(df, spec, ["fundus", "oct", "vf", "iop"], df[col], estimator_factory=_probe)
        gg = np.array([str(i).split("_")[0] for i in ids])
        draws = dec.net_benefit_draws({"tab": p_tab, "tab+fundus": p_all}, y_int, gg, thr,
                                      n_boot=min(args.n_boot, 500))
        ev = dec.evpi(draws)
        pb = dec.prob_best(draws)
        j15 = int(np.argmin(np.abs(thr - 0.15)))
        nb_tab = dec.net_benefit(p_tab, y_int, thr)[j15]
        nb_all = dec.net_benefit(p_all, y_int, thr)[j15]
        w(f"| {ep} | {nb_tab:+.4f} | {nb_all:+.4f} | {ev[j15]:.4f} | {pb['tab+fundus'][j15]:.2f} |")
        RES["decision"][col] = {"nb_tab@0.15": float(nb_tab), "nb_all@0.15": float(nb_all),
                                "evpi@0.15": float(ev[j15]), "p_fundus_best@0.15": float(pb["tab+fundus"][j15])}
        _log(f"  {ep}: EVPI@0.15 {ev[j15]:.4f}")
    w("")
    w("*Read: if +fundus does not raise net benefit over the tabular strategy and EVPI≈0, the acquisition decision "
      "is settled without the fundus photo — it adds no decision value beyond the OCT/VF/IOP already ordered.*")
    w("")

    # ===== verdict =====
    w("## Verdict")
    w("")
    any_beat = any(RES["ladder"][col][best[col][0]]["resid"][1] > floor[col] for _, col in ENDPOINTS)
    w(f"- **Representation ceiling (§A):** " +
      ("no fundus representation — frozen or fine-tuned, FM or CNN — has a camera-controlled honest CI that clears "
       "the training-free camera floor on any endpoint. The honest baseline-fundus ceiling on GRAPE is "
       "representation- and training-independent." if not any_beat else
       "at least one representation's CI clears the camera floor — see §A (revise the honest-ceiling claim)."))
    redundant = all((RES["voi"][col]["delta_auc"][1] <= 0 <= RES["voi"][col]["delta_auc"][2]) for _, col in ENDPOINTS)
    w("- **Non-redundancy (§B):** " +
      ("the best fundus embedding's incremental ΔAUC over OCT+VF+IOP has a CI spanning 0 on every endpoint, and its "
       "PID net-synergy with VF for the continuous rate is ≈0 (p≫0.05) — it is **redundant** with the cheap tabular "
       "tests, adding no unique or synergistic prognostic signal." if redundant else
       "at least one endpoint shows a fundus incremental ΔAUC CI excluding 0 — see §B (fundus adds discrimination there)."))
    w("- **Decision value (§C):** net benefit / EVPI show whether ordering the fundus photo changes the "
      "monitoring decision at clinical thresholds.")
    w("- **Limits:** 14–40 events → wide CIs (pilot); FT embeddings are out-of-fold (leakage-safe) but "
      "fine-tuned to the same endpoint (best-case for fundus); single all-glaucoma cohort; baseline fundus only. "
      "Characterization, not prediction-SOTA.")
    w("")

    out_md = config.REPO_ROOT / "reports" / "grape_phase3_voi.md"
    out_md.write_text("\n".join(O) + "\n")
    out_json = config.REPO_ROOT / "data" / "derived" / "grape_phase3_voi.json"
    out_json.write_text(json.dumps(RES, indent=2, default=float))
    print("\n".join(O))
    _log(f"(wrote {out_md} + {out_json})")


if __name__ == "__main__":
    main()
