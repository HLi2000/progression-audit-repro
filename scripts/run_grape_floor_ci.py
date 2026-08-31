#!/usr/bin/env python
"""Floor CI + paired non-inferiority test (addresses the reviewer's 'the floor has no CI, and
"sits at the floor" is CI-vs-point eyeballing, not a paired test').

For each endpoint we take the SAME fixed per-eye scores used in F4 — the camera-only floor
(is-topcon, deterministic, no pixels) and the camera-residualized RETFound-CFP frozen-probe OOF
probabilities — and run ONE patient-cluster bootstrap that, on each resample of patients, recomputes
BOTH AUCs on the same rows and takes their difference. This gives:

  * a 95% CI on the floor itself (previously a bare point 0.561/0.600/0.580),
  * a 95% CI on the representation, and
  * a 95% CI on the PAIRED difference d = AUC(fundus) - AUC(floor).

"Fundus does not clear the floor" is then a tested statement: the paired-d CI includes 0 / its
upper bound does not exceed a small margin, rather than two separate CIs eyeballed for overlap.

    .venv/bin/python scripts/run_grape_floor_ci.py [--n-boot 2000]
Reuses the honest-representation probe/bootstrap discipline verbatim. CPU, ~1 min.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from glaucoma_triage import config

ENDPOINTS = [("PLR2", "plr2"), ("PLR3", "plr3"), ("MD", "md")]
REP = "retfound_cfp"  # the full-fundus RETFound probe used in F4 (0.576/0.579/0.584)


def _probe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced"))])


def _repeated_oof(X, y, g, seeds=8):
    acc = np.zeros((len(y), seeds))
    for s in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            acc[te, s] = _probe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return acc.mean(axis=1)


def _residualize(emb, cam):
    e = emb.copy()
    for c in np.unique(cam):
        i = cam == c
        e[i] = (e[i] - e[i].mean(0)) / (e[i].std(0) + 1e-6)
    return e


def _paired_ci(p_rep, p_floor, y, g, n_boot, seed=0):
    """Floor/rep point AUCs + patient-cluster-bootstrap CIs, and the CI of the PAIRED difference
    d = AUC(rep) - AUC(floor) recomputed on the same resampled patients each draw."""
    auc_rep = float(roc_auc_score(y, p_rep))
    auc_flo = float(roc_auc_score(y, p_floor))
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in pats}
    b_rep, b_flo, b_d = [], [], []
    for _ in range(n_boot):
        rows = np.concatenate([idx[u] for u in rng.choice(pats, size=len(pats), replace=True)])
        if y[rows].min() == y[rows].max():
            continue
        ar = roc_auc_score(y[rows], p_rep[rows])
        af = roc_auc_score(y[rows], p_floor[rows])
        b_rep.append(ar); b_flo.append(af); b_d.append(ar - af)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    d = np.array(b_d)
    return {
        "rep_auc": auc_rep, "rep_ci": ci(b_rep),
        "floor_auc": auc_flo, "floor_ci": ci(b_flo),
        "delta": float(auc_rep - auc_flo), "delta_ci": ci(b_d),
        # one-sided evidence that fundus EXCEEDS the floor:
        "p_rep_gt_floor": float((d > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-pca", type=int, default=16)
    args = ap.parse_args()

    d = np.load(config.REPO_ROOT / "data" / "derived" / f"grape_{REP}_emb.npz", allow_pickle=True)
    emb = d["emb"].astype(float)
    eye = d["eye_id"].astype(str)
    cam = d["camera"].astype(str)
    g = np.array([e.split("_")[0] for e in eye])
    resid_pca = PCA(args.n_pca, random_state=0).fit_transform(_residualize(emb, cam))
    p_floor = (cam == "topcon").astype(float)

    out = {"representation": REP, "n_pca": args.n_pca, "n_boot": args.n_boot, "endpoints": {}}
    print(f"paired floor test — representation = {REP} (camera-residualized), {args.n_boot} patient-bootstrap draws\n")
    for EL, ep in ENDPOINTS:
        y = d[f"prog_{ep}"].astype(int)
        p_rep = _repeated_oof(resid_pca, y, g)
        r = _paired_ci(p_rep, p_floor, y, g, args.n_boot)
        r["n_pos"] = int(y.sum())
        out["endpoints"][ep] = r
        print(f"{EL:4s} (n_pos={int(y.sum()):2d}):  floor {r['floor_auc']:.3f} [{r['floor_ci'][0]:.2f},{r['floor_ci'][1]:.2f}]"
              f"  |  fundus {r['rep_auc']:.3f} [{r['rep_ci'][0]:.2f},{r['rep_ci'][1]:.2f}]"
              f"  |  d=fundus-floor {r['delta']:+.3f} [{r['delta_ci'][0]:+.2f},{r['delta_ci'][1]:+.2f}]"
              f"  |  P(fundus>floor)={r['p_rep_gt_floor']:.2f}")

    Path(config.REPO_ROOT / "data" / "derived" / "grape_floor_ci.json").write_text(json.dumps(out, indent=2))
    print("\n[floor-ci] wrote data/derived/grape_floor_ci.json")


if __name__ == "__main__":
    main()
