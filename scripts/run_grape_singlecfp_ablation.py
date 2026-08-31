"""Single baseline-CFP ablation (Huang / Fouda regime) with the SAME estimator as the serial models, so Fig 4 covers all
five published methods consistently. One frozen model throughout -- a RETFound-CFP disc-embedding probe on the single
baseline image -- evaluated as: reproduced (eye-group OOF, camera-kept, standard per-eye ROC-AUC = the leaky read),
-device (camera-residualized, eye-group), -fair split (patient-grouped), -all (patient-grouped + residualized). There is
no serial window, so -non-circularity is N/A. A single baseline image carries no temporal signal: every controlled read
sits at the camera floor, so the class fails the DEVICE check.

  .venv/bin/python scripts/run_grape_singlecfp_ablation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from glaucoma_triage.progression import survival as S

ROOT = Path(__file__).resolve().parent.parent
LABELS = [("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"), ("MD", "prog_md")]


def _resid_within_camera(X, cam):
    Xr = X.astype(float).copy()
    for c in np.unique(cam):
        m = cam == c
        mu = np.nanmean(X[m], axis=0); sd = np.nanstd(X[m], axis=0) + 1e-8
        Xr[m] = (X[m] - mu) / sd
    return np.nan_to_num(Xr)


def _cell(X, y, cv_group, patient, n_boot=1000):
    p = S._event_oof_proba(X, y, cv_group)
    return [round(v, 3) for v in S._auc_cluster_ci(p, y, patient, n_boot=n_boot)] if p is not None else [float("nan")] * 3


def _eyelevel_max(X, y, seeds=200):
    """Leaky eye-level ceiling: max standard per-eye ROC-AUC over random 80/20 eye-level splits (the best single split)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for tr, te in StratifiedShuffleSplit(seeds, test_size=0.2, random_state=0).split(X, y):
        if y[te].min() == y[te].max():
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000))])
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return round(float(np.max(aucs)), 3) if aucs else float("nan")


def main():
    d = np.load(ROOT / "data" / "derived" / "grape_retfound_cfp_disc_emb.npz", allow_pickle=True)
    X = PCA(16, random_state=0).fit_transform(np.nan_to_num(d["emb"].astype(np.float32)))   # (263,16) single baseline disc
    eye = d["eye_id"].astype(str); cam = d["camera"].astype(str)
    pat = np.array([e.split("_")[0] for e in eye])
    eye_g = np.arange(len(eye)).astype(str)          # eye-level CV groups (each eye unique) = leaky
    Xr = _resid_within_camera(X, cam)

    out = {}
    for ep, col in LABELS:
        y = d[col].astype(int)
        cell = {"reproduced": [_eyelevel_max(X, y)] * 3,    # eye-level split ceiling (best single split, standard AUC)
                "reproduced_oof": _cell(X, y, eye_g, pat),  # eye-group pooled OOF (kept for reference)
                "device":     _cell(Xr, y, eye_g, pat),     # -device (eye-group, camera-residualized)
                "fair_split": _cell(X, y, pat, pat),        # -fair split (patient-grouped)
                "circular":   None,                          # single baseline image: no serial window
                "all":        _cell(Xr, y, pat, pat),        # -all (patient-grouped + residualized) = floor
                "events":     int(y.sum())}
        out[ep] = cell
        print(f"[single-CFP {ep}] ev{int(y.sum())}: reproduced {cell['reproduced'][0]:.2f} | "
              f"-device {cell['device'][0]:.2f} | -fair split {cell['fair_split'][0]:.2f} | all {cell['all'][0]:.2f}",
              flush=True)

    (ROOT / "data" / "derived" / "grape_singlecfp_ablation.json").write_text(json.dumps(out, indent=2))
    print("wrote data/derived/grape_singlecfp_ablation.json")


if __name__ == "__main__":
    main()
