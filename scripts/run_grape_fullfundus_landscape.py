"""§02 (endpoint-sensitivity landscape + fusion) and §03 (device confound) on the FULL-FUNDUS RETFound embedding.

The artifact is standardized on the full-fundus (`fov`) RETFound-CFP embedding — the in-distribution crop for
RETFound (pretrained on whole CFPs). This computes, per endpoint (PLR2/PLR3/MD), from ONE embedding:

  §02  single-modality camera-controlled AUC[CI] for VF · OCT-RNFL · IOP · fundus(FM, camera-residualized PCA-16)
       + an all-modality FUSION AUC + the training-free camera-only floor.  (NO morphometry.)
  §03  device confound — camera-only floor, camera-decode AUC (RAW embedding → topcon/canon), naive-FM vs resid-FM.

Accepts either the local frozen npz (`grape_retfound_cfp_emb.npz`: keys emb/eye_id/camera) or the Colab
full-fundus npz (`phase3_fullfundus_reps.npz`: key `retfound_cfp__lp_emb`). → data/derived/grape_fullfundus_landscape.json

    .venv/bin/python scripts/run_grape_fullfundus_landscape.py [--emb data/derived/grape_retfound_cfp_emb.npz] [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json

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

ENDPOINTS = [("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"), ("MD", "prog_md")]
N_PCA = 16


def _probe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced"))])


def _repeated_oof(X, y, g, seeds=8):
    acc = np.zeros((len(y), seeds))
    for s in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            acc[te, s] = _probe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return acc.mean(axis=1)


def _auc_ci(p, y, g, n_boot, seed=0):
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in pats}
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[u] for u in rng.choice(pats, size=len(pats), replace=True)])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        draws.append(roc_auc_score(yy, p[rows]))
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (np.nan, np.nan)
    return float(auc), lo, hi


def _resid(emb, cam):
    emb = np.asarray(emb, float).copy()
    for c in np.unique(cam):
        i = cam == c
        emb[i] = (emb[i] - emb[i].mean(0)) / (emb[i].std(0) + 1e-6)
    return emb


def _camera_decode_auc(raw, cam, g):
    y = (cam == "canon").astype(int)
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(raw, y, g):
        pipe = Pipeline([("pca", PCA(32, random_state=0)), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.5, max_iter=3000))])
        oof[te] = pipe.fit(raw[tr], y[tr]).predict_proba(raw[te])[:, 1]
    return float(roc_auc_score(y, oof))


def _load_emb(path):
    d = np.load(path, allow_pickle=True)
    if "emb" in d.files:
        return d["emb"].astype(float), d["eye_id"].astype(str), d["camera"].astype(str)
    return d["retfound_cfp__lp_emb"].astype(float), d["eye_id"].astype(str), d["camera"].astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", default="data/derived/grape_retfound_cfp_emb.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-pca", type=int, default=N_PCA)
    args = ap.parse_args()
    raw = config.raw_dir("grape")
    base = grape.load_grape(raw)

    emb, eye, cam = _load_emb(args.emb)
    tab = pd.DataFrame({"eye_id": eye}).merge(base, on="eye_id", how="left")
    assert (tab["eye_id"].to_numpy() == eye).all()
    g = np.array([e.split("_")[0] for e in eye])
    for c in grape.OCT_FEATURES + grape.VF_FEATURES + ["iop"]:
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
        tab[c] = tab[c].fillna(tab[c].median())

    raw_pca = PCA(args.n_pca, random_state=0).fit_transform(np.nan_to_num(emb))          # naive (no residualization)
    resid_pca = PCA(args.n_pca, random_state=0).fit_transform(_resid(np.nan_to_num(emb), cam))
    TAB = {"vf": tab[grape.VF_FEATURES].to_numpy(), "oct": tab[grape.OCT_FEATURES].to_numpy(),
           "iop": tab[["iop"]].to_numpy(), "fundus": resid_pca}
    fusion_X = np.hstack([TAB["vf"], TAB["oct"], TAB["iop"], TAB["fundus"]])

    RES = {"emb": args.emb, "n_eyes": int(len(eye)), "landscape": {}, "device": {},
           "camera_decode_auc": _camera_decode_auc(np.nan_to_num(emb), cam, g)}
    print(f"[full-fundus] camera-decode AUC {RES['camera_decode_auc']:.3f}")
    for name, col in ENDPOINTS:
        y = tab[col].astype(int).to_numpy()
        floor = roc_auc_score(y, (cam == "topcon").astype(int))
        cells = {m: _auc_ci(_repeated_oof(X, y, g), y, g, args.n_boot) for m, X in TAB.items()}
        fus = _auc_ci(_repeated_oof(fusion_X, y, g), y, g, args.n_boot)
        naive = _auc_ci(_repeated_oof(raw_pca, y, g), y, g, args.n_boot)
        RES["landscape"][col] = {"floor": float(floor), "fusion": list(fus),
                                 **{m: list(v) for m, v in cells.items()}}
        RES["device"][col] = {"floor": float(floor), "naive_fm": list(naive), "resid_fm": list(cells["fundus"])}
        lead = max(cells, key=lambda m: cells[m][0])
        print(f"  {name}: floor {floor:.3f} | vf {cells['vf'][0]:.3f} oct {cells['oct'][0]:.3f} "
              f"iop {cells['iop'][0]:.3f} fundus {cells['fundus'][0]:.3f} | fusion {fus[0]:.3f} | lead {lead}")

    out = config.REPO_ROOT / "data" / "derived" / "grape_fullfundus_landscape.json"
    out.write_text(json.dumps(RES, indent=2, default=float))
    print(f"[full-fundus] wrote {out}")


if __name__ == "__main__":
    main()
