"""Well-powered BASELINE PROSPECTIVE forecast: predict the shipped PLR2/PLR3/MD labels from the FIRST (baseline) visit,
patient-grouped repeated-OOF + patient cluster-bootstrap CI, NAIVE and camera-RESIDUALIZED. Full events (40/14/27) --
the well-powered replacement for the underpowered future-held-out landmark.

The baseline visit is the starting state, before any decline, so predicting the future-trajectory label from it is a
genuine PROSPECTIVE forecast (the Mohammadzadeh 0.84 / OHTS 0.74 design) and NON-CIRCULAR (one point can't compute the
label slope, which needs >=3 visits). We run it two ways, both using the SAME frozen RETFound-CFP disc embedding for the
fundus so the comparison is fair:
  * IMAGE-ONLY  = PCA(16) of the baseline RETFound-CFP disc embedding (= the fm_disc probe).
  * FUSION      = that same fundus embedding + baseline visual field (mean-sens, SD) + OCT-RNFL (mean + 4 sectors) +
                  IOP + structure (age, CCT).
Both predict the shipped label; the claim is that adding VF+OCT+IOP to the fundus embedding still can't beat the floor.

  .venv/bin/python scripts/run_grape_prospective_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from glaucoma_triage.datasets import grape
from glaucoma_triage.progression import survival as S

ROOT = Path(__file__).resolve().parent.parent
LABELS = [("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"), ("MD", "prog_md")]
TAB_COLS = ["vf_mean_sens", "vf_sd", "iop", "rnfl_mean", "rnfl_s", "rnfl_n", "rnfl_i", "rnfl_t", "age", "cct"]


def _resid_within_camera(X, cam):
    Xr = X.astype(float).copy()
    for c in np.unique(cam):
        m = cam == c
        mu = np.nanmean(X[m], axis=0); sd = np.nanstd(X[m], axis=0) + 1e-8
        Xr[m] = (X[m] - mu) / sd
    return np.nan_to_num(Xr)


def _fit(X, y, g, n_boot=1000):
    p = S._event_oof_proba(X, y, g)
    return [round(v, 3) for v in S._auc_cluster_ci(p, y, g, n_boot=n_boot)] if p is not None else [float("nan")] * 3


def main():
    raw = ROOT / "data" / "raw" / "grape"
    base = grape.load_grape(raw)                                             # baseline (visit-1) tabular + shipped labels
    d = np.load(ROOT / "data" / "derived" / "grape_retfound_cfp_disc_emb.npz", allow_pickle=True)
    emb_pca = PCA(16, random_state=0).fit_transform(np.nan_to_num(d["emb"].astype(np.float32)))   # (263, 16)
    eid = d["eye_id"].astype(str); cam_arr = d["camera"].astype(str)
    emb_by = {e: emb_pca[i] for i, e in enumerate(eid)}; cam_by = {e: cam_arr[i] for i, e in enumerate(eid)}

    eyes = base["eye_id"].astype(str).to_numpy()
    g = pd.Series(eyes).str.split("_").str[0].to_numpy()
    cam = np.array([cam_by.get(e, "topcon") for e in eyes])
    X_img = np.array([emb_by.get(e, np.zeros(16)) for e in eyes])                                  # fundus embedding
    X_tab = np.nan_to_num(base[TAB_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float))    # VF+OCT+IOP+struct
    X_fus = np.column_stack([X_img, X_tab])

    out = {}
    for regime, X in (("image", X_img), ("fusion", X_fus)):
        Xr = _resid_within_camera(X, cam)
        block = {}
        for ep, col in LABELS:
            y = base[col].to_numpy().astype(int)
            naive = _fit(X, y, g); resid = _fit(Xr, y, g)
            block[ep] = {"events": int(y.sum()), "naive": naive, "resid": resid}
            print(f"[{regime} first-1] {ep} (ev {int(y.sum())}): naive {naive[0]:.2f} [{naive[1]:.2f},{naive[2]:.2f}] "
                  f"| resid {resid[0]:.2f} [{resid[1]:.2f},{resid[2]:.2f}]", flush=True)
        out[f"{regime}_first1"] = block

    (ROOT / "data" / "derived" / "grape_prospective_baseline.json").write_text(json.dumps(out, indent=2))
    print("wrote data/derived/grape_prospective_baseline.json")


if __name__ == "__main__":
    main()
