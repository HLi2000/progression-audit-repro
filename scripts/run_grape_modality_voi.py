"""Per-modality value-of-information for GRAPE progression: each modality's INCREMENTAL ΔAUC over the other three
(leave-one-out), with a PAIRED patient-cluster bootstrap CI — the honest "does this test add non-redundant value?"
read for EVERY modality (not just fundus). Fundus = full-fundus RETFound, camera-residualized PCA-16 (matches §02).

  .venv/bin/python scripts/run_grape_modality_voi.py [--n-boot 1000] [--n-pca 16]
  -> data/derived/grape_modality_voi.json  {endpoint: {full_auc, delta:{modality:[dAUC,lo,hi]}}}
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from glaucoma_triage import config
from glaucoma_triage.datasets import grape

ENDPOINTS = [("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"), ("MD", "prog_md")]
MODS = ["vf", "oct", "fundus", "iop"]


def _load_landscape():
    spec = importlib.util.spec_from_file_location("lscape", "scripts/run_grape_fullfundus_landscape.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-pca", type=int, default=16)
    ap.add_argument("--emb", default="data/derived/grape_retfound_cfp_emb.npz")
    args = ap.parse_args()
    L = _load_landscape()

    base = grape.load_grape(config.raw_dir("grape"))
    emb, eye, cam = L._load_emb(args.emb)
    tab = pd.DataFrame({"eye_id": eye}).merge(base, on="eye_id", how="left")
    assert (tab["eye_id"].to_numpy() == eye).all()
    g = np.array([e.split("_")[0] for e in eye])
    for c in grape.OCT_FEATURES + grape.VF_FEATURES + ["iop"]:
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
        tab[c] = tab[c].fillna(tab[c].median())
    fpca = PCA(args.n_pca, random_state=0).fit_transform(L._resid(np.nan_to_num(emb), cam))
    BLOCK = {"vf": tab[grape.VF_FEATURES].to_numpy(), "oct": tab[grape.OCT_FEATURES].to_numpy(),
             "iop": tab[["iop"]].to_numpy(), "fundus": fpca}

    pats = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in pats}

    def paired_dauc(pf, pl, y, seed=0):
        d = roc_auc_score(y, pf) - roc_auc_score(y, pl)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(args.n_boot):
            rows = np.concatenate([idx[u] for u in rng.choice(pats, len(pats), True)])
            yy = y[rows]
            if yy.min() == yy.max():
                continue
            draws.append(roc_auc_score(yy, pf[rows]) - roc_auc_score(yy, pl[rows]))
        return float(d), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    RES = {}
    for ep, col in ENDPOINTS:
        y = tab[col].astype(int).to_numpy()
        Xfull = np.hstack([BLOCK[m] for m in MODS])
        pf = L._repeated_oof(Xfull, y, g)
        RES[ep] = {"events": int(y.sum()), "full_auc": float(roc_auc_score(y, pf)), "delta": {}}
        for m in MODS:
            pl = L._repeated_oof(np.hstack([BLOCK[k] for k in MODS if k != m]), y, g)
            RES[ep]["delta"][m] = list(paired_dauc(pf, pl, y))
            d, lo, hi = RES[ep]["delta"][m]
            print(f"  {ep} +{m:6s}|rest: {d:+.3f} [{lo:+.2f},{hi:+.2f}]", flush=True)

    out = config.REPO_ROOT / "data" / "derived" / "grape_modality_voi.json"
    out.write_text(json.dumps(RES, indent=2, default=float))
    print(f"[modality-voi] wrote {out}")


if __name__ == "__main__":
    main()
