"""The modality × #visits surface for GRAPE progression: OOF AUC predicting the FIXED shipped MD label from each
modality's features computed over the first ``k`` visits (all 263 eyes at every k — eyes with <k visits use what
they have, so the cohort is fixed and there is no power wall). Lines: ALL modalities (multimodal), then VF-alone,
fundus-alone, IOP-alone, OCT-alone. Serial modalities (VF, IOP) accrue with k; baseline-only modalities (fundus,
OCT) are flat — so the multimodal climb is decomposed into the label-defining VF trajectory vs the flat imaging.
This is the CIRCULAR surface (input window grows into the label window); the honest non-circular landmark is null
(run_grape_landmark_interaction.py).

  .venv/bin/python scripts/run_grape_visit_surface.py [--n-boot 200] [--n-pca 16]  -> data/derived/grape_visit_surface.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from glaucoma_triage import config
from glaucoma_triage.datasets import grape

MODS = ["all", "vf", "fundus", "iop", "oct"]


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _iop_window(fu, k):
    rows = []
    for eid, g in fu.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        io = pd.to_numeric(g["iop"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(io)
        t, io = t[ok], io[ok]
        if k is not None:
            t, io = t[:k], io[:k]
        slope = float(np.polyfit(t, io, 1)[0]) if len(io) >= 2 else 0.0
        rows.append({"eye_id": eid, "iw_slope": slope, "iw_now": float(io[-1]) if len(io) else np.nan,
                     "iw_mean": float(np.mean(io)) if len(io) else np.nan,
                     "iw_sd": float(np.std(io)) if len(io) else 0.0})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-pca", type=int, default=16)
    ap.add_argument("--emb", default="data/derived/grape_retfound_cfp_emb.npz")
    args = ap.parse_args()
    SE = _imp("scripts/run_grape_serial_event.py", "se")
    LS = _imp("scripts/run_grape_fullfundus_landscape.py", "ls")

    raw = config.raw_dir("grape")
    base = grape.load_grape(raw)
    fu = grape.load_grape_followup(raw)
    # SERIAL fundus: per-CFP camera-residualised RETFound embedding (baseline + follow-up CFPs) → PCA, then
    # mean-pool per eye over visits ≤ k (accrues with k, symmetric with VF/IOP). Camera residualised per-visit
    # because the acquisition device switches between visits.
    fd = np.load("data/derived/grape_retfound_followup_emb.npz", allow_pickle=True)
    femb = LS._resid(np.nan_to_num(fd["emb"].astype(float)), fd["camera"].astype(str))
    fpca_all = PCA(args.n_pca, random_state=0).fit_transform(femb)
    feye, fvis = fd["eye_id"].astype(str), fd["visit"].astype(int)
    fcols = [f"f{j:02d}" for j in range(args.n_pca)]

    def fundus_window(k, eyes):
        rows = []
        for eid in eyes:
            m = feye == eid
            mk = (m & (fvis <= k)) if k is not None else m
            if mk.any():
                vec = fpca_all[mk].mean(0)
            elif m.any():
                vec = fpca_all[m][int(np.argmin(fvis[m]))]
            else:
                vec = np.full(args.n_pca, np.nan)
            rows.append(vec)
        out = pd.DataFrame(rows, columns=fcols)
        out.insert(0, "eye_id", list(eyes))
        return out

    octcols = list(grape.OCT_FEATURES)
    octdf = base[["eye_id"] + octcols].copy()
    for c in octcols:
        octdf[c] = pd.to_numeric(octdf[c], errors="coerce")
        octdf[c] = octdf[c].fillna(octdf[c].median())
    EPS = [("prog_md", "MD"), ("prog_plr2", "PLR2"), ("prog_plr3", "PLR3")]
    lab = base[["eye_id"] + [c for c, _ in EPS]].copy()

    KS = [1, 2, 3, 4, 5, 6, None]
    klabels = ["1", "2", "3", "4", "5", "6", "full"]
    SET = {"vf": SE._DIAL_COLS, "iop": ["iw_slope", "iw_now", "iw_mean", "iw_sd"], "fundus": fcols, "oct": octcols}
    SET["all"] = SET["vf"] + SET["iop"] + SET["fundus"] + SET["oct"]
    RES = {"k": klabels, "endpoints": {name: {m: [] for m in MODS} for _, name in EPS}}
    for k, kl in zip(KS, klabels):
        vf = SE._window_features(fu, k)
        fdf = fundus_window(k, vf["eye_id"].tolist())
        feat = vf.merge(_iop_window(fu, k), on="eye_id").merge(fdf, on="eye_id").merge(octdf, on="eye_id").merge(lab, on="eye_id")
        for c in fcols:
            feat[c] = feat[c].fillna(feat[c].median())
        line = [f"k={kl:>4}:"]
        for col, name in EPS:
            y = feat[col].to_numpy()
            for m in MODS:
                auc, lo, hi = SE._circular_auc(feat, y, SET[m], n_boot=args.n_boot)
                RES["endpoints"][name][m].append(round(float(auc), 3))
            line.append(f"{name}[all {RES['endpoints'][name]['all'][-1]:.2f} vf {RES['endpoints'][name]['vf'][-1]:.2f} fun {RES['endpoints'][name]['fundus'][-1]:.2f}]")
        print(" ".join(line), flush=True)

    out = config.REPO_ROOT / "data" / "derived" / "grape_visit_surface.json"
    out.write_text(json.dumps(RES, indent=2))
    print(f"[visit-surface] wrote {out}")


if __name__ == "__main__":
    main()
