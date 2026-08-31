"""Is the camera->progression link genuine baseline case-mix, or an exposure-time artifact?

A device confound could be a spurious imaging shortcut OR a real case-mix gradient the camera happens to index (if one
camera photographed sicker eyes, a fundus model reading severity would be doing legitimate prognosis). We separate the
two by stratifying baseline severity AND follow-up length by baseline camera (binary TOPCON vs Canon).

Finding: baseline severity is matched (VF mean sensitivity, RNFL, IOP, age, CCT all p>0.05), so the camera does not
index more-advanced eyes; but TOPCON eyes are followed markedly longer (median 3.12 vs 2.03 yr, p<1e-3), and PLR/MD-slope
significance is mechanically easier to reach with longer follow-up. So the camera-progression association is an
exposure-time artifact, not baseline case-mix -- still a data artifact, not retinal prognosis.

    .venv/bin/python scripts/run_grape_camera_casemix.py [--raw data/raw/grape]
-> data/derived/grape_camera_casemix.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from glaucoma_triage.datasets import grape

METRICS = ["age", "vf_mean_sens", "vf_sd", "rnfl_mean", "iop", "cct", "n_visits", "fu_years"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/grape")
    ap.add_argument("--emb", default="data/derived/grape_retfound_cfp_disc_emb.npz")
    args = ap.parse_args()
    base = grape.load_grape(args.raw)
    fu = grape.load_grape_followup(args.raw)
    emb = np.load(args.emb, allow_pickle=True)
    cam = pd.Series(np.where(emb["camera"].astype(str) == "topcon", "TOPCON", "Canon"),
                    index=emb["eye_id"].astype(str), name="cam")
    b = base.set_index("eye_id").copy()
    b["cam"] = cam
    b["n_visits"] = fu.groupby("eye_id")["visit"].nunique()
    b["fu_years"] = pd.to_numeric(fu["interval_years"], errors="coerce").groupby(fu["eye_id"]).max()
    b = b[b["cam"].notna()]
    for c in METRICS:
        b[c] = pd.to_numeric(b[c], errors="coerce")

    out = {"n": {c: int((b["cam"] == c).sum()) for c in ("TOPCON", "Canon")}, "metrics": {}, "prog": {}}
    for c in METRICS:
        t = b.loc[b.cam == "TOPCON", c].dropna(); k = b.loc[b.cam == "Canon", c].dropna()
        p = float(stats.mannwhitneyu(t, k).pvalue) if len(t) and len(k) else float("nan")
        out["metrics"][c] = {"TOPCON_median": round(float(t.median()), 2),
                             "Canon_median": round(float(k.median()), 2), "p_mwu": round(p, 4)}
    for c in ("TOPCON", "Canon"):
        m = b.cam == c
        out["prog"][c] = {ep: round(float(b.loc[m, f"prog_{ep}"].mean()), 3) for ep in ("plr2", "plr3", "md")}
    Path("data/derived/grape_camera_casemix.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote data/derived/grape_camera_casemix.json")


if __name__ == "__main__":
    main()
