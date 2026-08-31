"""NON-CIRCULAR multimodal landmark interaction for GRAPE progression.

Input = modalities summarised over the EARLY visits 1..L only; target = progression on the DISJOINT held-out
FUTURE window L+1..end — so the input never touches the label window (unlike the field's whole-series serial
models). Modalities: VF-history (recent/slope/variability over 1..L), IOP-history (recent/slope), baseline OCT-RNFL
(no serial OCT in GRAPE), baseline fundus (full-fundus RETFound, camera-residualised PCA-16). Reads:

  • PRIMARY (well-powered): PID net-synergy on the CONTINUOUS future MD-slope rate (every eye with ≥L+3 visits) —
    the honest non-circular interaction. Single-modality + fusion R² beside it.
  • SECONDARY (power-walled): future binary EVENT AUC per modality + fusion (13/5 events at L=1/2 — wide CIs).

  .venv/bin/python scripts/run_grape_landmark_interaction.py [--landmarks 1,2] [--n-boot 500] [--n-pca 16]
  -> data/derived/grape_landmark_interaction.json
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
from glaucoma_triage.interaction import pid
from glaucoma_triage.progression import survival

VF_HIST = ["vf_now", "vf_slope_hist", "vf_sd_hist"]
IOP_HIST = ["iop_now", "iop_slope_hist"]
PAIRS = [("oct", "vf"), ("fundus", "vf"), ("fundus", "oct")]


def _landscape():
    spec = importlib.util.spec_from_file_location("lscape", "scripts/run_grape_fullfundus_landscape.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _future_rate(followup, L, min_future=3):
    out = {}
    for eid, g in followup.sort_values("visit").groupby("eye_id"):
        g = g.reset_index(drop=True)
        if len(g) < L + min_future:
            continue
        fut = g.iloc[L:]
        tf = pd.to_numeric(fut["interval_years"], errors="coerce").to_numpy()
        sf = pd.to_numeric(fut["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(tf) & np.isfinite(sf)
        out[str(eid)] = float(np.polyfit(tf[ok], sf[ok], 1)[0]) if ok.sum() >= 2 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", default="1,2")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-pca", type=int, default=16)
    ap.add_argument("--emb", default="data/derived/grape_retfound_cfp_emb.npz")
    args = ap.parse_args()
    L = _landscape()
    raw = config.raw_dir("grape")
    base = grape.load_grape(raw)
    fu = grape.load_grape_followup(raw)

    emb, eye, cam = L._load_emb(args.emb)
    fpca = PCA(args.n_pca, random_state=0).fit_transform(L._resid(np.nan_to_num(emb), cam))
    fcols = [f"fundus_{j:02d}" for j in range(fpca.shape[1])]
    fdf = pd.DataFrame({"eye_id": eye, **{c: fpca[:, j] for j, c in enumerate(fcols)}})

    octcols = list(grape.OCT_FEATURES)
    RES = {"n_pca": args.n_pca, "landmarks": {}}
    for Lm in [int(x) for x in args.landmarks.split(",")]:
        ev = survival.landmark_event_dataset(fu, base, Lm, min_future=3)   # VF/IOP history + baseline + future event
        rate = _future_rate(fu, Lm)
        ev = ev.merge(fdf, on="eye_id", how="left")
        miss_oct = [c for c in octcols if c not in ev.columns]
        if miss_oct:
            ev = ev.merge(base[["eye_id"] + miss_oct], on="eye_id", how="left")
        for c in octcols + VF_HIST + IOP_HIST:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
            ev[c] = ev[c].fillna(ev[c].median())
        ev = ev[ev[fcols[0]].notna()].reset_index(drop=True)
        ev["y_rate"] = ev["eye_id"].map(rate)
        g = ev["patient_id"].to_numpy()

        spec = {"vf": VF_HIST, "oct": octcols, "fundus": fcols, "iop": IOP_HIST}
        rec = {"n_eyes": int(len(ev)), "events": int(ev["event"].sum()),
               "single_rate_r2": {}, "single_event_auc": {}, "pid_rate": {}}

        # continuous future-rate: single-modality R² + PID interaction (the non-circular, well-powered read)
        ms = ev["y_rate"].to_numpy(dtype=float)
        okm = np.isfinite(ms)
        for m, cols in spec.items():
            rec["single_rate_r2"][m] = survival.grouped_oof_r2(ev.loc[okm, cols], ms[okm], g[okm])
        for a, b in PAIRS:
            pa = pid.pair_analysis(ms[okm], ev.loc[okm, spec[a]].to_numpy(), ev.loc[okm, spec[b]].to_numpy(),
                                   g[okm], n_boot=args.n_boot, n_perm=args.n_boot, cross_fit=True)
            rec["pid_rate"][f"{a}x{b}"] = {"net_synergy": pa.pid.net_synergy, "p": pa.p_perm.get("net_synergy")}

        # future binary EVENT: single-modality + fusion AUC (power-walled — report with caveat)
        y = ev["event"].astype(int).to_numpy()
        if 0 < y.sum() < len(y):
            for m, cols in spec.items():
                rec["single_event_auc"][m] = float(roc_auc_score(y, L._repeated_oof(ev[cols].to_numpy(), y, g)))
            Xf = np.hstack([ev[spec[m]].to_numpy() for m in spec])
            rec["event_fusion_auc"] = float(roc_auc_score(y, L._repeated_oof(Xf, y, g)))
        RES["landmarks"][str(Lm)] = rec
        print(f"[L={Lm}] n={rec['n_eyes']} ev={rec['events']} | "
              f"rate-R2 " + " ".join(f"{m} {rec['single_rate_r2'][m]:+.3f}" for m in spec) +
              " | PID " + " ".join(f"{a}x{b} {rec['pid_rate'][f'{a}x{b}']['net_synergy']:+.3f}"
                                   f"(p{rec['pid_rate'][f'{a}x{b}']['p']:.2f})" for a, b in PAIRS))
        if rec["single_event_auc"]:
            print(f"       event-AUC " + " ".join(f"{m} {rec['single_event_auc'][m]:.2f}" for m in spec) +
                  f" | fusion {rec.get('event_fusion_auc', float('nan')):.2f}")

    out = config.REPO_ROOT / "data" / "derived" / "grape_landmark_interaction.json"
    out.write_text(json.dumps(RES, indent=2, default=float))
    print(f"[landmark-interaction] wrote {out}")


if __name__ == "__main__":
    main()
