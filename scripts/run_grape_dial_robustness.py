"""Circularity-dial ROBUSTNESS: is the k=3 leakage onset a property of the LABEL, or an artifact of counting visits?

The dial (run_grape_serial_event.py) predicts the fixed shipped label from the first-k-visit VF statistics; on the
MD-slope label it jumps at k=3, exactly when a slope and its p-value first become computable. If that onset is
predicted a priori by the label's DEFINITION (not the arithmetic of "more visits = more signal"), then:

  * loosening/tightening the slope's significance threshold (p_thr) must LEAVE the jump at k=3 (still a 3-point slope), and
  * a placebo label whose defining statistic is computable at k=2 must have its jump MOVE to k=2.

We recompute the dial for each alternative label, reusing the exact window-feature + patient-grouped-OOF pipeline from
run_grape_serial_event.py. Sanity: the shipped-MD dial reproduces the committed panel (0.47/0.47/0.78/.../0.90).

    .venv/bin/python scripts/run_grape_dial_robustness.py [--raw data/raw/grape] [--n-boot 1000]
-> data/derived/grape_dial_robustness.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from glaucoma_triage.datasets import grape
from glaucoma_triage.progression import survival as SV

KS = [1, 2, 3, 4, 5, 6, None]


def _dial_module():
    spec = importlib.util.spec_from_file_location("sde", str(Path(__file__).with_name("run_grape_serial_event.py")))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _two_visit_decline(fu: pd.DataFrame, pct: float = 15.0) -> pd.Series:
    """Placebo label computable at k=2: the steepest-declining `pct`% of eyes by the FIRST-TWO-VISIT slope."""
    rows = []
    for eid, g in fu.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s); t, s = t[ok], s[ok]
        rows.append({"eye_id": eid, "slope2": float(np.polyfit(t[:2], s[:2], 1)[0]) if len(s) >= 2 else 0.0})
    d = pd.DataFrame(rows).set_index("eye_id")
    return (d["slope2"] < np.nanpercentile(d["slope2"], pct)).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/grape")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    m = _dial_module()
    base = grape.load_grape(args.raw)
    fu = grape.load_grape_followup(args.raw)

    labels = {
        "shipped_md_slope_p05": base.set_index("eye_id")["prog_md"].astype(int),          # slope, p<.05 -> onset k=3
        "md_slope_p10": SV.md_slope_event_table(fu, p_thr=0.10).set_index("eye_id")["event"].astype(int),  # k=3
        "placebo_2visit_k2": _two_visit_decline(fu),                                       # 2-point change -> onset k=2
    }
    out = {"ks": ["full" if k is None else k for k in KS], "n_eyes": int(len(base)),
           "prevalence": {n: int(l.sum()) for n, l in labels.items()}, "dial": {}}
    for name, lab in labels.items():
        curve = []
        for k in KS:
            fk = m._window_features(fu, k)
            a, lo, hi = m._circular_auc(fk, lab.reindex(fk["eye_id"]).to_numpy(), m._DIAL_COLS, n_boot=args.n_boot)
            curve.append([round(a, 3), round(lo, 3), round(hi, 3)])
            print(f"[dial-robust] {name:24s} k={'full' if k is None else k}: {a:.2f}", flush=True)
        out["dial"][name] = curve
    Path("data/derived/grape_dial_robustness.json").write_text(json.dumps(out, indent=2))
    print("wrote data/derived/grape_dial_robustness.json")


if __name__ == "__main__":
    main()
