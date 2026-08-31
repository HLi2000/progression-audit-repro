"""Release a FIXED patient-grouped five-fold split for GRAPE, so any re-evaluation scores on identical folds.

Both eyes of a patient always land in the same fold (fellow-eye leakage is impossible), the assignment is
deterministic (sklearn GroupKFold, no randomness), and it is endpoint-agnostic (one split usable for PLR2/PLR3/MD).
Paired with the reporting template (App.~\ref{app:template}), this turns "re-evaluate on GRAPE" into a single,
unambiguous protocol.

    .venv/bin/python scripts/run_grape_standard_splits.py [--raw data/raw/grape]
-> data/derived/grape_standard_splits.csv  (eye_id, patient_id, fold, prog_plr2, prog_plr3, prog_md)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from glaucoma_triage.datasets import grape


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/grape")
    args = ap.parse_args()
    base = grape.load_grape(args.raw).copy()
    base["patient_id"] = base["eye_id"].astype(str).str.split("_").str[0]
    eyes = base["eye_id"].to_numpy()
    groups = base["patient_id"].to_numpy()

    fold = np.full(len(eyes), -1, dtype=int)
    for f, (_, te) in enumerate(GroupKFold(n_splits=5).split(eyes, groups=groups)):
        fold[te] = f
    assert (fold >= 0).all()
    # sanity: no patient split across folds
    assert base.assign(fold=fold).groupby("patient_id")["fold"].nunique().max() == 1

    out = base[["eye_id", "patient_id"]].assign(fold=fold)
    for ep in ("prog_plr2", "prog_plr3", "prog_md"):
        if ep in base:
            out[ep] = base[ep].astype("Int64")
    dst = Path("data/derived/grape_standard_splits.csv")
    out.to_csv(dst, index=False)
    n_pat = base["patient_id"].nunique()
    sizes = out.groupby("fold").size().to_dict()
    print(f"wrote {dst}  ({len(out)} eyes / {n_pat} patients; fold sizes {sizes})")


if __name__ == "__main__":
    main()
