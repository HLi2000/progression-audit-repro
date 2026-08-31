"""TSDF eye-level split CEILING under a STANDARD per-eye ROC-AUC: the max (and p90) over random 80/20 eye-level splits,
so Fig 4's `reproduced` column is defined consistently with the other methods (best single leaky split, standard AUC).
Reuses TSDF's released DSMIL + features. This is the honest leaky ceiling; it is not expected to reach the reported
value under a standard AUC.

  .venv-torch/bin/python scripts/run_grape_tsdf_eyelevel_max.py
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from run_grape_tsdf_repro import ENDPOINTS, _epoch_train, _new_model, _predict, load_bags

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "derived", "tsdf_grape_feats")


def eyelevel_dist(bags, n_splits=30, max_epochs=40):
    y = np.array([b["y"] for b in bags]); idx = np.arange(len(bags)); aucs = []
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
    for si, (tr, te) in enumerate(sss.split(idx, y)):
        if y[te].min() == y[te].max():
            continue
        net, crit, opt, sch = _new_model(seed=si)
        for _ in range(max_epochs):
            _epoch_train(net, crit, opt, [bags[i] for i in tr]); sch.step()
        probs, ys = _predict(net, [bags[i] for i in te])
        aucs.append(roc_auc_score(ys, probs[:, 1]))                    # STANDARD per-eye ROC-AUC (positive-class prob)
    aucs = np.array(aucs)
    return {"max": round(float(aucs.max()), 3), "p90": round(float(np.percentile(aucs, 90)), 3),
            "median": round(float(np.median(aucs)), 3), "n": int(len(aucs))}


def main():
    out = {}
    for e in ENDPOINTS:
        bags = load_bags(DATA, e)
        d = eyelevel_dist(bags)
        out[e] = d
        print(f"[TSDF eye-level {e}] standard-AUC over {d['n']} splits: max {d['max']:.3f} "
              f"p90 {d['p90']:.3f} median {d['median']:.3f}", flush=True)
    from pathlib import Path
    Path(__file__).resolve().parent.parent.joinpath("data", "derived", "grape_tsdf_eyelevel_max.json").write_text(json.dumps(out, indent=2))
    print("wrote data/derived/grape_tsdf_eyelevel_max.json")


if __name__ == "__main__":
    main()
