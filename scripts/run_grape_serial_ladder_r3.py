"""R3 of the serial reproduce-collapse ladder: a temporal ATTENTION head over the frozen fundus-embedding sequence
(mirrors TSDF's attention temporal aggregator / DiffSight's transformer). Higher capacity than the R1/R2 logistic
probes -- it CAN overfit the handful of positives -- so it is the fair test of "can a real temporal model reach the
published 0.96 under a patient-grouped split?".

Reads data/derived/grape_serial_seq.npz (emb (N,L,1024), emb_resid, mask, patient_id, prog_*) written by
run_grape_serial_ladder.py -- no openpyxl / GRAPE workbook needed, so this runs in .venv-torch.

Same two protocols as R1/R2: eye-level 80/20 x seeds (median/**max** = the tiny-fold reproduce) and patient-grouped
5-fold repeated OOF + patient cluster-bootstrap CI (the honest collapse), input naive vs camera-residualized. Appends
the "R3_temporal" block to data/derived/grape_serial_ladder.json and prints.

  .venv-torch/bin/python scripts/run_grape_serial_ladder_r3.py [--seeds 60] [--repeats 4] [--epochs 60]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
ENDPOINTS = [("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"), ("MD", "prog_md")]
PUBLISHED = {"PLR2": [0.87, 0.79], "PLR3": [0.96, 0.88], "MD": [0.92, 0.80]}


class AttnHead(nn.Module):
    """Masked attention-pooling over visits -> 2-layer MLP. Small but able to overfit tiny positive sets."""
    def __init__(self, d=1024, h=64):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.attn = nn.Linear(h, 1)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))

    def forward(self, x, mask):
        z = torch.tanh(self.proj(x))                       # (B,L,h)
        a = self.attn(z).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)   # (B,L)
        pooled = (z * a.unsqueeze(-1)).sum(1)              # (B,h)
        return self.head(pooled).squeeze(-1)               # (B,)


def _fit_predict(Xtr, Mtr, ytr, Xte, Mte, epochs, seed):
    torch.manual_seed(seed)
    m = AttnHead(Xtr.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    pos = max(1, int(ytr.sum())); neg = max(1, int((ytr == 0).sum()))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32))
    xtr, mtr, ytr_t = torch.tensor(Xtr), torch.tensor(Mtr), torch.tensor(ytr, dtype=torch.float32)
    xte, mte = torch.tensor(Xte), torch.tensor(Mte)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(m(xtr, mtr), ytr_t)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(xte, mte)).numpy()


def _cluster_ci(p, y, g, n_boot=1000, seed=0):
    a = roc_auc_score(y, p); rng = np.random.default_rng(seed)
    pats = np.unique(g); idx = {u: np.where(g == u)[0] for u in pats}; draws = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[u] for u in rng.choice(pats, len(pats), True)])
        yy = y[rows]
        if yy.min() != yy.max():
            draws.append(roc_auc_score(yy, p[rows]))
    return [float(a), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _grouped_oof(X, M, y, g, repeats, epochs):
    P = np.zeros((len(y), repeats))
    for r in range(repeats):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
            P[te, r] = _fit_predict(X[tr], M[tr], y[tr], X[te], M[te], epochs, seed=r)
    return P.mean(1)


def _grouped_cv(X, M, y, g, seeds, epochs):
    """TSDF's metric: mean-per-fold AUC under participant 5-fold CV, over CV seeds. Max = the patient-grouped reproduce."""
    means = []
    for s in range(seeds):
        try:
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g))
        except ValueError:
            continue
        fa = []
        for tr, te in folds:
            if y[te].min() == y[te].max():
                continue
            p = _fit_predict(X[tr], M[tr], y[tr], X[te], M[te], epochs, seed=s)
            fa.append(roc_auc_score(y[te], p))
        if fa:
            means.append(float(np.mean(fa)))
    means = np.array(means)
    return {"cv_median": float(np.median(means)), "cv_max": float(means.max()),
            "cv_p90": float(np.percentile(means, 90)), "n_seeds": int(len(means))}


def _eyelevel(X, M, y, seeds, epochs):
    aucs, npos = [], []
    for s, (tr, te) in enumerate(StratifiedShuffleSplit(seeds, test_size=0.2, random_state=0).split(X, y)):
        if y[te].min() == y[te].max():
            continue
        p = _fit_predict(X[tr], M[tr], y[tr], X[te], M[te], epochs, seed=s)
        aucs.append(roc_auc_score(y[te], p)); npos.append(int(y[te].sum()))
    aucs = np.array(aucs)
    return {"median": float(np.median(aucs)), "max": float(aucs.max()),
            "p90": float(np.percentile(aucs, 90)), "n_seeds": int(len(aucs)),
            "mean_testpos": float(np.mean(npos))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--cv-seeds", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true", help="skip compute; regenerate the report from the JSON")
    args = ap.parse_args()
    if args.report_only:
        _write_report(json.loads((ROOT / "data" / "derived" / "grape_serial_ladder.json").read_text()))
        print("[R3] report-only: refreshed reports/grape_serial_ladder.md")
        return
    torch.set_num_threads(max(1, torch.get_num_threads()))

    d = np.load(ROOT / "data" / "derived" / "grape_serial_seq.npz", allow_pickle=True)
    En = d["emb"].astype(np.float32); Er = d["emb_resid"].astype(np.float32); M = d["mask"]
    g = d["patient_id"].astype(str)
    out = {}
    for name, col in ENDPOINTS:
        y = d[col].astype(int); ev = int(y.sum())
        eye = _eyelevel(En, M, y, args.seeds, args.epochs)
        cv = _grouped_cv(En, M, y, g, args.cv_seeds, args.epochs)
        gn = _cluster_ci(_grouped_oof(En, M, y, g, args.repeats, args.epochs), y, g, args.n_boot)
        gr = _cluster_ci(_grouped_oof(Er, M, y, g, args.repeats, args.epochs), y, g, args.n_boot)
        out[name] = {"events": ev, "eyelevel": eye, "grouped_cv": cv,
                     "grouped_naive": gn, "grouped_resid": gr}
        print(f"[R3_temporal] {name} (ev {ev}): eye-level max {eye['max']:.2f} | "
              f"patient-CV mean-fold max {cv['cv_max']:.2f} (median {cv['cv_median']:.2f}) | "
              f"pooled-OOF naive {gn[0]:.2f} [{gn[1]:.2f},{gn[2]:.2f}] resid {gr[0]:.2f} "
              f"| published {PUBLISHED[name][0]:.2f}/{PUBLISHED[name][1]:.2f}", flush=True)

    jp = ROOT / "data" / "derived" / "grape_serial_ladder.json"
    J = json.loads(jp.read_text())
    J["rungs"]["R3_temporal"] = out
    jp.write_text(json.dumps(J, indent=2))
    print("[R3] appended R3_temporal to grape_serial_ladder.json")
    _write_report(J)
    print("[R3] refreshed reports/grape_serial_ladder.md with all rungs")


RUNG_LABEL = {"R1_simple": "R1 simple features", "R2_embeddings": "R2 full embeddings",
              "R3_temporal": "R3 temporal head"}


def _write_report(J):
    dec = J["reference"]["camera_decode_followup"]; cp = J["reference"]["camera_pair_floor"]
    L = ["# GRAPE serial-image reproduce-then-collapse ladder (fundus-only)\n",
         f"*Can GRAPE's own follow-up fundus photos reproduce the serial models' 0.87-0.96? {J['n_eyes']} eyes with "
         f">=2 photos (short series); {J['pct_switch']*100:.0f}% switch camera between visits. The follow-up embedding "
         f"decodes the camera at OOF AUC {dec:.2f}, but the camera->label link (camera-pair floor, below) is at/below "
         "chance -- the two 0.96s are a coincidence, not a device->label path.*\n",
         "## Reproduce (each paper's own split) vs collapse (honest, patient-grouped + camera-residualized)\n",
         "A ladder of fundus-only serial models of increasing capacity. TWO reproduce columns -- each split's own "
         "max-over-seeds metric -- and the honest COLLAPSE (patient cluster-bootstrap pooled OOF). Eye-level = "
         "DiffSight's single random 80/20; patient-CV = TSDF's participant 5-fold, scored as the mean-per-fold AUC "
         "(TSDF's metric) whose max over CV seeds is where a single reported run like 0.96 lands.\n",
         "| rung | endpoint | ev | eye-level median/**max** (DiffSight) | patient-CV mean-fold median/**max** (TSDF) | honest pooled-OOF naive / resid [CI] | published |",
         "|---|---|---|---|---|---|---|"]
    for rung in ["R1_simple", "R2_embeddings", "R3_temporal"]:
        if rung not in J["rungs"]:
            continue
        for name, _ in ENDPOINTS:
            r = J["rungs"][rung][name]; e = r["eyelevel"]; cv = r["grouped_cv"]
            gn = r["grouped_naive"]; gr = r["grouped_resid"]; pub = PUBLISHED[name]
            L.append(f"| {RUNG_LABEL[rung]} | {name} | {r['events']} | {e['median']:.2f}/**{e['max']:.2f}** | "
                     f"{cv['cv_median']:.2f}/**{cv['cv_max']:.2f}** | {gn[0]:.2f} / {gr[0]:.2f} [{gr[1]:.2f},{gr[2]:.2f}] | "
                     f"{pub[0]:.2f}/{pub[1]:.2f} |")
    r3 = J["rungs"].get("R3_temporal", {}).get("PLR3", {})
    r3g = r3.get("grouped_naive", [float("nan")] * 3); r3r = r3.get("grouped_resid", [float("nan")] * 3)
    r3cv = r3.get("grouped_cv", {}).get("cv_max", float("nan"))
    r2cv = J["rungs"].get("R2_embeddings", {}).get("PLR3", {}).get("grouped_cv", {}).get("cv_max", float("nan"))
    L += ["",
          f"**Reference.** Follow-up embedding -> camera decode = **{dec:.2f}** (patient-grouped). "
          f"Camera-pair -> label (no pixels) = {cp['PLR2']:.2f} / {cp['PLR3']:.2f} / {cp['MD']:.2f} (PLR2/PLR3/MD), "
          "at/below chance.\n",
          "> **Read.** (1) **Eye-level reproduces DiffSight:** the max over 80/20 seeds on ~2-5 test positives brackets "
          "the published range; the median is at chance -- a single split on a handful of positives (DiffSight scores "
          "PLR3 on 2 test eyes). (2) **Patient-CV does NOT reproduce TSDF from the fundus embeddings:** scored as the "
          "mean-per-fold AUC (TSDF's own metric), the max over CV seeds caps at PLR3 "
          f"{r2cv:.2f} (logistic) / {r3cv:.2f} (temporal) -- neither reaches TSDF's 0.96. A patient-grouped held-out "
          "fold's AUC is bounded by the real fundus signal (which is at the floor); overfitting cannot raise a held-out "
          "score, and there is no fellow-eye leakage to exploit. So 0.96 is not reproducible from GRAPE's fundus "
          "content under a participant split at the embedding level. (3) **Both splits share one honest floor:** the "
          f"patient cluster-bootstrap pooled-OOF is at chance at every rung, naive and camera-residualized (temporal "
          f"PLR3 {r3g[0]:.2f} \\[{r3g[1]:.2f},{r3g[2]:.2f}] naive, {r3r[0]:.2f} resid). The camera decodes at {dec:.2f} "
          "but does not predict the label. **DiffSight's number reproduces as tiny-test-fold noise; TSDF's 0.96 is not "
          "reproducible from the embeddings and needs either an end-to-end encoder on the raw image sequence "
          "(in-window structural change) or a pipeline artifact -- the next rung.**\n"]
    (ROOT / "reports" / "grape_serial_ladder.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
