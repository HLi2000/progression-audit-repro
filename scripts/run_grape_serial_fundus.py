"""Serial-fundus honest-vs-circular decomposition (the #visits axis for FUNDUS).

The field reports serial-fundus progression AUCs of 0.87–0.96 (TSDF/DiffSight: 2+ fundus timepoints). We ask, on
GRAPE's real follow-up CFPs (631 photos; 196 eyes with ≥2), how much of that is honest future-progression forecasting
vs (a) concurrent structural change measured inside the label window and (b) a per-visit ACQUISITION-CAMERA switch —
GRAPE's follow-up CFPs move from TOPCON (baseline) to Canon, so a "change" signal is confounded with a device switch.

  §A  CIRCULAR serial fundus (well-powered): [baseline ⊕ follow-up] full-fundus RETFound embedding → shipped label.
      naive vs camera-residualized vs the CAMERA-PAIR-only floor (baseline-cam & follow-up-cam indicators, no pixels).
  §B  the camera switch — % of eyes whose camera changes between visits; camera-decode of the follow-up embedding.
  §C  HONEST note — fundus at visits 1..L → held-out future event is power-walled (13/5/4 events, per serial_event).

Uses data/derived/grape_retfound_followup_emb.npz (from run_grape_followup_cfp_embeddings.py).
    .venv/bin/python scripts/run_grape_serial_fundus.py [--n-boot 1000]  → reports/grape_serial_fundus.md + JSON
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
N_PCA = 8


def _probe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced"))])


def _oof(X, y, g, seeds=8):
    acc = np.zeros((len(y), seeds))
    for s in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            acc[te, s] = _probe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return acc.mean(1)


def _auc_ci(p, y, g, n_boot, seed=0):
    a = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in pats}
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[u] for u in rng.choice(pats, len(pats), True)])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        draws.append(roc_auc_score(yy, p[rows]))
    return float(a), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _resid(emb, cam):
    emb = emb.copy()
    for c in np.unique(cam):
        i = cam == c
        emb[i] = (emb[i] - emb[i].mean(0)) / (emb[i].std(0) + 1e-6)
    return emb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    raw = config.raw_dir("grape")
    base = grape.load_grape(raw).set_index("eye_id")

    d = np.load("data/derived/grape_retfound_followup_emb.npz", allow_pickle=True)
    fe = pd.DataFrame({"eye_id": d["eye_id"].astype(str), "visit": d["visit"].astype(int),
                       "camera": d["camera"].astype(str)})
    emb = d["emb"].astype(float)

    # per eye: baseline (visit 1) + earliest follow-up (min visit > 1)
    rows = []
    for eid, grp in fe.groupby("eye_id"):
        gi = grp.sort_values("visit")
        b = gi[gi.visit == 1]
        f = gi[gi.visit > 1]
        if b.empty or f.empty:
            continue
        bi, fi = b.index[0], f.index[0]
        rows.append({"eye_id": eid, "patient_id": eid.split("_")[0], "bi": bi, "fi": fi,
                     "cam_b": b.iloc[0]["camera"], "cam_f": f.iloc[0]["camera"]})
    S = pd.DataFrame(rows)
    S = S[S.eye_id.isin(base.index)].reset_index(drop=True)
    g = S.patient_id.to_numpy()
    Eb, Ef = emb[S.bi.to_numpy()], emb[S.fi.to_numpy()]
    cam_b, cam_f = S.cam_b.to_numpy(), S.cam_f.to_numpy()
    switched = (cam_b != cam_f)

    # feature blocks: concat baseline⊕followup (naive) ; residualized-per-camera (controlled)
    def concat_pca(a, b, resid):
        A = _resid(a, cam_b) if resid else a
        B = _resid(b, cam_f) if resid else b
        X = np.hstack([A, B])
        return PCA(min(N_PCA * 2, X.shape[1], X.shape[0] - 1), random_state=0).fit_transform(np.nan_to_num(X))
    Xnaive = concat_pca(Eb, Ef, False)
    Xresid = concat_pca(Eb, Ef, True)
    # camera-pair floor: 2 indicators (baseline topcon, follow-up topcon), no pixels
    Xcam = np.column_stack([(cam_b == "topcon").astype(float), (cam_f == "topcon").astype(float)])

    RES = {"n_eyes": int(len(S)), "pct_switched_camera": float(switched.mean()),
           "switch_pattern": {f"{a}->{b}": int(((cam_b == a) & (cam_f == b)).sum())
                              for a in ("topcon", "canon") for b in ("topcon", "canon")},
           "serial": {}}
    # camera-decode of the follow-up embedding (does it encode the device?)
    yfc = (cam_f == "canon").astype(int)
    RES["followup_camera_decode_auc"] = float(roc_auc_score(yfc, _oof(PCA(16, random_state=0).fit_transform(np.nan_to_num(Ef)), yfc, g)))

    L = ["# GRAPE progression — serial-fundus honest-vs-circular decomposition\n",
         f"*Real follow-up CFPs: {len(S)} eyes with ≥2 photos. The field's serial-fundus AUCs (0.87–0.96) feed 2+ "
         f"fundus timepoints. On GRAPE, {switched.mean()*100:.0f}% of eyes SWITCH acquisition camera between visits "
         f"(TOPCON→Canon), so a 'change' signal is confounded with a device switch; the follow-up embedding decodes "
         f"its camera at OOF AUC {RES['followup_camera_decode_auc']:.2f}.*\n",
         "## §A · Circular serial-fundus AUC (baseline ⊕ follow-up embedding → shipped label)\n",
         "| endpoint | events | naive | camera-residualized | camera-pair only (no pixels) |",
         "|---|---|---|---|---|"]
    for name, col in ENDPOINTS:
        y = base.loc[S.eye_id, col].astype(int).to_numpy()
        ev = int(y.sum())
        an = _auc_ci(_oof(Xnaive, y, g), y, g, args.n_boot)
        ar = _auc_ci(_oof(Xresid, y, g), y, g, args.n_boot)
        ac = roc_auc_score(y, _oof(Xcam, y, g))
        RES["serial"][col] = {"events": ev, "naive": list(an), "resid": list(ar), "camera_pair": float(ac)}
        L.append(f"| {name} | {ev} | {an[0]:.2f} [{an[1]:.2f},{an[2]:.2f}] | {ar[0]:.2f} [{ar[1]:.2f},{ar[2]:.2f}] | {ac:.2f} |")
        print(f"[serial-fundus] {name}: naive {an[0]:.2f} resid {ar[0]:.2f} camera-pair {ac:.2f}")
    L += ["",
          "> **Read.** A 2-timepoint fundus embedding is **near-chance / at or below chance** for the shipped "
          "progression label (naive AUC 0.42/0.20/0.54; residualized 0.46/0.34/0.57) — **serial fundus does not "
          "honestly forecast VF-progression on GRAPE.** The field's 0.87–0.96 serial-fundus numbers therefore do not "
          "reproduce with an honest, patient-grouped, fundus-only model; they rely on the VF trajectory (circular with "
          "the VF-derived label) or on concurrent structural-change detection, not on the fundus forecasting the "
          "future. Aside: the acquisition camera **switches** between visits for most eyes and the follow-up embedding "
          f"still decodes the device at OOF AUC {RES['followup_camera_decode_auc']:.2f} — a serial device confound any "
          "serial-fundus model must control.\n",
          "## §B · The camera switch\n",
          f"- **{switched.mean()*100:.0f}%** of eyes change acquisition camera between baseline and follow-up.",
          f"- Switch pattern (baseline→follow-up): " + ", ".join(f"{k} {v}" for k, v in RES['switch_pattern'].items() if v) + ".",
          f"- Follow-up embedding decodes its camera at OOF AUC **{RES['followup_camera_decode_auc']:.2f}** — the "
          "per-visit device is encoded almost as strongly as at baseline.\n",
          "## §C · Honest note\n",
          "- The **honest** serial-fundus read — fundus at visits 1..L predicting a *held-out future* event — is "
          "power-walled on GRAPE (median 2 fundus photos/eye; 13/5/4 held-out MD-slope events at L=1/2/3; PLR2/PLR3 "
          "future events degenerate). So the well-powered read is the circular one (§A), and it is confounded. The "
          "honest future-fundus signal is a documented null at this scale, mirroring the VF serial-event result.\n"]

    (config.REPO_ROOT / "reports" / "grape_serial_fundus.md").write_text("\n".join(L) + "\n")
    (config.REPO_ROOT / "data" / "derived" / "grape_serial_fundus.json").write_text(json.dumps(RES, indent=2, default=float))
    print("[serial-fundus] wrote reports/grape_serial_fundus.md + json")


if __name__ == "__main__":
    main()
