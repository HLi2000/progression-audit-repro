"""GRAPE progression — the MULTI-MODAL value + INTERACTION decomposition (the surface headline).

Modalities {vf, oct, fundus_fm, iop} (fundus_fm = full-fundus RETFound embedding, camera-residualized PCA-4).
Per endpoint (PLR2/PLR3/MD): single-modality + all-modality FUSION AUC, SAGE-Shapley φ (which modality adds value),
Grabisch–Roubens interaction I[a,b] + cluster-boot CI (binary — caveated, noisy at 14–40 events). Plus:

  • PID (redundancy/unique/synergy) on the WELL-POWERED continuous MD-slope rate (263 eyes) — the trustworthy
    interaction read: oct×vf (structure×function), fundus×vf, fundus×oct.
  • InterSHAP — an OUTPUT-based cross-modal Shapley interaction (does the model *combine* modalities in its output,
    performance-agnostic; complementary to the performance-based SAGE), reusing interaction.shapley.interaction_matrix.

Three triangulating interaction reads. → data/derived/grape_modality_interaction.json + reports/grape_modality_interaction.md
    .venv/bin/python scripts/run_grape_modality_interaction.py [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from glaucoma_triage import config
from glaucoma_triage.datasets import grape
from glaucoma_triage.interaction import pid
from glaucoma_triage.interaction.report import decompose
from glaucoma_triage.interaction.shapley import coalitions, interaction_matrix

ENDPOINTS = [("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"), ("MD", "prog_md")]
PLAYERS = ["vf", "oct", "fundus", "iop"]
PAIRS = [("oct", "vf"), ("fundus", "vf"), ("fundus", "oct")]
N_FUNDUS_PCA = 4


def _probe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced"))])


def _resid_pca(emb, cam, n):
    emb = np.nan_to_num(np.asarray(emb, float)).copy()
    for c in np.unique(cam):
        i = cam == c
        emb[i] = (emb[i] - emb[i].mean(0)) / (emb[i].std(0) + 1e-6)
    return PCA(min(n, emb.shape[1]), random_state=0).fit_transform(emb)


def _md_slope(fu):
    out = {}
    for eid, g in fu.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        out[str(eid)] = float(np.polyfit(t[ok], s[ok], 1)[0]) if ok.sum() >= 2 else np.nan
    return out


def _intershap(X, y, spec_idx, players):
    """OUTPUT-based cross-modal Shapley interaction: fit the full multimodal model, then decompose its per-eye
    output logit into pairwise interaction via the Grabisch–Roubens index (interaction.shapley), masking the
    complement modalities to the background (feature mean). Signed mean over eyes: >0 synergy, <0 redundancy."""
    pipe = _probe().fit(X, y)
    bg = X.mean(0)
    co = coalitions(players)
    acc = {}
    for xi in X:
        vtab = {}
        for S in co:
            xm = bg.copy()
            for m in S:
                xm[spec_idx[m]] = xi[spec_idx[m]]
            vtab[S] = float(pipe.decision_function(xm[None])[0])
        for pair, val in interaction_matrix(vtab, players).items():
            acc.setdefault(pair, []).append(val)
    return {pair: float(np.mean(v)) for pair, v in acc.items()}


def _ic(dec, a, b):
    for x, y in ((a, b), (b, a)):
        if (x, y) in dec.interaction:
            return dec.interaction[(x, y)], dec.ci.get(f"I[{x},{y}]", (float("nan"), float("nan")))
    return float("nan"), (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--emb", default="data/derived/phase3_fullfundus_reps.npz")
    ap.add_argument("--n-fundus-pca", type=int, default=N_FUNDUS_PCA)
    args = ap.parse_args()
    raw = config.raw_dir("grape")
    base = grape.load_grape(raw)
    fu = grape.load_grape_followup(raw)

    P = np.load(args.emb, allow_pickle=True)
    eye = P["eye_id"].astype(str)
    cam = P["camera"].astype(str)
    femb = P["retfound_cfp__lp_emb"] if "retfound_cfp__lp_emb" in P.files else P["emb"]
    fpca = _resid_pca(femb, cam, args.n_fundus_pca)
    fcols = [f"fundus_{j:02d}" for j in range(fpca.shape[1])]
    fdf = pd.DataFrame({"eye_id": eye, **{c: fpca[:, j] for j, c in enumerate(fcols)}})

    df = base.merge(fdf, on="eye_id", how="left")
    for c in grape.VF_FEATURES + grape.OCT_FEATURES + ["iop"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df = df[df[fcols[0]].notna()].reset_index(drop=True)
    g = df["patient_id"].to_numpy()

    spec = {"vf": grape.VF_FEATURES, "oct": grape.OCT_FEATURES, "fundus": fcols, "iop": ["iop"]}
    order = grape.VF_FEATURES + grape.OCT_FEATURES + fcols + ["iop"]
    spec_idx, off = {}, 0
    for m in PLAYERS:
        spec_idx[m] = list(range(off, off + len(spec[m])))
        off += len(spec[m])
    Xall = df[order].to_numpy()

    ms = np.array([_md_slope(fu).get(e, np.nan) for e in df["eye_id"]])
    okm = np.isfinite(ms)

    RES = {"n_eyes": int(len(df)), "players": PLAYERS, "endpoints": {}, "pid_rate": {}, "intershap_rate": {}}

    # ---- PID on the continuous MD-slope rate (well-powered) ----
    for a, b in PAIRS:
        pa = pid.pair_analysis(ms[okm], df.loc[okm, spec[a]].to_numpy(), df.loc[okm, spec[b]].to_numpy(),
                               g[okm], n_boot=min(args.n_boot, 500), n_perm=min(args.n_boot, 500), cross_fit=True)
        RES["pid_rate"][f"{a}x{b}"] = {"net_synergy": pa.pid.net_synergy, "redundancy": pa.pid.redundancy,
                                       "unique_a": pa.pid.unique1, "unique_b": pa.pid.unique2, "synergy": pa.pid.synergy,
                                       "p": pa.p_perm.get("net_synergy")}
        print(f"[PID rate] {a}x{b}: net-syn {pa.pid.net_synergy:+.3f} p={pa.p_perm.get('net_synergy'):.3f}")

    # ---- InterSHAP (output-based) on the rate-predicting full model (use MD binary as the supervised target) ----
    ish = _intershap(Xall, df["prog_md"].astype(int).to_numpy(), spec_idx, PLAYERS)
    RES["intershap_rate"] = {f"{a}x{b}": ish.get((a, b), ish.get((b, a))) for a, b in PAIRS}
    print("[InterSHAP] " + " ".join(f"{a}x{b} {RES['intershap_rate'][f'{a}x{b}']:+.3f}" for a, b in PAIRS))

    # ---- per-endpoint: single + fusion AUC, SAGE-Shapley φ, Grabisch–Roubens (binary, caveated) ----
    for name, col in ENDPOINTS:
        d = decompose(df, spec, lambda dd, c=col: dd[c], name, n_boot=args.n_boot, n_repeats=3,
                      estimator_factory=_probe, progress=f"[{name}]")
        fa_lo, fa_hi = d.ci.get("full_auc", (float("nan"), float("nan")))
        RES["endpoints"][col] = {
            "events": int(df[col].sum()), "full_auc": d.full_auc, "full_auc_ci": [fa_lo, fa_hi],
            "single_auc": d.single_auc, "shapley": d.shapley,
            "grabisch": {f"{a}x{b}": {"I": _ic(d, a, b)[0], "ci": list(_ic(d, a, b)[1])} for a, b in PAIRS},
        }
        print(f"[{name}] fusion {d.full_auc:.3f} | φ " + " ".join(f"{m} {d.shapley[m]:+.3f}" for m in PLAYERS))

    out = config.REPO_ROOT / "data" / "derived" / "grape_modality_interaction.json"
    out.write_text(json.dumps(RES, indent=2, default=float))
    print(f"[modality-interaction] wrote {out}")


if __name__ == "__main__":
    main()
