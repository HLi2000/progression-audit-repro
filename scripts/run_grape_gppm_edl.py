"""Faithful-enough GPPM: the trusted multi-view EVIDENTIAL-FUSION architecture (not a plain logistic), fed the
summarized inputs we already have -- VF two-field statistics + frozen RETFound disc-ROI features -- instead of rendering
Voronoi images and training two ResNet-18 backbones. Reproduces GPPM's actual contribution (Zheng et al. 2025):

  * Two evidential branches (Sensoy et al. 2018 EDL): features -> MLP -> softplus evidence e -> Dirichlet alpha=e+1,
    belief b_k=e_k/S, uncertainty u=2/S (S=sum alpha).  [GPPM Eqs 1-3]
  * Dempster's rule of combination fuses the VF and fundus opinions with a conflict term.  [GPPM Eq 4]
  * Multi-task EDL loss L = L_edl(fused) + L_edl(VF) + lambda_F * L_edl(fundus), lambda_F=0.01 (fundus down-weighted);
    L_edl = expected cross-entropy (Bayes risk under the Dirichlet) + annealed KL to the uniform Dirichlet + a
    class-balanced BCE (progression up-weighted 0.8/0.2).  [GPPM Eqs 8-9]
  * Prediction = fused expected progression probability alpha_P / S.

Same reproduce-then-audit harness as the logistic version: eye-level p90 reproduce + the item-6 cells (device=AAROC,
fair split=patient-grouped, circularity=early VF+CFP, all=(1)+(2)+(3) combined, IECV twin, residualized twins).
Overwrites the ``gppm_fusion`` block of data/derived/grape_two_vft.json. Two envs (data has openpyxl, torch env has torch):
  .venv/bin/python       scripts/run_grape_gppm_edl.py --prep     # build gppm_edl_inputs.npz
  .venv-torch/bin/python scripts/run_grape_gppm_edl.py            # train the EDL model + write gppm_fusion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
INPUTS = ROOT / "data" / "derived" / "gppm_edl_inputs.npz"


# ----------------------------------------------------------------------------- scoring helpers (numpy only) --------
def _aaroc(p, y, cam):
    p = np.asarray(p, float); y = np.asarray(y); cam = np.asarray(cam)
    ok = np.isfinite(p); p, y, cam = p[ok], y[ok], cam[ok]; pv = []
    for i in np.where(y == 1)[0]:
        ctrl = p[(y == 0) & (cam == cam[i])]
        if len(ctrl):
            pv.append((np.sum(ctrl < p[i]) + 0.5 * np.sum(ctrl == p[i])) / len(ctrl))
    return float(np.mean(pv)) if pv else float("nan")


def _boot_ci(score, p, y, aux, pat, n_boot):
    """point estimate + patient cluster-bootstrap 95% CI; `score(p,y,aux)` -> float (aux = camera for AAROC, ignored for AUC)."""
    p = np.asarray(p, float); ok = np.isfinite(p)
    p, y, aux, pat = p[ok], np.asarray(y)[ok], np.asarray(aux)[ok], np.asarray(pat)[ok]
    if len(np.unique(y)) < 2:
        return [float("nan")] * 3
    rng = np.random.default_rng(0); pats = np.unique(pat); pidx = {q: np.where(pat == q)[0] for q in pats}; bs = []
    for _ in range(n_boot):
        idx = np.concatenate([pidx[q] for q in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            bs.append(score(p[idx], y[idx], aux[idx]))
        except Exception:
            pass
    lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if bs else (float("nan"),) * 2)
    return [round(score(p, y, aux), 3), round(float(lo), 3), round(float(hi), 3)]


def _aaroc_cell(p, y, cam, pat, nb):
    return [float("nan")] * 3 if p is None else _boot_ci(_aaroc, p, y, cam, pat, nb)


def _auc_cell(p, y, pat, nb):
    return [float("nan")] * 3 if p is None else _boot_ci(lambda pp, yy, _a: roc_auc_score(yy, pp), p, y, y, pat, nb)


def _resid_within_camera(X, cam):
    Xr = X.astype(float).copy()
    for c in np.unique(cam):
        m = cam == c
        mu = np.nanmean(X[m], axis=0); sd = np.nanstd(X[m], axis=0) + 1e-8
        Xr[m] = (X[m] - mu) / sd
    return np.nan_to_num(Xr)


# --------------------------------------------------------------------------- torch EDL model (lazy) ---------------
def _make_fit():
    """Build the EDL fit_predict closure. torch is imported HERE so the module imports without torch (for --prep)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(4)

    class _EvidBranch(nn.Module):
        def __init__(self, d, h=32):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 2), nn.Softplus())

        def forward(self, x):
            e = self.net(x); a = e + 1.0; Ssum = a.sum(1, keepdim=True)
            return a, e / Ssum, (2.0 / Ssum).squeeze(1)             # Dirichlet alpha, belief, uncertainty (K=2)

    class _GPPM(nn.Module):
        def __init__(self, d_vf, d_cfp):
            super().__init__(); self.vf = _EvidBranch(d_vf); self.cfp = _EvidBranch(d_cfp)

        def forward(self, xvf, xcfp):
            a_v, bv, uv = self.vf(xvf); a_f, bf, uf = self.cfp(xcfp)
            bvS, bvP = bv[:, 0], bv[:, 1]; bfS, bfP = bf[:, 0], bf[:, 1]
            C = bvS * bfP + bvP * bfS                               # Dempster conflict
            den = (1.0 - C).clamp(min=1e-6)
            bS = (bvS * bfS + bvS * uf + bfS * uv) / den
            bP = (bvP * bfP + bvP * uf + bfP * uv) / den
            u = (uv * uf) / den; Sf = 2.0 / u.clamp(min=1e-6)
            return torch.stack([bS * Sf + 1.0, bP * Sf + 1.0], 1), a_v, a_f

    def _edl_loss(alpha, y, lamK, cw=(0.2, 0.8), lamB=0.1):
        Ssum = alpha.sum(1); yoh = torch.stack([1.0 - y, y], 1)
        ece = (yoh * (torch.digamma(Ssum).unsqueeze(1) - torch.digamma(alpha))).sum(1).mean()
        a_t = yoh + (1.0 - yoh) * alpha; St = a_t.sum(1)           # evidence removed from the true class
        kl = (torch.lgamma(St) - torch.lgamma(a_t).sum(1)
              + ((a_t - 1.0) * (torch.digamma(a_t) - torch.digamma(St).unsqueeze(1))).sum(1)).mean()
        p = (alpha[:, 1] / Ssum).clamp(1e-6, 1.0 - 1e-6); w = cw[1] * y + cw[0] * (1.0 - y)
        return ece + lamK * kl + lamB * F.binary_cross_entropy(p, y, weight=w)

    def fit_predict(Xvf_tr, Xcfp_tr, ytr, Xvf_te, Xcfp_te, epochs, seed):
        torch.manual_seed(seed)
        def _std(tr, te):
            mu = tr.mean(0); sd = tr.std(0) + 1e-6
            return ((tr - mu) / sd).astype(np.float32), ((te - mu) / sd).astype(np.float32)
        Xvf_tr, Xvf_te = _std(Xvf_tr, Xvf_te); Xcfp_tr, Xcfp_te = _std(Xcfp_tr, Xcfp_te)   # train-fold standardization
        m = _GPPM(Xvf_tr.shape[1], Xcfp_tr.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
        xv = torch.tensor(Xvf_tr); xc = torch.tensor(Xcfp_tr); yt = torch.tensor(ytr, dtype=torch.float32)
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            a_fused, a_v, a_f = m(xv, xc); lamK = min(1.0, ep / 10.0)   # KL annealing
            (_edl_loss(a_fused, yt, lamK) + _edl_loss(a_v, yt, lamK) + 0.01 * _edl_loss(a_f, yt, lamK)).backward()
            opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            a_fused, _, _ = m(torch.tensor(Xvf_te), torch.tensor(Xcfp_te))
            return (a_fused[:, 1] / a_fused.sum(1)).numpy()
    return fit_predict


# --------------------------------------------------------------------------------- OOF + eye-level ----------------
def _oof(fit, Xvf, Xcfp, y, groups, reps, epochs):
    n = len(y); P = np.zeros(n); Cn = np.zeros(n)
    nsp = min(5, int(y.sum()), int((y == 0).sum()), len(np.unique(groups)))
    if nsp < 2:
        return None
    for r in range(reps):
        for tr, te in StratifiedGroupKFold(nsp, shuffle=True, random_state=r).split(Xvf, y, groups):
            P[te] += fit(Xvf[tr], Xcfp[tr], y[tr], Xvf[te], Xcfp[te], epochs, r); Cn[te] += 1
    return np.where(Cn > 0, P / np.maximum(Cn, 1), np.nan)


def _eyelevel(fit, Xvf, Xcfp, y, seeds, epochs):
    a = []
    for s, (tr, te) in enumerate(StratifiedShuffleSplit(seeds, test_size=0.2, random_state=0).split(Xvf, y)):
        if y[te].min() == y[te].max():
            continue
        a.append(roc_auc_score(y[te], fit(Xvf[tr], Xcfp[tr], y[tr], Xvf[te], Xcfp[te], epochs, s)))
    a = np.array(a)
    return round(float(a.max()), 3), round(float(np.percentile(a, 90)), 3), round(float(np.percentile(a, 95)), 3)


# ------------------------------------------------------------------------------ prep (data env) ------------------
def _prep(raw):
    sys.path.insert(0, str(ROOT / "scripts"))
    from sklearn.decomposition import PCA
    from run_grape_two_vft import PAIR_COLS, _pair_features
    from glaucoma_triage.datasets import grape
    base = grape.load_grape(raw); fu = grape.load_grape_followup(raw)
    fe = np.load(ROOT / "data" / "derived" / "grape_retfound_followup_emb.npz", allow_pickle=True)
    femb = fe["emb"].astype(np.float32); feid = fe["eye_id"].astype(str); fvis = np.asarray(fe["visit"]); fcam = fe["camera"].astype(str)
    base_emb, last_emb, base_cam = {}, {}, {}
    for e in np.unique(feid):
        idx = np.where(feid == e)[0]
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(fvis[idx])]
        base_emb[e], last_emb[e], base_cam[e] = femb[idx[0]], femb[idx[-1]], fcam[idx[0]]
    cfp_eyes = sorted(base_emb)
    pca = PCA(16, random_state=0).fit(np.nan_to_num(np.array([last_emb[e] for e in cfp_eyes] + [base_emb[e] for e in cfp_eyes])))

    def _fusion(vf_feat, cfp_dict):
        vf = vf_feat[vf_feat["eye_id"].astype(str).isin(set(cfp_eyes))]
        eyes = vf["eye_id"].astype(str).to_numpy()
        Xvf = vf[PAIR_COLS].to_numpy(float).astype(np.float32)
        Xcfp = pca.transform(np.nan_to_num(np.array([cfp_dict[e] for e in eyes]))).astype(np.float32)
        return eyes, Xvf, Xcfp, vf["patient_id"].to_numpy().astype(str), np.array([base_cam[e] for e in eyes]).astype(str)

    eyes_iw, Xvf_iw, Xcfp_iw, pat_iw, cam_iw = _fusion(_pair_features(fu, "last"), last_emb)
    eyes_ec, Xvf_ec, Xcfp_ec, pat_ec, cam_ec = _fusion(_pair_features(fu, "2"), base_emb)
    lk_iw = base.set_index("eye_id").reindex(eyes_iw); lk_ec = base.set_index("eye_id").reindex(eyes_ec)
    out = {"eyes_iw": eyes_iw.astype(str), "Xvf_iw": Xvf_iw, "Xcfp_iw": Xcfp_iw, "pat_iw": pat_iw, "cam_iw": cam_iw,
           "eyes_ec": eyes_ec.astype(str), "Xvf_ec": Xvf_ec, "Xcfp_ec": Xcfp_ec, "pat_ec": pat_ec, "cam_ec": cam_ec}
    for ep in ("prog_plr2", "prog_plr3", "prog_md"):
        out[f"y_iw_{ep}"] = lk_iw[ep].to_numpy().astype(int); out[f"y_ec_{ep}"] = lk_ec[ep].to_numpy().astype(int)
    np.savez(INPUTS, **out)
    print(f"[prep] wrote {INPUTS.name}: {len(eyes_iw)} eyes, VF{Xvf_iw.shape[1]}+CFP{Xcfp_iw.shape[1]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/grape")
    ap.add_argument("--prep", action="store_true", help="build the input npz (run in the data env)")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if args.prep:
        _prep(Path(args.raw)); return

    d = np.load(INPUTS, allow_pickle=True)
    fit = _make_fit()
    nb, R, E = args.n_boot, args.reps, args.epochs
    Xvf_iw, Xcfp_iw, pat_iw, cam_iw, eyes_iw = d["Xvf_iw"], d["Xcfp_iw"], d["pat_iw"].astype(str), d["cam_iw"].astype(str), d["eyes_iw"].astype(str)
    Xvf_ec, Xcfp_ec, pat_ec, cam_ec, eyes_ec = d["Xvf_ec"], d["Xcfp_ec"], d["pat_ec"].astype(str), d["cam_ec"].astype(str), d["eyes_ec"].astype(str)
    Xcfp_iw_dev = _resid_within_camera(Xcfp_iw, cam_iw).astype(np.float32)
    Xcfp_ec_dev = _resid_within_camera(Xcfp_ec, cam_ec).astype(np.float32)

    tv = json.loads((ROOT / "data" / "derived" / "grape_two_vft.json").read_text())
    gppm = {"n_eyes": int(len(eyes_iw)), "_arch": "evidential two-branch (VF-stats + RETFound-ROI) + Dempster fusion, lambda_F=0.01"}
    for ep, EL in [("prog_plr2", "PLR2"), ("prog_plr3", "PLR3"), ("prog_md", "MD")]:
        y_iw = d[f"y_iw_{ep}"].astype(int); y_ec = d[f"y_ec_{ep}"].astype(int)
        emax, ep90, ep95 = _eyelevel(fit, Xvf_iw, Xcfp_iw, y_iw, args.seeds, E)
        p_iw = _oof(fit, Xvf_iw, Xcfp_iw, y_iw, eyes_iw, R, E)                # eye-group reproduce OOF (leaky, in-window)
        p_ec_pat = _oof(fit, Xvf_ec, Xcfp_ec, y_ec, pat_ec, R, E)            # (4) patient-grouped + non-circular
        cell = {
            "reproduce":    _auc_cell(p_iw, y_iw, pat_iw, nb),
            "m_aaroc":      _aaroc_cell(p_iw, y_iw, cam_iw, pat_iw, nb),                              # (1) AAROC
            "m_device":     _auc_cell(_oof(fit, Xvf_iw, Xcfp_iw_dev, y_iw, eyes_iw, R, E), y_iw, pat_iw, nb),  # (1) resid twin
            "m_split":      _auc_cell(_oof(fit, Xvf_iw, Xcfp_iw, y_iw, pat_iw, R, E), y_iw, pat_iw, nb),       # (2) patient
            "m_circ":       _auc_cell(_oof(fit, Xvf_ec, Xcfp_ec, y_ec, eyes_ec, R, E), y_ec, pat_ec, nb),      # (3) early
            "all_combined": _aaroc_cell(p_ec_pat, y_ec, cam_ec, pat_ec, nb),                          # (4) = (1)+(2)+(3)
            "iecv":         _auc_cell(_oof(fit, Xvf_ec, Xcfp_ec, y_ec, cam_ec, R, E), y_ec, pat_ec, nb),       # (4) IECV twin
            "all":          _auc_cell(_oof(fit, Xvf_ec, Xcfp_ec_dev, y_ec, pat_ec, R, E), y_ec, pat_ec, nb),   # (4) resid twin
            "eyelevel_max": emax, "eyelevel_p90": ep90, "eyelevel_p95": ep95,
            "events": int(y_iw.sum())}
        gppm[EL] = cell
        print(f"[GPPM-EDL {EL}] ev{int(y_iw.sum())}: reproduce p90 {ep90:.2f} max {emax:.2f} | (1)AAROC {cell['m_aaroc'][0]:.2f} | "
              f"-split {cell['m_split'][0]:.2f} | -circ {cell['m_circ'][0]:.2f} | (4)comb {cell['all_combined'][0]:.2f} | "
              f"IECV {cell['iecv'][0]:.2f}", flush=True)
    tv["gppm_fusion"] = gppm
    (ROOT / "data" / "derived" / "grape_two_vft.json").write_text(json.dumps(tv, indent=2, default=float))
    print("[GPPM-EDL] overwrote gppm_fusion in grape_two_vft.json", flush=True)


if __name__ == "__main__":
    main()
