#!/usr/bin/env python
"""Baseline-MATCH then COLLAPSE: reproduce the published GRAPE CFP→progression AUCs with the lit's own
architectures + imbalance recipe, then show each is a protocol artifact.

The published GRAPE baseline fine-tunes ResNet-50 on a single baseline CFP with an unstated split and no
camera control → 0.71/0.80/0.73 (PLR2/PLR3/MD); Fouda et al. add a soft-vote ENSEMBLE (EfficientNet-B4 /
DenseNet-121 / ConvNeXt-Tiny) + SMOTE + focal loss + ROI-crop → ≈0.86 (PLR3). This script fine-tunes the
SAME architectures + recipe under two protocols:

  * ``published`` — single random EYE-level 80/20 split (no patient grouping, no camera control), several
    seeds → the AUC *distribution* whose upper band brackets the published single number (which had no CV/CI).
  * ``honest``    — patient-grouped 5-fold CV (fellow eyes kept together) → the reproducible signal, with a
    camera-only reference on the same eyes (the device batch-effect floor).

Recipe (Fouda-faithful): ROI disc-crop input, focal loss (α=0.75, γ=2.0), minority oversampling, per-arch
fine-tune + soft-vote ensemble. Writes data/derived/grape_cfp_finetune_results.json + a table the report reads.

    .venv-torch/bin/python scripts/run_grape_cfp_finetune.py --archs resnet50,efficientnet_b4,densenet121,convnext_tiny \
        --endpoints plr2,plr3,md --crop roi --seeds 5
CPU-heavy (~hours for the full 4-arch sweep); use a GPU box / Colab for the full run, or --archs resnet50 to smoke.
"""

from __future__ import annotations

import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_ARCH = {  # short name → timm id
    "resnet50": "resnet50.a1_in1k", "efficientnet_b4": "efficientnet_b4",
    "densenet121": "densenet121.ra_in1k", "convnext_tiny": "convnext_tiny.fb_in1k",
}
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
DEV = torch.device("mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))   # GPU sweep (recipe/metric unchanged)
_AMP = DEV.type == "cuda"                                                    # fp16 autocast on CUDA: same recipe, ~2-3x faster
if _AMP:
    torch.backends.cudnn.benchmark = True                                   # autotune convs (fixed 224 input)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


class FocalLoss(nn.Module):
    """Binary focal loss (Lin 2017) with the Fouda-recipe defaults α=0.75, γ=2.0."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * y + (1 - p) * (1 - y)
        a_t = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (a_t * (1 - p_t).clamp(min=1e-6) ** self.gamma * ce).mean()


def _roi_path(cfp_path: str) -> str:
    """Map a baseline CFP path → its released ROI disc-crop (same basename under ROI/ROI images/)."""
    p = Path(cfp_path)
    roi = p.parents[1] / "ROI" / "ROI images" / p.name
    return str(roi if roi.exists() else p)


def _load_images(paths, res=224):
    X = np.empty((len(paths), 3, res, res), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((res, res))
        a = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
        X[i] = a.transpose(2, 0, 1)
    return torch.from_numpy(X)


def _finetune(arch, Xtr, ytr, Xte, *, epochs, steps=6, lr=3e-4, bs=32, unfreeze_frac=0.30, seed=0,
              loss="focal", full_epoch=False, early_stop=False, patience=40, val_frac=0.3):
    """Fine-tune a timm ``arch``. Returns ``(held_out_probs, val_auc)`` (val_auc is None without early stopping).
    Three recipes:
    * default (loss='focal', full_epoch=False, early_stop=False): unfreeze last ``unfreeze_frac`` + head, focal loss,
      class-balanced 50/50 batches for ``epochs×steps`` batches -- the imbalance-robust reproduction.
    * Huang (loss='ce', full_epoch=True, unfreeze_frac=1.0, bs=4, lr=1e-4): full fine-tune, cross-entropy, full-epoch
      shuffled batches for ``epochs`` epochs, cosine LR lr→lr/10 -- GRAPE Sci-Data recipe (no early stopping).
    * Fouda (loss='focal', early_stop=True, bs=8, lr=4e-5): focal loss, class-balanced batches, cosine LR, and
      early stopping on a stratified validation carve (``patience`` epochs, best-val-AUC restore) -- IMSA-2025
      training controls. ``val_auc`` is returned so the ensemble can weight by validation performance."""
    import timm
    torch.manual_seed(seed)
    model = timm.create_model(_ARCH[arch], pretrained=True, num_classes=1).to(DEV)
    if _AMP:
        model = model.to(memory_format=torch.channels_last)
    params = list(model.named_parameters())
    cut = int(len(params) * (1.0 - unfreeze_frac))
    for i, (_, p) in enumerate(params):
        p.requires_grad_(i >= cut)
    lossf = FocalLoss() if loss == "focal" else (lambda lo, y: F.binary_cross_entropy_with_logits(lo, y))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=_AMP)
    Xtr = Xtr.to(DEV, non_blocking=True); Xte = Xte.to(DEV, non_blocking=True)   # GPU-resident: no per-step host->device copy
    yt = torch.tensor(ytr, dtype=torch.float32).to(DEV)
    rng = np.random.default_rng(seed)

    def _step(idx):                                                         # one optimiser step with AMP (recipe unchanged)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=_AMP):
            loss = lossf(model(Xtr[idx]).squeeze(1), yt[idx])
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

    def _predict(X):
        model.eval(); ebs = max(bs, 128)
        with torch.no_grad(), torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=_AMP):
            return np.concatenate([torch.sigmoid(model(X[j:j + ebs]).squeeze(1)).float().cpu().numpy()
                                   for j in range(0, len(X), ebs)])

    val_auc = None
    model.train()
    if full_epoch:                                                          # Huang recipe: full-epoch batch-bs, cosine LR
        yv = np.asarray(ytr)
        if early_stop and yv.sum() >= 2 and (yv == 0).sum() >= 2:           # optional early stopping (speed; not strict Huang)
            from sklearn.model_selection import StratifiedShuffleSplit
            tri, vai = next(StratifiedShuffleSplit(1, test_size=val_frac, random_state=seed).split(yv.reshape(-1, 1), yv))
            yva = yv[vai]
        else:
            tri, vai = np.arange(len(yv)), None
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs * int(np.ceil(len(tri) / bs))), eta_min=lr / 10)
        best, best_state, bad = -1.0, None, 0
        for _e in range(epochs):
            model.train(); order = tri.copy(); rng.shuffle(order)
            for j in range(0, len(order), bs):
                _step(order[j:j + bs]); sched.step()
            if vai is not None:                                             # early-stop on held-out val AUC
                auc = roc_auc_score(yva, _predict(Xtr[vai])) if yva.min() != yva.max() else 0.5
                if auc > best:
                    best, best_state, bad = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if vai is not None and best_state is not None:
            model.load_state_dict(best_state); val_auc = float(best)
    elif early_stop:                                                        # Fouda: balanced batches + cosine + early stop
        yv = np.asarray(ytr)
        if yv.sum() >= 2 and (yv == 0).sum() >= 2:                          # stratified val carve from TRAIN only
            from sklearn.model_selection import StratifiedShuffleSplit
            tri, vai = next(StratifiedShuffleSplit(1, test_size=val_frac, random_state=seed).split(yv.reshape(-1, 1), yv))
        else:
            tri = vai = np.arange(len(yv))
        pos, neg = tri[yv[tri] == 1], tri[yv[tri] == 0]; yva = yv[vai]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs * steps), eta_min=lr / 10)
        best, best_state, bad = -1.0, None, 0
        for _e in range(epochs):
            model.train()
            for _ in range(steps):
                ip = rng.choice(pos, bs // 2, replace=len(pos) < bs // 2)
                ineg = rng.choice(neg, bs - bs // 2, replace=len(neg) < bs - bs // 2)
                _step(np.concatenate([ip, ineg])); sched.step()
            auc = roc_auc_score(yva, _predict(Xtr[vai])) if yva.min() != yva.max() else 0.5
            if auc > best:
                best, best_state, bad = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        val_auc = float(best)
    else:                                                                   # balanced-sampler reproduction (no early stop)
        pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
        for _ in range(epochs * steps):
            ip = rng.choice(pos, bs // 2, replace=len(pos) < bs // 2)
            ineg = rng.choice(neg, bs - bs // 2, replace=len(neg) < bs - bs // 2)
            _step(np.concatenate([ip, ineg]))
    return _predict(Xte), val_auc


def _aaroc(p, y, cam):
    """(1) -device: COVARIATE-ADJUSTED AUC (Janes & Pepe 2009, covariate-adjusted ROC) with camera as the covariate.
    Mean over cases of the case's placement value (percentile among SAME-camera controls) -> discrimination holding the
    device fixed. Uniform across methods (predictions only), keeps the split. NaN-safe (mid-rank for ties)."""
    p = np.asarray(p, float); y = np.asarray(y); cam = np.asarray(cam)
    ok = np.isfinite(p); p, y, cam = p[ok], y[ok], cam[ok]
    cases = np.where(y == 1)[0]; pv = []
    for i in cases:
        ctrl = p[(y == 0) & (cam == cam[i])]
        if len(ctrl):
            pv.append((np.sum(ctrl < p[i]) + 0.5 * np.sum(ctrl == p[i])) / len(ctrl))
    return float(np.mean(pv)) if pv else float("nan")


def _stratified_auc(p, y, cam):
    """(1) robustness twin of _aaroc: AUC WITHIN each camera stratum, pooled by n_pos*n_neg (stratified concordance,
    van Elteren lineage). Reported alongside the AAROC to show the device verdict is method-robust."""
    p = np.asarray(p); y = np.asarray(y); cam = np.asarray(cam)
    ok = np.isfinite(p); p, y, cam = p[ok], y[ok], cam[ok]
    num = den = 0.0
    for c in np.unique(cam):
        m = cam == c; yy = y[m]
        if yy.min() != yy.max():
            w = float(yy.sum() * (len(yy) - yy.sum()))
            num += w * roc_auc_score(yy, p[m]); den += w
    return num / den if den else float("nan")


def _iecv(archs, X, y, g, cam, *, epochs, steps, t0, recipe=None):
    """(4) -all as INTERNAL-EXTERNAL cross-validation (Debray 2013 / Ahmed 2014): leave-one-camera-out -- train on the
    other device(s), predict the held-out device; pooled OOF. Patients are nested in device, so this removes device AND
    patient in one external split. Baseline single-CFP = 2 devices -> 2-fold (coarse; reported as support). Per-arch +
    ensemble."""
    recipe = recipe or {}
    keys = list(archs) + ["ensemble"]
    oof = {a: np.full(len(y), np.nan) for a in archs}
    for c in np.unique(cam):
        tr = np.where(cam != c)[0]; te = np.where(cam == c)[0]
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        for a in archs:
            oof[a][te], _ = _finetune(a, X[tr], y[tr], X[te], epochs=epochs, steps=steps, seed=0, **recipe)
        print(f"    (4)IECV leave-out {c}: train {len(tr)} predict {len(te)} (ev_te {int(y[te].sum())})  [{(time.time()-t0)/60:.1f}m]", flush=True)
    out = {}; okall = np.all([np.isfinite(oof[a]) for a in archs], axis=0)
    for a in archs:
        ok = np.isfinite(oof[a])
        out[a] = float(roc_auc_score(y[ok], oof[a][ok])) if ok.sum() and len(np.unique(y[ok])) == 2 else float("nan")
    oof["ensemble"] = np.where(okall, np.mean([oof[a] for a in archs], axis=0), np.nan)
    out["ensemble"] = float(roc_auc_score(y[okall], oof["ensemble"][okall])) if okall.sum() and len(np.unique(y[okall])) == 2 else float("nan")
    return out, oof


def _all_camstrat(archs, X, y, g, cam, *, epochs, steps, t0, recipe=None, reps=2):
    """(4) -all: camera-stratified, patient-grouped GroupKFold pooled OOF -- device+patient removed in ONE
    well-powered split (circularity is n/a for a single baseline CFP). Per-arch + soft-vote ensemble."""
    recipe = recipe or {}
    strat = np.array([f"{c}|{v}" for c, v in zip(cam, y)])                  # stratify on camera x label
    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    keys = list(archs) + ["ensemble"]
    if n_splits < 2:
        nanv = {k: float("nan") for k in keys}
        return nanv, {k: np.full(len(y), np.nan) for k in keys}
    oof = {a: np.zeros(len(y)) for a in archs}; cnt = np.zeros(len(y))
    for r in range(reps):
        try:
            folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=r).split(X.numpy(), strat, g))
        except ValueError:
            folds = list(StratifiedGroupKFold(max(2, n_splits - 1), shuffle=True, random_state=r).split(X.numpy(), y, g))
        for fi, (tr, te) in enumerate(folds):
            for a in archs:
                p, _ = _finetune(a, X[tr], y[tr], X[te], epochs=epochs, steps=steps, seed=r, **recipe)
                oof[a][te] += p
            cnt[te] += 1
            print(f"    (4)all camstrat rep{r} fold{fi+1}/{len(folds)}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    out = {}; okall = cnt > 0
    oofp = {a: np.where(okall, oof[a] / np.maximum(cnt, 1), np.nan) for a in archs}   # per-eye OOF probabilities
    for a in archs:
        out[a] = float(roc_auc_score(y[okall], oofp[a][okall])) if len(np.unique(y[okall])) == 2 else float("nan")
    oofp["ensemble"] = np.where(okall, np.mean([oofp[a] for a in archs], axis=0), np.nan)
    out["ensemble"] = float(roc_auc_score(y[okall], oofp["ensemble"][okall])) if len(np.unique(y[okall])) == 2 else float("nan")
    return out, oofp


def _published(archs, X, y, g, cam, *, seeds, epochs, steps, t0, recipe=None, weighted=False):
    """Eye-level 80/20 over several seeds; per-arch + soft-vote ensemble AUC distribution.
    ``weighted``: Fouda's validation-weighted soft-vote (weight each arch by its held-out-val AUC) instead of
    the equal-weight mean."""
    recipe = recipe or {}
    per = {a: [] for a in archs}; per["ensemble"] = []
    dev = {a: [] for a in archs}; dev["ensemble"] = []                      # (1) -device: within-camera stratified AUC
    keys = list(archs) + ["ensemble"]
    roof = {k: np.zeros(len(y)) for k in keys}; rc = np.zeros(len(y))       # eye-level OOF (avg of test preds over seeds)
    for s in range(seeds):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=s, stratify=y)
        if y[te].sum() < 1 or y[te].sum() == len(te):
            continue
        probs = {}; wt = {}
        for a in archs:
            probs[a], va = _finetune(a, X[tr], y[tr], X[te], epochs=epochs, steps=steps, seed=s, **recipe)
            wt[a] = max(va, 1e-6) if (weighted and va is not None) else 1.0
            per[a].append(float(roc_auc_score(y[te], probs[a])))
            dev[a].append(_stratified_auc(probs[a], y[te], cam[te]))
            roof[a][te] += probs[a]
        w = np.array([wt[a] for a in archs]); w = w / w.sum()               # equal, or validation-weighted (Fouda)
        ens = np.sum([w[i] * probs[a] for i, a in enumerate(archs)], axis=0)
        per["ensemble"].append(float(roc_auc_score(y[te], ens)))
        dev["ensemble"].append(_stratified_auc(ens, y[te], cam[te]))
        roof["ensemble"][te] += ens; rc[te] += 1
        print(f"    reproduce seed{s}: " + ", ".join(f"{a} {per[a][-1]:.3f}" for a in archs)
              + f", ENS {per['ensemble'][-1]:.3f} | (1)dev {dev['ensemble'][-1]:.3f}  [{(time.time()-t0)/60:.1f}m]", flush=True)

    def _dist(d):
        return {k: {"aucs": v, "median": float(np.nanmedian(v)) if v else float("nan"),
                    "p90": float(np.nanpercentile(v, 90)) if v else float("nan"),
                    "max": float(np.nanmax(v)) if v else float("nan")} for k, v in d.items()}
    oof = {k: np.where(rc > 0, roof[k] / np.maximum(rc, 1), np.nan) for k in keys}   # per-eye reproduce OOF
    return {"reproduce": _dist(per), "device": _dist(dev), "oof": oof}


def _honest(archs, X, y, g, cam, *, epochs, steps, t0):
    """Patient-grouped 5-fold pooled OOF; per-arch + soft-vote ensemble AUC + camera-only reference."""
    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    oof = {a: np.full(len(y), np.nan) for a in archs}
    if n_splits >= 2:
        for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=0).split(X.numpy(), y, g):
            for a in archs:
                oof[a][te], _ = _finetune(a, X[tr], y[tr], X[te], epochs=epochs, steps=steps, seed=0)
    out = {}
    for a in archs:
        ok = np.isfinite(oof[a])
        out[a] = float(roc_auc_score(y[ok], oof[a][ok])) if ok.sum() and len(np.unique(y[ok])) == 2 else float("nan")
    okall = np.all([np.isfinite(oof[a]) for a in archs], axis=0)
    ens_oof = np.where(okall, np.mean([oof[a] for a in archs], axis=0), np.nan)
    out["ensemble"] = float(roc_auc_score(y[okall], ens_oof[okall])) if okall.sum() and len(np.unique(y[okall])) == 2 else float("nan")
    out["camera_only"] = float(roc_auc_score(y, (cam == "topcon").astype(int)))
    print(f"    HONEST: " + ", ".join(f"{a} {out[a]:.3f}" for a in archs)
          + f", ENS {out['ensemble']:.3f}, camera-only {out['camera_only']:.3f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    oof["ensemble"] = ens_oof
    return out, oof                                                         # (2) per-eye patient-grouped OOF


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="resnet50,efficientnet_b4,densenet121,convnext_tiny")
    ap.add_argument("--endpoints", default="plr2,plr3,md")
    ap.add_argument("--crop", choices=["roi", "full"], default="roi")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6, help="batches per epoch (balanced sampler)")
    ap.add_argument("--seeds", type=int, default=5, help="# eye-level splits for the published regime")
    ap.add_argument("--faithful", action="store_true",
                    help="Huang GRAPE recipe: full fine-tune + cross-entropy + full-epoch batch-4 + cosine LR (use with --crop full --epochs 264)")
    ap.add_argument("--fouda", action="store_true",
                    help="Fouda IMSA-2025 training controls: focal loss + cosine LR + early stopping (patience 40) + "
                         "batch-8 + lr 4e-5 + validation-weighted soft-vote (use with --crop roi; PLR3 only)")
    ap.add_argument("--bs", type=int, default=0,
                    help="override batch size for GPU throughput; LR is linearly rescaled from the recipe's base "
                         "(Huang bs4/lr1e-4, Fouda bs8/lr4e-5, default bs32/lr3e-4) so the effective recipe is preserved. 0 = recipe default.")
    ap.add_argument("--patience", type=int, default=40, help="early-stopping patience in epochs (Fouda; Huang when --early-stop)")
    ap.add_argument("--early-stop", dest="early_stop", action="store_true",
                    help="enable early stopping for the Huang recipe too (speed; deviates from the paper's fixed 264-epoch / full-cosine run)")
    ap.add_argument("--loss", choices=["focal", "ce"], default="focal",
                    help="loss for the default (non-faithful) recipe; lets you mix e.g. ROI + cross-entropy + partial fine-tune")
    ap.add_argument("--unfreeze", type=float, default=0.30,
                    help="fraction of the backbone to unfreeze in the default recipe (1.0 = full fine-tune)")
    ap.add_argument("--manifest", default="data/derived/grape_cfp_manifest.csv")
    ap.add_argument("--out", default="data/derived/grape_cfp_finetune_results.json")
    args = ap.parse_args()
    archs = args.archs.split(",")
    base_bs, base_lr = (4, 1e-4) if args.faithful else ((8, 4e-5) if args.fouda else (32, 3e-4))   # each recipe's native batch/LR
    bs = args.bs or base_bs
    lr = base_lr * bs / base_bs                                             # linear LR-scaling rule (preserves dynamics)
    if args.faithful:
        recipe = dict(loss="ce", full_epoch=True, unfreeze_frac=1.0, bs=bs, lr=lr,
                      early_stop=args.early_stop, patience=args.patience)   # early_stop off => strict fixed-264
    elif args.fouda:
        recipe = dict(loss="focal", early_stop=True, patience=args.patience, bs=bs, lr=lr)   # cosine + early stop inside
    else:
        recipe = dict(bs=bs, lr=lr, early_stop=args.early_stop, patience=args.patience,
                      loss=args.loss, unfreeze_frac=args.unfreeze)   # mix loss/crop/unfreeze freely
    if args.bs or args.fouda:
        print(f"[ft] batch={bs} (base {base_bs}) -> LR-scaled lr={lr:.2e} (base {base_lr:.0e}); "
              f"fouda={args.fouda} AMP={'on' if _AMP else 'off'}", flush=True)

    man = pd.read_csv(args.manifest)
    man = man[man["exists"]].reset_index(drop=True)
    paths = [_roi_path(p) for p in man["cfp_path"]] if args.crop == "roi" else man["cfp_path"].tolist()
    n_roi = sum(1 for a, b in zip(paths, man["cfp_path"]) if a != b)
    print(f"[ft] {len(man)} eyes; crop={args.crop} ({n_roi} ROI hits); archs={archs}; preprocessing…", flush=True)
    X = _load_images(paths)
    g = man["eye_id"].str.split("_").str[0].to_numpy()
    cam = man["camera"].to_numpy()
    results = {"_meta": {"archs": archs, "crop": args.crop, "epochs": args.epochs, "steps": args.steps, "seeds": args.seeds}}
    oof_save = {"eye_id": man["eye_id"].to_numpy().astype(str), "patient": g.astype(str), "camera": cam.astype(str)}
    oof_path = str(Path(args.out).with_suffix("")) + "_oof.npz"
    t0 = time.time()

    for ep in args.endpoints.split(","):
        y = man[f"prog_{ep}"].astype(int).to_numpy()
        print(f"[ft] === {ep.upper()} ({int(y.sum())} pos / {len(y)}) ===", flush=True)
        print(f"[ft] {ep.upper()} :: reproduce (eye-level seeds) + (1)-device seed-wise stratified AUC", flush=True)
        pub = _published(archs, X, y, g, cam, seeds=args.seeds, epochs=args.epochs, steps=args.steps, t0=t0, recipe=recipe, weighted=args.fouda)
        keys = list(archs) + ["ensemble"]
        aaroc = {a: _aaroc(pub["oof"][a], y, cam) for a in keys}                  # (1) covariate-adjusted AUC (Janes-Pepe)
        strat = {a: _stratified_auc(pub["oof"][a], y, cam) for a in keys}         # (1) robustness twin (stratified concordance)
        print(f"[ft] {ep.upper()} :: (2)-patient (patient-grouped 5-fold OOF) + camera-only floor", flush=True)
        patient, patient_oof = _honest(archs, X, y, g, cam, epochs=args.epochs, steps=args.steps, t0=t0)
        # headline (4)-all = (1)+(2)+(3): AAROC on the patient-grouped OOF (single baseline CFP has no (3)); this is the
        # cleanest within-GRAPE floor, nested under (1)-device, and it is FREE -- scored on the (2)-patient OOF already
        # computed (no extra training). The IECV leave-one-camera-out is kept as the transport twin (an extra refit).
        all_combined = {a: _aaroc(patient_oof[a], y, cam) for a in keys}          # (4) = (1)+(2): AAROC on patient-grouped OOF
        print(f"[ft] {ep.upper()} :: (4)-all combined = AAROC on the patient-grouped OOF; IECV transport twin; (3)-circularity n/a", flush=True)
        allc, all_oof = _iecv(archs, X, y, g, cam, epochs=args.epochs, steps=args.steps, t0=t0, recipe=recipe)
        results[ep] = {"n_pos": int(y.sum()),
                       "published": pub["reproduce"],                             # reproduce (eye-level dist)
                       "device_aaroc": aaroc, "device_strat": strat,              # (1) covariate-adjusted AUC + robustness twin
                       "device": pub["device"],                                   # (1) seed-wise stratified dist (kept)
                       "honest": patient, "all_combined": all_combined, "all": allc}  # (2) patient; (4) combined headline + IECV twin
        oof_save[f"{ep}_y"] = y                                                   # per-eye OOF probs for every regime
        for regime, od in (("repro", pub["oof"]), ("patient", patient_oof), ("all", all_oof)):
            for a, arr in od.items():
                oof_save[f"{ep}_{regime}_{a}"] = np.asarray(arr, dtype=np.float32)
        def _q(d, k, e="ensemble"):
            v = d.get(e, {}); return v.get(k) if isinstance(v, dict) else v
        print(f"[ft] {ep.upper()} SUMMARY (ensemble): reproduce p90 {_q(pub['reproduce'],'p90'):.3f} | "
              f"(1)device AAROC {aaroc['ensemble']:.3f} (strat {strat['ensemble']:.3f}) | "
              f"(2)patient {patient.get('ensemble', float('nan')):.3f} | (4)combined {all_combined.get('ensemble', float('nan')):.3f} "
              f"(IECV {allc.get('ensemble', float('nan')):.3f}) | "
              f"floor(camera) {patient.get('camera_only', float('nan')):.3f}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))   # checkpoint after each endpoint
        np.savez(oof_path, **oof_save)                             # checkpoint per-eye OOF too

    print(f"[ft] wrote {args.out} + {oof_path} [{(time.time()-t0)/60:.1f} min]", flush=True)
    print(f"[ft] OOF npz keys: eye_id/patient/camera + per endpoint {{ep}}_y and {{ep}}_{{repro,patient,all}}_{{arch,ensemble}}", flush=True)


if __name__ == "__main__":
    main()
