"""Torch full-resolution readers for OCT-FormatShift (the v3 build, Phase A/B).

⚠ TORCH-ONLY — deliberately NOT imported by ``oct_format/__init__`` (which imports nothing), so the CPU
harness stays torch-free; only the ``.venv-torch`` panel scripts import this explicitly. A ``FullResReader``
is a plain ``OCTModel`` (``pipeline.OCTModel``): its referral probability is the model's glaucoma
probability DIRECTLY (Route 1 — no normative-percentile detour), so it is scored by ``degradation`` exactly
like the 64² map-readers, on the SAME per-eye shifts (fair same-axis comparison). Reused by the Phase-B
supervised B-scan CNN via the ``in_ch`` / ``prep`` hooks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def prep_maps(maps: np.ndarray) -> torch.Tensor:
    """(N,H,W) RNFLT map -> (N,2,H,W): [thickness/100 with NaN→0, validity mask]. The map-reader input."""
    m = np.asarray(maps, dtype=np.float32)
    finite = np.isfinite(m)
    thick = np.where(finite, m, 0.0) / 100.0
    return torch.from_numpy(np.stack([thick, finite.astype(np.float32)], axis=1))


def prep_gray(images: np.ndarray) -> torch.Tensor:
    """(N,H,W) 0-255 grayscale B-scan/thickness image -> (N,1,H,W) in [-1,1]. The Phase-B supervised
    B-scan CNN input (use with ``SmallCNN(in_ch=1)``)."""
    x = (np.asarray(images, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(x[:, None, :, :])


class SmallCNN(nn.Module):
    """Compact CNN: N (Conv-GroupNorm-ReLU-MaxPool) blocks → GAP → linear logit. GroupNorm = batch-size-safe
    (works for the small CPU batches + tiny bootstrap resamples). ``in_ch`` lets Phase B reuse it for B-scans."""

    def __init__(self, in_ch: int = 2, ch: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        c_in, blocks = in_ch, []
        for c in ch:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1), nn.GroupNorm(min(8, c), c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(ch[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x).mean(dim=(2, 3))).squeeze(1)


def forward_prob(net: nn.Module, maps: np.ndarray, *, prep: Callable = prep_maps, bs: int = 128) -> np.ndarray:
    """Per-eye glaucoma probability in [0,1] (batched, no-grad)."""
    net.eval()
    out = []
    with torch.no_grad():
        X = prep(maps)
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(net(X[i:i + bs])).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def clean_auc(net: nn.Module, maps: np.ndarray, y: np.ndarray, *, prep: Callable = prep_maps) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, forward_prob(net, maps, prep=prep)))


class FullResReader:
    """An ``OCTModel``: a trained full-resolution net whose referral proba is its glaucoma probability directly."""

    def __init__(self, net: nn.Module, name: str, adaptation: str, *, representation: str = "map-fullres",
                 prep: Callable = prep_maps):
        self.net, self.name, self.adaptation, self.representation, self._prep = net, name, adaptation, representation, prep

    def predict_referral_proba(self, eyes: np.ndarray) -> np.ndarray:
        return forward_prob(self.net, eyes, prep=self._prep)


def train_cnn(train_maps: np.ndarray, train_y: np.ndarray, val_maps: np.ndarray, val_y: np.ndarray, *,
              epochs: int, augment: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
              seed: int = 0, lr: float = 1e-3, bs: int = 32, in_ch: int = 2, consistency: float = 0.0,
              prep: Callable = prep_maps) -> nn.Module:
    """Train the compact CNN (BCE with class-balancing pos_weight, Adam), early-stopping on val AUC.

    ``augment(batch_maps, rng) -> batch_maps`` (optional) applies per-batch training augmentation — the
    device-shift domain-randomisation regime (Phase A) or a B-scan perturbation (Phase B). ``None`` = the
    clean supervised baseline.

    ``consistency`` > 0 turns plain domain-randomisation into **stability training** (Zheng 2016): each step is
    scored on BOTH the clean and the shifted view, plus a penalty on the disagreement between their referral
    probabilities — which optimises the referral-FLIP directly (plain domain-rand only trains on shifted samples,
    so it need not reduce clean↔shifted disagreement). Returns the net loaded with its best-val-AUC weights.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SmallCNN(in_ch=in_ch)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(np.mean(np.asarray(train_y) == 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-3)]))
    ty = np.asarray(train_y).astype(np.float32)
    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(train_maps))
        for i in range(0, len(train_maps) - bs + 1, bs):        # drop the last partial batch (keeps GN stable)
            idx = perm[i:i + bs]
            batch = train_maps[idx]
            yb = torch.from_numpy(ty[idx])
            opt.zero_grad()
            if augment is not None and consistency > 0:          # stability training: accurate on clean AND shifted,
                zc, za = net(prep(batch)), net(prep(augment(batch, rng)))   # and their referral probs must AGREE
                loss = 0.5 * (lossf(zc, yb) + lossf(za, yb)) \
                    + consistency * ((torch.sigmoid(zc) - torch.sigmoid(za)) ** 2).mean()
            else:
                b = augment(batch, rng) if augment is not None else batch
                loss = lossf(net(prep(b)), yb)
            loss.backward()
            opt.step()
        auc = clean_auc(net, val_maps, val_y, prep=prep)
        if np.isfinite(auc) and auc > best_auc:
            best_auc, best_state = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net
