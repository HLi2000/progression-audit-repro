"""Paired clean->shifted degradation — the PRIMARY v3 metric (OCT-FormatShift), plus the neutral-third
magnitude-match verifier.

Why this replaces effective-robustness (plan A4/A5): effective robustness needs hundreds of models to fit
the ID->OOD line and is weakest for synthetic + medical shift, and the Shi-2023 multi-ID correction needs
each model's true in-distribution set (absent for a macular-pretrained FM applied to a disc task). The
honest, model-count-agnostic read of a device shift is the PAIRED degradation of ONE decision pipeline on
the SAME eyes, clean vs shifted:

  * referral-flip          — fraction of eyes whose clinical call (proba>0.5) changes clean->shifted. The
                             operating-point-free stability of the decision itself.
  * ΔAUC                    — AUC(clean) - AUC(shift). Discrimination loss. NOTE: a pure calibration/location
                             device offset shifts every eye's score alike and BARELY moves the (rank-based)
                             AUC — a real, disclosed feature: a device bias is a calibration failure, not a
                             discrimination failure, so ΔAUC is intentionally the INSENSITIVE read here and
                             the operating-point metrics below carry the signal (measurement-science point).
  * Δsens@spec             — at the operating threshold that fixes clean specificity at ``spec`` (e.g. 0.90),
                             the SIGNED change in sensitivity under shift (sens_clean - sens_shift): + = the
                             SAME referral rule MISSES more disease after a device switch (the clinically-
                             decisive read); - = catches more, usually by over-referring (read WITH spec@shift).
  * detection-retention    — sens_shift / sens_clean at that fixed operating point (<=1). A sens-ONLY diagnostic
                             that is GAMEABLE (a reader flooding referrals under shift scores ~1.0 while spec
                             collapses) — DISPLAYED alongside, never a ranking key; read WITH spec@shift/flip/ΔAUC.

All are derived from the shared ``OCTModel.predict_referral_proba`` (``pipeline.py``), so every contender
— map CNN, B-scan FM, Eyened, ComBat-treated, Ridge floor — is scored by THIS code and each row is
representation-LABELLED. Eye-level percentile-bootstrap CIs (the eye is the unit; plan C8). A severity
sweep runs the whole thing across dose levels bracketing the published LoA.

Magnitude-match verifier (plan A1/A2): the SECONDARY cross-representation view is only licensed if a
Cirrus->Spectralis shift induces the SAME per-sector thickness change whether instantiated on the map or on
the B-scan. That match is measured with a NEUTRAL THIRD thickness estimator used by NO contender (a direct
pixel-mean sector reduction — not any reader's Ridge/FM/segmenter), so the check is not circular. It only
licenses the caveated secondary; matched MAGNITUDE is never claimed to be matched STRUCTURE (§8 proves a
location-scale vs a focal shift of equal magnitude reward different methods).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from . import canonical, transforms

__all__ = [
    "sens_at_spec", "paired_degradation", "build_shifted_maps", "severity_sweep",
    "neutral_sector_estimator", "magnitude_match",
]

_EPS = 1e-9


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, or NaN when a resample/split is single-class (AUC undefined)."""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _ece(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (Guo 2017): the bin-count-weighted gap between the referral probability
    (confidence) and the observed glaucoma frequency (accuracy). A pure device OFFSET moves every score alike
    and BARELY changes the rank-based AUC, but it DOES shift calibration — so ΔECE is the read that exposes
    'a device shift is a calibration failure, not a discrimination one' (the counterpart to the insensitive ΔAUC)."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (scores > lo) & (scores <= hi) if i > 0 else (scores >= lo) & (scores <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(labels[m].mean()) - float(scores[m].mean()))
    return float(ece)


def sens_at_spec(labels: np.ndarray, scores: np.ndarray, spec: float) -> tuple[float, float]:
    """Operating threshold fixing specificity at ``spec`` among negatives + the sensitivity there.

    Returns ``(sensitivity, threshold)``. The threshold = the ``spec`` quantile of the NEGATIVE scores (so
    a fraction ``spec`` of negatives fall at/below it -> specificity ``spec``); sensitivity = fraction of
    positives strictly above it. NaN sens if either class is empty.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    neg, pos = scores[labels == 0], scores[labels == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("inf")
    thr = float(np.quantile(neg, spec))
    return float(np.mean(pos > thr)), thr


def _point_metrics(pc: np.ndarray, ps: np.ndarray, labels: np.ndarray, spec: float,
                   thr: float | None = None) -> dict:
    """The paired-degradation reads from clean/shift probabilities + labels (no bootstrap).

    ``thr`` fixes the operating point on the CLEAN read; passing the point-estimate threshold into the
    bootstrap keeps every resample on the SAME clinical rule (else the CI would conflate threshold jitter
    with degradation).
    """
    call_c, call_s = pc > 0.5, ps > 0.5
    flip = float(np.mean(call_s != call_c))
    auc_c, auc_s = _auc(labels, pc), _auc(labels, ps)
    if thr is None:
        sens_c, thr = sens_at_spec(labels, pc, spec)
    else:
        pos = pc[labels == 1]
        sens_c = float(np.mean(pos > thr)) if pos.size else float("nan")
    pos_s, neg_s = ps[labels == 1], ps[labels == 0]
    sens_s = float(np.mean(pos_s > thr)) if pos_s.size else float("nan")
    spec_s = float(np.mean(neg_s <= thr)) if neg_s.size else float("nan")
    retention = float(min(sens_s / sens_c, 1.0)) if (np.isfinite(sens_c) and sens_c > _EPS) else float("nan")
    ece_c, ece_s = _ece(labels, pc), _ece(labels, ps)
    return {"referral_flip": flip, "auc_clean": auc_c, "auc_shift": auc_s,
            "delta_auc": (auc_c - auc_s) if (np.isfinite(auc_c) and np.isfinite(auc_s)) else float("nan"),
            "sens_at_spec_clean": sens_c, "sens_at_spec_shift": sens_s,
            "delta_sens_at_spec": (sens_c - sens_s) if (np.isfinite(sens_c) and np.isfinite(sens_s)) else float("nan"),
            "spec_realized_shift": spec_s, "retention": retention,
            "ece_clean": ece_c, "ece_shift": ece_s, "delta_ece": ece_s - ece_c, "op_threshold": thr}


def _bootstrap_ci(pc, ps, labels, spec, thr, rng, n_boot, alpha=0.05) -> dict:
    """Eye-level percentile CIs for the degradation reads (resample eyes, recompute at the FIXED clean op point)."""
    keys = ("auc_clean", "auc_shift", "delta_auc", "delta_sens_at_spec", "referral_flip", "retention", "delta_ece")
    if rng is None or n_boot <= 0:
        return {}
    n = len(labels)
    acc: dict[str, list] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = _point_metrics(pc[idx], ps[idx], np.asarray(labels)[idx], spec, thr=thr)
        for k in keys:
            if np.isfinite(m[k]):
                acc[k].append(m[k])
    return {k: [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]
            for k, v in acc.items() if v}


def paired_degradation(model: Any, clean_eyes: Any, shifted_eyes: Any, labels: np.ndarray, *,
                       spec: float = 0.90, n_boot: int = 1000, rng: np.random.Generator | None = None) -> dict:
    """Score ONE decision pipeline's clean->shifted degradation on the SAME eyes (the PRIMARY v3 metric).

    ``model`` is any ``OCTModel``; ``clean_eyes`` / ``shifted_eyes`` are its native inputs (canonical maps
    for a ``MapReaderPipeline``, an index array for a ``ProbaModel``); ``labels`` (0/healthy, >0/glaucoma).
    Returns the point reads + eye-level bootstrap CIs, tagged with the model's representation/adaptation so
    the panel stays representation-LABELLED.
    """
    labels = (np.asarray(labels) > 0).astype(int)
    pc = np.asarray(model.predict_referral_proba(clean_eyes), dtype=float)
    ps = np.asarray(model.predict_referral_proba(shifted_eyes), dtype=float)
    pt = _point_metrics(pc, ps, labels, spec)
    ci = _bootstrap_ci(pc, ps, labels, spec, pt["op_threshold"], rng, n_boot)
    return {"model": getattr(model, "name", "model"),
            "representation": getattr(model, "representation", "?"),
            "adaptation": getattr(model, "adaptation", "?"),
            "spec_target": spec, "n": int(len(labels)),
            "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
            **pt, "ci": ci}


# --- shift construction (mirrors run_benchmark's per-eye seeding so results are comparable) --------------
def build_shifted_maps(clean_maps: np.ndarray, level: float, rng: np.random.Generator, *,
                       pair: str = transforms.DEFAULT_PAIR, shift_fn: Callable = transforms.apply_format
                       ) -> np.ndarray:
    """Apply the calibrated shift per eye at ``level`` (each eye a distinct seeded draw, as run_benchmark)."""
    seed = int(rng.integers(0, 2**31 - 1))
    return np.stack([shift_fn(m, np.random.default_rng(seed + i), level, pair=pair)
                     for i, m in enumerate(clean_maps)])


def severity_sweep(model: Any, clean_maps: np.ndarray, labels: np.ndarray, *,
                   levels: tuple[float, ...] = (0.5, 1.0, 1.5), pair: str = transforms.DEFAULT_PAIR,
                   shift_fn: Callable = transforms.apply_format, spec: float = 0.90, n_boot: int = 1000,
                   rng: np.random.Generator | None = None) -> dict:
    """Run ``paired_degradation`` across dose levels bracketing the published LoA (severity curve, not a point).

    Map-axis only (the shift is a map transform). ``rng`` seeds both the per-level shift draw and the
    bootstrap; each level gets its own child stream so the curve is reproducible.
    """
    rng = rng or np.random.default_rng(0)
    per_level = {}
    for lv in levels:
        shifted = build_shifted_maps(clean_maps, lv, np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                                     pair=pair, shift_fn=shift_fn)
        per_level[lv] = paired_degradation(model, clean_maps, shifted, labels, spec=spec, n_boot=n_boot,
                                           rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
    return {"model": getattr(model, "name", "model"),
            "representation": getattr(model, "representation", "?"),
            "adaptation": getattr(model, "adaptation", "?"),
            "pair": pair, "spec_target": spec, "levels": list(levels), "per_level": per_level}


# --- neutral-third magnitude-match verifier (plan A1/A2) ------------------------------------------------
def neutral_sector_estimator(maps: np.ndarray) -> np.ndarray:
    """A NEUTRAL third thickness estimator used by NO contender: direct pixel-mean sector thickness (N, 5).

    Contenders read through a Ridge / FM / learned segmenter; this is a plain masked pixel mean per sector
    (``canonical.rnfl_sectors``), so using it to check that two representations' shifts induce the same
    sector-thickness change is not circular (the segmenter that defines the match is no player's own).
    """
    return np.array([[canonical.rnfl_sectors(m)[s] for s in canonical.SECTORS] for m in maps], dtype=float)


def magnitude_match(clean_maps: np.ndarray, shifted_maps: np.ndarray, *,
                    estimator: Callable = neutral_sector_estimator) -> dict:
    """Per-sector induced thickness change under the NEUTRAL estimator — the quantity to MATCH across
    representations before the caveated cross-representation view is licensed (plan A1/A2).

    Returns per-sector mean/SD of (shifted - clean) sector thickness and the mean |systematic shift|. In
    Phase 1 the map-axis and the B-scan-axis magnitudes are matched on this SAME neutral reduction; a match
    licenses the exploratory secondary only — matched magnitude is never claimed to be matched structure.
    """
    c, s = estimator(clean_maps), estimator(shifted_maps)
    d = s - c
    mean_d = np.nanmean(d, axis=0)
    return {"sectors": list(canonical.SECTORS),
            "mean_delta_um": [float(x) for x in mean_d],
            "sd_delta_um": [float(x) for x in np.nanstd(d, axis=0)],
            "abs_systematic_um": float(np.nanmean(np.abs(mean_d))),
            "global_systematic_um": float(abs(mean_d[0]))}
