"""Eyened/OCT-ONH read-stability under the B-scan perturbation — an INSTRUMENT QC read, NOT a contender rank.

⚠ ROLE CHANGE (docs §8/§9): Eyened is a SEGMENTATION model (B-scan → RNFL → cpRNFL thickness), not a
glaucoma-prediction model, so the benchmark uses it as the reference MEASUREMENT INSTRUMENT (the ruler +
the neutral thickness estimator for calibration/validation), NOT a scored contender — scoring the
measurement instrument as a competitor would be circular (A2). The scored contenders are the trained
PREDICTION models (RETFound + the deferred MIRAGE/DINOv2/supervised CNN).

This module keeps the read machinery: it turns the extracted Eyened cpRNFL reads (clean + per B-scan
perturbation level, from `scripts/run_eyened_contender.py`) into a recovery / stability / retention summary
— now interpreted as an **instrument QC property** ("is the ruler steady under a B-scan appearance
perturbation?"), reported axis-labelled and NOT placed on the contender leaderboard. The `benchmark.certify`
tier it computes is a convenience summary of that stability, not a competitive rank.

Honest scope: the B-scan axis is identified + physics-grounded but not calibrated to a device-pair's
same-eye stats (unlike the map offset); the calibrated vendor result stays on the map axis + V6.
"""

from __future__ import annotations

import numpy as np

from . import benchmark as B
from . import canonical
from .envelope import sector_matrix


def _mae(reads: np.ndarray, truth: np.ndarray, quad_truth: np.ndarray) -> tuple[float, float]:
    glob = float(np.nanmean(np.abs(reads - truth)))
    quad = sector_matrix(reads)[:, 1:]
    focal = float(np.nanmean(np.nanmax(np.abs(quad - quad_truth), axis=1)))
    return glob, focal


def score_eyened(clean_reads: np.ndarray, perturbed_by_level: dict[float, np.ndarray],
                 labels: np.ndarray, *, rng: np.random.Generator | None = None, test_frac: float = 0.5,
                 n_boot: int = 1000, headline_level: float | None = None) -> dict:
    """Eyened tolerance certificate on the B-scan perturbation axis, via the SHARED retention gate.

    ``clean_reads`` (N, n) = Eyened cpRNFL on clean B-scans (the per-eye reference the perturbed read must
    recover); ``perturbed_by_level`` = {level: (N, n)} Eyened reads on perturbed B-scans; ``labels`` (N,)
    with 0 = healthy, >0 = glaucoma (GLEAM). The normative reference is fit on a HELD-OUT TRAIN split (not
    in-sample — phase-review fix #6), the gate is detection RETENTION under perturbation, and the retention
    CI is bootstrapped → 'indeterminate' when the (underpowered GLEAM) gate straddles the floor.
    """
    labels = (np.asarray(labels) > 0).astype(int)                 # binarise glaucoma
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(clean_reads)
    idx = rng.permutation(n); cut = max(1, int((1.0 - test_frac) * n))
    tr, ev = idx[:cut], idx[cut:]
    ref = canonical.NormativeReference.fit(clean_reads[tr], labels[tr] == 0)   # held-out normative reference
    clean_ev = clean_reads[ev]
    truth = clean_ev                                             # the clean Eyened read IS the per-eye reference
    quad_truth = sector_matrix(truth)[:, 1:]
    const = np.repeat(clean_ev.mean(axis=0, keepdims=True), len(clean_ev), axis=0)  # constant reader
    const_mae, const_focal = _mae(const, truth, quad_truth)
    pos = labels[ev] == 1
    clean_call = ref.abnormal_batch(clean_ev)
    sens_clean = float(np.mean(clean_call[pos] == 1)) if pos.any() else float("nan")
    clean_pos = np.asarray(clean_call[pos] == 1)

    rows = {}
    for lv, reads in sorted(perturbed_by_level.items()):
        reads_ev = reads[ev]
        mae, focal = _mae(reads_ev, truth, quad_truth)
        call = ref.abnormal_batch(reads_ev)
        flip = float(np.mean(call != clean_call))
        sens_shift = float(np.mean(call[pos] == 1)) if pos.any() else float("nan")
        sys_bias = float(np.nanmean(np.abs(np.nanmean(reads_ev - truth, axis=0))))
        rci = B.retention_ci(clean_pos, np.asarray(call[pos] == 1), rng, n_boot=n_boot) if pos.any() else None
        earned = mae <= const_mae * (1.0 - B.EARNED_MARGIN)       # standalone: must beat the constant reader
        cert = B.certify(mae=mae, focal_mae=focal, const_mae=const_mae, const_focal=const_focal, flip=flip,
                         sens_clean=sens_clean, sens_shift=sens_shift, earned=earned, retention_ci=rci)
        rows[lv] = {"level": lv, "rnfl_mae": mae, "sys_bias_um": sys_bias, "flip_rate": flip, **cert}

    lv = headline_level if headline_level is not None else max(rows)
    return {"model": "eyened", "axis": "bscan_perturbation", "n": int(len(clean_reads)),
            "n_eval": int(len(ev)), "glaucoma_rate": float((labels == 1).mean()), "per_level": rows, **rows[lv]}
