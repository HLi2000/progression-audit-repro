"""Real cross-vendor cpRNFL corroboration (§6) — the Eyened-segmented torch track, scored on the main env.

The device axis is synthetic-and-identified (§0); this module is the BOUNDED real-data corroboration. It
consumes cpRNFL profiles that ``scripts/extract_eyened_cprnfl.py`` produced by running the ONE vendor-
neutral Eyened/OCT-ONH segmenter on each vendor's RAW B-scan (Spectralis=GLEAM, Topcon=GOALS). Holding the
segmentation source CONSTANT removes the expert-vs-device confound; the residual cross-vendor difference is
imaging (optics/geometry) + cohort.

⚠ This is an ANATOMY CORROBORATION, **NOT a model-benchmark** — and it cannot be one with this data:
  * the vendors' RAW inputs are DIFFERENT IMAGE TYPES (Spectralis en-face thickness map vs Topcon raw
    B-scan) from DIFFERENT COHORTS, so the only vendor-common representation is the segmented cpRNFL;
  * GOALS has NO glaucoma class labels → no detection metric is possible on it;
  * the challenge PNGs have an UNCERTAIN axial µm scale, so only the SCALE-INVARIANT profile SHAPE is
    cleanly comparable (a magnitude / referral read would be dominated by the pixel-scale mismatch + cohort).
So this CAN claim only that the reads measure the same anatomy (the double-hump aligns). Models are
BENCHMARKED on the identified synthetic device axis (§0) + real single-vendor GLEAM detection (the appearance
panel); a real cross-vendor DETECTION benchmark needs SAME-EYE cross-device data (AI-READI, deferred).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _shape(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / n if n > _EPS else x


def align_to_reference(prof: np.ndarray, ref: np.ndarray) -> tuple[int, bool]:
    """Best circular roll + flip mapping ``prof`` onto ``ref`` shape (device angular conventions differ)."""
    rs = _shape(ref)
    best = (-2.0, 0, False)
    for flip in (False, True):
        b = prof[::-1] if flip else prof
        for r in range(len(prof)):
            c = float(np.dot(rs, _shape(np.roll(b, r))))
            if c > best[0]:
                best = (c, r, flip)
    return best[1], best[2]


def _apply(prof: np.ndarray, roll: int, flip: bool) -> np.ndarray:
    return np.roll(prof[::-1] if flip else prof, roll)


def real_cross_vendor(profiles: dict[str, np.ndarray], reference: str = "spectralis") -> dict:
    """Scale-invariant SHAPE-agreement of real Eyened-segmented vendor cpRNFL profiles — a bounded ANATOMY
    CORROBORATION, NOT a model-benchmark (see the module docstring for why a benchmark is non-identified here).

    ``profiles`` = {vendor: (N, n) pixel-native RNFL profiles} from ``extract_eyened_cprnfl.py``. Each vendor's
    mean profile is angularly aligned to the reference and its shape correlation reported (mean-of-means +
    per-eye median). NO ComBat / magnitude harmonisation (that would be dominated by the pixel-scale unit
    mismatch, not device — and ComBat is a dropped finding, §8). The only honest read is the shape.
    """
    if reference not in profiles:
        reference = next(iter(profiles))
    ref = profiles[reference]
    ref_shape = _shape(ref.mean(axis=0))

    rows = {}
    for v, P in profiles.items():
        roll, flip = align_to_reference(P.mean(axis=0), ref.mean(axis=0))
        aligned = np.array([_apply(p, roll, flip) for p in P])
        per_eye = np.array([float(np.dot(ref_shape, _shape(p))) for p in aligned])   # per-eye corroboration spread
        rows[v] = {"n": int(len(P)), "global_px": float(P.mean()),
                   "shape_agreement_vs_ref": float(np.dot(ref_shape, _shape(aligned.mean(axis=0)))),
                   "per_eye_shape_agreement_median": float(np.median(per_eye))}
    return {"reference": reference, "per_vendor": rows,
            "min_shape_agreement": float(min(r["shape_agreement_vs_ref"] for r in rows.values())),
            "note": ("scale-invariant SHAPE agreement ONLY (the double-hump aligns → the vendors measure the "
                     "same anatomy, segmenter held constant). ⚠ NOT a model-benchmark: different raw image "
                     "types + cohorts, GOALS unlabeled, µm scale uncertain → any magnitude/referral read is "
                     "non-identified (§0). This CORROBORATES the synthetic-axis device result; models are "
                     "benchmarked on the synthetic axis + real single-vendor GLEAM detection. A real "
                     "cross-vendor DETECTION benchmark needs same-eye cross-device data (AI-READI, deferred).")}
