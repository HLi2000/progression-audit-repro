"""Synthetic disc-OCT format generator on the canonical grid + a measured ``format_distance``.

The powered backbone of OCT-FormatShift. Each map-level transform re-encodes an RNFLT map the way a
different scanner / centration / normative reference would, scaled by a single ``level`` in [0, 1]
(the dose knob). Composing them and measuring the distance from the clean map gives the dose-response
x-axis. This is the *identified* manipulation (we impose it on fixed content — confound-free by
construction; ImageNet-C / domain-randomization standard), so it — not the confounded cross-cohort
data — carries the benchmark's tolerance read (see docs/OCT_FORMATSHIFT_SCOPING.md §0).

Map-level transforms (B-scan-level speckle/resolution/segmentation-error live on the volume/B-scan
entry, NOT here — applying them to a finished thickness raster mis-models the mechanism):
  * scanner_offset  — the LITERATURE-CALIBRATED cross-vendor axis: a deterministic per-sector BIAS +
                      a PER-EYE, ANATOMY-COUPLED spread (segmentation ⊕ centration ⊕ magnification),
                      fit to published same-eye Bland-Altman bias + LoA (see calibrated_offset). The
                      spread is drawn once per eye (coherent across the map), so the between-eye LoA
                      survives eye-level averaging (an i.i.d. per-pixel noise would not). The primary axis.
  * centration      — decentred scan: the disc-centred map is translated, redistributing the S/I peaks
                      (real tolerance ~0.05 mm before sectors shift materially; PMC5081164).
  * normative_shift — a small global reference-baseline offset (a different device's normative DB),
                      distinct from the vendor measurement bias; moves the abnormal call w/o tissue change.

``format_distance`` is measured on the canonical grid: normalized mean thickness change on surviving
pixels + fraction of coverage lost, so it rises (in expectation) with ``level`` and is comparable
across transform mixes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as ndi_shift

from . import calibrated_offset, canonical

# level 1.0 maps to this severity of the calibrated SCANNER-OFFSET. Set to 1.0 so scanner_offset at level 1.0
# injects EXACTLY the published cross-vendor scanner gap (bias + LoA), not the earlier 1.5x over-stress
# (docs §5b, the pending dosing fix — now resolved). NB the COMPOSED apply_format also adds centration +
# normative_shift, so the composed panel shift at level 1.0 is a bit larger than the scanner gap; the panel's
# headline is AUTO-selected at the level whose induced systematic shift ≈ the published bias (run_oct_degradation_panel).
SEVERITY_AT_LEVEL1: float = 1.0
CENTER_MAX_FRAC: float = 0.05     # max decentration at level 1, as a fraction of grid width (~0.3 mm on 6 mm)
NORM_SHIFT_MAX: float = 3.0       # um, max global normative-baseline offset at level 1
DIST_REF: float = 20.0           # um thickness change normalised to distance ~1 (~ the cross-vendor LoA scale)

TRANSFORM_NAMES = ("scanner_offset", "centration", "normative_shift")
DEFAULT_PAIR = "cirrus_to_spectralis"   # nearest, best-calibrated vendor pair
# The calibrated cross-vendor envelope a deploy-anywhere model must span (Spectralis..Topcon..RTVue).
# The DR arm domain-randomises over ALL of these (Tobin 2017) so it is one vendor-envelope-robust model,
# not a per-pair specialist; the benchmark then scores tolerance ACROSS this envelope.
CALIBRATED_PAIRS = tuple(calibrated_offset.PUBLISHED.keys())


def scanner_offset(
    m: np.ndarray, rng: np.random.Generator, level: float, *, pair: str = DEFAULT_PAIR
) -> np.ndarray:
    """Literature-calibrated cross-vendor axis (per-quadrant BIAS + per-eye anatomy-coupled LoA), dosed by
    ``level`` (level 1.0 = the published gap). NB: the LoA spread is a PER-EYE draw, so on a SINGLE map the
    global shift is bias±spread (it averages to the bias only ACROSS eyes) — the correct between-eye model."""
    if level <= 0:
        return np.asarray(m, dtype=float).copy()
    severity = level * SEVERITY_AT_LEVEL1
    return calibrated_offset.apply_sectored_map(m, pair, rng, severity=severity)


def centration(m: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Simulate a decentred scan by translating the disc-centred map (NaN-aware); edges become NaN."""
    if level <= 0:
        return np.asarray(m, dtype=float).copy()
    m = canonical.clean(m)
    h, w = m.shape
    max_shift = CENTER_MAX_FRAC * level * min(h, w)
    dy = rng.uniform(-max_shift, max_shift)
    dx = rng.uniform(-max_shift, max_shift)
    valid = np.isfinite(m).astype(float)
    filled = np.where(valid > 0, np.nan_to_num(m, nan=0.0), 0.0)
    num = ndi_shift(filled, (dy, dx), order=1, mode="constant", cval=0.0)
    den = ndi_shift(valid, (dy, dx), order=1, mode="constant", cval=0.0)
    out = np.full_like(m, np.nan)
    ok = den >= 0.3
    out[ok] = num[ok] / np.maximum(den[ok], 1e-6)
    return out


def normative_shift(m: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """A small global reference-baseline offset (different normative DB), dosed by ``level``."""
    if level <= 0:
        return np.asarray(m, dtype=float).copy()
    m = canonical.clean(m)
    delta = rng.uniform(-NORM_SHIFT_MAX, NORM_SHIFT_MAX) * level
    return np.where(np.isfinite(m), m + delta, m)


_TRANSFORMS = {
    "scanner_offset": scanner_offset,
    "centration": centration,
    "normative_shift": normative_shift,
}


def apply_format(
    m: np.ndarray,
    rng: np.random.Generator,
    level: float,
    *,
    pair: str = DEFAULT_PAIR,
    exclude: tuple[str, ...] = (),
) -> np.ndarray:
    """Compose the non-excluded map-level transforms at ``level`` (deterministic in *which* fire).

    Order: scanner_offset -> centration -> normative_shift. Randomness is only in the noise/shift draws.
    """
    out = canonical.clean(m)
    for name in TRANSFORM_NAMES:
        if name in exclude:
            continue
        fn = _TRANSFORMS[name]
        out = fn(out, rng, level, pair=pair) if name == "scanner_offset" else fn(out, rng, level)
    return out


# --- generic (NON-calibrated) augmentation, magnitude-matched to the calibrated axis, for the ablation ---
# The mandatory control (Taori 2020 / BigAug lesson): any strong aug improves OOD generalization, so the
# calibrated transform's realism is credited only if it beats an EQUAL-STRENGTH generic perturbation.
GENERIC_OFFSET_MAX: float = 8.0    # um random global offset at level 1 (spans the real vendor-bias range)
GENERIC_SCALE_MAX: float = 0.10    # +/- random global multiplicative
GENERIC_NOISE_SD: float = 6.0      # um random noise SD at level 1 (~ the calibrated noise magnitude)


def generic_offset(m: np.ndarray, rng: np.random.Generator, level: float, *, pair: str = DEFAULT_PAIR) -> np.ndarray:
    """A magnitude-matched but VENDOR-AGNOSTIC random offset+scale+noise (replaces the calibrated offset)."""
    if level <= 0:
        return np.asarray(m, dtype=float).copy()
    m = canonical.clean(m)
    a = rng.uniform(-GENERIC_OFFSET_MAX, GENERIC_OFFSET_MAX) * level
    b = 1.0 + rng.uniform(-GENERIC_SCALE_MAX, GENERIC_SCALE_MAX) * level
    noise = rng.normal(0.0, GENERIC_NOISE_SD * level, size=m.shape)
    return np.where(np.isfinite(m), a + b * m + noise, m)


def apply_generic_format(m: np.ndarray, rng: np.random.Generator, level: float, *,
                         pair: str = DEFAULT_PAIR, exclude: tuple[str, ...] = ()) -> np.ndarray:
    """Same composition as :func:`apply_format` but with a generic (non-calibrated) offset — the ablation arm."""
    out = canonical.clean(m)
    if "scanner_offset" not in exclude:
        out = generic_offset(out, rng, level)
    if "centration" not in exclude:
        out = centration(out, rng, level)
    if "normative_shift" not in exclude:
        out = normative_shift(out, rng, level)
    return out


# --- NON-location-scale axis: focal segmentation error (the anti-ComBat control) ------------------------
# A location-scale corrector (ComBat) inverts the affine offset axis BY CONSTRUCTION. To show the
# certificate MATCHES method to corruption (rather than rubber-stamping ComBat), a FOCAL, per-eye-random,
# NON-affine corruption is needed: segmentation failure = focal RNFL drop-outs (under-seg → toward 0) and
# spikes (over-seg). ComBat's per-point location/scale cannot invert a per-eye-random FOCAL pattern; a
# spatial DENOISER can.
#
# Citation FIXED to CIRCUMPAPILLARY RNFL (not the old macular PMID 27191396): Mansberger 2019 (PMID 30855415
# / PMC6499633, Spectralis 6° circle, 406 eyes) — native−manual global error 95% CI −6.5..+13.2 µm, positive-
# skewed (over-seg native reads thicker slightly more often), errors correlated; Mansberger 2017 (PMC5548380)
# automated 1.6 µm thinner, changing global class in 8.5% of scans. Prevalence ~15% normal → 46–84% glaucoma.
# ROLE (honest scope): this is the ANTI-COMBAT CONTROL — ISOLATED focal spikes/drop-outs a spatial denoiser
# recovers but ComBat's per-point location/scale cannot. It is deliberately kept isolated (small patches) to
# preserve that property; a Mansberger-magnitude CONTIGUOUS sector error would defeat the denoiser too, so the
# per-eye direction is skewed to Mansberger but the induced GLOBAL magnitude is a focal STRESS dose, NOT a
# calibrated global axis (the calibrated global axis is the map/B-scan endpoint — Pillar 1).
SEG_ERR_MAX_PATCHES: int = 8      # ISOLATED focal error patches at level 1 (kept small ⇒ denoiser-recoverable)
SEG_ERR_MAX_RADIUS: int = 4       # px patch radius at level 1
SEG_ERR_SPIKE_UM: tuple[float, float] = (30.0, 80.0)   # over-segmentation thickness spike range
SEG_ERR_P_OVER_MEAN: float = 0.6  # per-eye P(over-seg vs under-seg) mean → Mansberger positive skew


def segmentation_error(m: np.ndarray, rng: np.random.Generator, level: float, *,
                       pair: str = DEFAULT_PAIR, exclude: tuple[str, ...] = ()) -> np.ndarray:
    """Isolated, non-affine RNFL segmentation error — the anti-ComBat control (Mansberger-circumpapillary).

    ``level``-many small isolated patches, each a drop-out (→0) or an over-seg spike (+30..80 µm). The
    over-vs-under split is a PER-EYE draw around ``P_OVER_MEAN`` (0.6 → the Mansberger positive skew: some
    eyes read net-thicker, some net-thinner), so the cross-eye direction matches the literature while each
    error stays ISOLATED (a spatial denoiser recovers it; ComBat's per-point location/scale cannot). The
    induced global magnitude is a focal stress dose — the CALIBRATED global axis is the map/B-scan endpoint.
    """
    if level <= 0:
        return canonical.clean(m).copy()
    m = canonical.clean(m)
    out = m.copy()
    valid = np.isfinite(m)
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return out
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    p_over = float(np.clip(rng.normal(SEG_ERR_P_OVER_MEAN, 0.25), 0.0, 1.0))   # per-eye over/under tendency
    for _ in range(max(1, int(round(level * SEG_ERR_MAX_PATCHES)))):
        j = int(rng.integers(len(ys)))
        r = int(rng.integers(1, 1 + max(1, int(round(level * SEG_ERR_MAX_RADIUS)))))
        patch = (np.hypot(yy - ys[j], xx - xs[j]) <= r) & valid
        if rng.random() < p_over:
            out[patch] = m[patch] + rng.uniform(*SEG_ERR_SPIKE_UM)         # over-segmentation spike
        else:
            out[patch] = 0.0                                               # under-segmentation drop-out
    return out


def format_distance(m0: np.ndarray, m1: np.ndarray) -> float:
    """Distance between two canonical-grid maps: normalized thickness change + coverage loss, in [0, 1]."""
    a, b = canonical.clean(m0), canonical.clean(m1)
    if a.shape != b.shape:
        raise ValueError("format_distance expects same-shape canonical maps")
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    val_change = float(np.mean(np.abs(a[both] - b[both]))) / DIST_REF if both.any() else 1.0
    cov0 = float(fa.mean())
    cov_lost = max(0.0, cov0 - float(fb.mean())) / max(cov0, 1e-6)
    return float(np.clip(0.5 * min(1.0, val_change) + 0.5 * cov_lost, 0.0, 1.0))
