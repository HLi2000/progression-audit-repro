"""Grounded format-variant generator for VF fields + a measured ``format_distance``.

Each transform re-encodes a field the way a different capture system / output version / normative
would, scaled by a single ``level`` in [0, 1] (the envelope's intensity knob). Composing them and
measuring the resulting distance from the clean field gives the dose-response x-axis. Transforms are
grounded (realistic perturbations of real fields), not synthetic eyes.

Transforms (Phase 1s — 5 lit-supported; 10-2 crop + generic dropout + diffuse-depression all REMOVED — the last as
a *signal-change* confound, ablation-flagged: a media/pupil-depressed field genuinely has a lower MD, so training to
"undo" it biases the readout; it is a real out-of-scope confound, not a *format* re-encoding of the same signal):
  * test-retest noise (``add_noise``)  — heteroscedastic Gaussian; SD rises exponentially as sensitivity drops
                                         (Henson 2000: ln SD = 3.62 − 0.098·sens, glaucoma), capped at a realistic
                                         retest bound (Artes 2002) → ~2 dB mild, ~5 dB deep
  * normative shift                    — systematic ± dB calibration offset (a different normative DB / device)
  * algorithm shift                    — SITA ↔ Full-Threshold: a small offset + a variance change (FT ~15% noisier)
  * device shift                       — HFA → home tablet: central-biased sparse coverage (~40% vs 85%) [Device axis]
  * quantization                       — coarser dB rounding (HFA integer-dB / FT 4-2 staircase ≈ 2 dB bins)

``format_distance`` is measured on the canonical grid: it combines the fraction of canonical
locations that went missing with the normalized magnitude of value change on surviving locations,
so it rises monotonically (in expectation) with ``level`` and is comparable across transform mixes.
"""

from __future__ import annotations

import numpy as np

from . import canonical

# --- lit-grounded transform parameters (values at level = 1.0; dB unless a fraction) ---
# Heteroscedastic test-retest noise: SD = NOISE_SCALE·level·exp(HENSON_A + HENSON_B·sensitivity_dB), capped.
# Henson et al. IOVS 2000 (glaucoma): ln SD = 3.62 − 0.098·sens → ~2 dB at 30 dB, ~5 at 20, ~14 at 10.
HENSON_A, HENSON_B = 3.62, -0.098
NORMAL_REF = 30.0        # TD → sensitivity: sens ≈ TD + age-normal central level (~30 dB)
NOISE_SD_CAP = 5.0       # cap SD at a realistic retest bound (Artes 2002 ~5–6 dB; the 15 dB FOS bound over-noised)
_MILD_SD = float(np.exp(HENSON_A + HENSON_B * NORMAL_REF))   # ~1.97 dB, Henson SD at sens 30
NOISE_SCALE = 1.0        # multiplier on the curve; set_noise_max grounds it so mild-point SD = measured retest SD
SHIFT_MAX = 2.0          # dB ± normative/device-DB calibration offset (SITA-Std↔FT ~0.8, HFA↔Octopus ~4)
ALGO_OFFSET = 0.8        # dB SITA-Standard↔Full-Threshold mean offset (Bengtsson–Heijl)
ALGO_VAR = 0.15          # FT ≈ 15% noisier than SITA above 25 dB (Artes 2002)
DEVICE_COVERAGE = 0.40   # home-tablet canonical coverage at level 1 (VERIFIED Eyecatcher 0.40 vs HFA 0.85)
QUANT_MAX = 1.0          # extra dB of rounding step on top of 1 dB → ~2 dB bins (HFA integer / FT 4-2 staircase)
_DIST_REF = 10.0         # dB scale that normalizes value-change into ~[0,1] for format_distance


def set_noise_max(sd: float) -> None:
    """Ground the noise curve on real data: scale it so the mild-point (sens≈30 dB) SD equals the measured
    short-term test-retest SD (e.g. vfpwgRetest24d2 ~1.8 dB). Deeper points then scale up per Henson."""
    global NOISE_SCALE
    NOISE_SCALE = float(sd) / _MILD_SD


def _henson_sd(td: np.ndarray) -> np.ndarray:
    """Per-location test-retest SD (dB) from the Henson sensitivity–variability curve, capped."""
    sens = np.clip(td + NORMAL_REF, 0.0, 35.0)
    return np.minimum(np.exp(HENSON_A + HENSON_B * sens), NOISE_SD_CAP)


def add_noise(values: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Heteroscedastic test-retest noise (Henson 2000): SD rises steeply as sensitivity falls."""
    v = values.copy()
    m = ~np.isnan(v)
    sd = NOISE_SCALE * level * _henson_sd(v)
    v[m] = v[m] + rng.normal(0.0, 1.0, size=m.sum()) * sd[m]
    return v


def normative_shift(values: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Systematic ± dB calibration offset — a different normative database / device dB-scale (SITA-Std↔FT ~0.8 dB,
    HFA↔Octopus/OCCP↔SAP ~4 dB via the apostilb reference). Bidirectional; distinct from the algorithm/strategy
    offset and the one-directional media depression."""
    offset = SHIFT_MAX * level * (1.0 if rng.random() < 0.5 else -1.0)
    return values + offset


def algorithm_shift(values: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Thresholding-strategy change (SITA-Standard ↔ Full-Threshold): the observable output difference is a small
    global offset (~0.8 dB; Bengtsson–Heijl) **plus a variance change** (FT ≈15% noisier above 25 dB; Artes 2002).
    Models a shift away from the training convention (SITA-Standard) toward a noisier strategy."""
    v = values.copy()
    m = ~np.isnan(v)
    d = 1.0 if rng.random() < 0.5 else -1.0
    v[m] = v[m] + d * ALGO_OFFSET * level
    extra = ALGO_VAR * level * _henson_sd(v[m])
    v[m] = v[m] + rng.normal(0.0, 1.0, size=int(m.sum())) * extra
    return v


def device_shift(values: np.ndarray, coords: np.ndarray, rng: np.random.Generator,
                 level: float) -> np.ndarray:
    """HFA → home-tablet switch (Device axis): a short/adaptive unsupervised home test measures only a
    **central-biased subset** of the 24-2 — VERIFIED Eyecatcher tablet ~40% canonical coverage vs HFA ~85%, ecc
    mean 11° vs 17°. Drop toward that coverage, keeping central points preferentially (drop probability rises with
    eccentricity, with jitter). NOT the removed generic dropout (this is a real device property, not a normal-24-2
    artifact) and NOT a strict 10-2 crop (it extends to ~17°)."""
    v = values.copy()
    valid = np.where(~np.isnan(v))[0]
    if len(valid) == 0 or level <= 0.0:
        return v
    keep_frac = 1.0 - (1.0 - DEVICE_COVERAGE) * min(level, 1.0)      # level 1 → ~DEVICE_COVERAGE
    k_drop = int(round((1.0 - keep_frac) * len(valid)))
    if k_drop <= 0:
        return v
    ecc = np.hypot(coords[valid, 0], coords[valid, 1])
    order = np.argsort(ecc + rng.normal(0.0, 1.0, size=len(valid)))  # central first, periphery last (+jitter)
    v[valid[order[len(valid) - k_drop:]]] = np.nan                   # drop the most-peripheral k_drop
    return v


def quantize(values: np.ndarray, level: float) -> np.ndarray:
    """Coarser dB rounding — HFA stores integer dB and the Full-Threshold 4-2 staircase gives ≈2 dB resolution."""
    step = 1.0 + QUANT_MAX * level
    return np.round(values / step) * step


TRANSFORM_NAMES = ("normative_shift", "algorithm_shift", "add_noise", "device_shift", "quantize")

# Transforms that genuinely CHANGE the field's level/MD (a global dB shift) → an augmented row's own MD ≠ the clean
# MD, so the MD head must be trained on the transformed field's OWN MD for these (well-posed; see envelope._augment).
# The others (add_noise zero-mean, device_shift coverage, quantize) leave the underlying level intact → the clean MD
# stays the correct, well-posed recovery target.
LEVEL_CHANGING = ("normative_shift", "algorithm_shift")


def apply_format(values: np.ndarray, coords: np.ndarray, rng: np.random.Generator,
                 level: float, *, exclude: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compose the format transforms at a given ``level``; returns the re-encoded ``(values, coords)``.

    ``level <= 0`` is a true no-op (the clean anchor of the envelope sweep). Coords are unchanged; ``device_shift``
    encodes its sparse coverage as NaN on the same grid (which ``format_distance`` reads as the coverage term).
    ``exclude`` skips named transforms — used for leave-one-out **ablation** and to build the **tablet simulator**
    (``exclude={'algorithm_shift','quantize'}`` → the device+calibration+noise bundle).

    NOTE: ``apply_format`` is fully DETERMINISTIC in which transforms it applies (every non-excluded one fires). The
    per-transform *application probability* (device_shift Bernoulli p, Phase 1s) lives in ``envelope._augment`` only,
    NOT here, so the tablet simulator / R3 test re-encoding / ``format_distance`` stay deterministic."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(coords, dtype=float)
    if level <= 0.0:
        return v.copy(), c.copy()
    ex = exclude or set()
    if "normative_shift" not in ex:
        v = normative_shift(v, rng, level)
    if "algorithm_shift" not in ex:
        v = algorithm_shift(v, rng, level)
    if "add_noise" not in ex:
        v = add_noise(v, rng, level)
    if "device_shift" not in ex:
        v = device_shift(v, c, rng, level)
    if "quantize" not in ex:
        v = quantize(v, level)
    return v, c


def format_distance(v0: np.ndarray, c0: np.ndarray, v1: np.ndarray, c1: np.ndarray) -> float:
    """Measured distance in [0, 1] between a clean field and its re-encoding, on the canonical grid."""
    a = canonical.resample(v0, c0)
    b = canonical.resample(v1, c1)
    orig_valid = ~np.isnan(a)
    if orig_valid.sum() == 0:
        return 0.0
    now_missing = orig_valid & np.isnan(b)
    miss_frac = now_missing.sum() / orig_valid.sum()
    common = orig_valid & ~np.isnan(b)
    shift = float(np.mean(np.abs(a[common] - b[common]))) / _DIST_REF if common.any() else 0.0
    return float(np.clip(0.5 * miss_frac + 0.5 * min(shift, 1.0), 0.0, 1.0))
