"""Controlled B-scan perturbation axis — the IDENTIFIED same-input tolerance test for a B-scan model.

The map-offset axis (``transforms.py``) scores map-readers; a B-scan segmenter (Eyened/OCT-ONH) reads raw
B-scans, so its identified tolerance test is a controlled perturbation of THAT input. These are the
B-scan-level imaging effects that distinguish devices / acquisition quality (docs §G), graded by a scalar
``level`` in [0, 1] and confound-free by construction (we impose them on fixed content):

  * speckle   — OCT speckle is MULTIPLICATIVE (Schmitt 1999, DOI 10.1117/1.429925): a random gain field.
                The contrast ratio (SD/mean) is **0.52 for the AMPLITUDE/envelope (Rayleigh)** speckle and
                **1.0 for fully-developed INTENSITY (negative-exponential)** speckle (Sci Rep 2017,
                PMC5665955; Goodman). We inject the AMPLITUDE contrast (0.52) as a per-pixel Gamma gain of
                mean 1 — a deliberately SUB-fully-developed (multi-look, k=1/CV²≈3.7) stress dose applied to
                the 8-bit display B-scan (a stress perturbation, not a physically-exact linear-intensity
                speckle simulation). For fully-developed intensity speckle use CV≈1.0 (an exponential field).
  * contrast  — global intensity/contrast scaling (vendor "look" / gain differences).
  * blur      — Gaussian blur = axial/lateral resolution loss (a weaker, §G-flagged axis).

⚠ Honest scope: identified + physics-grounded, but NOT calibrated to a specific device-pair's same-eye
statistics the way the map offset is (`calibrated_offset`). So this scores imaging-quality/appearance
tolerance; the calibrated *vendor* result stays on the map axis + the V6 cross-vendor corroboration.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from . import device_physics

SPECKLE_CONTRAST: float = 0.52   # AMPLITUDE/Rayleigh speckle contrast (SD/mean) at level 1 (intensity=1.0)
CONTRAST_MAX: float = 0.40       # max +/- global contrast scaling at level 1
BLUR_MAX_SIGMA: float = 2.0      # max Gaussian blur sigma (px) at level 1 (resolution loss)
DIST_REF: float = 40.0           # intensity change normalised to distance ~1 (0-255 B-scan scale)
CIRRUS_CUBE_UM_PER_PX: float = 10.0   # ~2 mm axial / 200 px for the Harvard-GF/GDP Cirrus cube

TRANSFORM_NAMES = ("speckle", "contrast", "blur")


def _valid_range(img: np.ndarray) -> tuple[float, float]:
    """Valid B-scan intensity range for clipping — non-negative up to the 0-255 display max (never
    degenerate, so a constant region still receives the multiplicative speckle before saturation)."""
    return 0.0, float(max(255.0, np.nanmax(img)))


def speckle(img: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Multiplicative speckle: a per-pixel Gamma gain field (mean 1, CV = 0.52*level = the AMPLITUDE contrast;
    a sub-fully-developed multi-look stress dose — fully-developed intensity speckle would be CV=1.0)."""
    if level <= 0:
        return np.asarray(img, dtype=float).copy()
    cv = SPECKLE_CONTRAST * level
    k = 1.0 / max(cv * cv, 1e-6)          # Gamma shape: CV = 1/sqrt(k)
    field = rng.gamma(k, 1.0 / k, size=img.shape)   # mean 1, CV = cv
    lo, hi = _valid_range(img)
    return np.clip(img * field, lo, hi)


def contrast(img: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Global contrast/intensity scaling about the image mean (device gain / 'look')."""
    if level <= 0:
        return np.asarray(img, dtype=float).copy()
    m = float(np.nanmean(img))
    factor = 1.0 + rng.uniform(-CONTRAST_MAX, CONTRAST_MAX) * level
    lo, hi = _valid_range(img)
    return np.clip(m + (img - m) * factor, lo, hi)


def blur(img: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Gaussian blur = axial/lateral resolution loss (dosed by level)."""
    if level <= 0:
        return np.asarray(img, dtype=float).copy()
    return gaussian_filter(np.asarray(img, dtype=float), sigma=BLUR_MAX_SIGMA * level)


_TRANSFORMS = {"speckle": speckle, "contrast": contrast, "blur": blur}


def apply_bscan_format(img: np.ndarray, rng: np.random.Generator, level: float, *,
                       exclude: tuple[str, ...] = ()) -> np.ndarray:
    """Compose the non-excluded B-scan perturbations at ``level`` (deterministic in *which* fire)."""
    out = np.asarray(img, dtype=float)
    for name in TRANSFORM_NAMES:
        if name in exclude:
            continue
        out = _TRANSFORMS[name](out, rng, level)
    return out


def bscan_format_distance(clean: np.ndarray, perturbed: np.ndarray) -> float:
    """Normalised mean absolute intensity change — the B-scan dose-response x-axis (rises with level)."""
    clean, perturbed = np.asarray(clean, dtype=float), np.asarray(perturbed, dtype=float)
    return float(np.nanmean(np.abs(perturbed - clean)) / DIST_REF)


# --- PHYSICS-DERIVED cross-device B-scan shift (OCT-FormatShift Pillar 1) --------------------------------
# Unlike the generic hand-set path above, this composes a device-pair shift whose magnitudes are DERIVED
# from published device specs (`device_physics`): the speckle-contrast change (the DEPLOYED pair is Spectralis
# 16-frame ART → Topcon single-frame, i.e. MORE grain injected), the added axial blur (coarser device
# resolution; near-INERT at the deployed scale — the stress is speckle+contrast dominated), and the
# display-contrast change (wavelength/backscatter). `level` is the overall dose. Defaults are the deployed
# Spectralis→Topcon pair (every call site passes them explicitly regardless).
def apply_bscan_device_shift(img: np.ndarray, rng: np.random.Generator, level: float, *,
                             ref: str = "spectralis", target: str = "topcon",
                             um_per_px: float = CIRRUS_CUBE_UM_PER_PX,
                             exclude: tuple[str, ...] = ()) -> np.ndarray:
    """Physics-derived ref->target B-scan appearance shift (speckle change + added blur + contrast), dosed by
    ``level``. The magnitudes come from ``device_physics.cross_device_appearance`` (device specs), so the
    only free knob is the overall dose."""
    out = np.asarray(img, dtype=float)
    if level <= 0:
        return out.copy()
    a = device_physics.cross_device_appearance(ref, target)
    # speckle-contrast change. Direction-aware (the substrate keeps its own speckle; we move TOWARD the target):
    #   * target NOISIER (ratio>1, e.g. Spectralis 16-frame ART -> Topcon/Cirrus single-frame): INJECT an
    #     independent multiplicative Gamma gain field. OCT speckle is multiplicative (Schmitt 1999) and
    #     independent multiplicative fields compose in QUADRATURE (Goodman), so injecting contrast
    #     c_add = sqrt(C_target^2 - C_ref^2) makes the composed amplitude contrast reach C_target. This ADDS
    #     real speckle grain (physically right) rather than amplifying the existing residual (which would just
    #     scale the anatomy too).
    #   * target SMOOTHER (ratio<1, e.g. -> ART-averaged Spectralis): suppress by blending the high-frequency
    #     fluctuation toward the local mean (there is nothing to "remove" but the existing texture).
    if "speckle" not in exclude and not np.isclose(a.speckle_ratio, 1.0):
        if a.speckle_ratio > 1.0:
            c_add = np.sqrt(max(0.0, a.speckle_contrast_target ** 2 - a.speckle_contrast_ref ** 2)) * level
            if c_add > 1e-4:
                k = 1.0 / (c_add * c_add)                       # Gamma shape: mean 1, CV = c_add
                out = out * rng.gamma(k, 1.0 / k, size=out.shape)
        else:
            local_mean = gaussian_filter(out, sigma=1.0)
            w = max(0.0, 1.0 + (a.speckle_ratio - 1.0) * level)
            out = local_mean + (out - local_mean) * w
    # added axial blur to reach the coarser device resolution (sigma in px = derived um / um-per-px * dose)
    if "blur" not in exclude and a.added_blur_sigma_um > 0.0:
        sigma_px = a.added_blur_sigma_um / max(um_per_px, 1e-6) * level
        if sigma_px > 1e-3:
            out = gaussian_filter(out, sigma=sigma_px)
    # display-contrast change from relative backscatter (longer wavelength -> dimmer)
    if "contrast" not in exclude and not np.isclose(a.contrast_ratio, 1.0):
        m = float(np.nanmean(out))
        out = m + (out - m) * (1.0 + (a.contrast_ratio - 1.0) * level)
    lo, hi = _valid_range(img)
    return np.clip(out, lo, hi)
