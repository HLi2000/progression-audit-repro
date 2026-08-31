"""Octopus → Humphrey total-deviation harmonization (the cross-instrument bridge for the GRAPE rung).

The GRAPE rung "breaks" not because of the spatial grid but because of **units/sign/normative**: GRAPE
ships raw Octopus differential-light **sensitivity** (higher = better, centred ~+25 dB), whereas the
interpretation layer is trained on Humphrey **total deviation** (loss = negative, centred near 0). This
module converts Octopus sensitivity to an HFA-equivalent TD so the two are commensurable.

Grounding (literature):
- **dB-scale offset.** 0 dB = 10,000 asb on the HFA vs 4,000 asb on Octopus, so an HFA sensitivity reads
  ``10·log10(10000/4000) ≈ 3.98 dB`` higher than the same threshold on Octopus (empirically ~+2.5–3.4 dB
  for SITA strategies; e.g. 19 dB HFA ≈ 15.3 dB Octopus). We use the physical scale offset.
- **Total deviation** = measured sensitivity − age-matched normal at each location (the "hill of vision",
  which declines with eccentricity and ~0.6 dB/decade with age; Heijl normative). We use an *approximate*
  hill-of-vision normal (a documented simplification, NOT a device normative DB).

**Honest ceiling:** Octopus↔HFA agreement is good for mild loss and degrades in moderate–severe disease
(Bland–Altman variability rises) — exactly the advanced, all-glaucoma regime GRAPE occupies — so expect
this to make GRAPE *commensurable and partially recovered*, not perfectly aligned.
"""

from __future__ import annotations

import numpy as np

# 10*log10(10000/4000): HFA sensitivity dB is ~3.98 higher than Octopus for the same threshold luminance.
OCTOPUS_TO_HFA_DB: float = 3.98

# Approximate HFA hill-of-vision normal: central ~34 dB, ~0.25 dB/deg eccentricity fall-off,
# ~0.065 dB/yr age decline referenced at age 45. Labelled approximate — not a vendor normative DB.
_NORMAL_CENTRE_DB: float = 34.0
_ECC_SLOPE_DB_PER_DEG: float = 0.25
_AGE_SLOPE_DB_PER_YR: float = 0.065
_AGE_REF_YR: float = 45.0


def octopus_to_hfa_db(sens: np.ndarray) -> np.ndarray:
    """Octopus sensitivity (dB) → HFA-equivalent sensitivity (dB) via the apostilb-scale offset."""
    return np.asarray(sens, dtype=float) + OCTOPUS_TO_HFA_DB


def hfa_age_normal(coords: np.ndarray, age: float) -> np.ndarray:
    """Approximate age-corrected normal HFA sensitivity (dB) at each ``(x, y)`` location (hill of vision)."""
    coords = np.asarray(coords, dtype=float)
    ecc = np.hypot(coords[:, 0], coords[:, 1])
    age = _AGE_REF_YR if age is None or not np.isfinite(age) else float(age)
    normal = (_NORMAL_CENTRE_DB - _ECC_SLOPE_DB_PER_DEG * ecc
              - _AGE_SLOPE_DB_PER_YR * (age - _AGE_REF_YR))
    return np.clip(normal, 0.0, 40.0)


def sensitivity_to_td(sens: np.ndarray, coords: np.ndarray, age: float,
                      normative: dict | None = None) -> np.ndarray:
    """Octopus raw sensitivity → HFA total-deviation analog (loss = negative), NaNs preserved.

    Default: ``TD = octopus_to_hfa_db(sens) − hfa_age_normal(coords, age)`` (approximate hill-of-vision).
    If ``normative`` (from :func:`datasets.vf_extra.fit_hill_of_vision`, keys ``level_normal_age45`` +
    ``age_slope``) is given, use a **spatially-flat empirical** normal fitted to real SUNY-IU controls —
    exact for MD (the trustworthy metric) and free of a fabricated eccentricity curve. Sign matches HFA."""
    if normative is not None:
        age_v = 45.0 if age is None or not np.isfinite(age) else float(age)
        normal = normative["level_normal_age45"] + normative["age_slope"] * (age_v - 45.0)
        return octopus_to_hfa_db(sens) - normal
    return octopus_to_hfa_db(sens) - hfa_age_normal(coords, age)
