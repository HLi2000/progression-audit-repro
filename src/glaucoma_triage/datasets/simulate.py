"""Synthetic augmentation of VF-MD for healthy PAPILA eyes.

PAPILA runs 30-2 perimetry but collects it only for eyes with IOP > 22 mmHg (Kovalyk et al.,
Sci Data 2022), so VF mean deviation (MD) is present almost only for non-healthy eyes
(324/333 healthy eyes lack it) and its *presence* leaks the label. We remove that missingness
leak by imputing MD for the healthy eyes that lack it, drawn from a normal-eye MD distribution.

Grounding (see ``config/datasets.yaml`` for the tunable parameters):
  * **Distribution = Gaussian.** The test/retest distribution of MD in healthy observers is
    not significantly different from normal (Wall et al., PubMed 27580755), so a parametric
    normal is the literature-supported choice.
  * **mean ≈ 0 dB, no age term.** MD is scored against the age-matched normative database, so
    a healthy eye's expected MD is ~0 by construction; the ~0.9 dB/decade age effect in raw
    sensitivity is already removed (PubMed 11919433). 30-2 vs 24-2 is immaterial for this
    global, center-weighted index.
  * **std ≈ 1.0 dB.** The published normal band of −2 to +2 dB is the *95% interval*
    (StatPearls NBK585112), so 2σ ≈ 2 dB ⇒ σ ≈ 1.0. This is deliberately ≥ the Harvard-GF
    non-glaucoma real MD std (~0.70 dB, which is compressed by VF-based selection): we must
    not make the simulated distribution narrower than real, or "is it simulated" becomes a
    new leak. See :func:`glaucoma_triage.eda.data_status.vf_md_distribution_check`, which
    derives the Harvard reference from data instead of asserting it.
  * **Clip = ±4 dB (~4σ).** A wide *physiological* guard, NOT the −2/+2 clinical band. That
    band is the 5th–95th percentile; clipping there would discard the legitimate lower-5%
    normal tail (mild cataract/fatigue/learning) and leave a too-clean spike separable in
    shape from real MD. Humphrey convention: loss = negative.

Only HEALTHY eyes are imputed — simulating a normal MD for a glaucoma/suspect eye would be
wrong. Real measured MD is never overwritten. Imputed rows are flagged ``vf_md_simulated``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _truncated_normal(rng: np.random.Generator, mean: float, std: float,
                      lo: float, hi: float, size: int) -> np.ndarray:
    """Draw ``size`` samples from N(mean, std) truncated to [lo, hi] by rejection."""
    out = np.empty(size, dtype=float)
    filled = 0
    for _ in range(1000):  # guard against pathological (lo, hi)
        if filled >= size:
            break
        draw = rng.normal(mean, std, size)
        keep = draw[(draw >= lo) & (draw <= hi)]
        take = min(len(keep), size - filled)
        out[filled:filled + take] = keep[:take]
        filled += take
    if filled < size:  # fallback: clip (should not happen for sane bounds)
        out[filled:] = np.clip(rng.normal(mean, std, size - filled), lo, hi)
    return out


def simulate_healthy_vf_md(
    df: pd.DataFrame,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    trunc: tuple[float, float] = (-4.0, 4.0),
    seed: int = 0,
    rng: np.random.Generator | None = None,
    source: str = "papila",
) -> pd.DataFrame:
    """Impute normal-eye VF-MD for healthy ``source`` eyes that lack it.

    Returns a copy with ``vf_md`` filled for healthy eyes missing it and a boolean
    ``vf_md_simulated`` flag marking exactly those rows.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    out = df.copy()

    if "vf_md_simulated" not in out.columns:
        out["vf_md_simulated"] = False
    out["vf_md_simulated"] = out["vf_md_simulated"].fillna(False).astype(bool)

    vf = pd.to_numeric(out.get("vf_md"), errors="coerce")
    # Healthy eyes of this source that are missing MD (never touch non-healthy or real MD).
    mask = (out["source"] == source) & (out["diagnosis"] == "healthy") & vf.isna()
    n = int(mask.sum())
    if n:
        draws = _truncated_normal(rng, mean, std, trunc[0], trunc[1], n)
        out.loc[mask, "vf_md"] = draws
        out.loc[mask, "vf_md_simulated"] = True
    return out
