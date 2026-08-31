"""Learning-curve sample-size extrapolation — turns an underpowered atlas ranking into a sufficiency read.

At a fixed pilot size the input-sufficiency atlas is a list of AUCs/R² with wide CIs and no answer to "how
much data would each modality need to be *sufficient*". The classical fix (Figueroa et al., BMC Med Inform
Decis Mak 2012, PMC3307431; `pmsims`): measure performance at several *training sizes* by subsampling, then
fit the **inverse power law** ``score(n) = b − a·n^(−c)`` — whose asymptote ``b`` is the achievable ceiling
and which inverts to the ``n`` needed to reach a target. Interpolation is validated; far extrapolation is
not — so ceilings/`n`-to-target are reported as **projections** with the subsample spread as uncertainty.

Generic: :func:`learning_curve` takes a ``score_fn(row_indices) -> float`` (any patient-grouped CV metric)
and patient ``groups``; the caller supplies the metric (e.g. grouped-CV R² of a modality for MD-slope).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _ipl(n, a, b, c):
    return b - a * np.power(n, -c)


def learning_curve(score_fn, groups: np.ndarray, sizes, *, n_reps: int = 5, seed: int = 0) -> pd.DataFrame:
    """Performance vs #patients by subsampling whole patients ``n_reps`` times per size.

    ``score_fn`` receives the row indices of the sampled patients' eyes and returns a CV score (NaN allowed).
    Returns ``DataFrame(n_patients, score_mean, score_sd, n_reps)``."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows = []
    for nsz in sizes:
        nsz = int(min(nsz, len(uniq)))
        if nsz < 2:
            continue
        vals = []
        for _ in range(n_reps):
            pick = rng.choice(uniq, size=nsz, replace=False)
            idx = np.where(np.isin(groups, pick))[0]
            s = score_fn(idx)
            if s == s:                       # drop NaN scores (degenerate subsample)
                vals.append(float(s))
        if vals:
            rows.append({"n_patients": nsz, "score_mean": float(np.mean(vals)),
                         "score_sd": float(np.std(vals)), "n_reps": len(vals)})
    return pd.DataFrame(rows)


def fit_inverse_power_law(sizes, scores) -> dict | None:
    """Fit ``score(n) = b − a·n^(−c)``; return ``{a, b, c, ceiling}`` (ceiling = ``b``) or None if it fails."""
    sizes = np.asarray(sizes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(sizes) & np.isfinite(scores)
    sizes, scores = sizes[ok], scores[ok]
    if len(sizes) < 3:
        return None
    p0 = [max(0.1, float(scores.max() - scores.min())), float(scores.max()) + 0.05, 0.5]
    try:
        popt, _ = curve_fit(_ipl, sizes, scores, p0=p0, maxfev=10000,
                            bounds=([0.0, 0.0, 0.01], [np.inf, 1.5, 3.0]))
    except (RuntimeError, ValueError):
        return None
    a, b, c = (float(v) for v in popt)
    return {"a": a, "b": b, "c": c, "ceiling": b}


def predict(params: dict, n: float) -> float:
    """Projected score at ``n`` patients from a fitted inverse-power-law."""
    return float(_ipl(np.asarray(n, dtype=float), params["a"], params["b"], params["c"]))


def n_for_target(params: dict, target: float) -> float:
    """Projected #patients to reach ``target`` score; ``inf`` if the target is at/above the ceiling."""
    a, b, c = params["a"], params["b"], params["c"]
    if target >= b:
        return float("inf")
    return float((a / (b - target)) ** (1.0 / c))
