"""Partial Information Decomposition (PID) — a sample-efficient synergy/redundancy read for a pair of modalities.

At 27–40 events the Grabisch–Roubens Shapley *interaction* index (a difference-of-differences) is too noisy
to separate from zero. PID attacks the same synergy/redundancy question more directly, on a **continuous**
target, by decomposing the information two sources carry about the target into four non-negative parts:

    I(target ; X₁, X₂) = redundancy + unique₁ + unique₂ + synergy

For jointly-Gaussian variables with a univariate target, Barrett (Phys Rev E 2015) showed the sensible PID
measures collapse to the **minimum-MI (MMI)** redundancy ``R = min(I(T;X₁), I(T;X₂))`` — a closed form from
covariances, no binning (cf. the closed-form Gaussian estimators of Luo et al., arXiv:2605.09919). We use
the Gaussian MI ``I(T;X) = −½ ln(1 − R²)`` with the **adjusted** R² (a small-sample bias reduction, since raw
R² inflates MI at low N), a **permutation null** (shuffle the target → the chance level of each component,
which absorbs residual positive bias), and a **patient cluster-bootstrap** CI. As a cross-check we also report
the same-target R²-based pairwise interaction ``v(12) − v(1) − v(2)`` (>0 synergy / <0 redundancy) — PID and it
should agree in sign.

Caveats (state them): the Gaussian/linear assumption means only linear synergy/redundancy is captured; MI is
in nats; the estimates are pilot-grade and interpreted by sign + permutation significance, not magnitude alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold


def _oof_r2(y: np.ndarray, X: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    """Patient-grouped out-of-fold R² of OLS ``y ~ [1, X]`` (clipped to [0, 0.999)).

    Cross-fitted counterpart of :func:`_adj_r2` — the same discipline as the learning-curve atlas, so a
    PID built on it is comparable to the atlas OOF read (a tiny in-sample association that does not
    generalise yields OOF R²≈0 ⇒ MI≈0, rather than an inflated in-sample MI)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    n_splits = min(k, len(np.unique(groups)))
    if n_splits < 2 or n <= p + 2 or np.var(y) <= 0:
        return 0.0
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(len(te)), X[te]]) @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    r2 = 1.0 - float(np.nansum((y - pred) ** 2)) / ss_tot
    return float(np.clip(r2, 0.0, 0.999))


def _adj_r2(y: np.ndarray, X: np.ndarray) -> float:
    """Adjusted coefficient of determination of the OLS regression ``y ~ [1, X]`` (clipped to [0, 0.999))."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n = len(y)
    if n <= p + 1 or np.var(y) <= 0:
        return 0.0
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    denom = n - p - 1
    r2a = 1.0 - (1.0 - r2) * (n - 1) / denom if denom > 0 else r2
    return float(np.clip(r2a, 0.0, 0.999))


def gaussian_mi(y: np.ndarray, X: np.ndarray, groups: np.ndarray | None = None, k: int = 5) -> float:
    """Gaussian mutual information ``I(y;X) = −½ ln(1 − R²) ≥ 0`` (nats), via the multiple correlation.

    ``groups=None`` → in-sample adjusted-R² (fast, for tests); ``groups`` given → patient-grouped **OOF** R²
    (:func:`_oof_r2`), the defensible cross-fitted estimate consistent with the atlas."""
    r2 = _adj_r2(y, X) if groups is None else _oof_r2(y, X, groups, k)
    return max(0.0, -0.5 * float(np.log(1.0 - r2)))


@dataclass
class PIDResult:
    mi1: float
    mi2: float
    mi_joint: float
    redundancy: float
    unique1: float
    unique2: float
    synergy: float
    net_synergy: float          # synergy − redundancy (sign matches a Shapley interaction index)


def mmi_pid(y: np.ndarray, X1: np.ndarray, X2: np.ndarray,
            groups: np.ndarray | None = None, k: int = 5) -> PIDResult:
    """Gaussian minimum-MI PID of the information ``(X1, X2)`` carry about ``y`` (Barrett 2015).

    ``groups`` given → cross-fitted (OOF) MI; ``None`` → in-sample adjusted-R² MI."""
    X1 = np.asarray(X1, dtype=float); X2 = np.asarray(X2, dtype=float)
    if X1.ndim == 1:
        X1 = X1[:, None]
    if X2.ndim == 1:
        X2 = X2[:, None]
    i1 = gaussian_mi(y, X1, groups, k)
    i2 = gaussian_mi(y, X2, groups, k)
    ij = gaussian_mi(y, np.column_stack([X1, X2]), groups, k)
    red = min(i1, i2)
    u1, u2 = i1 - red, i2 - red
    syn = ij - i1 - i2 + red
    return PIDResult(i1, i2, ij, red, u1, u2, syn, syn - red)


def r2_interaction(y: np.ndarray, X1: np.ndarray, X2: np.ndarray,
                   groups: np.ndarray | None = None, k: int = 5) -> float:
    """Same-target cross-check: pairwise interaction ``v(12) − v(1) − v(2)`` with ``v`` = R² (OOF if
    ``groups`` given, else adjusted-R²). >0 synergy / <0 redundancy; should share the sign of
    :attr:`PIDResult.net_synergy`."""
    X1 = np.asarray(X1, dtype=float); X2 = np.asarray(X2, dtype=float)
    if X1.ndim == 1:
        X1 = X1[:, None]
    if X2.ndim == 1:
        X2 = X2[:, None]
    r2 = (lambda X: _adj_r2(y, X)) if groups is None else (lambda X: _oof_r2(y, X, groups, k))
    return float(r2(np.column_stack([X1, X2])) - r2(X1) - r2(X2))


def _cluster_resample(rng, groups: np.ndarray) -> np.ndarray:
    """Row indices from resampling whole patient groups with replacement (bilateral eyes stay together)."""
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([idx_by[uniq[i]] for i in pick])


@dataclass
class PairAnalysis:
    pid: PIDResult
    r2_interaction: float
    ci: dict                     # {component: (lo, hi)} patient cluster-bootstrap 95% CI
    p_perm: dict                 # {component: permutation-null p-value}
    n: int


def _pid_boot_draw(y, X1, X2, groups, cross_fit, k, draw_seed):
    """One cluster-bootstrap PID draw → (PID result, r2_interaction). Own RNG from ``draw_seed`` →
    reproducible under any ``n_jobs``."""
    rng = np.random.default_rng(draw_seed)
    idx = _cluster_resample(rng, groups)
    gb = groups[idx] if cross_fit else None
    return mmi_pid(y[idx], X1[idx], X2[idx], gb, k), r2_interaction(y[idx], X1[idx], X2[idx], gb, k)


def _pid_perm_draw(y, X1, X2, g_or_none, k, draw_seed):
    """One target-permutation null draw → (redundancy, synergy, net_synergy, r2_interaction)."""
    rng = np.random.default_rng(draw_seed)
    yp = rng.permutation(y)
    r = mmi_pid(yp, X1, X2, g_or_none, k)
    return r.redundancy, r.synergy, r.net_synergy, r2_interaction(yp, X1, X2, g_or_none, k)


def pair_analysis(y: np.ndarray, X1: np.ndarray, X2: np.ndarray, groups: np.ndarray, *,
                  n_boot: int = 500, n_perm: int = 500, seed: int = 0,
                  cross_fit: bool = True, k: int = 5, n_jobs: int = -1) -> PairAnalysis:
    """Full PID read for one modality pair: point estimate + cluster-bootstrap CI + permutation-null p-values.

    ``cross_fit=True`` (default) uses patient-grouped **OOF** MI (consistent with the atlas — a tiny in-sample
    association that does not generalise reads as ≈0); ``False`` uses in-sample adjusted-R² MI (faster).
    ``p_perm`` is the fraction of target-permuted draws whose component ≥ the observed (the chance floor;
    permutation preserves the estimator's bias, so it is a valid significance test for either estimator)."""
    y = np.asarray(y, dtype=float)
    X1 = np.asarray(X1, dtype=float); X2 = np.asarray(X2, dtype=float)
    if X1.ndim == 1:
        X1 = X1[:, None]
    if X2.ndim == 1:
        X2 = X2[:, None]
    groups = np.asarray(groups)
    ok = np.isfinite(y) & np.all(np.isfinite(X1), axis=1) & np.all(np.isfinite(X2), axis=1)
    y, X1, X2, groups = y[ok], X1[ok], X2[ok], groups[ok]
    g_or_none = groups if cross_fit else None

    obs = mmi_pid(y, X1, X2, g_or_none, k)
    obs_r2i = r2_interaction(y, X1, X2, g_or_none, k)

    comps = ["redundancy", "unique1", "unique2", "synergy", "net_synergy"]
    boot = {c: [] for c in comps}
    boot["r2_interaction"] = []
    # independent, reproducible per-draw seeds (distinct streams for bootstrap vs permutation)
    boot_seeds = np.random.SeedSequence(seed).spawn(n_boot)
    boot_res = Parallel(n_jobs=n_jobs)(
        delayed(_pid_boot_draw)(y, X1, X2, groups, cross_fit, k, s) for s in boot_seeds)
    for r, r2i in boot_res:
        for c in comps:
            boot[c].append(getattr(r, c))
        boot["r2_interaction"].append(r2i)
    ci = {c: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for c, v in boot.items()}

    perm = {c: 0 for c in ["redundancy", "synergy", "net_synergy"]}
    perm["r2_interaction"] = 0
    perm_seeds = np.random.SeedSequence(seed + 1).spawn(n_perm)
    perm_res = Parallel(n_jobs=n_jobs)(
        delayed(_pid_perm_draw)(y, X1, X2, g_or_none, k, s) for s in perm_seeds)
    for red, syn, nsyn, r2i in perm_res:
        perm["redundancy"] += red >= obs.redundancy
        perm["synergy"] += syn >= obs.synergy
        perm["net_synergy"] += nsyn >= obs.net_synergy
        perm["r2_interaction"] += r2i >= obs_r2i
    p_perm = {c: (v + 1) / (n_perm + 1) for c, v in perm.items()}
    return PairAnalysis(obs, obs_r2i, ci, p_perm, int(len(y)))
