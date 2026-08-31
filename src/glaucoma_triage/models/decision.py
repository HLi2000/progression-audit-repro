"""Decision-theoretic value-of-information layer (net benefit + EVPI) for the acquisition policy.

Rationale (Vickers, Van Calster, Wynants & Steyerberg, *Diagn Progn Res* 2023, PMC10243069): for a
clinical *decision* — here "order-or-skip the next test" — null-hypothesis testing and confidence
intervals on net benefit are of questionable value. One should choose the strategy with the highest
**expected net benefit now**, and express the remaining uncertainty as (a) the **probability each
strategy is the best** and (b) the **expected value of perfect information (EVPI)** — the expected loss
in net benefit from not knowing which strategy is truly best, i.e. the value of a larger study.

Net benefit (Vickers & Elkin 2006) at a decision threshold ``p_t``, flagging positive when the model's
probability ``p ≥ p_t``::

    NB = TP/n − (FP/n) · p_t/(1 − p_t)

with reference strategies **treat-all** (``NB = prev − (1−prev)·odds``) and **treat-none** (``NB = 0``).
EVPI is estimated by the **Bayesian-bootstrap Monte-Carlo** method (Sadatsafavi et al., *Med Decis
Making* 2023/2025; here the patient **cluster** bootstrap, matching the repo's CI discipline): treat the
bootstrap net-benefit draws as posterior draws, then per threshold

    EVPI = mean_b[ max_d NB_d^b ] − max_d[ mean_b NB_d^b ],   P(d best) = mean_b[ NB_d^b = max_d' NB ].

Pure numpy + scikit-learn isotonic. CPU-only. (Formal EVSI — the value of a *specific* future sample of
size m — uses the same machinery scaled to n+m; EVPI is its finite upper bound and is what we report for
this pilot.)
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

TREAT_ALL = "treat_all"
TREAT_NONE = "treat_none"


def net_benefit(p: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Vickers net benefit of a probability model at each ``p_t`` (flag positive when ``p ≥ p_t``)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty(len(thresholds), dtype=float)
    for j, pt in enumerate(thresholds):
        flag = (p >= pt).astype(float)
        tp = float(np.sum(flag * y))
        fp = float(np.sum(flag * (1.0 - y)))
        odds = pt / (1.0 - pt)
        out[j] = (tp - fp * odds) / n
    return out


def treat_all_net_benefit(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit of treating everyone as positive: ``prev − (1−prev)·p_t/(1−p_t)``."""
    y = np.asarray(y, dtype=float)
    prev = float(y.mean()) if len(y) else 0.0
    odds = np.asarray(thresholds, dtype=float) / (1.0 - np.asarray(thresholds, dtype=float))
    return prev - (1.0 - prev) * odds


def decision_curve(
    strategies: dict[str, np.ndarray], y: np.ndarray, thresholds: np.ndarray,
    *, include_all_none: bool = True,
) -> dict[str, np.ndarray]:
    """Net-benefit curve for each named probability strategy, plus treat-all / treat-none references."""
    curves = {name: net_benefit(p, y, thresholds) for name, p in strategies.items()}
    if include_all_none:
        curves[TREAT_ALL] = treat_all_net_benefit(y, thresholds)
        curves[TREAT_NONE] = np.zeros(len(thresholds), dtype=float)
    return curves


def net_benefit_draws(
    strategies: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, thresholds: np.ndarray,
    *, n_boot: int = 500, seed: int = 0, include_all_none: bool = True,
) -> dict[str, np.ndarray]:
    """Patient **cluster**-bootstrap net-benefit draws per strategy: ``{name: array(n_boot, n_thr)}``.

    Resamples whole patients (``groups``) with replacement — the same clustered discipline as
    ``interaction.report._cluster_bootstrap`` — so bilateral eyes stay together. Feeds :func:`evpi` /
    :func:`prob_best`."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    names = list(strategies)
    draws = {name: np.empty((n_boot, len(thresholds)), dtype=float) for name in names}
    if include_all_none:
        draws[TREAT_ALL] = np.empty((n_boot, len(thresholds)), dtype=float)
        draws[TREAT_NONE] = np.zeros((n_boot, len(thresholds)), dtype=float)
    probs = {name: np.asarray(p, dtype=float) for name, p in strategies.items()}
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_group[uniq[i]] for i in pick])
        yb = y[rows]
        for name in names:
            draws[name][b] = net_benefit(probs[name][rows], yb, thresholds)
        if include_all_none:
            draws[TREAT_ALL][b] = treat_all_net_benefit(yb, thresholds)
    return draws


def _stack(draws: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(draws)
    return names, np.stack([draws[n] for n in names], axis=0)  # (D, B, T)


def evpi(draws: dict[str, np.ndarray]) -> np.ndarray:
    """Per-threshold EVPI = ``mean_b max_d NB − max_d mean_b NB`` (≥ 0). Value of resolving uncertainty."""
    _, stack = _stack(draws)                       # (D, B, T)
    max_per_draw = stack.max(axis=0).mean(axis=0)  # E_b[max_d NB]  -> (T,)
    best_mean = stack.mean(axis=1).max(axis=0)     # max_d E_b[NB]  -> (T,)
    return np.maximum(max_per_draw - best_mean, 0.0)


def _contract(draws: dict[str, np.ndarray], r: float) -> dict[str, np.ndarray]:
    """Shrink each strategy's bootstrap draws toward its own mean by factor ``r`` (posterior contraction)."""
    out = {}
    for name, v in draws.items():
        mu = v.mean(axis=0, keepdims=True)
        out[name] = mu + r * (v - mu)
    return out


def evsi(draws: dict[str, np.ndarray], n_current: int, m: float) -> np.ndarray:
    """Expected value of sample information for ``m`` additional patients (per threshold), ``0 ≤ EVSI ≤ EVPI``.

    Normal-approximation projection: a study adding ``m`` patients contracts the posterior SD of each
    strategy's net benefit by ``r = sqrt(n/(n+m))`` (√n information scaling), so the *residual* decision
    uncertainty is ``EVPI`` on the contracted draws and ``EVSI(m) = EVPI(now) − EVPI(residual)``. Monotone in
    ``m``: →0 as ``m→0``, →EVPI as ``m→∞``. Labelled a projection — it assumes the current bootstrap posterior
    is representative and contracts smoothly; treat as an order-of-magnitude 'is more data worth it' read."""
    r = float(np.sqrt(n_current / (n_current + m)))
    return np.maximum(evpi(draws) - evpi(_contract(draws, r)), 0.0)


def prob_best(draws: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-threshold probability each strategy has the highest net benefit (ties split; sums to 1)."""
    names, stack = _stack(draws)                        # (D, B, T)
    is_max = stack == stack.max(axis=0, keepdims=True)
    frac = is_max / is_max.sum(axis=0, keepdims=True)   # split ties equally
    prob = frac.mean(axis=1)                            # (D, T)
    return {n: prob[i] for i, n in enumerate(names)}


def calibrate_isotonic(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Isotonic-recalibrated probabilities (monotone fit of ``y`` on ``p``, clipped to [0,1]).

    Note: fitting on the same out-of-fold probabilities it recalibrates is mildly optimistic — use for a
    calibration *sensitivity* read, not the headline decision curve (the logistic value function is
    already a proper-scoring, near-calibrated estimator)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return iso.fit_transform(p, y)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Brier score (mean squared error of the probability) — lower is better-calibrated + sharper."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over ``n_bins`` equal-width probability bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    n = len(p)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(err)


def brier_draws(p: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Patient **cluster**-bootstrap Brier draws on the FIXED out-of-fold probabilities → array(n_boot).

    Resamples whole patients with replacement (same discipline as ``net_benefit_draws``); recomputes the
    Brier score on the resampled rows without refitting (a cached-prediction bootstrap → captures
    evaluation-set variance, appropriate for a calibration-metric CI). Feed to ``np.percentile`` for a CI."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_group[uniq[i]] for i in pick])
        out[b] = brier(p[rows], y[rows])
    return out
