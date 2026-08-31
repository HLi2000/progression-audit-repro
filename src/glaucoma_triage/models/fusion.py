"""Modality-subset estimator + patient-grouped cross-validated value function (binary & multiclass).

The characteristic function ``v(S)`` for the coalition game is **SAGE-proper** (Covert et al.,
NeurIPS 2020, arXiv:2004.00668): the *loss reduction* of a calibrated model trained on the feature
columns of modality subset ``S`` relative to the no-information (class-prior) predictor:

    v(S) = L(∅) − L(S),   L = out-of-fold multinomial log-loss (or Brier)

so ``v(∅) = 0`` and Shapley values satisfy efficiency ``Σ_i φ_i = v(N) = L(∅) − L(N)``. A *loss*
objective (not AUC) is used deliberately — AUC is a rank statistic, not additively decomposable, so
Shapley efficiency would not hold. AUC is computed only for **descriptive** reporting
(:func:`subset_auc`; OVR-macro for >2 classes).

Splits are **patient-disjoint and class-stratified** (sklearn ``StratifiedGroupKFold``). The estimator
is a median-imputed, standardized logistic regression (NO ``class_weight``: the proper-scoring value
function rewards calibrated probabilities, which balanced re-weighting destroys). Coalition values are
averaged over several CV seeds (**repeated CV**). Both binary and multiclass ``y`` are supported
(``y`` is integer-coded; multiclass logistic is multinomial). CPU-only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..eda.fundus_label import roc_auc
from ..interaction.shapley import ValueTable, coalitions

# modality name -> the feature columns that modality contributes
ModalitySpec = dict[str, list[str]]

BASELINE_AUC: float = 0.5  # AUC of a no-information classifier — descriptive only
_EPS: float = 1e-6


def regularized_estimator(C: float = 0.2) -> Pipeline:
    """More strongly L2-regularized logistic (lower ``C`` = more shrinkage) — for low-event regimes
    (e.g. the ~47-event progression probe) where the default C=1.0 overfits."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=C)),
    ])


def _default_estimator() -> Pipeline:
    """Median-impute → standardize → L2 logistic regression. No class weighting (see module doc)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def _subset_columns(spec: ModalitySpec, subset: Iterable[str]) -> list[str]:
    cols: list[str] = []
    for m in subset:
        cols.extend(spec[m])
    return cols


def _class_prior(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Per-class frequency vector (the mean/prior predictor). Uniform if ``y`` empty."""
    if len(y) == 0:
        return np.full(len(classes), 1.0 / len(classes))
    return np.array([(y == c).mean() for c in classes], dtype=float)


def _align_proba(p: np.ndarray, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map a model's ``predict_proba`` columns onto the global ``classes`` order (0 for unseen)."""
    out = np.zeros((p.shape[0], len(classes)), dtype=float)
    idx = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        out[:, idx[c]] = p[:, j]
    return out


def _max_stratified_splits(y: np.ndarray, groups: np.ndarray) -> int:
    """Largest usable fold count: bounded by the rarest class's group count (need ≥1 per fold)."""
    if len(y) == 0:
        return 0
    counts = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    return max(0, min(counts)) if counts else 0


def _oof_proba(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, classes: np.ndarray,
    k: int, seed: int, factory: Callable[[], object], empty: bool,
) -> np.ndarray:
    """Out-of-fold class-probability matrix (n × |classes|) over patient-disjoint stratified folds.

    ``empty`` (or a single-class training fold) uses the train-fold class prior."""
    n, C = len(y), len(classes)
    proba = np.full((n, C), np.nan, dtype=float)
    n_splits = min(k, _max_stratified_splits(y, groups))
    if n_splits < 2:
        proba[:] = _class_prior(y, classes)
        return proba
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X, y, groups):
        y_tr = y[tr]
        if empty or len(np.unique(y_tr)) < 2:
            proba[te] = _class_prior(y_tr, classes)
            continue
        model = factory()
        model.fit(X.iloc[tr], y_tr)
        proba[te] = _align_proba(model.predict_proba(X.iloc[te]),
                                 np.asarray(model.classes_), classes)
    nan_rows = np.isnan(proba).any(axis=1)
    if nan_rows.any():
        proba[nan_rows] = _class_prior(y, classes)
    return proba


def _prepare(
    df: pd.DataFrame, spec: ModalitySpec, subset: list[str], y: pd.Series, patient_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, bool]:
    """``(X, y, groups, classes, empty)`` on the label-scorable rows for one coalition."""
    y_num = pd.to_numeric(y, errors="coerce")
    keep = y_num.notna().to_numpy()
    sub = df.loc[keep]
    cols = _subset_columns(spec, subset)
    X = (sub[cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
         if cols else pd.DataFrame(index=range(int(keep.sum()))))
    yv = y_num[keep].to_numpy(dtype=int)
    groups = pd.factorize(sub[patient_col])[0]
    classes = np.array(sorted(np.unique(yv)))
    return X, yv, groups, classes, (len(cols) == 0)


def _oof_proba_repeated(
    df: pd.DataFrame, spec: ModalitySpec, subset: list[str], y: pd.Series, *,
    patient_col: str, k: int, seed: int, n_repeats: int,
    estimator_factory: Callable[[], object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean OOF class-probabilities over ``n_repeats`` seeded stratified-group CV runs, labels, classes."""
    X, yv, groups, classes, empty = _prepare(df, spec, subset, y, patient_col)
    factory = estimator_factory or _default_estimator
    acc = np.zeros((len(yv), len(classes)), dtype=float)
    for r in range(max(1, n_repeats)):
        acc += _oof_proba(X, yv, groups, classes, k, seed + r, factory, empty)
    return acc / max(1, n_repeats), yv, classes


def _loss(y: np.ndarray, proba: np.ndarray, classes: np.ndarray, metric: str) -> float:
    """Multinomial log-loss (default) or multiclass Brier over the OOF probabilities."""
    p = np.clip(proba, _EPS, 1 - _EPS)
    p = p / p.sum(axis=1, keepdims=True)
    if metric == "brier":
        onehot = (y[:, None] == classes[None, :]).astype(float)
        return float(np.mean(np.sum((onehot - p) ** 2, axis=1)))
    return float(log_loss(y, p, labels=classes))


def subset_auc(
    df: pd.DataFrame, spec: ModalitySpec, subset: Iterable[str], y: pd.Series, *,
    patient_col: str = "patient_id", k: int = 5, seed: int = 0, n_repeats: int = 1,
    estimator_factory: Callable[[], object] | None = None,
) -> float:
    """Patient-grouped repeated out-of-fold **AUC** for ``subset`` — descriptive only.

    Binary: rank-identity AUC on the positive class. Multiclass: OVR-macro ``roc_auc_score``."""
    subset = list(subset)
    if not subset:
        return BASELINE_AUC
    proba, yv, classes = _oof_proba_repeated(
        df, spec, subset, y, patient_col=patient_col, k=k, seed=seed,
        n_repeats=n_repeats, estimator_factory=estimator_factory)
    if len(classes) < 2:
        return float("nan")
    if len(classes) == 2:
        return roc_auc(proba[:, 1], (yv == classes[1]).astype(float))
    try:
        return float(roc_auc_score(yv, proba, multi_class="ovr", average="macro", labels=classes))
    except ValueError:
        return float("nan")


def oof_proba(
    df: pd.DataFrame, spec: ModalitySpec, subset: Iterable[str], y: pd.Series, *,
    patient_col: str = "patient_id", id_col: str = "eye_id", k: int = 5, seed: int = 0,
    n_repeats: int = 3, estimator_factory: Callable[[], object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-eye out-of-fold **positive-class** probabilities for ``subset``, with aligned labels + ids.

    Public wrapper over :func:`_oof_proba_repeated` for the decision / acquisition layer (net benefit,
    order-or-skip policy), which needs per-sample calibrated probabilities rather than a pooled scalar.
    Returns ``(p_pos, y01, ids)`` over the label-scorable rows (``y`` non-NaN), in ``df`` order:
    ``p_pos`` = mean OOF probability of the positive class, ``y01`` = 0/1 label, ``ids`` = ``id_col``
    values (row index if ``id_col`` absent). **Binary labels only** (raises otherwise) — the same
    patient-disjoint stratified CV + repeated-seed discipline as :func:`subset_auc`/:func:`coalition_values`.
    """
    subset = list(subset)
    proba, yv, classes = _oof_proba_repeated(
        df, spec, subset, y, patient_col=patient_col, k=k, seed=seed,
        n_repeats=n_repeats, estimator_factory=estimator_factory)
    if len(classes) != 2:
        raise ValueError(f"oof_proba supports binary labels only; got classes={classes.tolist()}")
    keep = pd.to_numeric(y, errors="coerce").notna().to_numpy()
    ids = (df.loc[keep, id_col].to_numpy() if id_col in df.columns
           else np.arange(int(keep.sum())))
    return proba[:, 1], (yv == classes[1]).astype(int), ids


def per_class_auc(
    df: pd.DataFrame, spec: ModalitySpec, subset: Iterable[str], y: pd.Series, *,
    patient_col: str = "patient_id", k: int = 5, seed: int = 0, n_repeats: int = 1,
    estimator_factory: Callable[[], object] | None = None,
) -> dict[int, float]:
    """Per-class one-vs-rest AUC for a multiclass ``subset`` (descriptive; ``{}`` if <3 classes).

    Complements the OVR-*macro* :func:`subset_auc` by breaking it out per class — e.g. for the 3-class
    healthy/OHT/glaucoma read, so a reader can see the (circular) OHT axis separately from the clean
    glaucoma axis. Same patient-grouped repeated-CV OOF probabilities as :func:`subset_auc`."""
    subset = list(subset)
    if not subset:
        return {}
    proba, yv, classes = _oof_proba_repeated(
        df, spec, subset, y, patient_col=patient_col, k=k, seed=seed,
        n_repeats=n_repeats, estimator_factory=estimator_factory)
    if len(classes) < 3:
        return {}
    try:
        per = roc_auc_score(yv, proba, multi_class="ovr", average=None, labels=classes)
    except ValueError:
        return {}
    return {int(c): float(a) for c, a in zip(classes, per)}


def coalition_values(
    df: pd.DataFrame, spec: ModalitySpec, y: pd.Series, *,
    patient_col: str = "patient_id", k: int = 5, seed: int = 0, n_repeats: int = 3,
    metric: str = "log_loss", estimator_factory: Callable[[], object] | None = None,
) -> ValueTable:
    """SAGE-proper value table ``{frozenset(S): L(∅) − L(S)}`` (binary or multiclass).

    ``v(∅) = 0``; higher ``v(S)`` = more loss explained. Feeds
    :func:`interaction.shapley.shapley_values` / :func:`interaction_matrix`."""
    players = list(spec)
    p0, y0, classes0 = _oof_proba_repeated(df, spec, [], y, patient_col=patient_col, k=k, seed=seed,
                                           n_repeats=n_repeats, estimator_factory=estimator_factory)
    L0 = _loss(y0, p0, classes0, metric)
    table: ValueTable = {}
    for s in coalitions(players):
        if not s:
            table[s] = 0.0
            continue
        p, yv, classes = _oof_proba_repeated(
            df, spec, list(s), y, patient_col=patient_col, k=k, seed=seed,
            n_repeats=n_repeats, estimator_factory=estimator_factory)
        table[s] = L0 - _loss(yv, p, classes, metric)
    return table
