"""Fit the interpretation layer and characterize its format-tolerance envelope.

The interpretation model is trained on canonical-grid fields to recover the VF-intrinsic targets
(MD regression + abnormal-field classification). Two arms:

  * **baseline** — trained on clean canonical fields only;
  * **domain-randomized (DR)** — trained on clean fields + on-the-fly format variants.

Reads: (1) the tolerance **envelope** — quality (MD MAE, abnormal AUC) and calibration (ECE) vs
measured ``format_distance``, with a breaking point; (2) same-signal **invariance** — agreement of
the interpretation across two re-encodings of the same field; (3) the real held-out **ladder**.

Reuses ``models.fusion.regularized_estimator`` (classifier), ``models.decision.{ece,brier}``, and
``eda.fundus_label.roc_auc``; the train-on-A / test-on-B domain-shift protocol is not expressible
through fusion's CV-internal API, so the thin fit/eval glue lives here.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..eda.fundus_label import roc_auc
from ..models.decision import brier, ece
from ..models.fusion import regularized_estimator
from . import canonical, transforms


class GeneralHeightImputer(BaseEstimator, TransformerMixin):
    """Per-field **general-height** fill (Phase 1u): impute each missing location with that field's own mean of
    present TD — the VF "general height" depth estimate (Heijl general-height / pattern-deviation framework) — rather
    than a per-feature global median. For a sparse field (e.g. the ~40%-coverage home tablet, or a 10-2) the general
    height reconstructs the unmeasured periphery far closer to the truth than a global median, so a linear reader
    recovers its MD/PSD from a sparse input almost as well as from a full one (verified: tablet MD-MAE 2.2→0.2 dB,
    10-2 8.3→1.5 dB). It also de-biases the ``device_shift`` augmentation rows, keeping the DR weight vector near
    clean-optimal. Applied uniformly to every arm (baseline + dr) → a front-end quality choice, not per-input
    routing. Falls back to the fitted per-feature mean for a wholly-empty field."""

    def fit(self, X, y=None):  # noqa: D102
        col = np.nanmean(np.asarray(X, dtype=float), axis=0)
        self.col_ = np.where(np.isfinite(col), col, 0.0)
        return self

    def transform(self, X):  # noqa: D102
        X = np.asarray(X, dtype=float).copy()
        for i in range(X.shape[0]):
            m = np.isnan(X[i])
            if not m.any():
                continue
            present = X[i, ~m]
            X[i, m] = float(present.mean()) if present.size else self.col_[m]   # field general height, else column
        return X


def _regressor() -> Pipeline:
    """General-height-impute -> standardize -> ridge (Phase 1u: GH fill replaces the median for sparse-field MD/PSD
    recovery). Mirrors the fusion classifier's front end otherwise."""
    return Pipeline([
        ("impute", GeneralHeightImputer()),
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def _nonlinear_regressor() -> Pipeline:
    """Median-impute -> histogram gradient-boosted trees: the non-linear PSD (focal-pattern) head. Tuned deeper +
    slower than the default (Phase 1z dr-tune sweep) — ``max_iter=400, learning_rate=0.05, max_leaf_nodes=63``
    lowers the held-out real-rung PSD-MAE (focal-recovery axis 0.68→0.70) with no cost to the linear MD/call heads,
    the switch, or clean precision (verified: still wins the PSD focal read on all 5 rungs)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg", HistGradientBoostingRegressor(random_state=0, max_iter=400,
                                              learning_rate=0.05, max_leaf_nodes=63)),
    ])


def _nonlinear_classifier() -> Pipeline:
    """Median-impute -> histogram gradient-boosted trees: a simple non-linear abnormal classifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(random_state=0)),
    ])


def _constant_regressor() -> Pipeline:
    """Mean-impute -> predict-the-mean: the degenerate scalar regressor (MD + aux heads for the constant model)."""
    return Pipeline([("impute", SimpleImputer(strategy="mean")), ("reg", DummyRegressor(strategy="mean"))])


def build_interp_model(kind: str = "linear") -> "InterpModel":
    """Factory for the benchmark's model kinds (all share the ``InterpModel`` interface). ``aux_factory`` sets
    the estimator used for the auxiliary scalar heads (PSD, hemifield means) to match the kind:

      * ``linear`` — ridge MD/aux + logistic abnormal (the reference baseline);
      * ``hybrid`` — "right tool per target" (Phase 1w): **linear** Ridge MD head + **linear** logistic call (both
        optimal for the *linear* MD functional / its threshold label; a tree MD head is 3–7× worse) + a **HistGBM
        PSD** aux head (PSD is *nonlinear* → HGB is 2–4× better and DR wins the focal read). Only the PSD head is
        non-linear, so MD / the abnormal call / R1 / sens+spec are identical to ``linear``; only PSD-MAE changes;
      * ``nonlinear`` — histogram gradient-boosted trees on every head (a simple non-linear contender);
      * ``constant`` — ``DummyRegressor``(mean) + ``DummyClassifier``(prior): the DEGENERATE, gameable
        model (constant MD, constant abnormal proba → never flips) that the tolerance certificate must reject.
    """
    if kind == "linear":
        return InterpModel()
    if kind == "hybrid":
        return InterpModel(aux_factory=_nonlinear_regressor)   # linear MD + linear call + HGB PSD (Phase 1w)
    if kind == "nonlinear":
        return InterpModel(reg=_nonlinear_regressor(), clf=_nonlinear_classifier(),
                           aux_factory=_nonlinear_regressor)
    if kind == "constant":
        return InterpModel(reg=_constant_regressor(), clf=DummyClassifier(strategy="prior"),
                           aux_factory=_constant_regressor)
    raise ValueError(f"unknown model kind: {kind!r}")


def _label(field: dict) -> int:
    """Classification target for a field: the supplied clinical ``label`` (e.g. glaucoma 0/1) if present,
    else the VF-intrinsic abnormal-field flag derived from MD (screening endpoint)."""
    lab = field.get("label")
    if lab is not None:
        return int(lab)
    return canonical.abnormal_binary(field["md"])


def _md_gt(field: dict, x: np.ndarray) -> float:
    """Ground-truth / target MD for a field: the shipped **native** MD if present (the real weighted STATPAC MD
    — Harvard/Rotterdam/10-2), else the **HLO inverse-variance-weighted** MD on the canonical row for a
    coordinate-aligned source (:func:`canonical.md_hlo`; UWHVF/SUNY-IU/Iowa/Eyecatcher/vfpwg), else the unweighted
    mean-TD (by-order Harvard already covered by native; synth-coord GRAPE stays unweighted — permutation-safe)."""
    nm = field.get("native_md")
    if nm is not None and np.isfinite(nm):
        return float(nm)
    return canonical.md_hlo(x) if field.get("aligned") else float(field["md"])


def _psd_gt(field: dict, x: np.ndarray) -> float:
    """Target PSD for a field: the **HLO variance-weighted** PSD (the classical STATPAC PSD) on the canonical row
    for a coordinate-aligned source, else the unweighted SD-of-TD proxy (by-order/synth-coord rungs)."""
    return canonical.psd_hlo(x) if field.get("aligned") else canonical.psd(x)


def to_xy(fields: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, md_target, abn_target)`` from clean fields (X resampled to the canonical grid). ``md_target`` is the
    HLO-weighted / native MD (:func:`_md_gt`); ``abn_target`` is the clinical ``label`` where present, else the
    abnormal flag from the field's UNWEIGHTED mean-MD (the plain clinical call — decoupled from the weighted read)."""
    X = canonical.field_matrix(fields)
    md = np.array([_md_gt(f, x) for f, x in zip(fields, X)], dtype=float)
    abn = np.array([_label(f) for f in fields], dtype=int)
    return X, md, abn


def _encode_one(f: dict, level: float, rng: np.random.Generator, *,
                exclude: set[str] | None = None, apply_device: bool = True) -> tuple[np.ndarray, float, int, float]:
    """Re-encode one field at ``level`` and return ``(input_row, md_target, abn_target, distance)``.

    The MD target is the field's **own offset-adjusted level** (FGGDL "universal grader"), computed the well-posed
    way (Phase-1s soundness fix #2 / #5): the *level-changing* transforms (``normative_shift``/``algorithm_shift`` —
    a genuine global dB shift) are applied FIRST on the full grid → their mean IS the level the reader should
    report (so a calibration offset is not penalised as error); the *non-level-changing* transforms (``add_noise``
    zero-mean, ``device_shift`` coverage, ``quantize``) are then applied ON TOP to build the degraded input only —
    their target stays the offset-adjusted full-grid level (recovering the true level under coverage/noise IS the
    task). The abnormal-call target is the field's CLEAN label (format-invariance = the decision-stability goal).

    ``apply_device`` (default True) gates ``device_shift`` — training thins it to a Bernoulli in :func:`_augment`;
    the R3 test / simulator always apply it (deterministic full compound)."""
    ex = set(exclude or ())
    nonlvl = set(transforms.TRANSFORM_NAMES) - set(transforms.LEVEL_CHANGING)
    # 1) level-changing offsets on the FULL grid → the offset-adjusted level the reader should report (HLO-weighted
    #    for coordinate-aligned sources — the augmentation set is UWHVF/SUNY-IU/Iowa, all aligned)
    v_lvl, c_lvl = transforms.apply_format(f["values"], f["coords"], rng, level, exclude=nonlvl | ex)
    lvl_field = canonical.resample(v_lvl, c_lvl)
    md_target = canonical.md_hlo(lvl_field) if f.get("aligned") else canonical.md(lvl_field)
    if not np.isfinite(md_target):
        md_target = float(f["md"])
    # 2) coverage / noise / quantize ON TOP → the degraded model input (device optionally thinned in training)
    ex2 = set(transforms.LEVEL_CHANGING) | ex
    if not apply_device:
        ex2.add("device_shift")
    v_in, c_in = transforms.apply_format(v_lvl, c_lvl, rng, level, exclude=ex2)
    dist = transforms.format_distance(f["values"], f["coords"], v_in, c_in)
    return canonical.resample(v_in, c_in), md_target, _label(f), dist


def _augment(fields: list[dict], rng: np.random.Generator, n_aug: int,
             level_range: tuple[float, float], *, device_p: float = 0.5,
             exclude: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n_aug`` format-variant rows per field. MD label = the row's **own offset-adjusted level**
    (:func:`_encode_one`); abnormal label = the field's CLEAN flag. ``device_shift`` is applied per row with
    probability ``device_p`` (Phase-1s stochastic per-op augmentation — the only transform thinned; the others
    always fire so ``normative_shift`` stays present to hold the switch). ``exclude`` drops named transforms
    (leave-one-out ablation)."""
    rows, md, abn = [], [], []
    lo, hi = level_range
    for f in fields:
        for _ in range(n_aug):
            lvl = float(rng.uniform(lo, hi))
            apply_device = bool(rng.random() < device_p)
            row, m, a, _ = _encode_one(f, lvl, rng, exclude=exclude, apply_device=apply_device)
            rows.append(row)
            md.append(m)
            abn.append(a)
    if not rows:
        return np.empty((0, canonical.N_CANON)), np.empty(0), np.empty(0, dtype=int)
    return np.vstack(rows), np.array(md, dtype=float), np.array(abn, dtype=int)


class InterpModel:
    """The interpretation layer: an MD regressor + an abnormal-field classifier.

    Defaults to a ridge MD regressor + a logistic classifier; ``reg``/``clf`` may be injected to swap in
    other estimators (see :func:`build_interp_model` for the benchmark's linear/nonlinear/constant kinds).
    Any injected pair only has to implement sklearn's ``fit`` / ``predict`` (regressor) and
    ``fit`` / ``predict_proba`` (classifier)."""

    def __init__(self, reg=None, clf=None, aux_factory=None) -> None:
        self.reg = reg if reg is not None else _regressor()
        self.clf = clf if clf is not None else regularized_estimator()
        self.aux_factory = aux_factory if aux_factory is not None else _regressor
        self.aux_regs: dict = {}          # name -> fitted scalar regressor (PSD, hemifield means)
        self.single_class: int | None = None

    def fit(self, X: np.ndarray, md: np.ndarray, abn: np.ndarray, aux: dict | None = None) -> "InterpModel":
        self.reg.fit(X, md)
        if len(np.unique(abn)) < 2:
            self.single_class = int(abn[0]) if len(abn) else 0
        else:
            self.clf.fit(X, abn)
        # auxiliary scalar head(s) (the PSD proxy) — fit each on its finite-target rows
        for name, vals in (aux or {}).items():
            vals = np.asarray(vals, dtype=float)
            ok = np.isfinite(vals)
            if ok.sum() >= 2:
                self.aux_regs[name] = self.aux_factory().fit(X[ok], vals[ok])
        return self

    def predict_md(self, X: np.ndarray) -> np.ndarray:
        return self.reg.predict(X)

    def predict_aux(self, name: str, X: np.ndarray) -> np.ndarray:
        """Predicted auxiliary scalar (the ``psd`` proxy); NaNs if that head wasn't fit."""
        r = self.aux_regs.get(name)
        return r.predict(X) if r is not None else np.full(len(X), np.nan)

    def predict_psd(self, X: np.ndarray) -> np.ndarray:
        return self.predict_aux("psd", X)

    def predict_abn(self, X: np.ndarray) -> np.ndarray:
        if self.single_class is not None:
            return np.full(len(X), float(self.single_class))
        return self.clf.predict_proba(X)[:, 1]


def _aux_targets(X: np.ndarray, fields: list[dict] | None = None) -> dict:
    """Clean auxiliary scalar target(s) from canonical rows: the pattern-spread (PSD) target. PSD stays the CLEAN
    target (recover the true focal spread from a degraded input). It is the **HLO variance-weighted PSD** (the
    classical STATPAC PSD, :func:`canonical.psd_hlo`) for coordinate-aligned sources and the unweighted SD-of-TD
    proxy otherwise — via :func:`_psd_gt` when ``fields`` are supplied (else HLO-weighted, the aligned default).
    NaN-guarded, so sparse rows yield NaN (skipped at fit). Hemifield heads removed with R4."""
    if fields is None:
        return {"psd": np.array([canonical.psd_hlo(x) for x in X], dtype=float)}
    return {"psd": np.array([_psd_gt(f, x) for f, x in zip(fields, X)], dtype=float)}


def train_models(train_fields: list[dict], rng: np.random.Generator, *,
                 n_aug: int = 4, level_range: tuple[float, float] = (0.1, 0.8),
                 device_p: float = 0.25, clean_oversample: int = 3,
                 pooled_fields: list[dict] | None = None,
                 factory=InterpModel, exclude: set[str] | None = None) -> dict:
    """Fit the interpretation-model arms. Always: ``baseline`` (clean-only) and ``dr`` (clean + synthetic
    format variants). If ``pooled_fields`` is given: a third ``pooled`` arm (clean + **real** multi-device
    fields, no synthetic) — the FGGDL-style comparator isolating real vs synthetic diversity.

    DR augmentation (Phase 1s): MD label = the row's **own offset-adjusted level** (:func:`_augment`); abnormal
    label = the field's CLEAN flag (format-invariance); PSD head keeps the **clean** HLO-weighted target (recover the
    true spread). ``device_shift`` is thinned to a Bernoulli ``device_p`` per row (the switch-holding
    ``normative_shift`` stays always-on) and the clean rows are over-represented ×``clean_oversample`` in the DR
    mixture (Phase 1u **clean-anchoring** — keeps the single linear weight vector near clean-optimal so DR ties the
    clean/full-grid rungs instead of losing them, while still winning the shifted ones). ``level_range``,
    ``device_p`` and ``clean_oversample`` are the frozen-on-synthetic-envelope hyperparameters (NOT tuned on the real
    rungs — Phase-1s soundness fix #5).

    ``factory`` is a zero-arg callable returning a fresh :class:`InterpModel` (default = the linear reference);
    the benchmark passes ``lambda: build_interp_model(kind)`` to swap the estimator while keeping the data
    recipe (arm) fixed — so *model kind* (estimator) and *arm* (training data) stay orthogonal."""
    X, md, abn = to_xy(train_fields)
    aux = _aux_targets(X, train_fields)
    baseline = factory().fit(X, md, abn, aux)
    Xa, mda, abna = _augment(train_fields, rng, n_aug, level_range, device_p=device_p, exclude=exclude)
    if len(Xa):
        # clean-anchor: over-represent the clean rows ×clean_oversample in the DR mixture so the single (linear)
        # weight vector stays near clean-optimal while still seeing the augmented diversity (Phase 1u).
        os = max(1, int(clean_oversample))
        Xd = np.vstack([np.repeat(X, os, axis=0), Xa])
        mdd = np.concatenate([np.repeat(md, os), mda])
        abnd = np.concatenate([np.repeat(abn, os), abna])
        aux_d = {k: np.concatenate([np.repeat(v, os), np.repeat(v, n_aug)]) for k, v in aux.items()}
    else:
        Xd, mdd, abnd, aux_d = X, md, abn, aux
    models = {"baseline": baseline, "dr": factory().fit(Xd, mdd, abnd, aux_d)}
    if pooled_fields:
        Xp, mdp, abnp = to_xy(pooled_fields)
        aux_p = {k: np.concatenate([aux[k], _aux_targets(Xp, pooled_fields)[k]]) for k in aux}
        models["pooled"] = factory().fit(
            np.vstack([X, Xp]), np.concatenate([md, mdp]), np.concatenate([abn, abnp]), aux_p)
    return models


def _eval(model: InterpModel, X: np.ndarray, md_true: np.ndarray, abn_true: np.ndarray,
          aux_true: dict | None = None) -> dict:
    md_pred = model.predict_md(X)
    p_abn = model.predict_abn(X)
    out = {"md_mae": float(np.nanmean(np.abs(md_pred - md_true)))}
    if aux_true is not None and "psd" in aux_true and "psd" in model.aux_regs:
        out["psd_mae"] = float(np.nanmean(np.abs(model.predict_psd(X) - np.asarray(aux_true["psd"], float))))
    if len(np.unique(abn_true)) == 2:
        out["auc"] = float(roc_auc(p_abn, abn_true.astype(float)))
        out["ece"] = float(ece(p_abn, abn_true.astype(float)))
        out["brier"] = float(brier(p_abn, abn_true.astype(float)))
    else:
        out["auc"] = out["ece"] = out["brier"] = float("nan")
    return out


def _transform_test(test_fields: list[dict], level: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Re-encode every test field at ``level`` (full deterministic compound — device always applied); return
    (X, md_true, abn_true, mean_distance). ``md_true`` is the **own offset-adjusted level** (:func:`_encode_one`),
    so R3 eval agrees with the DR training target (a calibration offset is not scored as error; coverage/noise
    recovery is)."""
    rows, md, abn, dists = [], [], [], []
    for f in test_fields:
        row, m, a, d = _encode_one(f, level, rng, apply_device=True)
        rows.append(row)
        md.append(m)
        abn.append(a)
        dists.append(d)
    X = np.vstack(rows) if rows else np.empty((0, canonical.N_CANON))
    return X, np.array(md, dtype=float), np.array(abn, dtype=int), float(np.mean(dists)) if dists else 0.0


def envelope_curve(models: dict, test_fields: list[dict], levels: list[float],
                   rng: np.random.Generator) -> list[dict]:
    """One row per (arm, level): mean measured distance + quality (MD-MAE, PSD-MAE, AUC) + calibration + detection
    **sensitivity** on re-encoded tests. PSD-MAE recovers the CLEAN pattern-spread from the degraded input
    (focal-sensitive). ``sensitivity`` = recall on the truly-abnormal test fields (fraction still flagged
    `predict_abn ≥ 0.5` as format degrades) — the sensitivity-vs-distance read, whose healthy counterpart is the
    SUNY-IU false-positive curve."""
    # PSD target = the SAME read the head is trained on (:func:`_psd_gt` — HLO-weighted for aligned rows); using the
    # unweighted ``canonical.psd`` here would score the head against a different index (a ~1.4 dB systematic offset
    # that inflates + flattens the clean PSD-MAE). Matching them makes the envelope a real focal-recovery curve.
    Xt = canonical.field_matrix(test_fields)
    psd_true = np.array([_psd_gt(f, x) for f, x in zip(test_fields, Xt)], dtype=float)
    aux_true = {"psd": psd_true}
    out = []
    for level in levels:
        X, md_true, abn_true, dist = _transform_test(test_fields, level, rng)
        abn_mask = abn_true == 1
        for arm, model in models.items():
            m = _eval(model, X, md_true, abn_true, aux_true)
            m["sensitivity"] = (float(np.mean(model.predict_abn(X)[abn_mask] >= 0.5))
                                if abn_mask.any() else float("nan"))
            out.append({"arm": arm, "level": level, "distance": dist, **m})
    return out


def breaking_point(rows: list[dict], arm: str, *, metric: str = "auc",
                   tolerance: float = 0.75) -> float:
    """Smallest measured distance at which ``arm`` first drops below ``tolerance`` on ``metric``
    (nan if it never does). For ECE/MAE (lower=better) pass a metric handled by the driver instead."""
    pts = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["distance"])
    for r in pts:
        if not np.isnan(r[metric]) and r[metric] < tolerance:
            return float(r["distance"])
    return float("nan")


def invariance(model: InterpModel, test_fields: list[dict], level: float,
               rng: np.random.Generator) -> dict:
    """Same-signal invariance: agreement of the interpretation across two re-encodings at ``level``.

    Cohen's kappa on the abnormal-field flag + mean |Δ| of predicted MD between the two encodings."""
    def encode(seed_rng):
        rows = []
        for f in test_fields:
            v, c = transforms.apply_format(f["values"], f["coords"], seed_rng, level)
            rows.append(canonical.resample(v, c))
        return np.vstack(rows) if rows else np.empty((0, canonical.N_CANON))

    Xa, Xb = encode(rng), encode(rng)
    fa = (model.predict_abn(Xa) >= 0.5).astype(int)
    fb = (model.predict_abn(Xb) >= 0.5).astype(int)
    # κ is undefined for a constant/degenerate classifier — report nan, not a fabricated 1.0
    if model.single_class is not None or len(np.unique(np.concatenate([fa, fb]))) < 2:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(fa, fb))
    md_delta = float(np.mean(np.abs(model.predict_md(Xa) - model.predict_md(Xb))))
    return {"level": level, "kappa": kappa, "md_abs_delta": md_delta}


def pair_agreement(model: InterpModel, pairs: list[tuple]) -> dict:
    """Interpretation agreement across REAL paired encodings of the same eye (no synthetic transform).

    ``pairs`` = list of ``(vec_a, vec_b)`` or ``(vec_a, vec_b, patient_id)`` canonical-order TD vectors
    (two repeats of the same eye, or the same eye on two devices). Returns κ on the abnormal flag, mean
    |Δ| of predicted MD, raw data-level |Δ MD|, per-pair flips, and **patient ids** + **n_patients** so
    the caller can compute a patient-CLUSTERED CI (effective n = patients, not pairs)."""
    if not pairs:
        return {"n": 0, "n_patients": 0, "kappa": float("nan"), "md_abs_delta": float("nan"),
                "data_md_abs_delta": float("nan"), "flip_rate": float("nan"),
                "flips": np.empty(0), "pids": np.empty(0)}
    Xa = np.vstack([canonical.resample(p[0], canonical.CANON_COORDS) for p in pairs])
    Xb = np.vstack([canonical.resample(p[1], canonical.CANON_COORDS) for p in pairs])
    pids = np.array([p[2] if len(p) > 2 else i for i, p in enumerate(pairs)])
    fa = (model.predict_abn(Xa) >= 0.5).astype(int)
    fb = (model.predict_abn(Xb) >= 0.5).astype(int)
    if model.single_class is not None or len(np.unique(np.concatenate([fa, fb]))) < 2:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(fa, fb))
    md_delta = float(np.mean(np.abs(model.predict_md(Xa) - model.predict_md(Xb))))
    data_delta = float(np.nanmean([abs(canonical.md(p[0]) - canonical.md(p[1])) for p in pairs]))
    flips = (fa != fb).astype(int)
    return {"n": len(pairs), "n_patients": int(len(np.unique(pids))), "kappa": kappa,
            "md_abs_delta": md_delta, "data_md_abs_delta": data_delta,
            "flip_rate": float(np.mean(flips)), "flips": flips, "pids": pids}


def cluster_ci(values: np.ndarray, groups: np.ndarray, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile CI for the mean of per-item ``values``, via **patient-cluster** bootstrap (resample
    whole ``groups`` with replacement). Captures sampling uncertainty (NOT confounding)."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx[uniq[i]] for i in pick])
        boots.append(float(np.nanmean(values[rows])) if rows.size else float("nan"))
    est = float(np.nanmean(values)) if values.size else float("nan")
    lo, hi = (float(x) for x in np.nanpercentile(boots, [2.5, 97.5]))
    return {"est": est, "lo": lo, "hi": hi}


def wilson_ci(k: int, n: int, *, z: float = 1.96) -> dict:
    """Wilson score 95% CI for a binomial proportion ``k/n`` — the small-n / near-boundary appropriate interval
    (Brown, Cai & DasGupta 2001; a percentile bootstrap of a proportion under-covers at small n)."""
    k, n = int(k), int(n)
    if n <= 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": k, "n": n}
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"est": float(p), "lo": float(max(0.0, centre - half)),
            "hi": float(min(1.0, centre + half)), "k": k, "n": n}


def loa_tolerance_ci(diffs: np.ndarray, *, agree: float = 0.95, conf: float = 0.95) -> dict:
    """Bias + 95% limits-of-agreement with **exact** noncentral-t / tolerance-factor confidence intervals on each
    limit — the small-sample-correct method (Carkeet 2015; the Bland–Altman approximate CI is too permissive for
    n<40). Returns ``bias``, point limits ``loa_lo``/``loa_hi``, and each limit's CI (``lo_lo``/``lo_hi`` for the
    lower limit, ``hi_lo``/``hi_hi`` for the upper)."""
    from scipy.stats import nct, norm
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    n = len(d)
    nan = float("nan")
    if n < 2:
        return {"bias": nan, "loa_lo": nan, "loa_hi": nan, "lo_lo": nan, "lo_hi": nan,
                "hi_lo": nan, "hi_hi": nan, "n": int(n)}
    mean, s = float(d.mean()), float(d.std(ddof=1))
    zc = float(norm.ppf(1 - (1 - agree) / 2))          # 1.96 for 95% agreement
    a, df, ncp = 1 - conf, n - 1, np.sqrt(n) * float(norm.ppf(1 - (1 - agree) / 2))
    step = s / np.sqrt(n)
    hi_lo = mean + step * float(nct.ppf(a / 2, df, ncp))       # CI of the UPPER limit (97.5th pctile)
    hi_hi = mean + step * float(nct.ppf(1 - a / 2, df, ncp))
    lo_lo = mean - step * float(nct.ppf(1 - a / 2, df, ncp))   # CI of the LOWER limit (mirror)
    lo_hi = mean - step * float(nct.ppf(a / 2, df, ncp))
    return {"bias": mean, "loa_lo": mean - zc * s, "loa_hi": mean + zc * s,
            "lo_lo": lo_lo, "lo_hi": lo_hi, "hi_lo": hi_lo, "hi_hi": hi_hi, "n": int(n)}


def bland_altman_replicates(a_groups: list, b_reps: list) -> dict:
    """Variance-components single-measurement Bland–Altman for method comparison WITH replicates (Bland & Altman
    1999; Carstensen *MethComp*). Per eye ``i``: ``a_groups[i]`` = list of **same-session** replicate sub-lists
    for device A (e.g. HFA Baseline pair, Followup pair) so within-A variance is the pooled *within-session*
    variance (the clean repeatability, not interval-spanning); ``b_reps[i]`` = flat list of device-B replicates
    (e.g. tablet). Returns bias (B−A), within-device SDs ``sw_a``/``sw_b``, between-eye true-difference SD
    ``s_between``, the single-measurement LoA, and each device's repeatability coefficient ``rc_a``/``rc_b``."""
    biases, va, vb, na_reps, nb_reps = [], [], [], [], []
    for ag, br in zip(a_groups, b_reps):
        a_all = [x for sub in ag for x in sub if np.isfinite(x)]
        b = [x for x in br if np.isfinite(x)]
        if not a_all or not b:
            continue
        wsub = [np.var(np.asarray([x for x in sub if np.isfinite(x)], float), ddof=1)
                for sub in ag if len([x for x in sub if np.isfinite(x)]) >= 2]
        if wsub:
            va.append(float(np.mean(wsub)))
        if len(b) >= 2:
            vb.append(float(np.var(np.asarray(b, float), ddof=1)))
        biases.append(float(np.mean(b)) - float(np.mean(a_all)))
        na_reps.append(len(a_all))
        nb_reps.append(len(b))
    n = len(biases)
    nan = float("nan")
    if n < 2:
        return {"bias": nan, "sw_a": nan, "sw_b": nan, "s_between": nan, "loa_lo": nan,
                "loa_hi": nan, "rc_a": nan, "rc_b": nan, "n": int(n)}
    biases = np.asarray(biases, float)
    s2wa = float(np.mean(va)) if va else 0.0
    s2wb = float(np.mean(vb)) if vb else 0.0
    na = float(np.mean(na_reps)) or 1.0
    nb = float(np.mean(nb_reps)) or 1.0
    s2between = max(float(np.var(biases, ddof=1)) - s2wa / na - s2wb / nb, 0.0)
    sd_single = float(np.sqrt(s2between + s2wa + s2wb))
    bias = float(biases.mean())
    return {"bias": bias, "sw_a": float(np.sqrt(s2wa)), "sw_b": float(np.sqrt(s2wb)),
            "s_between": float(np.sqrt(s2between)), "loa_lo": bias - 1.96 * sd_single,
            "loa_hi": bias + 1.96 * sd_single, "rc_a": 1.96 * np.sqrt(2) * np.sqrt(s2wa),
            "rc_b": 1.96 * np.sqrt(2) * np.sqrt(s2wb), "n": int(n)}


def within_across_call_disagreement(a_call_groups: list, b_call_lists: list) -> dict:
    """Per-COMPARISON referral-call disagreement, three label-free ways, each as a ``k/n`` count (feed
    :func:`wilson_ci`) so the rates are comparable across devices with different #tests: A **same-session**
    repeat (consecutive pairs *within* each A session), B repeat (consecutive B pairs), and the **switch**
    (one per eye: A-majority vs B-majority). ``a_call_groups[i]`` = list of A session sub-lists of 0/1 calls;
    ``b_call_lists[i]`` = list of B 0/1 calls."""
    a_k = a_n = b_k = b_n = s_k = s_n = 0
    for ag, b in zip(a_call_groups, b_call_lists):
        for sub in ag:                                     # same-session consecutive A pairs
            for i in range(len(sub) - 1):
                a_n += 1
                a_k += int(sub[i] != sub[i + 1])
        for i in range(len(b) - 1):                        # consecutive B pairs
            b_n += 1
            b_k += int(b[i] != b[i + 1])
        a_all = [x for sub in ag for x in sub]
        if a_all and b:                                    # one switch comparison per eye (majority vs majority)
            s_n += 1
            s_k += int((np.mean(a_all) >= 0.5) != (np.mean(b) >= 0.5))
    def rate(k, n):
        return k / n if n else float("nan")
    return {"a_repeat": rate(a_k, a_n), "a_k": a_k, "a_n": a_n,
            "b_repeat": rate(b_k, b_n), "b_k": b_k, "b_n": b_n,
            "switch": rate(s_k, s_n), "switch_k": s_k, "switch_n": s_n}


def mcnemar_kappa(dec_a: np.ndarray, dec_b: np.ndarray) -> dict:
    """Paired binary-call agreement: full 2×2 (``a``=both-refer, ``b``=A-refer/B-not, ``c``=A-not/B-refer,
    ``d``=both-not), % agreement, Cohen's κ, and the **exact McNemar** test (binomtest on the discordant b/c;
    McNemar 1947). ``b`` vs ``c`` exposes directional disagreement (e.g. one device systematically under-calling)."""
    from scipy.stats import binomtest
    x, y = np.asarray(dec_a, int), np.asarray(dec_b, int)
    a = int(np.sum((x == 1) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    n = a + b + c + d
    p = float(binomtest(min(b, c), b + c).pvalue) if (b + c) > 0 else 1.0
    kappa = float(cohen_kappa_score(x, y)) if n and len(np.unique(np.concatenate([x, y]))) > 1 else float("nan")
    return {"a": a, "b": b, "c": c, "d": d, "n": int(n),
            "disagree": float((b + c) / n) if n else float("nan"),
            "pct_agree": float((a + d) / n) if n else float("nan"), "p_exact": p, "kappa": kappa}


def md_abs_errors(model: InterpModel, fields: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-eye |predicted − true| MD and patient groups — feeds :func:`cluster_ci` for ladder MD-MAE CIs.
    GT = :func:`_md_gt` (shipped native MD, else HLO-weighted for aligned rungs, else mean-TD)."""
    X = canonical.field_matrix(fields)
    md_true = np.array([_md_gt(f, x) for f, x in zip(fields, X)], dtype=float)
    errs = np.abs(model.predict_md(X) - md_true)
    groups = np.array([f.get("patient_id", i) for i, f in enumerate(fields)])
    return errs, groups


def psd_abs_errors(model: InterpModel, fields: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-eye |predicted − true| PSD and patient groups — the PSD sibling of :func:`md_abs_errors`, feeding
    :func:`cluster_ci` for a ladder PSD-MAE SE. GT = :func:`_psd_gt` (HLO variance-weighted PSD for aligned
    rungs, else the unweighted SD-of-TD proxy); rows where the PSD head wasn't fit carry NaN (nan-safe in CI)."""
    X = canonical.field_matrix(fields)
    psd_true = np.array([_psd_gt(f, x) for f, x in zip(fields, X)], dtype=float)
    errs = np.abs(model.predict_psd(X) - psd_true)
    groups = np.array([f.get("patient_id", i) for i, f in enumerate(fields)])
    return errs, groups


def decision_stability(model: InterpModel, fields: list[dict], levels: list[float],
                       rng: np.random.Generator) -> list[dict]:
    """Label-free "safe-to-switch": per format level, the **decision-flip rate** — the fraction of the
    same eyes whose abnormal/referral call (`predict_abn ≥ 0.5`) changes between the clean field and its
    re-encoding at that level. Measures whether the *decision moves*, not whether it is correct → no label."""
    Xc, _, _, _ = _transform_test(fields, 0.0, rng)  # level 0 = identity (clean)
    dc = (model.predict_abn(Xc) >= 0.5).astype(int)
    out = []
    for lvl in levels:
        Xs, _, _, dist = _transform_test(fields, lvl, rng)
        ds = (model.predict_abn(Xs) >= 0.5).astype(int)
        flips = (dc != ds).astype(int)
        out.append({"level": lvl, "distance": dist, "flip_rate": float(np.mean(flips)), "flips": flips})
    return out


def sensitivity(model: InterpModel, fields: list[dict]) -> dict:
    """Detection sensitivity (recall) of the referral call on a KNOWN-glaucoma cohort: fraction of eyes flagged
    abnormal (`predict_abn ≥ 0.5`) on the CLEAN canonical field. Non-circular (truth = clinical recruitment);
    the counterpart to specificity/over-referral. Returns ``{sens, k, n}``."""
    X = canonical.field_matrix(fields)
    if not len(X):
        return {"sens": float("nan"), "k": 0, "n": 0}
    ref = (model.predict_abn(X) >= 0.5).astype(int)
    return {"sens": float(np.mean(ref)), "k": int(ref.sum()), "n": int(len(ref))}


def over_referral_curve(model: InterpModel, fields: list[dict], levels: list[float],
                        rng: np.random.Generator) -> list[dict]:
    """Over-referral (false-positive) rate on a HEALTHY cohort vs format distance — the specificity counterpart to
    :func:`sensitivity`. Per level: fraction of healthy eyes wrongly flagged (`predict_abn ≥ 0.5`) after
    re-encoding. Distance-0 is ~trivial (SUNY-IU is the normative source, MD≈0); the shift-induced FP rise is the
    informative, non-circular signal."""
    out = []
    for lvl in levels:
        Xs, _, _, dist = _transform_test(fields, lvl, rng)
        fp = (model.predict_abn(Xs) >= 0.5).astype(float) if len(Xs) else np.array([])
        out.append({"level": lvl, "distance": dist,
                    "fp_rate": float(np.mean(fp)) if len(fp) else float("nan"), "fp": fp})
    return out


def ladder_eval(model: InterpModel, rungs: dict[str, list[dict]]) -> list[dict]:
    """Evaluate one fitted model on each real held-out rung (no synthetic transform). MD/PSD ground truth is the
    :func:`_md_gt`/:func:`_psd_gt` read: the **shipped native MD** where present (real device GT — Harvard-GDP/GF,
    Rotterdam, 10-2), else the **HLO inverse-variance-weighted** MD/PSD (the real STATPAC indices) for a
    coordinate-aligned rung, else the unweighted mean-TD / SD-of-TD (synth-coord GRAPE). Reported as raw MD-MAE
    (no "irreducible floor"; the arm contrast is the sound signal). ``sensitivity`` = fraction of the
    (all-glaucoma) rung referred (`predict_abn ≥ 0.5`)."""
    out = []
    for name, fields in rungs.items():
        if not fields:
            continue
        X, _, abn_true = to_xy(fields)
        has_native = any("native_md" in f for f in fields)
        md_true = np.array([_md_gt(f, x) for f, x in zip(fields, X)], dtype=float)   # native / HLO-weighted MD
        psd_true = np.array([_psd_gt(f, x) for f, x in zip(fields, X)], dtype=float)  # HLO-weighted PSD (aligned)
        row = {"rung": name, "n": len(fields), "md_mean": float(np.nanmean(md_true)),
               "abnormal_rate": float(np.mean(abn_true)), "native_gt": bool(has_native),
               "sensitivity": float(np.mean(model.predict_abn(X) >= 0.5))}
        row.update(_eval(model, X, md_true, abn_true, {"psd": psd_true}))
        out.append(row)
    return out
