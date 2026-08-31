"""Score a foundation model (RETFound) as an external contender through the SAME tolerance certificate.

A foundation model outputs an EMBEDDING, not a cpRNFL profile, so its clinically-meaningful "read" is the
**glaucoma probability from a FROZEN linear probe** on the frozen encoder — the *controlled adaptation
regime* the plan requires (no fine-tuning that could erase inherited robustness; ImageNet-RIB). Its
tolerance = does that probability / abnormal call survive a controlled B-scan perturbation
(`bscan_transforms`), scored on GLEAM Spectralis (real labels). This reuses the map-reader / Eyened
certificate machinery (`benchmark.capability_axes` + `tolerance_certificate`) on a SCALAR probability read
(so `recovery_focal = recovery_global`; there are no sectors) — the FM-vs-single-dataset comparison
realised as an axis/read-LABELLED leaderboard entry, not a false same-axis ranking.

Honest scope (docs §9): the FM read is a linear-probe glaucoma probability, distinct from cpRNFL; scored on
the identified B-scan axis (not the calibrated vendor offset). The reads (`data/derived/*_contender.npz`)
come from `scripts/run_retfound_contender.py` (torch); this scores them label-free-for-tolerance on the
main env.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import benchmark as B
from . import degradation as D
from . import pipeline as P


def _probe(n_pos: int | None = None, n_neg: int | None = None) -> Pipeline:
    """The COMMON frozen linear-probe head, applied uniformly to every FM's RECOMMENDED feature (extracted
    producer-side: RETFound GAP · DINOv2 [CLS]⊕avg-pooled patch tokens · MIRAGE pooled patch tokens) so the
    cross-model comparison varies the MODEL, not the probe head — the fair protocol (a per-paper tuning script
    would confound model-vs-protocol). The L2 reg strength is SELECTED on held-out folds over a small C grid
    (Oquab 2023's DINOv2 linear-eval likewise sweeps the probe LR; StandardScaler + L2-logreg is the standard
    linear probe); for the small-n GLEAM regime — a class too small to CV reliably — it falls back to a fixed
    strong-reg probe (the honest small-data guard; Kumar 2022: keep the probe simple, LP is the OOD-robust
    regime). ``n_pos``/``n_neg`` = the TRAIN-split class counts (pass them to enable the CV selection)."""
    if n_pos is not None and n_neg is not None and min(n_pos, n_neg) >= 9:
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegressionCV(Cs=[0.01, 0.05, 0.2, 1.0], cv=3, max_iter=5000,
                                                      scoring="roc_auc"))])
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.05, max_iter=2000))])


def score_fm_degradation(clean_emb: np.ndarray, perturbed_by_level: dict[float, np.ndarray],
                         labels: np.ndarray, rng: np.random.Generator, *, test_frac: float = 0.35,
                         n_boot: int = 1000, spec: float = 0.90, model: str = "retfound",
                         adaptation: str = "linear-probe", split: tuple | None = None) -> dict:
    """Bridge the FM's frozen linear-probe read onto the COMMON-GROUND paired-degradation harness (System B).

    The single biggest missing wire (plan §B1): ``score_fm`` emits the standalone System-A certificate, but the
    FM never entered the SAME paired-degradation scorer the map-readers use. Here the probe is fit on a TRAIN
    split (no leakage), then the held-out split's clean + per-level perturbed glaucoma probabilities are
    materialised as a ``pipeline.ProbaModel`` and scored by ``degradation.paired_degradation`` — referral-flip +
    ΔAUC + Δsens@spec + detection-retention with eye-level bootstrap CIs, IDENTICAL to the map-readers, only
    axis-LABELLED ``bscan`` (the appearance/quality-tolerance STRESS axis, NOT the calibrated device axis).

    ``clean_emb`` (N,D) FM embeddings of clean B-scans; ``perturbed_by_level`` {level: (N,D)}; ``labels`` (N,)
    0=healthy / >0=glaucoma (GLEAM). Per level a single ``ProbaModel`` holds ``[clean_prob, perturbed_prob]``
    and clean/shifted index arrays select the two halves, so ``paired_degradation`` reads the drift exactly as
    for a live model. Returns ``{model, axis, adaptation, per_level: {lv: degradation}}``.

    ``split`` (optional) = ``(train_idx, eval_idx)``: fit the probe on ``train_idx`` and evaluate on ``eval_idx``
    instead of an own random ``test_frac`` split. Pass the shared seed-0 ``(tr, te)`` so the frozen-LP rows train
    on the SAME eyes and are scored on the SAME held-out eyes as the fine-tune / supervised rows — a genuine
    head-to-head where ONLY the adaptation regime (frozen probe vs full fine-tune) differs, not the split.
    """
    labels = (np.asarray(labels) > 0).astype(int)
    n = len(clean_emb)
    if split is not None:
        # SHARED split: fit on the SAME train eyes + evaluate on the SAME held-out eyes as the fine-tune /
        # supervised arms (both carry the seed-0 tr/te) -> a genuine head-to-head where ONLY the adaptation
        # regime differs. GLEAM is one B-scan/eye, so this stays an eye-level split (no same-eye leakage).
        tr, te = np.asarray(split[0], dtype=int), np.asarray(split[1], dtype=int)
    else:
        # frozen-LP's own-protocol split (no shared tr/te available, e.g. the standalone producer)
        idx = rng.permutation(n)
        k = max(4, int((1.0 - test_frac) * n))
        tr, te = idx[:k], idx[k:]
    ytr = labels[tr]
    probe = _probe(int((ytr == 1).sum()), int((ytr == 0).sum())).fit(clean_emb[tr], ytr)
    y_te = labels[te]
    clean_prob = probe.predict_proba(clean_emb[te])[:, 1]
    n_te = len(te)
    clean_idx, shifted_idx = np.arange(n_te), np.arange(n_te, 2 * n_te)
    per_level = {}
    for lv, emb in sorted(perturbed_by_level.items()):
        pert_prob = probe.predict_proba(np.asarray(emb)[te])[:, 1]
        pm = P.ProbaModel(model, np.concatenate([clean_prob, pert_prob]),
                          representation="bscan", adaptation=adaptation)
        per_level[lv] = D.paired_degradation(pm, clean_idx, shifted_idx, y_te, spec=spec, n_boot=n_boot,
                                             rng=np.random.default_rng(int(rng.integers(2**31))))
    return {"model": model, "axis": "bscan_appearance_stress", "adaptation": adaptation, "n": int(n),
            "n_eval": int(n_te), "glaucoma_rate": float(labels.mean()),
            "levels": sorted(perturbed_by_level), "per_level": per_level}


def score_proba_degradation(clean_prob: np.ndarray, perturbed_by_level: dict[float, np.ndarray],
                            labels: np.ndarray, rng: np.random.Generator, *, n_boot: int = 1000,
                            spec: float = 0.90, model: str = "supervised", adaptation: str = "supervised",
                            representation: str = "bscan") -> dict:
    """Score a contender that outputs GLAUCOMA PROBABILITIES DIRECTLY (e.g. the Phase-B supervised B-scan CNN)
    on the common-ground paired-degradation harness — NO linear probe (the read IS the prob, already produced
    on the contender's own held-out eyes). ``clean_prob`` (N,) + ``perturbed_by_level`` {level: (N,)} → per
    level a ``pipeline.ProbaModel`` holds ``[clean_prob, perturbed_prob]`` and ``degradation.paired_degradation``
    reads the drift, IDENTICAL to the FM and map-reader panels (axis-LABELLED ``bscan``)."""
    labels = (np.asarray(labels) > 0).astype(int)
    clean_prob = np.asarray(clean_prob, dtype=float)
    n = len(clean_prob)
    clean_idx, shifted_idx = np.arange(n), np.arange(n, 2 * n)
    per_level = {}
    for lv, pert in sorted(perturbed_by_level.items()):
        pm = P.ProbaModel(model, np.concatenate([clean_prob, np.asarray(pert, dtype=float)]),
                          representation=representation, adaptation=adaptation)
        per_level[lv] = D.paired_degradation(pm, clean_idx, shifted_idx, labels, spec=spec, n_boot=n_boot,
                                             rng=np.random.default_rng(int(rng.integers(2**31))))
    return {"model": model, "axis": "bscan_appearance_stress", "adaptation": adaptation, "n": int(n),
            "n_eval": int(n), "glaucoma_rate": float(labels.mean()),
            "levels": sorted(perturbed_by_level), "per_level": per_level}


def score_fm(clean_emb: np.ndarray, perturbed_by_level: dict[float, np.ndarray], labels: np.ndarray,
             rng: np.random.Generator, *, headline_level: float | None = None, test_frac: float = 0.35,
             n_boot: int = 1000, model: str = "retfound") -> dict:
    """FM tolerance certificate on the B-scan perturbation axis via a frozen linear-probe glaucoma read.

    ``clean_emb`` (N, D) = FM embeddings of clean B-scans; ``perturbed_by_level`` = {level: (N, D)};
    ``labels`` (N,) 0=healthy / >0=glaucoma (GLEAM). The probe is fit on a TRAIN split; tolerance +
    sensitivity are measured on the held-out split (no probe-fit leakage). The gate is detection RETENTION
    (does the perturbation HIDE disease vs the clean read), bootstrapped → 'indeterminate' when the
    (underpowered GLEAM) gate CI straddles the floor — the SAME shared certificate as the map-readers.
    """
    labels = (np.asarray(labels) > 0).astype(int)
    n = len(clean_emb)
    idx = rng.permutation(n)
    k = max(4, int((1.0 - test_frac) * n))
    tr, te = idx[:k], idx[k:]
    ytr = labels[tr]
    probe = _probe(int((ytr == 1).sum()), int((ytr == 0).sum())).fit(clean_emb[tr], ytr)

    clean_prob = probe.predict_proba(clean_emb[te])[:, 1]
    clean_call = (clean_prob > 0.5).astype(int)
    const_mae = float(np.mean(np.abs(clean_prob.mean() - clean_prob)))   # a constant-probability predictor
    pos = labels[te] == 1
    sens_clean = float(np.mean(clean_call[pos] == 1)) if pos.any() else float("nan")
    clean_pos = np.asarray(clean_call[pos] == 1)

    rows = {}
    for lv, emb in sorted(perturbed_by_level.items()):
        prob = probe.predict_proba(emb[te])[:, 1]
        call = (prob > 0.5).astype(int)
        drift = float(np.mean(np.abs(prob - clean_prob)))               # the FM read is the glaucoma prob
        flip = float(np.mean(call != clean_call))
        sens_shift = float(np.mean(call[pos] == 1)) if pos.any() else float("nan")
        rci = B.retention_ci(clean_pos, np.asarray(call[pos] == 1), rng, n_boot=n_boot) if pos.any() else None
        earned = drift <= const_mae * (1.0 - B.EARNED_MARGIN)
        cert = B.certify(mae=drift, focal_mae=drift, const_mae=const_mae, const_focal=const_mae, flip=flip,
                         sens_clean=sens_clean, sens_shift=sens_shift, earned=earned, retention_ci=rci)
        rows[lv] = {"level": lv, "prob_drift": drift, "flip_rate": flip, **cert}

    lv = headline_level if headline_level is not None else max(rows)
    return {"model": model, "axis": "bscan_perturbation", "read": "glaucoma_prob",
            "n": int(n), "n_eval": int(len(te)), "glaucoma_rate": float((labels == 1).mean()),
            "per_level": rows, **rows[lv]}
