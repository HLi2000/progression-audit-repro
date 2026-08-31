"""Fundus-label experiment: clinician-diagnosis label vs the CDR/rim/ISNT-derived label.

For the PAPILA fundus arm we have two candidate structural labels — the clinician diagnosis
(``features.fundus_structural_label``) and an imaging-morphology rule
(``features.fundus_structural_label_derived``, built purely from disc geometry with cited
clinical thresholds). This module quantifies how far disc morphology alone recovers the clinical
label and *where* the two disagree (expected: the suspect class), with pure numpy/pandas
statistics (no scikit-learn): 2×2 agreement + Cohen's κ, per-metric ROC-AUC (Mann-Whitney rank
identity), Youden-optimal thresholds (reported only as an explicitly optimistic/circular
sensitivity check), the derived×clinician-3-class breakdown, and an inter-expert reliability κ.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets import features

# Continuous morphometry metrics scored against the clinician label. ``higher_is_disease``
# orients each so a larger oriented score means more glaucomatous (for ROC-AUC / Youden).
METRICS: dict[str, bool] = {
    "vcdr": True,
    "hcdr": True,
    "area_cdr": True,
    "rim_disc_area_ratio": False,  # larger rim fraction is healthier
}


# --------------------------------------------------------------------- paired arrays helper
def _aligned(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    a = pd.to_numeric(y_true, errors="coerce")
    b = pd.to_numeric(y_pred, errors="coerce")
    ok = a.notna() & b.notna()
    return a[ok].to_numpy(float), b[ok].to_numpy(float)


# --------------------------------------------------------------------- agreement statistics
def confusion_2x2(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """``{tp, fp, fn, tn}`` treating ``y_true`` (clinician) as reference, ``y_pred`` derived."""
    a, b = _aligned(y_true, y_pred)
    return {
        "tp": int(np.sum((a == 1) & (b == 1))),
        "fp": int(np.sum((a == 0) & (b == 1))),
        "fn": int(np.sum((a == 1) & (b == 0))),
        "tn": int(np.sum((a == 0) & (b == 0))),
    }


def agreement(y_true: pd.Series, y_pred: pd.Series) -> float:
    a, b = _aligned(y_true, y_pred)
    return float(np.mean(a == b)) if a.size else float("nan")


def cohens_kappa(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Cohen's κ for two binary raters (chance-corrected agreement)."""
    a, b = _aligned(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Ranks (1-based) with ties assigned the average of the tied positions."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1)
    # average tied ranks
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(scores: pd.Series | np.ndarray, labels: pd.Series | np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank identity; higher ``score`` ⇒ predicted positive."""
    s = pd.to_numeric(pd.Series(np.asarray(scores, dtype=float)), errors="coerce")
    y = pd.to_numeric(pd.Series(np.asarray(labels, dtype=float)), errors="coerce")
    ok = s.notna() & y.notna()
    s, y = s[ok].to_numpy(float), y[ok].to_numpy(float)
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def youden_threshold(scores: pd.Series | np.ndarray, labels: pd.Series | np.ndarray) -> dict:
    """Threshold maximising Youden's J (sens+spec-1); predict positive when ``score >= thr``."""
    s = pd.to_numeric(pd.Series(np.asarray(scores, dtype=float)), errors="coerce")
    y = pd.to_numeric(pd.Series(np.asarray(labels, dtype=float)), errors="coerce")
    ok = s.notna() & y.notna()
    s, y = s[ok].to_numpy(float), y[ok].to_numpy(float)
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return {"thr": float("nan"), "sens": float("nan"), "spec": float("nan"), "J": float("nan")}
    best = {"thr": float("nan"), "sens": 0.0, "spec": 0.0, "J": -1.0}
    for thr in np.unique(s):
        pred = s >= thr
        sens = float(np.sum(pred & pos) / n_pos)
        spec = float(np.sum(~pred & neg) / n_neg)
        j = sens + spec - 1
        if j > best["J"]:
            best = {"thr": float(thr), "sens": sens, "spec": spec, "J": float(j)}
    return best


# --------------------------------------------------------------------- breakdowns
def derived_by_clinician_class(df: pd.DataFrame, combine: str = "mean") -> pd.DataFrame:
    """Cross-tab of the derived binary label (rows) against the clinician 3-class (columns)."""
    derived = (features.fundus_structural_label_derived(df, combine=combine)
               .astype("Int64").rename("derived"))
    order = [c for c in ("healthy", "suspect", "glaucoma") if c in set(df["diagnosis"])]
    tab = pd.crosstab(derived, df["diagnosis"])
    return tab.reindex(columns=order, fill_value=0)


def metric_distribution_by_class(df: pd.DataFrame, metrics=tuple(METRICS)) -> pd.DataFrame:
    """n / mean / median / IQR of each metric within each clinician class."""
    rows = []
    for m in metrics:
        vals = pd.to_numeric(df[m], errors="coerce")
        for cls, sub in vals.groupby(df["diagnosis"], observed=True):
            s = sub.dropna()
            rows.append({"metric": m, "class": cls, "n": int(s.size),
                         "mean": s.mean(), "median": s.median(),
                         "q1": s.quantile(0.25), "q3": s.quantile(0.75)})
    return pd.DataFrame(rows)


def metric_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric ROC-AUC and Youden operating point vs the clinician binary label."""
    y = features.fundus_structural_label(df)
    rows = []
    for m, higher_is_disease in METRICS.items():
        raw = pd.to_numeric(df[m], errors="coerce")
        oriented = raw if higher_is_disease else -raw
        yj = youden_threshold(oriented, y)
        # translate the oriented Youden threshold back to the metric's own scale
        thr = yj["thr"] if higher_is_disease else -yj["thr"]
        rows.append({"metric": m, "auc": roc_auc(oriented, y),
                     "youden_thr": thr, "sens": yj["sens"], "spec": yj["spec"]})
    # ISNT violation is binary; report it as agreement-style AUC too.
    isnt = pd.Series(df["isnt_violation"], index=df.index).astype("boolean").astype("Int64")
    rows.append({"metric": "isnt_violation", "auc": roc_auc(isnt, y),
                 "youden_thr": float("nan"), "sens": float("nan"), "spec": float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- variant comparison
# Reported derived-label variants: how the two expert segmentations are combined.
VARIANTS: list[tuple[str, str]] = [
    ("expert 1", "e1"),
    ("expert 2", "e2"),
    ("mean (consensus)", "mean"),
    ("OR (union)", "or"),
]


def routing_refer_label(df: pd.DataFrame, combine: str = "mean", profile: str = "gold",
                        use_field: bool = True,
                        field_md_threshold: float = features.FIELD_ABNORMAL_MD_DEFAULT) -> pd.Series:
    """Collapse the Stage-1 route to binary refer(1)/discharge(0); ``referrer`` → ``<NA>`` (abstain).
    ``referrer`` is *not* a discharge, so it is a safe (non-FN) outcome excluded from scoring."""
    route = features.stage1_route(df, combine=combine, profile=profile, use_field=use_field,
                                  field_md_threshold=field_md_threshold)
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y[route == "discharge"] = 0
    y[route.isin(["GECFS", "GRMS"])] = 1
    return y


def comparison_labels(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """The label variants scored against the clinician label: structural expert-combinations plus
    the sensitive structural label and the IOP / field / Stage-1-routing labels."""
    labels = [(name, features.fundus_structural_label_derived(df, combine=key))
              for name, key in VARIANTS]
    labels += [
        ("sensitive structural", features.fundus_structural_label_sensitive(df)),
        ("IOP ≥ 24", features.iop_raised_label(df)),
        ("abnormal field MD<-2", features.field_abnormal_label(df)),
        # Stage-1 routing (structure + IOP + field) collapsed to refer-vs-discharge. Gold routes on
        # the OR structural read (the best gold variant); field is always paired with IOP.
        ("routing — gold (OR)", routing_refer_label(df, combine="or")),
        ("routing — sensitive", routing_refer_label(df, profile="sensitive")),
    ]
    return labels


# --------------------------------------------------------------------- false-negative-rate analysis
# The critical triage error is a discharged glaucoma/suspect eye (a false negative). These helpers
# quantify how far the sensitive label + routing lower it, and the specificity cost.
# Gold structural = the OR (union-of-experts) read — the best-agreeing gold variant. Routing always
# uses structure + IOP + field together (field is never used without IOP).
FNR_DECISIONS: list[tuple[str, str]] = [
    ("gold structural (OR)", "label_or"),
    ("sensitive structural", "label_sensitive"),
    ("routing — gold (OR)", "route_gold"),
    ("routing — sensitive", "route_sensitive"),
]


def _decision_refer(df: pd.DataFrame, key: str) -> pd.Series:
    """Binary refer/flag(1) vs clear(0) for a decision key (structural label or routing)."""
    if key == "label_or":
        return features.fundus_structural_label_derived(df, combine="or")
    if key == "label_sensitive":
        return features.fundus_structural_label_sensitive(df)
    if key == "route_gold":
        return routing_refer_label(df, profile="gold", combine="or")
    if key == "route_sensitive":
        return routing_refer_label(df, profile="sensitive")
    raise ValueError(key)


def fnr_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class false-negative rate + healthy specificity for each decision. FNR = fraction of a
    clinician class NOT flagged (for routing, 'not flagged' = discharged; referrer/abstain counts
    as flagged/safe). This is the headline safety table."""
    dx = df["diagnosis"]
    G, S, H = dx == "glaucoma", dx == "suspect", dx == "healthy"
    n_g = int(G.sum())
    rows = []
    for name, key in FNR_DECISIONS:
        flag = _decision_refer(df, key).fillna(1).astype(int).astype(bool)  # <NA>(abstain)=flagged/safe
        g_fnr = round(1 - flag[G].mean(), 3)
        rows.append({
            "decision": name,
            "glaucoma_FNR": g_fnr,
            "glaucoma_FN_n": int(round(g_fnr * n_g)),   # missed-glaucoma count (of n_g)
            "suspect_FNR": round(1 - flag[S].mean(), 3),
            "pos_FNR": round(1 - flag[S | G].mean(), 3),
            "healthy_spec": round(1 - flag[H].mean(), 3),
        })
    return pd.DataFrame(rows)


def operating_point_table(df: pd.DataFrame, score: str = "vcdr") -> pd.DataFrame:
    """Threshold on ``score`` needed for target glaucoma sensitivity, and the healthy specificity
    there — makes the sensitivity/specificity ceiling explicit (100% sens ⇒ ~0 specificity)."""
    dx = df["diagnosis"]
    s = pd.to_numeric(df[score], errors="coerce")
    G, H = (dx == "glaucoma") & s.notna(), (dx == "healthy") & s.notna()
    pos = s[G].to_numpy()
    rows = []
    for target in (0.90, 0.95, 1.00):
        thr = float(np.quantile(pos, 1 - target, method="lower"))  # catch target-fraction of glaucoma
        flag_h = (s[H] >= thr).mean()
        rows.append({"target_glaucoma_sens": target, f"{score}_threshold": round(thr, 3),
                     "achieved_sens": round(float((s[G] >= thr).mean()), 3),
                     "healthy_spec": round(float(1 - flag_h), 3)})
    return pd.DataFrame(rows)


def in_sample_contribution(df: pd.DataFrame) -> dict[str, object]:
    """What the in-sample normal-limit clauses (``features.IN_SAMPLE_SENSITIVE_CLAUSES``) buy the
    sensitive profile: the full label vs an **external-cuts-only** label (those clauses dropped), each
    as ``n_pos`` + per-class FNR + healthy specificity, plus the disease uniquely recovered by the
    in-sample clauses and the extra healthy FP they cost. NOTE: ``IN_SAMPLE_SENSITIVE_CLAUSES`` is now
    empty (the disc-size cut moved to an external Chákṣu normative), so ``full == external`` and the
    unique counts are 0 — the adopted read is fully external. Retained for provenance / regression."""
    dx = df["diagnosis"]
    vcdr = pd.to_numeric(df.get("vcdr"), errors="coerce")
    scor = vcdr.notna()
    G, S, H = (dx == "glaucoma") & scor, (dx == "suspect") & scor, (dx == "healthy") & scor
    clauses = _sensitive_clause_masks(df)   # per-expert union clause masks — matches the adopted label
    ext_keys = [k for k in clauses if k not in features.IN_SAMPLE_SENSITIVE_CLAUSES]

    def _union(keys: list[str]) -> pd.Series:
        u = pd.Series(False, index=df.index)
        for k in keys:
            u = u | clauses[k]
        return u & scor

    def _row(name: str, pos: pd.Series) -> dict[str, object]:
        return {"profile": name, "n_pos": int(pos.sum()),
                "glaucoma_FNR": round(1 - pos[G].mean(), 3),
                "suspect_FNR": round(1 - pos[S].mean(), 3),
                "healthy_spec": round(1 - pos[H].mean(), 3)}

    only_ins = _union(list(features.IN_SAMPLE_SENSITIVE_CLAUSES)) & ~_union(ext_keys)
    return {
        "rows": [_row("sensitive (full)", _union(list(clauses))),
                 _row("sensitive (external cuts only)", _union(ext_keys))],
        "unique_glaucoma": int((only_ins & G).sum()),
        "unique_suspect": int((only_ins & S).sum()),
        "unique_fp_healthy": int((only_ins & H).sum()),
    }


def _sensitive_clause_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-clause boolean masks for the adopted sensitive label under the **per-expert union** — a clause
    fires if EITHER expert's contours trip it (matches ``features.fundus_structural_label_sensitive`` with
    ``combine="or"``, the adopted read)."""
    e1 = features.sensitive_clauses(df, read="e1")
    e2 = features.sensitive_clauses(df, read="e2")
    return {k: (e1[k].fillna(False) | e2[k].fillna(False)) for k in e1}


def gold_vs_sensitive_diff(df: pd.DataFrame) -> dict[str, object]:
    """What the adopted ``sensitive`` label (== ``fundus_suspicious_disc``) adds over the strict ``gold``
    baseline, eye for eye — both on the per-expert **union** (``or``) read: ``gold`` = the ISGEO cut-offs
    (``fundus_structural_label_derived(combine="or")``); ``sensitive`` = {disc-size-adjusted cup, IST rim
    rule}. Returns the set sizes (gold, sensitive, intersection) and, for each direction of the difference,
    the count by clinician class + a per-clause attribution (both sides via the per-expert union masks)."""
    dx = df["diagnosis"]
    gold = (features.fundus_structural_label_derived(df, combine="or") == 1).fillna(False)
    sens = (features.fundus_structural_label_sensitive(df) == 1).fillna(False)
    sn = sens & ~gold   # sensitive-not-gold (what the sensitive label adds)
    gn = gold & ~sens   # gold-not-sensitive (what the sensitive label drops)

    def _by_class(mask: pd.Series) -> dict[str, int]:
        return {c: int((mask & (dx == c)).sum()) for c in ("glaucoma", "suspect", "healthy")}

    # per-clause attribution (overlapping), both sides via the per-expert union masks.
    scla = _sensitive_clause_masks(df)
    g_e1, g_e2 = features.gold_clauses(df, read="e1"), features.gold_clauses(df, read="e2")
    gcl = {k: (g_e1[k].fillna(False) | g_e2[k].fillna(False)) for k in g_e1}
    return {
        "n_gold": int(gold.sum()), "n_sensitive": int(sens.sum()),
        "n_intersection": int((gold & sens).sum()),
        "sensitive_not_gold": {"n": int(sn.sum()), "by_class": _by_class(sn),
                               "by_clause": {k: int((sn & m.fillna(False)).sum())
                                             for k, m in scla.items()}},
        "gold_not_sensitive": {"n": int(gn.sum()), "by_class": _by_class(gn),
                               "by_clause": {k: int((gn & m).sum()) for k, m in gcl.items()}},
    }


def nerve_vs_disc_summary(df: pd.DataFrame) -> dict[str, object]:
    """The **abnormal-nerve vs abnormal-disc** dissociation, quantified. The *nerve* axis is the clinician
    diagnosis (glaucoma = abnormal optic nerve); the *disc* axis is the dx-independent
    ``fundus_suspicious_disc`` label (per-expert union / or read). A fundus read sees only the structural (disc)
    window, so the two agree on most eyes but dissociate both ways — disc-quiet glaucoma (abnormal nerve,
    normal-looking disc; diagnosed on the field) and physiological cups (normal nerve, suspicious-looking
    disc). Returns the 2×2 counts, the adopted-label and strict-``gold`` sensitivities vs glaucoma (for the
    literature-alignment note), the raw nerve↔disc agreement, and — for the routing contrast — how the
    glaucoma eyes are dispatched under the adopted ``disc`` gate vs the strict ``gold`` gate (bilateral).
    All pure counts; the diagnosis enters only on the nerve axis and the sensitivity denominators, never the
    disc axis (which stays image-derivable)."""
    dx = df["diagnosis"]
    nerve = dx == "glaucoma"                                                # abnormal nerve = diagnosis
    disc = (features.fundus_suspicious_disc_label(df) == 1).fillna(False)    # abnormal disc = adopted label
    gold = (features.fundus_structural_label_derived(df, combine="or") == 1).fillna(False)
    n_g = int(nerve.sum())

    cells = {
        "nerve_abn_disc_abn": int((nerve & disc).sum()),
        "nerve_abn_disc_norm": int((nerve & ~disc).sum()),
        "nerve_norm_disc_abn": int((~nerve & disc).sum()),
        "nerve_norm_disc_norm": int((~nerve & ~disc).sum()),
    }

    def _glaucoma_route(profile: str, combine: str | None) -> dict[str, int]:
        kw: dict[str, object] = {"profile": profile, "use_field": True, "bilateral": True}
        if combine is not None:
            kw["combine"] = combine
        r = features.stage1_route(df, **kw)[nerve]
        return {k: int((r == k).sum()) for k in ("GECFS", "GRMS", "discharge", "referrer")}

    return {
        "cells": cells,
        "n_total": int(len(df)),
        "disc_sens": float(cells["nerve_abn_disc_abn"] / n_g),
        "gold_sens": float((nerve & gold).sum() / n_g),
        "agreement": float((((nerve & disc) | (~nerve & ~disc))).mean()),
        "glaucoma_routing": {"disc": _glaucoma_route("disc", None),
                             "gold": _glaucoma_route("gold", "or")},
    }


def clinical_class_confusion(df: pd.DataFrame, profile: str = "disc", **kw) -> dict[str, object]:
    """The 3-class **clinical-class classifier** (:func:`features.clinical_class_rule`) scored against the
    ``clinical_class`` label: the true×predicted confusion over ``healthy`` / ``OHT`` / ``glaucoma``, per-class
    recall & precision, overall accuracy, and the glaucoma-miss breakdown (true glaucoma predicted
    non-glaucoma, split → healthy vs OHT). ``profile`` picks the structural read (``"disc"`` adopted / ``"gold"``);
    extra kwargs pass through to :func:`features.clinical_class_rule` (e.g. ``bilateral=True``)."""
    order = ["healthy", "OHT", "glaucoma"]
    y = df["clinical_class"].astype(object)
    pred = features.clinical_class_rule(df, profile=profile, **kw)
    scored = y.isin(order) & pred.isin(order)
    yt, yp = y[scored], pred[scored]
    conf = {t: {p: int(((yt == t) & (yp == p)).sum()) for p in order} for t in order}
    col = {p: sum(conf[t][p] for t in order) for p in order}
    recall = {c: (conf[c][c] / sum(conf[c].values()) if sum(conf[c].values()) else float("nan")) for c in order}
    precision = {c: (conf[c][c] / col[c] if col[c] else float("nan")) for c in order}
    n = int(scored.sum())
    correct = sum(conf[c][c] for c in order)
    gmiss = conf["glaucoma"]
    return {
        "order": order,
        "confusion": conf,                     # confusion[true][pred] = count
        "recall": recall,
        "precision": precision,
        "accuracy": float(correct / n) if n else float("nan"),
        "n": n,
        "glaucoma_miss": {"healthy": gmiss["healthy"], "OHT": gmiss["OHT"],
                          "total": gmiss["healthy"] + gmiss["OHT"]},
    }


def routing_fn_breakdown(df: pd.DataFrame, combine: str = "or",
                         profile: str = "gold") -> dict[str, object]:
    """Why the Stage-1 routing still discharges diseased eyes (the routing FN). A discharge fires
    only when ALL THREE gates are silent — structural read negative, IOP < 24 (NICE), MD ≥ −2 — so a
    discharged glaucoma/suspect trips none of the router's signals (and none are lost to a data gap:
    an incomplete eye goes to ``referrer``, not discharge). Returns count buckets over the discharged
    diseased eyes (how near each is to a trigger), a **disc-detectable vs disc-silent** split — since
    PAPILA has no basis-of-diagnosis field, "diagnosed on abnormal disc" is proxied by whether the
    *sensitive* disc read fires (disc-detectable = gold cut too strict; disc-silent = cup/rim normal on
    both experts' contours → needs image review) — the detectable set's per-clause attribution, and
    the discharged-glaucoma detail rows tagged with that split."""
    route = features.stage1_route(df, combine=combine, profile=profile, use_field=True)
    dx = df["diagnosis"]
    disch = route == "discharge"
    dis = disch & dx.isin(["glaucoma", "suspect"])
    vcdr = pd.to_numeric(df["vcdr"], errors="coerce")
    iop = pd.to_numeric(df["iop"], errors="coerce")
    md = pd.to_numeric(df["vf_md"], errors="coerce")
    sens = features.fundus_structural_label_sensitive(df).fillna(0).astype(int).astype(bool)
    detect, silent = dis & sens, dis & ~sens   # disc-detectable (sensitive read fires) vs disc-silent
    counts = {
        "n_total": int(dis.sum()),
        "n_glaucoma": int((disch & (dx == "glaucoma")).sum()),
        "n_suspect": int((disch & (dx == "suspect")).sum()),
        "iop_borderline": int((dis & (iop >= 21) & (iop < 24)).sum()),  # OHT below the NICE ≥24 bar
        "iop_normotensive": int((dis & (iop < 21)).sum()),
        "md_near_normal": int((dis & (md >= -1)).sum()),                # global MD essentially normal
        "vcdr_small": int((dis & (vcdr < 0.4)).sum()),
        "vcdr_borderline": int((dis & (vcdr >= 0.5)).sum()),            # a lower cut would catch these
        "irreducible": int((dis & (iop < 21) & (md >= -1) & (vcdr < 0.5)).sum()),  # silent on all 3
        "disc_detectable": int(detect.sum()),
        "disc_detectable_glaucoma": int((detect & (dx == "glaucoma")).sum()),
        "disc_detectable_suspect": int((detect & (dx == "suspect")).sum()),
        "disc_silent": int(silent.sum()),
        "disc_silent_glaucoma": int((silent & (dx == "glaucoma")).sum()),
        "disc_silent_suspect": int((silent & (dx == "suspect")).sum()),
    }
    # which sensitive clause makes each disc-detectable FN visible (overlapping)
    attribution = {k: int((detect & m).sum()) for k, m in features.sensitive_clauses(df).items()
                   if int((detect & m).sum()) > 0}
    gmask = (disch & (dx == "glaucoma")).to_numpy()
    glaucoma = pd.DataFrame({
        "eye_id": (df["eye_id"].to_numpy()[gmask] if "eye_id" in df.columns
                   else np.arange(int(gmask.sum()))),
        "vcdr": vcdr.to_numpy()[gmask].round(2),
        "iop": iop.to_numpy()[gmask].round(0),
        "vf_md": md.to_numpy()[gmask].round(1),
        "disc": np.where(sens.to_numpy()[gmask], "detectable", "silent"),
    })
    return {"counts": counts, "attribution": attribution, "glaucoma": glaucoma}


def structure_gt_summary(df: pd.DataFrame) -> dict[str, object]:
    """Summary of the dx-independent `fundus_suspicious_disc` label: positives by clinical group
    (validation — healthy flagged = physiological FP; non-healthy flagged = caught), the 4-way
    concordance crosstab, n-flagged vs a single rule everywhere, and routing on it (the proposed gate)."""
    dx = df["diagnosis"]
    H, NH = dx == "healthy", dx.isin(("suspect", "glaucoma"))
    gold = (features.fundus_structural_label_derived(df, combine="or") == 1).fillna(False)
    sens = (features.fundus_structural_label_sensitive(df) == 1).fillna(False)
    lab = (features.fundus_suspicious_disc_label(df) == 1).fillna(False)
    construction = pd.DataFrame([
        {"group": "healthy", "n": int(H.sum()),
         "suspicious": int((lab & H).sum()), "not": int((~lab & H).sum())},
        {"group": "non-healthy", "n": int(NH.sum()),
         "suspicious": int((lab & NH).sum()), "not": int((~lab & NH).sum())},
    ])
    compare = {"gold everywhere": int(gold.sum()), "sensitive everywhere": int(sens.sum()),
               "chosen (fundus_suspicious_disc)": int(lab.sum())}
    conc = features.structure_concordance(df).rename("concordance")
    route = features.stage1_route(df, profile="disc", use_field=True, bilateral=True).rename("route")
    return {"construction": construction, "compare": compare,
            "crosstab": pd.crosstab(conc, dx),
            "route_crosstab": pd.crosstab(route, dx)}  # disc gate + bilateral (patient-level) rule


def bilateral_coverage(df: pd.DataFrame, profile: str = "disc") -> dict[str, object]:
    """Patient-level (bilateral) aggregation of the routing FN: of the discharged diseased eyes, how
    many are **recovered** because the fellow eye is flagged (`features.patient_referred`) vs remain a
    **bilaterally disc-silent abstain floor** (both eyes discharge). Glaucoma is asymmetric, so a
    structure-silent eye is still worked up when its fellow betrays disease; the floor rests on a focal
    VF defect / records not in the released data, not on a better disc rule."""
    dx = df["diagnosis"]
    route = features.stage1_route(df, profile=profile, use_field=True)
    fn = (route == "discharge") & dx.isin(("suspect", "glaucoma"))
    pref = features.patient_referred(df, profile=profile, use_field=True)
    recovered, abstain = fn & pref, fn & ~pref

    def by(mask: pd.Series) -> dict[str, int]:
        return {c: int((mask & (dx == c)).sum()) for c in ("glaucoma", "suspect")}

    has_pid = "patient_id" in df.columns
    return {
        "n_fn": int(fn.sum()),
        "n_recovered": int(recovered.sum()), "recovered": by(recovered),
        "n_abstain": int(abstain.sum()), "abstain": by(abstain),
        "abstain_patients": int(df.loc[abstain, "patient_id"].nunique()) if has_pid else None,
        "abstain_patient_dx": (df.loc[abstain].groupby("patient_id")["diagnosis"].first()
                               .value_counts().to_dict() if has_pid else {}),
    }


def gt_routing_fn_residual(df: pd.DataFrame, profile: str = "disc") -> dict[str, object]:
    """The routing FN 'silent on every COMPUTABLE signal' — discharged diseased eyes with the
    structural read negative, IOP < 21, and global MD ≥ −1 (a CCT-adjusted IOP recovers ~0 too, since
    the FN are not thin-cornea). CCT-adjusted IOP is thus ruled out; each residual is EITHER a
    fundus-image feature our cup/disc contour metrics discard (RNFL/PPA/haemorrhage/notch/pallor —
    determinable only by eyes-on the photo / a trained image model) OR a retrospective-records
    diagnosis absent from PAPILA — the two are not separable from tabular data. Reports two aligned
    cuts of the 41 per-eye FN: the **per-eye** silent set (``n``), and the **bilateral floor** (the 26
    still discharged after the patient-level rule) split into silent vs near-threshold. Returns the
    per-eye-silent eye rows (the review worklist), each flagged ``in_bilateral_floor``."""
    dx = df["diagnosis"]
    nh = dx.isin(("suspect", "glaucoma"))
    iop = pd.to_numeric(df["iop"], errors="coerce")
    md = pd.to_numeric(df["vf_md"], errors="coerce")
    vcdr = pd.to_numeric(df["vcdr"], errors="coerce")
    silent = (iop < 21) & (md >= -1) & (vcdr < 0.5)   # far from every computable trigger
    fn = (features.stage1_route(df, profile=profile, use_field=True) == "discharge") & nh
    floor = (features.stage1_route(df, profile=profile, use_field=True, bilateral=True)
             == "discharge") & nh                      # bilateral residual (the abstain floor)
    resid = fn & silent                                # per-eye silent-on-every-signal
    cols = [c for c in ("eye_id", "patient_id", "diagnosis", "vcdr", "min_rim_disc_ratio",
                        "iop", "vf_md", "cct", "disc_area") if c in df.columns]
    eyes = df.loc[resid, cols].reset_index(drop=True)
    eyes["in_bilateral_floor"] = floor[resid].to_numpy()
    return {"n": int(resid.sum()),
            "n_glaucoma": int((resid & (dx == "glaucoma")).sum()),
            "n_suspect": int((resid & (dx == "suspect")).sum()),
            "n_floor": int(floor.sum()),
            "n_floor_silent": int((floor & silent).sum()),
            "n_floor_near": int((floor & ~silent).sum()),
            "eyes": eyes}


# --------------------------------------------------------------------- FN/FP anatomy + QC
def interexpert_vcdr_spread(df: pd.DataFrame) -> pd.Series:
    """``|vcdr_e1 − vcdr_e2|`` per eye — the two experts' cup-measurement disagreement (reliability)."""
    e1 = pd.to_numeric(df.get("vcdr_e1"), errors="coerce")
    e2 = pd.to_numeric(df.get("vcdr_e2"), errors="coerce")
    return (e1 - e2).abs()


def structural_error_modes(df: pd.DataFrame, profile: str = "sensitive") -> dict[str, int]:
    """Counts characterising the structural label's FN and FP vs the clinician **glaucoma** reference
    (glaucoma=1; healthy ∨ suspect=0). **FN = glaucoma missed** (the safety-critical error); **FP = a
    negative flagged**, split into `FP_suspect` (disc-normal OHT — expected, low-harm) and `FP_healthy`
    (physiological cupping)."""
    dx = df["diagnosis"]
    clin = features.fundus_structural_label(df)
    pred = (features.fundus_structural_label_sensitive(df) if profile == "sensitive"
            else features.fundus_structural_label_derived(df, combine="mean"))
    vcdr = pd.to_numeric(df["vcdr"], errors="coerce")
    iop = pd.to_numeric(df["iop"], errors="coerce")
    md = pd.to_numeric(df["vf_md"], errors="coerce")
    da = pd.to_numeric(df["disc_area"], errors="coerce")
    FN = (clin == 1) & (pred == 0)   # glaucoma missed
    FP = (clin == 0) & (pred == 1)   # a negative (healthy or suspect) flagged
    other_signal = (iop >= features.IOP_RAISED_THRESHOLD) | (md < features.FIELD_ABNORMAL_MD_DEFAULT)
    large_edge = features.DISC_AREA_TERTILE_EDGES[1]
    clauses = _sensitive_clause_masks(df)   # per-expert union clause masks — matches the adopted label `pred`
    out = {
        "FN_total": int(FN.sum()),               # == FN_glaucoma (reference is glaucoma-only)
        "FN_glaucoma": int(FN.sum()),
        "FN_recoverable_by_iop_or_field": int((FN & other_signal).sum()),
        "FN_truly_silent": int((FN & ~other_signal).sum()),
        "FN_noncupping_glaucoma": int((FN & (md <= -6) & (vcdr < 0.4)).sum()),
        "FP_total": int(FP.sum()),
        "FP_suspect": int((FP & (dx == "suspect")).sum()),   # disc-normal OHT flagged (expected)
        "FP_healthy": int((FP & (dx == "healthy")).sum()),   # physiological cupping
        "FP_large_disc_tertile": int((FP & (da > large_edge)).sum()),
    }
    for name, mask in clauses.items():
        out[f"FP_clause[{name}]"] = int((FP & mask).sum())
    return out


def qc_outliers(df: pd.DataFrame, max_per_category: int = 12) -> pd.DataFrame:
    """Specific eyes worth clinical/data review: severe field loss with a tiny cup (structure-
    function dissociation), large physiological cups (healthy big-disc flagged), and high inter-
    expert disagreement. Tidy ``[eye_id, category, ...metrics]`` for the report."""
    dx = df["diagnosis"]
    vcdr = pd.to_numeric(df["vcdr"], errors="coerce")
    md = pd.to_numeric(df["vf_md"], errors="coerce")
    da = pd.to_numeric(df["disc_area"], errors="coerce")
    spread = interexpert_vcdr_spread(df)
    pred = features.fundus_structural_label_sensitive(df)
    large_edge = features.DISC_AREA_TERTILE_EDGES[1]
    cats = {
        "severe_field_small_cup": (md <= -6) & (vcdr < 0.4),
        "large_physiological_cup": (dx == "healthy") & (da > large_edge) & (pred == 1),
        "high_interexpert_disagreement": spread > 0.15,
    }
    cols = ["eye_id", "diagnosis", "vcdr", "min_rim_disc_ratio", "disc_area", "iop", "vf_md"]
    frames = []
    for cat, mask in cats.items():
        sub = df.loc[mask, cols].copy()
        sub.insert(1, "category", cat)
        sub["interexpert_spread"] = spread[mask].round(3)
        sort_col = {"severe_field_small_cup": "vf_md", "large_physiological_cup": "vcdr",
                    "high_interexpert_disagreement": "interexpert_spread"}[cat]
        asc = cat == "severe_field_small_cup"
        frames.append(sub.sort_values(sort_col, ascending=asc).head(max_per_category))
    out = pd.concat(frames, ignore_index=True)
    for c in ("vcdr", "min_rim_disc_ratio", "disc_area", "vf_md"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out


def variant_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-label positives + agreement/κ/sens/spec vs the clinician label (reference)."""
    clin = features.fundus_structural_label(df)
    rows = []
    for name, dv in comparison_labels(df):
        cm = confusion_2x2(clin, dv)
        rows.append({
            "label": name,
            "n_pos": int((dv == 1).sum()),
            "agreement": agreement(clin, dv),
            "kappa": cohens_kappa(clin, dv),
            "sens": cm["tp"] / max(cm["tp"] + cm["fn"], 1),
            "spec": cm["tn"] / max(cm["tn"] + cm["fp"], 1),
        })
    return pd.DataFrame(rows)


def routing_crosstab(df: pd.DataFrame, combine: str = "mean", profile: str = "gold",
                     use_field: bool = True,
                     field_md_threshold: float = features.FIELD_ABNORMAL_MD_DEFAULT,
                     bilateral: bool = False) -> pd.DataFrame:
    """Stage-1 route (rows, in GECFS/GRMS/discharge/referrer order) × clinician 3-class (columns)."""
    route = features.stage1_route(df, combine=combine, profile=profile, use_field=use_field,
                                  field_md_threshold=field_md_threshold, bilateral=bilateral)
    cols = [c for c in ("healthy", "suspect", "glaucoma") if c in set(df["diagnosis"])]
    tab = pd.crosstab(route.rename("route"), df["diagnosis"])
    rows = [r for r in features.ROUTES if r in tab.index]
    return tab.reindex(index=rows, columns=cols, fill_value=0)


def inter_expert_kappa(df: pd.DataFrame,
                       thresholds: dict[str, float] | None = None) -> tuple[float, int]:
    """Cohen's κ between the two experts' per-eye derived labels (``combine="e1"`` vs ``"e2"``);
    also returns the count of eyes scorable by both experts."""
    l1 = features.fundus_structural_label_derived(df, thresholds, combine="e1")
    l2 = features.fundus_structural_label_derived(df, thresholds, combine="e2")
    both = l1.notna() & l2.notna()
    if not both.any():
        return float("nan"), 0
    return cohens_kappa(l1[both], l2[both]), int(both.sum())


# --------------------------------------------------------------------- report + figures
# ------------------------------------------------ label-definition combination search
# Search the DX-INDEPENDENT suspicious-disc label over the full literature-grounded candidate clause
# pool (:func:`features.candidate_clauses` — every recognised criterion derivable from the two-expert
# 2-D cup/disc contours, fixed literature thresholds), reporting ALL metrics both DISC-only (label vs
# the clinician glaucoma reference) and CLASSIFIER (the clinical_class cascade vs clinical_class), at
# EYE and PATIENT (bilateral) level, under PATIENT-level cross-validation, for BOTH expert-combine
# reads (``or`` per-expert union / ``mean`` averaged). The diagnosis is only the yardstick for picking
# a clinically-useful rule — the resulting label is a fixed rule applied without dx (no leakage). Every
# candidate clause now uses a fixed EXTERNAL threshold (:data:`features.IN_SAMPLE_CANDIDATE_CLAUSES` is
# empty — the rim/disc healthy limits are Chákṣu-derived), so no clause is re-derived out-of-fold; the
# out-of-fold machinery below is vestigial. This is an EXPERIMENT — it does NOT change the frozen
# :data:`features.FUNDUS_SUSPICIOUS_DISC_CLAUSES`.
_INSAMPLE_CLAUSES = features.IN_SAMPLE_CANDIDATE_CLAUSES
_CLF_ORDER = ("healthy", "OHT", "glaucoma")


def _patient_kfold(patient_ids: pd.Series, k: int, seed: int) -> list[np.ndarray]:
    """Patient-disjoint K-fold as a list of boolean test-masks over rows (grouped by patient)."""
    uniq = pd.unique(patient_ids)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[u]: i % k for i, u in enumerate(order)}
    assign = patient_ids.map(fold_of).to_numpy()
    return [assign == f for f in range(k)]


def _healthy_limits(train: pd.DataFrame, read: str) -> dict[str, object]:
    """Healthy normal limits for the in-sample clauses, derived from ``train``'s HEALTHY eyes only
    (out-of-fold) under the chosen expert ``read`` — the disc-area tertile edges + per-tertile cup
    (upper 97.5th), the DDLS per-tertile narrowest-rim (lower 2.5th), and the flat min-rim (5th) and
    rim-disc-area (2.5th) limits. **VESTIGIAL:** ``features.IN_SAMPLE_CANDIDATE_CLAUSES`` is now empty
    (every candidate clause uses a fixed EXTERNAL Chákṣu limit), so the search no longer substitutes
    these out-of-fold masks — kept for provenance / potential re-use."""
    h = train[train["diagnosis"].astype(str) == "healthy"]
    da = features._read_series(h, "disc_area", read)
    vc = features._read_series(h, "vcdr", read)
    mr = features._read_series(h, "min_rim_disc_ratio", read)
    ra = features._read_series(h, "rim_disc_area_ratio", read)
    lo, hi = da.quantile(1 / 3), da.quantile(2 / 3)

    def _q(s: pd.Series, p: float, mask: pd.Series | None = None) -> float:
        v = (s if mask is None else s[mask]).dropna()
        return float(v.quantile(p)) if len(v) else np.nan

    tert = {"small": da <= lo, "med": (da > lo) & (da <= hi), "large": da > hi}
    return {
        "edges": (lo, hi),
        "cup": {t: _q(vc, 0.975, m) for t, m in tert.items()},
        "ddls": {t: _q(mr, 0.025, m) for t, m in tert.items()},
        "min_rim_p5": _q(mr, 0.05),
        "rim_area_p2_5": _q(ra, 0.025),
    }


def _insample_masks(ev: pd.DataFrame, lim: dict[str, object], read: str) -> dict[str, np.ndarray]:
    """The in-sample candidate clauses evaluated on ``ev`` using the out-of-fold ``lim`` normal limits
    (from :func:`_healthy_limits`). Keys match :data:`features.IN_SAMPLE_CANDIDATE_CLAUSES`."""
    da = features._read_series(ev, "disc_area", read)
    vc = features._read_series(ev, "vcdr", read)
    mr = features._read_series(ev, "min_rim_disc_ratio", read)
    ra = features._read_series(ev, "rim_disc_area_ratio", read)
    lo, hi = lim["edges"]
    tert = [(da <= lo, "small"), ((da > lo) & (da <= hi), "med"), (da > hi, "large")]

    def _tcut(by: dict[str, float]) -> pd.Series:
        cut = pd.Series(np.nan, index=ev.index, dtype=float)
        for mask, t in tert:
            cut[mask] = by[t]
        return cut

    return {
        "disc-size-adjusted cup": (vc >= _tcut(lim["cup"])).fillna(False).to_numpy(),
        "DDLS-adjusted narrowest rim": (mr <= _tcut(lim["ddls"])).fillna(False).to_numpy(),
        "min-rim-disc<=p5": (mr <= lim["min_rim_p5"]).fillna(False).to_numpy(),
        "rim-disc-area<=p2.5": (ra <= lim["rim_area_p2_5"]).fillna(False).to_numpy(),
    }


def _binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Full binary-metric suite for a suspicious-disc mask ``pred`` vs the glaucoma reference ``y``
    (0/1) over the evaluated rows: sens, spec, PPV, NPV, accuracy, F1, MCC, Youden J, balanced acc,
    positivity, and the healthy-FP count."""
    tp = float(np.sum((y == 1) & pred)); fp = float(np.sum((y == 0) & pred))
    fn = float(np.sum((y == 1) & ~pred)); tn = float(np.sum((y == 0) & ~pred))
    tot = tp + fp + fn + tn
    se = tp / (tp + fn) if tp + fn else np.nan
    sp = tn / (tn + fp) if tn + fp else np.nan
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "sensitivity": se, "specificity": sp,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "accuracy": (tp + tn) / tot if tot else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
        "mcc": (tp * tn - fp * fn) / den if den else np.nan,
        "youden_j": (se + sp - 1) if (se == se and sp == sp) else np.nan,
        "balanced_acc": ((se + sp) / 2) if (se == se and sp == sp) else np.nan,
        "positivity": (tp + fp) / tot if tot else np.nan,
        "healthy_fp": fp,
    }


def _clf_metrics(yc: np.ndarray, pc: np.ndarray) -> dict[str, float]:
    """Metrics for the 3-class clinical_class cascade prediction ``pc`` vs truth ``yc`` over the
    evaluated rows: per-class recall (glaucoma recall = glaucoma **sensitivity** / OHT / healthy), glaucoma
    **specificity** (non-glaucoma correctly not called glaucoma), glaucoma precision, overall accuracy, and
    the glaucoma-missed count (true glaucoma predicted non-glaucoma — the safety metric)."""
    def _recall(c: str) -> float:
        d = yc == c
        return float(np.sum(pc[d] == c)) / int(d.sum()) if int(d.sum()) else np.nan

    gl_pred = pc == "glaucoma"
    gl_true = yc == "glaucoma"
    neg = ~gl_true                                        # non-glaucoma truth (healthy ∪ OHT)
    return {
        "clf_glaucoma_recall": _recall("glaucoma"),      # glaucoma sensitivity
        "clf_glaucoma_specificity": float(np.sum(neg & ~gl_pred)) / int(neg.sum())
        if int(neg.sum()) else np.nan,                   # non-glaucoma NOT called glaucoma
        "clf_glaucoma_precision": float(np.sum(gl_true & gl_pred)) / int(gl_pred.sum())
        if int(gl_pred.sum()) else np.nan,
        "clf_oht_recall": _recall("OHT"),
        "clf_healthy_recall": _recall("healthy"),
        "clf_accuracy": float(np.sum(yc == pc)) / len(yc) if len(yc) else np.nan,
        "clf_glaucoma_missed": float(np.sum(gl_true & ~gl_pred)),
    }


def label_combination_search(
    df: pd.DataFrame, spec_floor: float = 0.80, k: int = 5, seed: int = 0,
    combines: tuple[str, ...] = ("or", "mean"),
) -> pd.DataFrame:
    """Search unions of the full literature-grounded candidate clause pool
    (:func:`features.candidate_clauses`) for the best dx-independent suspicious-disc label. Each unique
    combination (deduplicated by its prediction vector — the simplest clause set per distinct label
    behaviour) is scored under patient-disjoint ``k``-fold CV, for each expert-combine read in
    ``combines`` (``"or"`` per-expert union / ``"mean"`` averaged), on **all** metrics at two levels:

    * **disc-only** — the candidate label vs the clinician **glaucoma** reference (glaucoma=1;
      healthy ∨ suspect=0), :func:`_binary_metrics`, at eye and patient (bilateral) level (``*_pt``);
    * **classifier** — the ``clinical_class`` cascade (:func:`features._clinical_class_from_disc`, damage
      = disc OR field → glaucoma; else IOP≥24 → OHT; else healthy) vs ``clinical_class``,
      :func:`_clf_metrics` (``clf_*`` / ``clf_*_pt``).

    Every clause now uses a fixed EXTERNAL threshold (no in-sample re-derivation — the out-of-fold path is
    vestigial); inter-expert κ (per-expert union e1 vs e2) is reported per combination. EXPERIMENT ONLY —
    the frozen ``FUNDUS_SUSPICIOUS_DISC_CLAUSES`` is unchanged.
    Returns one row per (unique combination × combine), sorted by out-of-fold disc Youden J."""
    df = df.reset_index(drop=True)
    n = len(df)
    dx = df["diagnosis"].astype(str)
    yt = np.where(dx == "glaucoma", 1.0, np.where(dx.isin(("healthy", "suspect")), 0.0, np.nan))
    yc = df["clinical_class"].astype(object).to_numpy()
    pid = df["patient_id"]
    scorable = pd.to_numeric(df["vcdr"], errors="coerce").notna().to_numpy() & ~np.isnan(yt)
    clf_scorable = np.isin(yc, _CLF_ORDER)
    folds = _patient_kfold(pid, k, seed)
    insample = set(features.IN_SAMPLE_CANDIDATE_CLAUSES)

    pool = list(features.candidate_clauses(df, "mean").keys())
    # per-expert candidate masks (for inter-expert κ of any clause set)
    cand_e1 = {nm: m.to_numpy() for nm, m in features.candidate_clauses(df, "e1").items()}
    cand_e2 = {nm: m.to_numpy() for nm, m in features.candidate_clauses(df, "e2").items()}

    rows = []
    for combine in combines:
        reads = ("e1", "e2") if combine == "or" else ("mean",)
        per_read = [features.candidate_clauses(df, r) for r in reads]
        full = {nm: np.zeros(n, dtype=bool) for nm in pool}
        for cc in per_read:
            for nm in pool:
                full[nm] |= cc[nm].to_numpy()
        full_arr = np.stack([full[nm] for nm in pool])  # (npool, n)

        # out-of-fold in-sample masks per fold (union over reads)
        fold_ins = []
        for tm in folds:
            train = df[~tm]
            acc: dict[str, np.ndarray] = {}
            for r in reads:
                lim = _healthy_limits(train, r)
                for nm, mm in _insample_masks(df, lim, r).items():
                    acc[nm] = acc.get(nm, np.zeros(n, dtype=bool)) | mm
            fold_ins.append(acc)

        # baseline predictions — the adopted label (fundus_suspicious_disc) and its gold comparison are both
        # the DEPLOYED per-expert union (or) read; matched by prediction equality (robust to clause-key
        # naming) and tagged only on the or panel below.
        adopted = (features.fundus_suspicious_disc_label(df) == 1).fillna(False).to_numpy()
        gold = (features.fundus_structural_label_derived(df, combine="or") == 1).fillna(False).to_numpy()

        # enumerate all non-empty subsets, dedup by nominal (full-data) prediction — keep the simplest
        # clause set per distinct label behaviour (every operating point is retained).
        best: dict[bytes, tuple[int, tuple[str, ...]]] = {}
        for r in range(1, len(pool) + 1):
            for combo in combinations(range(len(pool)), r):
                pred = full_arr[list(combo)].any(axis=0)
                names = tuple(pool[i] for i in combo)
                key = pred.tobytes()
                prev = best.get(key)
                if prev is None or (len(names), names) < (prev[0], prev[1]):
                    best[key] = (len(names), names)

        for key, (nc, names) in best.items():
            lists: dict[tuple[str, str], list[float]] = defaultdict(list)
            for fi, tm in enumerate(folds):
                pred = np.zeros(n, dtype=bool)
                for nm in names:
                    pred |= fold_ins[fi][nm] if nm in insample else full[nm]
                pred_s = pd.Series(pred, index=df.index)
                pred_bil = pred_s.groupby(pid).transform("any").to_numpy()
                te = tm & scorable
                if te.any():
                    for kk, vv in _binary_metrics(yt[te], pred[te]).items():
                        lists[("eye", kk)].append(vv)
                    for kk, vv in _binary_metrics(yt[te], pred_bil[te]).items():
                        lists[("pt", kk)].append(vv)
                tc = tm & clf_scorable
                if tc.any():
                    pc = features._clinical_class_from_disc(df, pred_s).to_numpy()
                    pcb = features._clinical_class_from_disc(df, pred_s, bilateral=True).to_numpy()
                    for kk, vv in _clf_metrics(yc[tc], pc[tc]).items():
                        lists[("eye", kk)].append(vv)
                    for kk, vv in _clf_metrics(yc[tc], pcb[tc]).items():
                        lists[("pt", kk)].append(vv)

            # inter-expert κ: per-expert union of the clause set (e1 vs e2), on scorable eyes
            p1 = np.zeros(n, dtype=bool); p2 = np.zeros(n, dtype=bool)
            for nm in names:
                p1 |= cand_e1[nm]; p2 |= cand_e2[nm]
            iek = cohens_kappa(pd.Series(p1[scorable]).astype(int), pd.Series(p2[scorable]).astype(int))

            row: dict[str, object] = {
                "clauses": " ∪ ".join(names), "n_clauses": nc, "combine": combine,
                "interexpert_kappa": iek,
            }
            for (lvl, kk), vals in lists.items():
                col = kk if lvl == "eye" else f"{kk}_pt"
                clean = [v for v in vals if v == v]  # drop NaN folds (e.g. a class absent in a test fold)
                row[col] = float(np.mean(clean)) if clean else np.nan
            row["meets_spec_floor"] = bool(row.get("specificity", np.nan) >= spec_floor)
            # tag ONLY the deployed per-expert union (or) read as the baseline — the adopted label and its
            # gold comparison are both or-read, so tagging the mean panel too would give two "adopted" / two
            # "gold" leaderboard rows.
            row["baseline"] = ("adopted" if combine == "or" and key == adopted.tobytes()
                               else "gold" if combine == "or" and key == gold.tobytes() else "")
            rows.append(row)

    return pd.DataFrame(rows).sort_values("youden_j", ascending=False).reset_index(drop=True)


def search_leaderboard(search: pd.DataFrame) -> pd.DataFrame:
    """Flag the notable rows of :func:`label_combination_search` for the compact leaderboard: the
    best combination per key metric (disc + classifier, eye + patient), the named baselines, and the
    Pareto frontier on (classifier patient-level glaucoma recall) vs (disc specificity). Adds a
    ``why`` column naming why each row is kept; returns just the flagged rows."""
    key_metrics = [
        "youden_j", "balanced_acc", "f1", "mcc", "sensitivity", "specificity",
        "clf_glaucoma_recall", "clf_glaucoma_specificity", "clf_glaucoma_recall_pt",
        "clf_glaucoma_specificity_pt", "clf_accuracy", "clf_accuracy_pt",
    ]
    why: dict[int, list[str]] = defaultdict(list)
    for m in key_metrics:
        if m in search.columns and search[m].notna().any():
            # glaucoma_missed would be argmin; all listed here are argmax metrics
            why[int(search[m].idxmax())].append(f"best {m}")
    for i in search.index[search["baseline"] != ""]:
        why[int(i)].append(f"baseline:{search.at[i, 'baseline']}")
    # Pareto frontier: maximise clf patient glaucoma recall AND disc specificity
    if {"clf_glaucoma_recall_pt", "specificity"} <= set(search.columns):
        cand = search[["clf_glaucoma_recall_pt", "specificity"]].dropna()
        for i, (rec, spec) in cand.iterrows():
            dominated = ((cand["clf_glaucoma_recall_pt"] >= rec) & (cand["specificity"] >= spec)
                         & ((cand["clf_glaucoma_recall_pt"] > rec) | (cand["specificity"] > spec))).any()
            if not dominated:
                why[int(i)].append("pareto")
    out = search.loc[sorted(why)].copy()
    out.insert(0, "why", [", ".join(why[int(i)]) for i in out.index])
    return out.sort_values("clf_glaucoma_recall_pt", ascending=False).reset_index(drop=True)


def build_markdown(df: pd.DataFrame) -> str:
    """Assemble the human-readable experiment report as a markdown string."""
    clin = features.fundus_structural_label(df)
    n = int(clin.notna().sum())
    labels = {key: features.fundus_structural_label_derived(df, combine=key)
              for _, key in VARIANTS}

    lines = [
        "# Fundus-label experiment — clinician diagnosis vs CDR/rim/ISNT (proposal v0.45)",
        "",
        f"PAPILA fundus arm, {n} eyes with a clinician label. Candidate structural labels for the "
        "Stage-1 fundus classifier are compared against the **clinician glaucoma reference** "
        "(`glaucoma=1`; `healthy ∨ suspect=0` — PAPILA suspect is disc-normal OHT, counted **negative**, "
        "since a disc rule cannot detect it and glaucoma is the safety-critical class): an "
        "**imaging-morphology** rule derived purely from the expert "
        "cup/disc contours (vertical/area CDR, inter-eye CDR asymmetry, ISNT rule) with cited "
        "thresholds — independent of the clinician, so the comparison is not circular — reported "
        "for **four expert-combination variants** (each expert alone, their *mean* consensus, their "
        "*OR* union). The comparison also adds **IOP** and the **Stage-1 routing** labels "
        "(v0.32 decision tree, Step 3; below).",
        "",
        "## How PAPILA assigned the diagnosis (provenance)",
        "Per the dataset paper (Kovalyk et al., *Scientific Data* 2022), the classes fall out of the "
        "**recruitment scheme** (comprehensive clinical evaluation, *sometimes with retrospective records*; "
        "the paper does **not** state how many clinicians assigned the label — two experts, 2 & 27 yr, drew "
        "only the **segmentations**, hence 7 patients have discordant OD/OS labels):",
        "- **Group 1 → glaucoma** — *\"patients diagnosed with simple chronic glaucoma recruited in the "
        "Glaucoma Area\"*; VF loss = *\"at least three points in the same hemifield … 5% below normal, "
        "excluding the pericecal and peripheral rows\"* (a **focal** criterion — met even with a *normal "
        "global MD*).",
        "- **Group 2** — *\"patients from primary care who … did not show any ocular pathology that could "
        "influence the morphology of the optic nerve\"*, split by IOP into **2.a → healthy** (*\"IOP ≤ 22 "
        "mmHg\"*) and **2.b → suspect** (*\"ocular hypertensive … IOP 22–28 mmHg without visual field "
        "affectation\"*).",
        "So **\"suspect\" ≈ ocular hypertension** — an **IOP-driven** category, *narrower* than the clinical "
        "\"glaucoma suspect\" (no suspicious-disc / family-history arm). *\"No patients with IOP > 22 mmHg "
        "and lesions suggestive of glaucomatous neuropathy were detected\"* → **no suspect eye is glaucoma**. "
        "And because all of Group 2 was recruited **free of optic-nerve-morphology pathology**, suspect is an "
        "**IOP category, not a disc category**: a suspect our disc rule flags has a *physiological* large cup, "
        "not disease — which is why suspects route to GRMS/discharge, essentially never GECFS-on-disc.",
        "Recorded variables: refractive error, IOP, CCT, axial length, and **MD only** (no pattern/threshold "
        "map); a visual field was retrieved **only when a Group-2 eye had IOP > 22 mmHg**; **no OCT and no "
        "gonioscopy**. Two consequences: (a) VF-MD missingness is real (VF only for IOP>22 → healthy VF-MD "
        "is simulated, below); (b) the glaucoma criterion is **focal** but only the **global MD** was "
        "released, and the diagnosis also read the **fundus image** (RNFL defects, peripapillary atrophy, "
        "disc haemorrhage, notch, pallor) — features our cup/disc **contour** metrics discard — which is "
        "the main reason for the structural false negatives below.",
        "",
        "## Label balance",
        f"- Clinician positive (**glaucoma**): {int((clin == 1).sum())} / {n} "
        f"(suspect + healthy = negative)",
    ]
    for name, key in VARIANTS:
        dv = labels[key]
        lines.append(f"- Derived positive [{name}]: {int((dv == 1).sum())} / {int(dv.notna().sum())}")

    extra = int(((labels["or"] == 1) & (labels["mean"] == 0)).sum())
    disagree = int(((labels["e1"] == 1) != (labels["e2"] == 1)).sum())
    lines += [
        "",
        "## Variant comparison vs the clinician label (reference)",
        "```",
        variant_comparison_table(df).round(3).to_string(index=False),
        "```",
        f"- Experts 1 and 2 give a different derived call on {disagree} eyes; the OR variant flags "
        f"{extra} eyes positive that the averaged (mean) rule does not (union ≥ mean ≥ each expert "
        "is not guaranteed, but union ≥ each expert by construction).",
    ]

    for name, key in [("mean (consensus)", "mean"), ("OR (union)", "or")]:
        lines += [
            "",
            f"## Derived label × clinician 3-class — [{name}]",
            "```",
            derived_by_clinician_class(df, combine=key).to_string(),
            "```",
        ]

    # --- gold vs sensitive: what the adopted label adds over the strict baseline ---
    gsd = gold_vs_sensitive_diff(df)
    sng, gns = gsd["sensitive_not_gold"], gsd["gold_not_sensitive"]

    def _diff_class(d: dict) -> str:
        return " + ".join(f"{d['by_class'][c]} {c}" for c in ("glaucoma", "suspect", "healthy"))

    def _diff_clause(d: dict) -> str:
        return ", ".join(f"{k} ×{v}" for k, v in d["by_clause"].items() if v)

    n_avg = int((features.fundus_structural_label_sensitive(df, combine="mean") == 1).sum())
    gold_bullet = (
        f"- **gold keeps {gns['n']} eyes** ({_diff_class(gns)}) sensitive drops — via {_diff_clause(gns)} "
        "(gold's **unfloored** asymmetry ≥ 0.20, which the adopted label no longer includes)."
        if gns["n"] else
        "- **gold keeps 0 eyes** sensitive drops — the adopted label **fully contains** the strict gold "
        "read (every gold-flagged eye is also caught by the adopted disc-size-adjusted cup)."
    )
    lines += [
        "",
        "## Gold vs sensitive — what the adopted label adds",
        "Both reads are diagnosis-independent and both are computed on the **per-expert union (`or`) "
        "read**. **gold** (strict ISGEO) flags "
        f"{gsd['n_gold']}; the adopted **sensitive** (== `fundus_suspicious_disc`) flags "
        f"{gsd['n_sensitive']}; they share {gsd['n_intersection']}. The difference is asymmetric:",
        f"- **sensitive adds {sng['n']} eyes** ({_diff_class(sng)}) gold misses — via {_diff_clause(sng)}. "
        "The disc-size-adjusted cup recovers small-disc disease the flat "
        "0.70 cut cannot, and the IST rim rule (I≥S≥T) catches inferior/superior neuroretinal-rim loss "
        "the ISGEO cup/asymmetry cuts miss. Of these, "
        f"{sng['by_class']['glaucoma'] + sng['by_class']['suspect']} are diseased (the sensitivity gain), "
        f"{sng['by_class']['healthy']} healthy (the specificity cost).",
        gold_bullet,
        "> Net: sensitive trades a little specificity to recover diseased discs the strict gold cut misses "
        "— the intended low-FNR direction for triage.",
        f"> **Read choice (union vs mean).** The adopted label is the **per-expert union (`or`) "
        f"read** — fire if EITHER grader's contours trip a clause — flagging {gsd['n_sensitive']}; it is "
        "the operating point at the label's maximum patient-level glaucoma sensitivity, with a bilateral "
        "routing floor of **26 (24 suspect + 2 glaucoma)**: the residual glaucoma is the single **RET026** "
        "bilateral normal-tension patient no contour rule reaches, and the union **recovers RET042** — a "
        "single grader's borderline trace the averaged read dilutes away. The **expert-averaged (mean) "
        f"consensus read** (flagging only {n_avg}) is the lower-sensitivity alternative we do **not** "
        "adopt (it misses RET042).",
    ]

    # --- Stage-1 routing (v0.32 decision tree, Step 3) ---
    nl = features.FIELD_ABNORMAL_MD["normal_limit"]
    # GRMS = disc-negative & (raised IOP OR abnormal field); decompose it + the tension mix by class so
    # the report shows WHY GRMS exceeds the OHT (IOP>=24) count (GIRFT admits abnormal fields too).
    _iop = pd.to_numeric(df["iop"], errors="coerce")
    _md = pd.to_numeric(df["vf_md"], errors="coerce")
    _bar = features.IOP_RAISED_THRESHOLD
    _grms = features.stage1_route(df, combine="or", profile="gold", use_field=True) == "GRMS"
    _grms_iop = int((_grms & (_iop >= _bar)).sum())
    _grms_field = int((_grms & (_md < nl) & (_iop < _bar)).sum())
    _dxg = df["diagnosis"] == "glaucoma"
    _ng = int(_dxg.sum())
    _ntg = round(100 * int((_dxg & (_iop < 21)).sum()) / max(_ng, 1))
    _htg = round(100 * int((_dxg & (_iop >= _bar)).sum()) / max(_ng, 1))
    _susp = df["diagnosis"] == "suspect"
    _susp_oht = int((_susp & (_iop >= _bar)).sum())
    _cct = pd.to_numeric(df["cct"], errors="coerce")
    _h = df["diagnosis"] == "healthy"
    _h_oht = int((_h & (_iop >= _bar)).sum())          # "healthy" eyes with raw IOP >= 24 (OHT-range)
    _h_cct = _cct[_h & (_iop >= _bar)].median()        # their CCT (thick cornea over-reads non-contact IOP)
    lines += [
        "",
        "## Stage-1 routing (v0.32 decision tree, Step 3): structure + IOP + field → "
        "GECFS / GRMS / discharge / referrer",
        "Rules (the \"Read & route\" node; structure = **gold OR** read): suspicious structure → "
        "**GECFS**; normal structure + "
        f"raised IOP (≥ {features.IOP_RAISED_THRESHOLD:g}) and/or abnormal field → **GRMS**; normal "
        f"structure + reliable normal field + IOP < {features.IOP_RAISED_THRESHOLD:g} → **discharge** "
        "(clear-negative, met no trigger); missing structure/IOP/field → **referrer** (return to "
        "referrer, filing gap). IOP and field are always read together. Per v0.32 (and the LOCSU CMGs) "
        "the two community routes are **GECFS** = Glaucoma Enhanced Case-Finding, a full work-up incl. "
        "OCT for a disc suspect; **GRMS** = Glaucoma Repeat Measures, the community repeat-measures / "
        "case-finding pathway on a **normal disc** for *possible OHT (IOP ≥ 24 with no other glaucoma "
        "signs) and/or an abnormal or unreliable visual-field defect* (GIRFT), confirmed by repeat "
        "Goldmann IOP + repeat field. A **bilateral (patient-level) rule** then escalates a discharge eye "
        "to its flagged fellow's route — glaucoma is asymmetric, so the patient is worked up together — "
        "cutting diseased discharges **52 → 38**. route × clinician 3-class (bilateral):",
        "```",
        routing_crosstab(df, combine="or", use_field=True, field_md_threshold=nl,
                         bilateral=True).to_string(),
        "```",
        f"> **GRMS ⊋ OHT (why it's larger, and valid).** GRMS admits *both* GIRFT entries, so it exceeds "
        f"the OHT count: of its {_grms_iop + _grms_field} per-eye entries **{_grms_iop} come via raised "
        f"IOP** and **{_grms_field} via an abnormal field** (only {int((_iop >= _bar).sum())} eyes have "
        f"IOP ≥ {_bar:g} at all). **Tension terms matter:** there is *no OHT within glaucoma* — OHT is "
        f"raised IOP **without** damage. PAPILA glaucoma is ~{_ntg}% normal-tension (NTG) and ~{_htg}% "
        f"high-tension (HTG, all with disc/field signs, so not OHT). **OHT (raised IOP, no damage) spans "
        f"two labels:** the suspect group ({_susp_oht}/{int(_susp.sum())} at IOP ≥ {_bar:g}) is PAPILA's "
        f"labelled OHT-type, but {_h_oht} \"healthy\" eyes also read IOP ≥ {_bar:g} — mostly **thick-cornea** "
        f"(median CCT {_h_cct:.0f} µm vs 533 for healthy), because our `iop` is **raw non-contact** (not "
        "CCT-adjusted), which over-reads. The routing sends both to **GRMS** — the correct repeat-with-"
        "Goldmann / CCT-correct confirmation (NICE NG81: don't refer on non-contact IOP alone), not "
        "discharge. Glaucoma itself enters routing via disc/field, since 86% of it is NTG the IOP gate "
        "cannot see.",
        f"> **Alignment with PAPILA's label criteria (Kovalyk 2022).** The routing mirrors NICE/LOCSU and is "
        f"broadly consistent with the PAPILA classes, but is **not** a faithful reproduction of the label "
        f"rule, in two measurable ways. (1) **IOP boundary:** PAPILA defines OHT/suspect at IOP > 22; the "
        f"routing uses the NICE referral bar ≥ {_bar:g}, so "
        f"{int((_susp & (_iop >= 22) & (_iop < _bar)).sum())} suspects at IOP 22–23 sit below it — "
        "deliberate (≥ 24 is the *referral* threshold, and ≥ 22 would sweep in "
        f"{int((_h & (_iop > 22) & (_iop < _bar)).sum())} thick-cornea 'healthy' over-reads), and it does "
        "**not** affect the glaucoma floor. (2) **Field:** PAPILA glaucoma is a **focal** VF criterion (≥3 "
        f"points, one hemifield) but the routing has only global MD, so **{int((_dxg & (_md >= -2)).sum())} "
        f"of {int(_dxg.sum())} glaucoma eyes are field-silent** to the gate (RET026 among them) — the "
        "data-driven gap that creates the floor. (3) The **disc gate** (GECFS) has no PAPILA counterpart "
        "(PAPILA used VF + records, not a cup/disc cut) — it is our added, image-derivable structural gate.",
        "> IOP cut-off 24 mmHg = NICE NG81 (v0.32 §8 confirm-item; grounded in GIRFT May 2026 + NICE "
        "NG81 §1.1). The abnormal-field term is an **MD-only proxy** for GIRFT's *“abnormal or "
        "unreliable visual-field defect”* — we have only global-mean MD, no PSD/GHT or reliability "
        "indices, so `MD < −2` stands in for both the *abnormal* and *unreliable* field (the gold "
        "standard is PSD p<5% / GHT outside normal limits; Hodapp-Parrish-Anderson 1993 stages MD: "
        "mild ≥ −6, moderate −6…−12, severe < −12), which PAPILA lacks. VF-MD is a "
        "**validated functional feature** here — the raw missingness leak (MD present only for "
        "non-healthy) was removed by the VF-MD simulation and validated as distributionally faithful "
        "(`data_status.vf_md_distribution_check`; simulated healthy have a realistic ~2.5% MD<−2 "
        "rate) — so the field versions are legitimate, not leaky.",
        "> **Non-contact-IOP caveat.** PAPILA's IOP is non-contact (Nidek NT-2000), which tends to "
        "*overestimate* IOP; NICE **NG81 §1.1.2** says *“do not base a decision to refer solely on "
        "IOP measurement using non-contact tonometry.”* So the raised-IOP branch here rests on a "
        "reading that cannot, on its own, drive a **referral** — it needs Goldmann confirmation, which "
        "is exactly the **GRMS repeat-measures** role. The routing demonstrates the mechanism, not a "
        "deployable decision (a deployed pathway needs Goldmann-quality IOP, per v0.32).",
    ]

    # (The per-eye gold routing-FN tour was removed — the FN-after-routing deep-dive lives in the
    #  disc-label sections below: "Suspicious-disc label" + "Why FN persist after routing".)

    # --- reducing the false-negative rate (the critical triage error) ---
    lines += [
        "",
        "## Reducing the glaucoma false-negative rate",
        "FNR = fraction of **glaucoma** NOT flagged for attention (for routing, 'not flagged' "
        "= discharged; a `referrer`/abstain is a safe, non-discharge outcome); `glaucoma_FN_n` is the "
        "missed-glaucoma count (of 87). The **sensitive** "
        "structural read is the **adopted label** (== `fundus_suspicious_disc`), computed on the "
        "per-expert union (`or`) read from **two clauses**: the disc-size-adjusted cup (~0.49–0.54; "
        "external healthy 97.5th-pct from Chákṣu, Kumar 2023) and the IST rim rule (I≥S≥T & vcdr ≥ 0.50; "
        "Higashide 2022) — it differs from gold by adding the disc-size-adjusted cup and the IST rim rule "
        "and dropping gold's unfloored asymmetry (the flat ISGEO 0.70 CDR cuts are subsumed by the "
        "disc-size cup; the strict full-ISNT is dropped). "
        "Routing keeps the NICE IOP bar (≥24). **Note:** the disc-size normal limit is now EXTERNAL "
        "(Chákṣu consensus-normal, not fit to PAPILA), so the healthy specificity below is a fair, "
        "non-circular estimate.",
        "> **Why the sensitive read is the train/eval label.** (a) It is **diagnosis-independent and "
        "image-derivable** — a fixed rule on the disc morphometry, so a fundus-image model can be trained "
        "and tested against it with no diagnosis leakage; (b) it is built entirely from **recognised "
        "clinical criteria** (ISGEO CDR/asymmetry, Jonas/Blue-Mountains disc-size); (c) it is the "
        "**low-FNR** read — it lowers the bar below the strict 0.70 gold cut to catch borderline disc "
        "suspects, matching the triage priority that a *missed* suspicious disc is the dangerous error; "
        "and (d) it was **chosen empirically** (label-definition experiment below) as the best clause "
        "union against the clinician diagnosis under patient-level CV.",
        "```",
        fnr_table(df)[["decision", "glaucoma_FNR", "glaucoma_FN_n", "healthy_spec"]].to_string(index=False),
        "```",
        "**Operating-point ceiling** — `vcdr` threshold for a target glaucoma sensitivity and the "
        "healthy specificity there:",
        "```",
        operating_point_table(df).to_string(index=False),
        "```",
        "> A single structural rule **cannot** reach ~0 FNR: glaucoma/suspect cups overlap healthy "
        "(vcdr down to ~0.12), so 100% sensitivity forces specificity ≈ 0 (flag nearly everyone) — "
        "the Cochrane/Michelessi 2015 & Mowatt 2008 ceiling (single structural tests ~70% sens at "
        "high spec). Suspects are the hard residual — many have structurally **normal** discs "
        "(ocular hypertension / pre-perimetric), invisible to any disc rule. Near-zero positive FNR "
        "therefore comes from the **multi-signal routing (structure OR raised-IOP OR abnormal-field) "
        "+ safety-first abstain/DAVRAP** (NICE NG81 'refer when in doubt'), not a more aggressive "
        "disc cut-off. Achieved PAPILA sensitivity stays below the source studies (mild cohort + "
        "contour-derived CDR).",
    ]

    # --- disc-size normal limit: external (Chákṣu) normative ---
    lines += [
        "",
        "### Disc-size-adjusted cup: an external (Chákṣu) normal limit",
        "The sensitive read's `disc-size-adjusted cup` clause uses the healthy 97.5th-pct vcdr per "
        "disc-area tertile derived from **Chákṣu** (Kumar 2023), an independent fundus dataset with the "
        "same 2-D cup/disc contour method (consensus-normal eyes, within-camera tertiles). It is therefore "
        "**external and non-circular** — the reported healthy specificity is a fair held-out estimate, not "
        "fit to PAPILA. Reassuringly, the external cut {0.494, 0.527, 0.540} reproduces the retired PAPILA "
        "in-sample 97.5th-pct {0.505, 0.528, 0.558} within 0.02 on every tertile (mutual corroboration).",
    ]

    # --- suspicious-disc label (dx-independent target) ---
    gts = structure_gt_summary(df)
    cmp = gts["compare"]
    rfg = routing_fn_breakdown(df, profile="disc")["counts"]  # routing FN = the disc-silent floor
    bc = bilateral_coverage(df, profile="disc")               # per-eye FN -> fellow-rescued vs abstain floor
    lines += [
        "",
        "## Suspicious-disc label (dx-independent — the chosen fundus target)",
        "The `features.fundus_suspicious_disc_label` selected above, applied to **every** eye as a fixed "
        "rule (no diagnosis) — positives by clinical group (a validation view: healthy flagged = "
        "physiological cups, the specificity cost; non-healthy flagged = caught):",
        "```",
        gts["construction"].to_string(index=False),
        "```",
        "Concordance with the clinician diagnosis (validation tag × dx):",
        "```",
        gts["crosstab"].to_string(),
        "```",
        "Stage-1 routing on this label (the structural gate, in place of gold) **with the bilateral "
        f"rule** — route × clinician class. Per-eye it discharges {rfg['n_total']} diseased; the "
        f"bilateral (patient-level) rule cuts that to **{bc['n_abstain']}** (vs gold's bilateral 38). "
        "Being **dx-independent**, this is the **proposed Stage-1 structural gate** — deployable once an "
        "image model predicts the label (not an oracle):",
        "```",
        gts["route_crosstab"].to_string(),
        "```",
        f"> Per-eye, the {rfg['n_total']} discharged diseased eyes ({rfg['n_glaucoma']} glaucoma + "
        f"{rfg['n_suspect']} suspect) are **the disc-silent set** — the label's clauses do not fire; "
        f"{rfg['irreducible']} are silent on every measured signal (IOP<21 & MD≥−1 & vcdr<0.5). These "
        "cannot be reached from the released data: PAPILA's **glaucoma criterion is a focal VF defect** "
        "(≥3 points in one hemifield), but only the **global MD** was released (no pattern map), and the "
        "diagnosis also drew on the fundus image + records — evidence our *contour* metrics discard. A "
        "trained **fundus-image model** is the only in-dataset lever that could recover the subset with "
        "visible RNFL/disc signs.",
        f"vs the strict gold read applied everywhere (the per-expert **OR union**, "
        f"{cmp['gold everywhere']} positives): this label flags {cmp['chosen (fundus_suspicious_disc)']} "
        "— the same per-expert OR-union read as the clause-level gold-vs-sensitive breakdown above.",
    ]

    # --- FOCUS: why false negatives persist after routing (even routing on the label) ---
    em = structural_error_modes(df, profile="sensitive")  # label-vs-clinician, before routing
    ab_pat = bc["abstain_patient_dx"]  # patient-level dx of the abstain floor
    lines += [
        "",
        "## Why glaucoma false negatives persist after routing",
        "With **glaucoma the positive class** (suspect = OHT negative), trace the safety-critical funnel:",
        f"1. **The label alone misses {em['FN_glaucoma']} glaucoma** (of {int((clin == 1).sum())}) — these "
        "discs simply do not cup (detail below).",
        f"2. **Routing adds two gates** (raised IOP ≥ {features.IOP_RAISED_THRESHOLD:g}, abnormal field "
        f"MD < {nl:g}); the field gate rescues most, leaving **{rfg['n_glaucoma']} glaucoma discharged** "
        "per-eye — all disc-silent, and slipping the IOP/field gates too.",
        f"3. **The bilateral (fellow-eye) rule** leaves the **glaucoma floor at "
        f"{bc['abstain']['glaucoma']}** — RET026 (bilateral normal-tension).",
        "4. **That floor is irreducible from the released data**: PAPILA's glaucoma criterion is a **focal "
        "VF defect** (≥3 points, one hemifield), but only the **global MD** was released — RET026 has a "
        "*normal* global MD (+0.7 / −0.2) and, on eyes-on review, no frank cupping/haemorrhage. A trained "
        "fundus-image model is the only in-dataset lever (detail below).",
        f"> The routing also discharges {rfg['n_suspect']} **suspects** per-eye ({bc['abstain']['suspect']} "
        "in the bilateral floor) — but these are **OHT negatives** (disc-normal, IOP < 24, normal field), so "
        "discharge/GRMS-monitor is the appropriate low-risk disposition, **not a miss**.",
        "",
        "### Bilateral coverage & the abstain floor",
        "Glaucoma is a patient-level, often-asymmetric disease, so a structure-silent eye is still worked "
        "up when its **fellow** eye is flagged. Aggregating the routing to the patient "
        "(`features.patient_referred` — refer the patient if either eye is GECFS/GRMS) recovers "
        f"**{bc['n_recovered']} of the {bc['n_fn']}** discharged diseased eyes "
        f"({bc['recovered']['glaucoma']} glaucoma + {bc['recovered']['suspect']} suspect) via the fellow.",
        f"The residual **{bc['n_abstain']} eyes across {bc['abstain_patients']} patients** "
        f"({bc['abstain']['glaucoma']} glaucoma + {bc['abstain']['suspect']} suspect eyes; "
        f"{ab_pat.get('glaucoma', 0)} glaucoma + {ab_pat.get('suspect', 0)} suspect patients) are "
        "**bilaterally disc-silent** — no structural / IOP / field signal in either eye. These are the "
        "**abstain floor**: our *contour* read cannot see them (they are not thin-cornea, not myopic, "
        "not older). PAPILA labelled them from a **focal VF defect** (released only as a normal-looking "
        "global MD), the fundus image + retrospective records — evidence our contour metrics discard. A "
        "trained **fundus-image model** could recover the subset with visible RNFL/disc signs; the rest "
        "(e.g. RET026, no frank image sign) are an honest abstain — their per-eye label stays 0.",
    ]

    # --- what causes the residual FN (CCT-IOP computable; the rest needs eyes-on) ---
    res = gt_routing_fn_residual(df)
    lines += [
        "",
        "### What causes the residual FN — and can we attribute it?",
        f"Two aligned cuts of the {rfg['n_total']} per-eye routing FN: **{res['n']}** are silent on *every "
        f"computable signal* (structural read, IOP < 21, global MD ≥ −1) — the **per-eye** cut "
        f"({res['n_glaucoma']} glaucoma + {res['n_suspect']} suspect); and the **bilateral floor** (the "
        f"**{res['n_floor']}** still discharged after the patient-level rule) splits into "
        f"**{res['n_floor_silent']} silent** + **{res['n_floor_near']} near-threshold** (borderline IOP "
        f"21–23 / MD −2…−1 / vcdr ≥ 0.5). The per-eye {res['n']} silent = {res['n_floor_silent']} in the "
        f"floor + {res['n'] - res['n_floor_silent']} that the fellow eye rescues. Attribution to the "
        "three candidate causes is only partly possible:",
        f"- **CCT-adjusted IOP — ruled out**: correcting IOP by pachymetry recovers ~0 (0 of "
        f"{rfg['n_total']} at a moderate correction; the FN are not thin-cornea, median CCT 544).",
        "- **Fundus-image features vs retrospective records — not separable from the tables.** An "
        f"indicative, **non-expert** eyes-on review of all {res['n']} per-eye-silent disc photos (worklist "
        "exported to `reports/gt_routing_fn_residual.csv`, flagged `in_bilateral_floor`) found uniformly "
        "small / non-cupped discs, no frank "
        "cupping and no obvious disc haemorrhage; peripapillary brightening was common but could **not** "
        "be reliably distinguished from photographic illumination artifact. So the images plausibly "
        "carry rim/peripapillary/RNFL texture our *contour* metrics discard, but reliable attribution "
        "needs an ophthalmologist or a trained fundus-image model (the project's Stage-1 classifier) — "
        "not this contour pipeline. A share are likely records/history or borderline-suspect calls, "
        "which PAPILA does not record and are therefore unknowable.",
    ]

    # --- the label's own errors, aggregate (the pre-routing FN the funnel above starts from) ---
    spread = interexpert_vcdr_spread(df)
    lines += [
        "",
        "### The label's own errors, before routing (aggregate FN/FP)",
        f"Step 1 of the funnel, unpacked. **Reliability first:** the two experts agree on `vcdr` to a "
        f"median {spread.median():.3f} (p90 {spread.quantile(0.9):.3f}; {100 * (spread > 0.15).mean():.1f}% "
        "of eyes disagree > 0.15), so the errors are **not** measurement noise — the metrics faithfully "
        "reflect the traced contours.",
        "",
        f"**False negatives ({em['FN_glaucoma']} glaucoma missed) — disc-invisible disease, not a metric "
        "bug:**",
        f"- {em['FN_recoverable_by_iop_or_field']}/{em['FN_glaucoma']} are recoverable by raised IOP "
        f"(≥24, NICE) or abnormal field (MD<−2) — this is exactly what the routing gates rescue; "
        f"**{em['FN_truly_silent']} are truly silent** on our disc-contour + IOP + field metrics (early / "
        "pre-perimetric, or defined by a focal VF defect / records our contours + global MD cannot see — "
        "see the provenance note above).",
        f"- {em['FN_noncupping_glaucoma']} have **severe field loss (MD ≤ −6) but a small cup both experts "
        "agree on** — non-cupping / functional-predominant disease (e.g. disc pallor, small disc), which "
        "optic-disc morphology cannot encode.",
        "",
        f"**False positives ({em['FP_total']} negatives flagged = {em['FP_suspect']} suspect + "
        f"{em['FP_healthy']} healthy):**",
        f"- **{em['FP_suspect']} suspects** — disc-normal **OHT** flagged by a physiological large cup; "
        "*expected and low-harm* (raised-IOP eyes reasonably get a look), not a disc error.",
        f"- **{em['FP_healthy']} healthy** ({em['FP_large_disc_tertile']} in the largest disc-area tertile) "
        "— big healthy discs with proportionally big cups the size-adjusted clause only partly removes; the "
        "rest is the disc-size-normative clause budget (the specificity cost).",
        "> Root cause: the disc read is a proxy for a partly non-structural diagnosis. FN = glaucoma the "
        "disc cannot show (focal-field / non-cupping); FP = physiological cupping in disc-normal negatives "
        "(OHT-suspects + big healthy discs). The safety net is the multi-signal routing + bilateral rule + "
        "abstain, not a better disc rule.",
        "",
        "## Clinical validity & limitations (sensitive structural metrics)",
        "The metrics are clinically recognised signs (vCDR/area-CDR **and inter-eye asymmetry** — ISGEO "
        "Category I, Foster 2002; disc-size adjustment — Blue Mountains/Jonas), computed from two-expert "
        "manual segmentations with tight "
        "agreement — so they are clinically valid *concepts*. **The fundus classifier's train/eval target "
        "is the diagnosis-independent `fundus_suspicious_disc` structural label** — a deterministic, "
        "reproducible function of the expert contours (no records, no ambiguity), so the *training* labels "
        "are clean. The **clinician diagnosis is a validation *reference*, not the training target**, and it "
        "is deliberately not fully reproducible from the image — it can rest on a **focal VF defect or "
        "records** (e.g. RET026) — so the routing floor is a **completeness** limit of the structural read, "
        "not a label-accuracy defect. The structural read is a compared, safety-oriented high-sensitivity "
        "operating point — **never a standalone decision, not deployable as-is**. Limitations: (1) 2D "
        "en-face proxies, not stereo (`min_rim_disc_ratio` is a BMO-MRW analog); (2) the disc-size "
        "97.5th-pct normal limit is EXTERNAL (Chákṣu consensus-normal) → non-circular specificity; the "
        "candidate-library `min_rim_disc_ratio` p5 stays in-sample but is not in the adopted label; (3) "
        "disc size is pixel-scaled (relative tertiles, not mm cut-offs); (4) the "
        "strict full-ISNT (I≥S≥N≥T) is **dropped** — degenerate (nasal steps fail in ~1/3 of normals) — "
        "but the **IST variant (I≥S≥T) is an adopted clause**, and inter-eye asymmetry ≥ 0.20 was "
        "evaluated and dropped (a search candidate, not adopted); (5) a rules "
        "label on expert contours here vs a trained fundus-image classifier in deployment.",
    ]

    lines += [
        "",
        "## Per-metric ROC-AUC vs the clinician label",
        "```",
        metric_auc_table(df).round(3).to_string(index=False),
        "```",
        "> Youden columns are the *data-optimal* threshold on THIS label — reported only as an "
        "optimistic/circular sensitivity check. The derived label itself uses fixed, cited "
        "clinical cut-offs, never these.",
        "",
        "## Metric distribution by clinician class",
        "```",
        metric_distribution_by_class(df).round(3).to_string(index=False),
        "```",
    ]

    healthy = df[df["diagnosis"] == "healthy"]
    sect = {s: round(pd.to_numeric(healthy[f"isnt_{s.lower()}"], errors="coerce").mean(), 1)
            for s in ("I", "S", "N", "T")}
    viol = (pd.Series(df["isnt_violation"]).astype("boolean")
            .groupby(df["diagnosis"], observed=True).mean().round(3))
    lines += [
        "",
        "## ISNT rule (finding)",
        f"- Healthy mean sector rim width (px): I {sect['I']} · S {sect['S']} · N {sect['N']} · "
        f"T {sect['T']} — **nasal ({sect['N']}) ≈ superior ({sect['S']})**, so the strict rule's "
        "S≥N step fails constantly; inferior is thickest and temporal thinnest.",
        f"- Strict I≥S≥N≥T violation rate by clinician class: {viol.to_dict()} — near-universal "
        "(healthy included), barely discriminating (AUC ≈ 0.54), consistent with Poon 2017 (AJO).",
        "- **Fix (adopted):** the literature-recommended **IST rule (I≥S≥T)** drops the "
        "nasal steps and roughly doubles specificity (Higashide 2022; Poon 2017; Maupin 2020); floored at "
        "vcdr ≥ 0.50 it is one of the **two adopted clauses** (with the disc-size-adjusted cup) on the "
        "per-expert union read — the strict full-ISNT stays dropped.",
    ]

    disc_by_class = (pd.to_numeric(df["disc_area"], errors="coerce")
                     .groupby(df["diagnosis"], observed=True).mean().round(1))
    lines += [
        "",
        "## Disc-size caveat",
        "CDR must be read against disc size (larger discs have larger cups; Blue Mountains Eye "
        "Study). Mean `disc_area` (px²) by clinician class — inspect for size confounding:",
        "```",
        disc_by_class.to_string(),
        "```",
    ]

    kappa, k_eyes = inter_expert_kappa(df)
    lines += [
        "",
        "## Inter-expert reliability",
        f"- Cohen's κ between the two experts' single-eye derived labels: {kappa:.3f} "
        f"(n={k_eyes} eyes scorable by both experts). A floor on how reproducible the morphology "
        "label can be, and the ceiling the OR/mean variants operate under.",
    ]

    lines += [
    ]

    # --- label-definition experiment: choosing the dx-independent suspicious-disc target ---
    # The search now spans the full candidate pool under both expert-combine reads; the legacy tables
    # below show the per-expert-union ("or") read — which IS the adopted read (the adopted label is the
    # "or" combine; the "mean" read is the lower-sensitivity alternative, see the leaderboard/scatter).
    search = label_combination_search(df)
    search = search[search["combine"] == "or"].reset_index(drop=True)
    _ABBR = {
        "vcdr>=0.70": "V70", "vcdr>=0.60": "V60", "area-cdr>=0.70": "A70", "hcdr>=0.70": "H70",
        "asymmetry>=0.20": "ASY", "asymmetry>=0.20 & vcdr>=0.5": "ASYf5",
        "disc-size-adjusted cup": "DSC", "IST violation & vcdr>=0.5": "IST",
        "full-ISNT violation": "ISNT", "IS violation": "IS", "rim-disc-area<=p2.5": "RDA",
        "min-rim-disc<=p5": "MRD", "DDLS-adjusted narrowest rim": "DDLS", "focal pole rim<0.1DD": "FOC",
    }

    def _ab(s: str) -> str:
        return "+".join(_ABBR.get(c, c) for c in s.split(" ∪ "))

    def _view(frame: pd.DataFrame) -> str:
        v = frame.copy()
        v["clauses"] = v["clauses"].map(_ab)
        cols = ["clauses", "n_clauses", "sensitivity", "specificity", "youden_j", "f1",
                "healthy_fp", "interexpert_kappa"]
        return v[cols].round(3).to_string(index=False)

    base = search[search["baseline"] != ""].assign(clauses=lambda d: d["baseline"])
    # simplest rule at each distinct (sens, spec) operating point, best Youden first
    uniq = (search.assign(_k=list(zip(search["sensitivity"].round(3), search["specificity"].round(3))))
            .sort_values(["_k", "n_clauses"]).drop_duplicates("_k")
            .sort_values("youden_j", ascending=False))
    floor = uniq[uniq["meets_spec_floor"]].sort_values("sensitivity", ascending=False)
    lines += [
        "",
        "## Label-definition experiment — choosing the dx-independent suspicious-disc target",
        "For an image model that predicts **“is this disc suspicious?”** the target must be a "
        "**binary, diagnosis-independent** structural label. This searches all binary unions of the "
        "cited gold + sensitive clauses (below) and scores each **against the clinician diagnosis** "
        "under **patient-level 5-fold CV** (every clause now uses a fixed EXTERNAL threshold — the "
        "rim/disc healthy limits are Chákṣu-derived — so there is no in-sample re-derivation). The "
        "diagnosis is only the **yardstick** for "
        "picking a clinically-useful rule — the chosen label is a fixed rule applied **without** the "
        "diagnosis, so there is no leakage for the image classifier.",
        "**Chosen target** (`features.fundus_suspicious_disc_label`): **DSC ∪ IST** — the "
        "disc-size-adjusted cup (Blue Mountains/Jonas; external Chákṣu normal limit) unioned with "
        "the IST rim rule (I≥S≥T & vcdr ≥ 0.50; Higashide 2022), evaluated on the **per-expert union "
        "(`or`) read** (fire if EITHER grader's contours trip a clause). The flat ISGEO CDR cuts "
        "(V70/A70) are **subsumed** by the disc-size cup (they add no positives); the focal-min-rim "
        "(in-sample), flat-cup≥0.60 (redundant), larger-eye-asymmetry (non-standard), inter-eye "
        "asymmetry ≥ 0.20 (floored & unfloored) and the strict full-ISNT rim rule were **evaluated and "
        "pruned** from the adopted label (they remain search candidates). We adopt the per-expert union "
        "as the operating point at the label's maximum patient-level glaucoma sensitivity — it **recovers "
        "RET042**, a single grader's borderline trace the averaged read dilutes away; the "
        "**expert-averaged (mean) consensus read** is the lower-sensitivity alternative we do **not** "
        "adopt (it misses RET042). The bilateral routing floor is therefore **26 (24 suspect + 2 "
        "glaucoma)** — the residual RET026 bilateral NTG core is all that no contour rule reaches (see "
        "“Why false negatives persist after routing”).",
        f"Clause codes: {', '.join(f'{v}={k}' for k, v in _ABBR.items())}.",
        "",
        "**Named baselines:**",
        "```",
        _view(base),
        "```",
        "**Best rules by Youden J** (simplest rule shown per distinct sensitivity/specificity point):",
        "```",
        _view(uniq.head(8)),
        "```",
        f"**Most sensitive rules at specificity ≥ 0.80** (n={int(floor.shape[0])} combinations clear "
        "the floor):",
        "```",
        _view(floor.head(8)),
        "```",
        "",
        "![Label-combination search — disc sensitivity vs specificity of every clause union (out-of-fold "
        "5-fold CV); Pareto frontier + spec ≥ 0.80 floor + adopted/gold baselines](figures/fundus_label_search_pareto.png)",
        "*The search space the adopted label was chosen from — each point is one clause-union candidate "
        "(`or` vs `mean` expert-combine); the adopted read (DSC ∪ IST, **`or`** per-expert union) reaches "
        "higher sensitivity than the `mean` (expert-averaged) alternative.*",
        "> **Read honestly.** The label is a **geometry-only proxy** — it is blind to disc haemorrhage, "
        "RNFL defects, peripapillary atrophy and pallor (features an image model could still learn if a "
        "richer label were available), and CDR-based reads have a known accuracy ceiling (~70% sens at "
        "high spec; Mowatt 2008 / Michelessi 2015). Inter-expert κ ≈ 0.67 (gold) bounds achievable "
        "agreement. Metrics are patient-clustered and out-of-fold; `healthy_fp` = healthy eyes flagged "
        "(the physiological-cup specificity cost). Full ranking: `reports/label_combination_search.csv`.",
        "",
        "## Clinical standards for the derived-label thresholds",
        "1. Foster PJ et al. Br J Ophthalmol 2002;86:238 (ISGEO) — Category I case-definition: VCDR ≥ 0.7 "
        "(97.5th-percentile; area-CDR analogue used too) **and inter-eye VCDR asymmetry ≥ 0.20**.",
        "2. Soh ZD et al. Clin Exp Ophthalmol 2020;48:1210 — corroborates inter-eye VCDR asymmetry ≥ 0.20 "
        "as a referral-refinement criterion.",
        "3. Harizman N et al. Arch Ophthalmol 2006;124:1579 (ISNT rule; Jonas 1988) — intact in "
        "79% normal vs 28% glaucoma → specific but insensitive, used only as a support clause. "
        "Poon LY et al. Am J Ophthalmol 2017 — the strict quantitative rule holds in only a "
        "minority of normal disc photos (context for the high violation rate above).",
        "4. Blue Mountains Eye Study (Healey et al.) — disc-diameter dependence of VCDR.",
    ]
    return "\n".join(lines)


def make_figures(df: pd.DataFrame, out_dir: Path, search: pd.DataFrame | None = None) -> list[Path]:
    """Write the experiment's PNG figures; returns the paths written. ``search`` is the
    :func:`label_combination_search` result — pass the already-computed frame to avoid re-running the
    CV search for the Pareto-scatter figure; if ``None`` it is computed here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    classes = [c for c in ("healthy", "suspect", "glaucoma") if c in set(df["diagnosis"])]
    y = features.fundus_structural_label(df)

    # 1. metric distribution by clinician class (boxplots)
    metrics = list(METRICS)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        data = [pd.to_numeric(df.loc[df["diagnosis"] == c, m], errors="coerce").dropna()
                for c in classes]
        ax.boxplot(data, tick_labels=classes)
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Disc morphometry by clinician class")
    fig.tight_layout()
    p = out_dir / "fundus_metric_by_class.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 2. ROC curves for continuous metrics
    fig, ax = plt.subplots(figsize=(5, 5))
    for m, higher in METRICS.items():
        raw = pd.to_numeric(df[m], errors="coerce")
        oriented = (raw if higher else -raw)
        fpr, tpr = _roc_curve(oriented, y)
        ax.plot(fpr, tpr, label=f"{m} (AUC {roc_auc(oriented, y):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title("Metric ROC vs clinician label")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fundus_metric_roc.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 3. derived × clinician 3-class stacked bar — both expert-combination variants
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (name, key) in zip(np.atleast_1d(axes), [("mean", "mean"), ("OR", "or")]):
        tab = derived_by_clinician_class(df, combine=key)
        bottom = np.zeros(len(tab.columns))
        for lab in tab.index:
            ax.bar(tab.columns, tab.loc[lab].to_numpy(), bottom=bottom, label=f"derived={lab}")
            bottom += tab.loc[lab].to_numpy()
        ax.set_title(f"derived [{name}] within clinician class")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("eyes")
    fig.tight_layout()
    p = out_dir / "fundus_derived_by_class.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 4. Stage-1 route composition by clinician class (structure + IOP + field): gold (OR) vs sensitive
    route_colors = {"GECFS": "#B5410E", "GRMS": "#E08E45", "discharge": "#4C9A6B", "referrer": "#888"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    panels = [("gold (OR)", dict(profile="gold", combine="or", use_field=True)),
              ("sensitive", dict(profile="sensitive", use_field=True))]
    for ax, (title, kw) in zip(np.atleast_1d(axes), panels):
        tab = routing_crosstab(df, **kw)
        bottom = np.zeros(len(tab.columns))
        for r in tab.index:
            ax.bar(tab.columns, tab.loc[r].to_numpy(), bottom=bottom,
                   label=r, color=route_colors.get(r))
            bottom += tab.loc[r].to_numpy()
        ax.set_title(f"Stage-1 route ({title})")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("eyes")
    fig.tight_layout()
    p = out_dir / "fundus_stage1_routing.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 5. per-class FNR by decision (the headline safety chart) — lower is safer
    tab = fnr_table(df).set_index("decision")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tab))
    w = 0.38
    ax.bar(x - w / 2, tab["glaucoma_FNR"], w, label="glaucoma FNR", color="#B5410E")
    ax.bar(x + w / 2, tab["suspect_FNR"], w, label="suspect FNR", color="#E08E45")
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("false-negative rate")
    ax.set_title("Suspect/glaucoma FNR by decision (lower = safer)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fundus_fnr_by_decision.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 6. label-combination search — sensitivity vs specificity of every clause-union candidate (out-of-fold
    # 5-fold CV), Pareto frontier + adopted/gold baselines. THREE views: disc, classifier per-eye, classifier
    # bilateral. Visualises the search space the adopted label was chosen from.
    s = label_combination_search(df) if search is None else search
    n = len(s)

    def _pareto_fig(xcol: str, ycol: str, fname: str, xlabel: str, title: str, floor: float | None) -> None:
        fig, ax = plt.subplots(figsize=(6.6, 5))
        for cmb, col, mk in [("or", "#0C6A72", "o"), ("mean", "#B0761A", "^")]:
            sub = s[s["combine"] == cmb].dropna(subset=[xcol, ycol])
            ax.scatter(sub[xcol], sub[ycol], s=13, c=col, marker=mk, alpha=0.4,
                       edgecolors="none", label=f"{cmb} union ({len(sub)})")
        order = s.dropna(subset=[xcol, ycol]).sort_values(xcol, ascending=False)[[xcol, ycol]].to_numpy()
        front, best = [], -1.0
        for xv, yv in order:                       # upper (spec,sens)-Pareto envelope
            if yv > best:
                front.append((xv, yv))
                best = yv
        front = np.array(front)
        ax.plot(front[:, 0], front[:, 1], "-", color="#333", lw=1.3, zorder=4, label="Pareto frontier")
        if floor is not None:
            ax.axvline(floor, ls="--", color="#999", lw=1, label=f"spec ≥ {floor:.2f} floor")
        for lbl, colr in [("adopted", "#C0433B"), ("gold", "#2C8A69")]:
            b = s[(s["baseline"] == lbl) & (s["combine"] == "or")]
            if len(b):
                r = b.iloc[0]
                ax.scatter([r[xcol]], [r[ycol]], s=110, marker="*", c=colr, zorder=6,
                           edgecolors="white", linewidths=0.7)
                ax.annotate(f" {lbl}", (r[xcol], r[ycol]), fontsize=8, va="center",
                            fontweight="bold", color=colr)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sensitivity (out-of-fold)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        pth = out_dir / fname
        fig.savefig(pth, dpi=120)
        plt.close(fig)
        paths.append(pth)

    _pareto_fig("specificity", "sensitivity", "fundus_label_search_pareto.png",
                "specificity (out-of-fold)", f"Label-combination search — {n} clause unions (disc vs glaucoma)", 0.80)
    return paths


def _roc_curve(scores: pd.Series, labels: pd.Series):
    """(fpr, tpr) arrays over descending thresholds for plotting."""
    s = pd.to_numeric(scores, errors="coerce")
    y = pd.to_numeric(labels, errors="coerce")
    ok = s.notna() & y.notna()
    s, y = s[ok].to_numpy(float), y[ok].to_numpy(float)
    n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1])
    thr = np.unique(s)[::-1]
    tpr = [0.0]
    fpr = [0.0]
    for tv in thr:
        pred = s >= tv
        tpr.append(float(np.sum(pred & (y == 1)) / n_pos))
        fpr.append(float(np.sum(pred & (y == 0)) / n_neg))
    return np.array(fpr), np.array(tpr)
