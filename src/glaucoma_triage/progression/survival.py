"""Time-axis for GRAPE progression — landmark dynamic prediction + a transparent time-to-event endpoint.

GRAPE ships only *binary* progression flags (PLR2/PLR3/MD) computed by an undocumented criterion that is
**not reproducible** from the released per-visit VF (best-effort concordance ≈0.83), and gives **no event
times**. A survival / time-axis analysis therefore has to *construct* its endpoint from the VF series. Two
complementary, honest constructions live here:

**PRIMARY — continuous future-MD landmark dynamic prediction** (well-powered: every eye contributes, no
rare-event bottleneck). At a landmark ``L`` (using only visits ``1..L``) we predict the eye's *future*
mean sensitivity (last visit), and read cross-validated R² as a function of (a) how much visit history is
used and (b) which modality set — the temporal face of the input-sufficiency question ("does accumulating
history help, and does the modality that carries the signal change with history length?").

**SECONDARY — transparent PLR time-to-event.** A pre-specified, literature-standard pointwise-linear-
regression event (a location progresses if its sensitivity slope ≤ ``slope_thr`` dB/yr at ``p < p_thr``; an
eye progresses once ``≥ min_points`` locations do), recomputed reproducibly with the event *time* = first
visit the criterion is met (else censored at the last visit). We report its concordance with GRAPE's
shipped flag (imperfect by construction) and fit a Cox model with Harrell's c-index + time-dependent AUC.
This is *not* GRAPE's official label — it is a transparent, reproducible surrogate with recoverable timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# baseline structural/clinical modalities (fixed at visit 1) vs VF trajectory summarised up to landmark L
BASELINE_STRUCT = ["fundus_vcdr", "rnfl_mean", "iop", "cct", "age"]
VF_HISTORY = ["vf_now", "vf_slope_hist", "vf_sd_hist"]
IOP_HISTORY = ["iop_now", "iop_slope_hist"]
FEATURE_SETS = {
    "baseline_struct": BASELINE_STRUCT,
    "vf_recency": ["vf_now"],                 # only the most-recent VF value (isolates recency)
    "vf_history": VF_HISTORY,                 # recent value + trajectory slope + variability
    "iop_history": IOP_HISTORY,               # IOP trajectory (also repeats in GRAPE)
    "all_history": VF_HISTORY + IOP_HISTORY,  # both longitudinal modalities (NO fundus — gated out)
    "both": BASELINE_STRUCT + VF_HISTORY,
    "fundus_only": ["fundus_vcdr"],
    "oct_only": ["rnfl_mean"],
}


# --------------------------------------------------------------------------- landmark (continuous) ---
def _reg() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])


def grouped_oof_r2(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, *, k: int = 5) -> float:
    """Patient-grouped pooled out-of-fold R² (Ridge). NaN if too few patient groups to split."""
    y = np.asarray(y, dtype=float)
    n_splits = min(k, len(np.unique(groups)))
    if n_splits < 2 or len(y) < n_splits:
        return float("nan")
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pred[te] = _reg().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return float(r2_score(y, pred))


def landmark_dataset(followup: pd.DataFrame, baseline: pd.DataFrame, landmark: int) -> pd.DataFrame:
    """One row per eye with ``≥ landmark+1`` visits: VF history over visits ``1..landmark`` + baseline
    structure + the future outcome (mean sensitivity at the last visit)."""
    rows = []
    for eid, g in followup.sort_values("visit").groupby("eye_id"):
        g = g.reset_index(drop=True)
        if len(g) < landmark + 1:                        # need at least one future visit beyond L
            continue
        hist = g.iloc[:landmark]
        t = pd.to_numeric(hist["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(hist["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        slope = float(np.polyfit(t[ok], s[ok], 1)[0]) if ok.sum() >= 2 else 0.0
        # IOP history over the same landmark window (IOP also repeats in GRAPE)
        io = pd.to_numeric(hist["iop"], errors="coerce").to_numpy() if "iop" in hist else np.full(len(t), np.nan)
        oki = np.isfinite(t) & np.isfinite(io)
        iop_slope = float(np.polyfit(t[oki], io[oki], 1)[0]) if oki.sum() >= 2 else 0.0
        # future outcomes: LEVEL = last-visit mean sensitivity; RATE = full-series MD-slope
        sf = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        tf = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        okf = np.isfinite(sf) & np.isfinite(tf)
        rows.append({
            "eye_id": eid, "patient_id": str(eid).split("_")[0],
            "vf_now": float(s[ok][-1]) if ok.any() else np.nan,
            "vf_slope_hist": slope, "vf_sd_hist": float(np.std(s[ok])) if ok.any() else 0.0,
            "iop_now": float(io[oki][-1]) if oki.any() else np.nan, "iop_slope_hist": iop_slope,
            "y_future": float(sf[okf][-1]) if okf.any() else np.nan,
            "y_rate": float(np.polyfit(tf[okf], sf[okf], 1)[0]) if okf.sum() >= 2 else np.nan,
        })
    lm = pd.DataFrame(rows)
    cols = ["eye_id"] + [c for c in BASELINE_STRUCT if c in baseline.columns]
    return lm.merge(baseline[cols], on="eye_id", how="left")


def landmark_curve(followup: pd.DataFrame, baseline: pd.DataFrame, *,
                   landmarks=(1, 2, 3), k: int = 5, fixed_cohort: bool = True,
                   target: str = "level") -> pd.DataFrame:
    """R² of predicting future mean sensitivity by (landmark × modality feature set) — the time-axis.

    ``fixed_cohort=True`` (default) restricts EVERY landmark to the same eyes (those with ≥ max(landmarks)+1
    visits) so a rising R² reflects *more history*, not a shifting/shrinking cohort — the confound in the
    earlier variable-cohort sweep. The prediction outcome is held fixed per eye (mean sensitivity at that
    eye's last visit); note that ``vf_now`` at a larger L is also temporally closer to the outcome, so the
    curve mixes 'more history' with 'more recent measurement' (recency) — read them together, not as pure
    history-length. ``fixed_cohort=False`` reproduces the old per-L maximal cohort."""
    out = []
    cohort_ids = None
    if fixed_cohort:
        need = max(landmarks) + 1
        vc = followup.groupby("eye_id").size()
        cohort_ids = set(vc[vc >= need].index)
    for L in landmarks:
        lm = landmark_dataset(followup, baseline, L)
        if cohort_ids is not None:
            lm = lm[lm["eye_id"].isin(cohort_ids)].reset_index(drop=True)
        ycol = "y_rate" if target == "rate" else "y_future"     # LEVEL (autocorrelated) vs RATE (progression)
        lm = lm[lm[ycol].notna()].reset_index(drop=True)
        y = lm[ycol].to_numpy(dtype=float)
        groups = lm["patient_id"].to_numpy()
        row = {"landmark": L, "n_eyes": len(lm)}
        for name, cols in FEATURE_SETS.items():
            X = lm[cols].apply(pd.to_numeric, errors="coerce")
            row[name] = grouped_oof_r2(X, y, groups, k=k)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------- HONEST serial EVENT prediction (landmark) ---
# The novel headline: a *non-circular* accuracy-vs-#visits curve for a progression EVENT. Input features
# come only from the early visits ``1..L``; the label is a transparent MD-slope progression event computed
# on the DISJOINT, held-out FUTURE window ``L+1..end`` — so the input is never part of the label window
# (unlike the field's serial models, which feed the whole VF series that DEFINES the label ⇒ circular ≈0.9).
EVENT_FEATURE_SETS = {
    "vf_history": VF_HISTORY,                                        # recent VF + trajectory + variability
    "iop_history": IOP_HISTORY,                                      # IOP trajectory (also repeats)
    "baseline_struct": BASELINE_STRUCT,                              # baseline fundus/OCT/IOP/CCT/age
    "all": list(dict.fromkeys(VF_HISTORY + IOP_HISTORY + BASELINE_STRUCT)),
}


def landmark_event_dataset(followup: pd.DataFrame, baseline: pd.DataFrame, landmark: int, *,
                           slope_thr: float = 0.0, p_thr: float = 0.05,
                           min_future: int = 3) -> pd.DataFrame:
    """One row per eye with ``≥ landmark + min_future`` visits (the honest serial-event design).

    INPUT features are summarised over the early visits ``1..landmark`` ONLY (VF recent value / trajectory
    slope / variability + IOP recent value / slope) plus baseline structure — none of which touches the
    label window. The LABEL is a transparent MD-slope progression **event** (:func:`md_slope_progression`)
    recomputed on the DISJOINT held-out FUTURE window (visits ``landmark+1..end``); ``min_future`` (≥3) future
    visits are required for a valid regression p-value. This is the non-circular counterpart to the field's
    serial models (which feed the whole VF series that *defines* the label)."""
    if min_future < 3:
        raise ValueError("min_future must be ≥3 (MD-slope event needs ≥3 future visits for a valid p-value)")
    rows = []
    for eid, g in followup.sort_values("visit").groupby("eye_id"):
        g = g.reset_index(drop=True)
        if len(g) < landmark + min_future:                 # need L input visits + ≥min_future future visits
            continue
        hist, fut = g.iloc[:landmark], g.iloc[landmark:]
        t = pd.to_numeric(hist["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(hist["vf_mean_sens"], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(s)
        slope = float(np.polyfit(t[ok], s[ok], 1)[0]) if ok.sum() >= 2 else 0.0
        io = pd.to_numeric(hist["iop"], errors="coerce").to_numpy() if "iop" in hist else np.full(len(t), np.nan)
        oki = np.isfinite(t) & np.isfinite(io)
        iop_slope = float(np.polyfit(t[oki], io[oki], 1)[0]) if oki.sum() >= 2 else 0.0
        # LABEL = MD-slope progression event on the disjoint future window (visits L+1..end)
        tf = pd.to_numeric(fut["interval_years"], errors="coerce").to_numpy()
        sf = pd.to_numeric(fut["vf_mean_sens"], errors="coerce").to_numpy()
        ev, _ = md_slope_progression(tf, sf, slope_thr=slope_thr, p_thr=p_thr)
        rows.append({
            "eye_id": eid, "patient_id": str(eid).split("_")[0],
            "vf_now": float(s[ok][-1]) if ok.any() else np.nan,
            "vf_slope_hist": slope, "vf_sd_hist": float(np.std(s[ok])) if ok.any() else 0.0,
            "iop_now": float(io[oki][-1]) if oki.any() else np.nan, "iop_slope_hist": iop_slope,
            "event": int(ev), "n_future": int(len(fut)),
        })
    lm = pd.DataFrame(rows)
    if lm.empty:
        return lm
    cols = ["eye_id"] + [c for c in BASELINE_STRUCT if c in baseline.columns]
    return lm.merge(baseline[cols], on="eye_id", how="left")


def _event_oof_proba(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                     k: int = 5, n_repeats: int = 3, seed: int = 0) -> np.ndarray | None:
    """Patient-grouped, repeated-CV pooled out-of-fold positive-class probability (regularized, balanced
    logistic). Returns ``None`` if the sample can't be split into ≥2 stratified patient-disjoint folds."""
    y = np.asarray(y, dtype=int)
    n_splits = min(k, int(y.sum()), int((y == 0).sum()), len(np.unique(groups)))
    if n_splits < 2:
        return None
    acc = np.zeros(len(y))
    for r in range(n_repeats):
        pred = np.full(len(y), np.nan)
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            folds = list(splitter.split(X, y, groups))
        except ValueError:
            return None
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2:
                pred[te] = y[tr].mean()
                continue
            pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler()),
                             ("clf", LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000))])
            pipe.fit(X[tr], y[tr])
            pred[te] = pipe.predict_proba(X[te])[:, 1]
        nan = np.isnan(pred)                                 # any eye never held out (degenerate) → prior
        if nan.any():
            pred[nan] = y.mean()
        acc += pred
    return acc / n_repeats


def _auc_cluster_ci(p: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                    n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUC + patient-cluster bootstrap 95% CI on FIXED out-of-fold probabilities (resample patients)."""
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {gg: np.where(groups == gg)[0] for gg in uniq}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[gg] for gg in pick])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        draws.append(roc_auc_score(yy, p[rows]))
    if not draws:
        return auc, float("nan"), float("nan")
    return auc, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def landmark_event_curve(followup: pd.DataFrame, baseline: pd.DataFrame, *,
                         landmarks=(1, 2, 3), min_future: int = 3, slope_thr: float = 0.0,
                         p_thr: float = 0.05, feature_sets: dict | None = None, k: int = 5,
                         n_repeats: int = 3, n_boot: int = 1000, seed: int = 0,
                         fixed_cohort: bool = True) -> pd.DataFrame:
    """The HONEST accuracy-vs-#visits EVENT curve: patient-grouped OOF AUC [95% CI] of predicting the
    held-out future-window MD-slope event from the first ``L`` visits, per (landmark × modality set).

    ``fixed_cohort=True`` restricts every landmark to the eyes with ``≥ max(landmarks)+min_future`` visits
    so the L-comparison is within the same eyes (a rising AUC reflects *more input visits*, not a shifting
    cohort). ``False`` uses each L's maximal eligible cohort (more power at small L; report n/events per L).
    Returns one row per landmark: ``n_eyes``, ``n_events``, and ``<set>`` / ``<set>_lo`` / ``<set>_hi``."""
    if feature_sets is None:
        feature_sets = EVENT_FEATURE_SETS
    cohort_ids = None
    if fixed_cohort:
        need = max(landmarks) + min_future
        vc = followup.groupby("eye_id").size()
        cohort_ids = set(vc[vc >= need].index)
    out = []
    for L in landmarks:
        lm = landmark_event_dataset(followup, baseline, L, slope_thr=slope_thr,
                                    p_thr=p_thr, min_future=min_future)
        if not lm.empty and cohort_ids is not None:
            lm = lm[lm["eye_id"].isin(cohort_ids)].reset_index(drop=True)
        n_eyes = int(len(lm))
        n_events = int(lm["event"].sum()) if n_eyes else 0
        row = {"landmark": L, "n_eyes": n_eyes, "n_events": n_events}
        if n_events >= 3 and (n_eyes - n_events) >= 3:      # need both classes with a few each to score
            y = lm["event"].to_numpy(int)
            groups = lm["patient_id"].to_numpy()
            for name, cols in feature_sets.items():
                X = lm[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
                p = _event_oof_proba(X, y, groups, k=k, n_repeats=n_repeats, seed=seed)
                if p is None:
                    row[name] = row[f"{name}_lo"] = row[f"{name}_hi"] = float("nan")
                    continue
                auc, lo, hi = _auc_cluster_ci(p, y, groups, n_boot=n_boot, seed=seed)
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = auc, lo, hi
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------- transparent PLR time-to-event ---
def plr_progression(t: np.ndarray, P: np.ndarray, *, slope_thr: float = -1.0,
                    p_thr: float = 0.01, min_points: int = 2) -> tuple[bool, float]:
    """Transparent pointwise-linear-regression event: (progressed, time).

    A location progresses if its sensitivity slope ≤ ``slope_thr`` (dB/yr) at ``p < p_thr`` over the visits
    seen so far; the eye's event time is the first visit at which ``≥ min_points`` locations qualify. If it
    never happens the eye is censored at its last visit time. −1 sentinels are masked per location."""
    order = np.argsort(t)
    t, P = t[order], P[order]
    for L in range(2, len(t) + 1):
        tt, PP = t[:L], P[:L]
        count = 0
        for j in range(PP.shape[1]):
            y = PP[:, j]
            m = np.isfinite(y) & (y >= 0) & np.isfinite(tt)
            if m.sum() < 2:
                continue
            lr = stats.linregress(tt[m], y[m])
            if lr.slope <= slope_thr and lr.pvalue < p_thr:
                count += 1
        if count >= min_points:
            return True, float(t[L - 1])
    return False, float(t[-1])


def plr_event_table(points: dict[str, tuple], **kw) -> pd.DataFrame:
    """Per-eye transparent PLR (event, time) over ``{eye_id: (t, P)}`` from ``grape.followup_vf_points``.

    NOTE: on GRAPE's short (3–9-visit), high-test-retest-variability Octopus series the pointwise criterion
    is degenerate — ≥2 of 61 locations cross "significant" by chance for almost every eye — so it flags ~all
    eyes (spec≈0). Kept for completeness / clean-data tests; the GRAPE survival endpoint uses MD-slope
    (:func:`md_slope_event_table`) instead, which is far better behaved and concords ≈0.93 with the shipped MD flag."""
    recs = []
    for eid, (t, P) in points.items():
        ev, tm = plr_progression(np.asarray(t, dtype=float), np.asarray(P, dtype=float), **kw)
        recs.append({"eye_id": eid, "patient_id": str(eid).split("_")[0], "event": bool(ev), "time": tm})
    return pd.DataFrame(recs)


def md_slope_progression(t: np.ndarray, s: np.ndarray, *, slope_thr: float = 0.0,
                         p_thr: float = 0.05) -> tuple[bool, float]:
    """Transparent MD-slope event: (progressed, time). The eye progresses at the first visit ``L`` where the
    cumulative regression of mean sensitivity on time (visits ``1..L``) has slope ``< slope_thr`` at
    ``p < p_thr`` (Octopus sensitivity: higher = better, so worsening ⇒ negative slope). Else censored at
    the last visit. Behaves sensibly on short noisy series, unlike the pointwise PLR."""
    order = np.argsort(t)
    t, s = t[order], s[order]
    m0 = np.isfinite(t) & np.isfinite(s)
    t, s = t[m0], s[m0]
    for L in range(3, len(t) + 1):            # need ≥3 points for a valid p-value (2 points → 0 resid df)
        lr = stats.linregress(t[:L], s[:L])
        if lr.slope < slope_thr and lr.pvalue < p_thr:
            return True, float(t[L - 1])
    return (False, float(t[-1])) if len(t) else (False, 0.0)


def md_slope_event_table(followup: pd.DataFrame, **kw) -> pd.DataFrame:
    """Per-eye transparent MD-slope (event, time) from the long follow-up frame (uses ``vf_mean_sens``)."""
    recs = []
    for eid, g in followup.sort_values("visit").groupby("eye_id"):
        t = pd.to_numeric(g["interval_years"], errors="coerce").to_numpy()
        s = pd.to_numeric(g["vf_mean_sens"], errors="coerce").to_numpy()
        ev, tm = md_slope_progression(t, s, **kw)
        recs.append({"eye_id": eid, "patient_id": str(eid).split("_")[0], "event": bool(ev), "time": tm})
    return pd.DataFrame(recs)


def event_concordance(events: pd.DataFrame, shipped: pd.Series) -> dict:
    """Agreement of the transparent 'ever-progressed' flag with GRAPE's shipped binary flag."""
    e = events.set_index("eye_id")["event"].astype(int)
    g = shipped.reindex(e.index).fillna(0).astype(int)
    tp = int(((e == 1) & (g == 1)).sum()); fp = int(((e == 1) & (g == 0)).sum())
    fn = int(((e == 0) & (g == 1)).sum()); tn = int(((e == 0) & (g == 0)).sum())
    pos_g, pos_e = int((g == 1).sum()), int((e == 1).sum())
    return {"concordance": (tp + tn) / len(e), "sens": tp / pos_g if pos_g else float("nan"),
            "spec": tn / (tn + fp) if (tn + fp) else float("nan"),
            "n_shipped_pos": pos_g, "n_transparent_pos": pos_e, "n": len(e)}


def cox_time_dependent(events: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str], *,
                       horizons=None, k: int = 5, seed: int = 0) -> dict:
    """Patient-grouped Cox on ``feature_cols``: pooled OOF Harrell c-index + time-dependent AUC at horizons.

    Returns ``{c_index, td_auc: {horizon: auc}, n, n_events}``. Uses scikit-survival; risk scores are pooled
    out-of-fold (patient-disjoint ``GroupKFold``), and time-dependent AUC is computed against the pooled
    train survival for the censoring estimate."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
    from sksurv.util import Surv

    d = events.merge(features, on="eye_id", how="inner").dropna(subset=feature_cols).reset_index(drop=True)
    ev = d["event"].to_numpy(bool)
    tm = d["time"].to_numpy(float)
    X = d[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    groups = d["patient_id"].to_numpy()
    n_splits = min(k, len(np.unique(groups)))
    y_all = Surv.from_arrays(event=ev, time=tm)

    risk = np.full(len(d), np.nan)
    if n_splits >= 2:
        scaler_cols = X.columns
        for tr, te in GroupKFold(n_splits=n_splits).split(X, tm, groups):
            if len(np.unique(ev[tr])) < 2:
                continue
            mu, sd = X.iloc[tr].mean(), X.iloc[tr].std().replace(0, 1.0)
            Xtr = ((X.iloc[tr] - mu) / sd)[scaler_cols]
            Xte = ((X.iloc[te] - mu) / sd)[scaler_cols]
            model = CoxPHSurvivalAnalysis(alpha=1.0)
            model.fit(Xtr.to_numpy(), Surv.from_arrays(event=ev[tr], time=tm[tr]))
            risk[te] = model.predict(Xte.to_numpy())
    ok = np.isfinite(risk)
    c_index = float("nan")
    if ok.sum() > 1 and len(np.unique(ev[ok])) == 2:
        c_index = float(concordance_index_censored(ev[ok], tm[ok], risk[ok])[0])

    td = {}
    if horizons is None:
        ev_times = tm[ev]
        horizons = list(np.quantile(ev_times, [0.5, 0.75])) if ev.any() else []
    if ok.sum() > 5 and ev[ok].any():
        # keep horizons strictly inside the observed follow-up of the scored set
        hz = [float(h) for h in horizons if tm[ok].min() < h < tm[ok].max()]
        if hz:
            try:
                auc, _ = cumulative_dynamic_auc(y_all[ok], y_all[ok], risk[ok], hz)
                td = {round(h, 2): float(a) for h, a in zip(hz, np.atleast_1d(auc))}
            except (ValueError, ArithmeticError):
                td = {}
    return {"c_index": c_index, "td_auc": td, "n": int(len(d)), "n_events": int(ev.sum())}
