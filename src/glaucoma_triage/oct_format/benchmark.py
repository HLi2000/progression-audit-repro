"""Anti-gaming tolerance certificate + leaderboard + clinical shift-taxonomy grid for OCT-FormatShift.

Certificate: the vf_format geometric-mean design (one zero axis -> score 0, so a constant reader can't
win) with the review fix — the stability reward is **gated behind detection RETENTION** (sens under shift
/ sens on the clean read): the shift must not HIDE disease relative to the clean call. Retention (not the
absolute sensitivity) is the honest gate — a specific normative-percentile call is inherently
low-sensitivity, so gating a harmoniser on the READER's absolute sensitivity would penalise it for the
reader's insensitivity, not for intolerance (docs/OCT_FORMATSHIFT_SCOPING.md §8; phase-review fix #4/#6).

Taxonomy grid (the clinically-actionable output): for each format/device shift tier, decide whether ONE
shared model is SAFE (read-change within the test-retest noise floor / MDC) or the device must be
SPECIALISED — the per-device deployment decision (§ SHIFT TAXONOMY in the plan; MDC ~4 um global,
PMC6882427).
"""

from __future__ import annotations

import numpy as np

from . import canonical, envelope, transforms

USEFULNESS_RATIO = 0.7
SENSITIVITY_FLOOR = 0.60   # min detection RETENTION under shift (sens_shift/sens_clean) to earn the stability reward
TOLERANCE_SCORE_FLOOR = 0.55
EARNED_MARGIN = 0.02
MDC_GLOBAL_UM = 4.0  # minimal-detectable-change of global RNFL (test-retest floor; PMC6882427)
_SENS_EPS = 1e-3     # clean detection below this -> the reader detects ~nothing even clean -> gate undefined

# The clinical shift taxonomy over the calibrated vendor pairs (severity-ordered; T3 < T4).
TAXONOMY = (
    ("T3_cirrus_spectralis", "cirrus_to_spectralis", 1.0),
    ("T4_cirrus_topcon", "cirrus_to_topcon", 1.0),
    ("T4_cirrus_rtvue", "cirrus_to_rtvue", 1.0),
)


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _recovery_axis(model_mae: float, constant_mae: float) -> float:
    if not np.isfinite(constant_mae) or constant_mae <= 0:
        return 0.0
    return _clamp01(1.0 - model_mae / constant_mae)


def capability_axes(*, recovery_global: float, recovery_focal: float, stability: float,
                    sensitivity: float) -> dict:
    """[0,1] axes + geometric-mean tolerance score; stability is GATED by ``sensitivity`` (the detection
    RETENTION under shift) clearing the floor — an intolerant read that HIDES disease earns no stability."""
    stability_gated = stability if (np.isfinite(sensitivity) and sensitivity >= SENSITIVITY_FLOOR) else 0.0
    axes = {"recovery_global": _clamp01(recovery_global), "recovery_focal": _clamp01(recovery_focal),
            "stability": _clamp01(stability_gated)}
    vals = np.array(list(axes.values()), dtype=float)
    score = float(np.exp(np.mean(np.log(np.clip(vals, 1e-9, 1.0)))))
    return {"axes": axes, "tolerance_score": score, "sensitivity": sensitivity}


def tolerance_certificate(*, cap: dict, useful: bool, earned: bool) -> dict:
    score = cap["tolerance_score"]
    if not useful:
        tier, label = 0, "not useful"
    elif useful and earned and score >= TOLERANCE_SCORE_FLOOR:
        tier, label = 2, "certified"
    else:
        tier, label = 1, "useful, not certified"
    return {"tier": tier, "tier_label": label, "certified": tier == 2,
            "tolerance_score": score, **cap["axes"], "sensitivity": cap["sensitivity"]}


def _sensitivity(model: envelope.InterpModel, abn_true: np.ndarray, X_shift: np.ndarray) -> float:
    pos = abn_true == 1
    if pos.sum() == 0:
        return float("nan")
    call = model.predict_abn(X_shift)
    return float(np.mean(call[pos] == 1))


def _sens_call_vector(model: envelope.InterpModel, abn_true: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Boolean 'called abnormal' over the POSITIVE (disease) eyes — the paired unit for bootstrapping retention."""
    pos = np.asarray(abn_true) == 1
    if pos.sum() == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(model.predict_abn(X)[pos] == 1)


def retention_ci(clean_pos_calls: np.ndarray, shift_pos_calls: np.ndarray, rng: np.random.Generator | None, *,
                 n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile-bootstrap CI for detection RETENTION (sens_shift/sens_clean) over the positive eyes.

    Resamples the paired (clean-call, shift-call) positive eyes → the CI on how much detection survives the
    shift. Used to REFUSE a hard tier when the gate is underpowered (CI straddles the floor)."""
    cp, sp = np.asarray(clean_pos_calls, dtype=float), np.asarray(shift_pos_calls, dtype=float)
    n = len(cp)
    if n == 0 or rng is None or n_boot <= 0 or len(sp) != n:
        return None
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sc = float(cp[idx].mean())
        if sc > _SENS_EPS:
            boots.append(min(float(sp[idx].mean()) / sc, 1.0))
    if not boots:
        return None
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def certify(*, mae: float, focal_mae: float, const_mae: float, const_focal: float, flip: float,
            sens_clean: float, sens_shift: float, earned: bool,
            retention_ci: tuple[float, float] | None = None) -> dict:
    """The ONE tolerance certificate used by every contender: recovery + stability, gated by detection
    RETENTION (sens_shift/sens_clean). If ``retention_ci`` straddles the floor the gate is underpowered →
    tier is capped to 'indeterminate' (refuse to certify on a noisy gate; phase-review fix #6)."""
    retention = (float(min(sens_shift / sens_clean, 1.0))
                 if (np.isfinite(sens_clean) and sens_clean > _SENS_EPS) else float("nan"))
    cap = capability_axes(recovery_global=_recovery_axis(mae, const_mae),
                          recovery_focal=_recovery_axis(focal_mae, const_focal),
                          stability=1.0 - flip, sensitivity=retention)
    useful = (mae <= USEFULNESS_RATIO * const_mae) and np.isfinite(sens_clean) and (sens_clean > _SENS_EPS)
    cert = tolerance_certificate(cap=cap, useful=bool(useful), earned=bool(earned))
    indeterminate = False
    if retention_ci is not None and useful:
        lo, hi = retention_ci
        if np.isfinite(lo) and np.isfinite(hi) and lo < SENSITIVITY_FLOOR <= hi:   # CI spans the gate floor
            indeterminate = True
            cert = {**cert, "tier": min(cert["tier"], 1), "certified": False,
                    "tier_label": "indeterminate (gate CI spans floor)"}
    cert.update({"sensitivity_clean": float(sens_clean), "sensitivity_shift": float(sens_shift),
                 "sensitivity_retention": retention, "indeterminate": indeterminate,
                 "retention_ci": [float(retention_ci[0]), float(retention_ci[1])] if retention_ci else None})
    return cert


def _arms(models: dict) -> dict:
    return {k: v for k, v in models.items() if not k.startswith("_")}


def run_benchmark(models: dict, clean_test_maps: np.ndarray, rng: np.random.Generator, *,
                  level: float = 1.0, pairs: tuple[str, ...] | None = None,
                  baseline_key: str = "baseline", glaucoma_labels: np.ndarray | None = None,
                  shift_fn=None, n_boot: int = 0) -> dict:
    """Score each model's tolerance ACROSS the calibrated vendor envelope + attach a certificate.

    The headline read is the format ENVELOPE, not a single pair: a Spectralis-only test would let every
    model pass (near-interchangeable), so recovery/stability are averaged over ``pairs`` (the whole
    calibrated vendor envelope by default). Within each pair all models see the SAME seeded shifted maps.
    Per-pair breakdown for the deployment decision lives in :func:`taxonomy_grid`.

    ``glaucoma_labels`` (0/1) makes the sensitivity gate a REAL detection sensitivity — of the truly-
    glaucomatous eyes, how many the model still flags on the SHIFTED read (structure->function under shift,
    consistent with the Eyened/RETFound contenders). Without it, sensitivity falls back to abnormal-CALL
    RETENTION vs the clean normative call (a self-consistency proxy — weaker; flagged in the scorecard).
    """
    pairs = pairs or transforms.CALIBRATED_PAIRS
    shift_fn = shift_fn or transforms.apply_format   # the shift axis (offset by default; seg-error etc. via override)
    arms = _arms(models)
    ref = models.get("_reference") or next(iter(arms.values())).reference
    truth = envelope.profile_true(clean_test_maps)
    abn = ref.abnormal_batch(truth)
    sens_pos = (np.asarray(glaucoma_labels) == 1).astype(int) if glaucoma_labels is not None else abn
    # gate semantics: real detection-retention (vs the glaucoma label) when labels are given, else the
    # label-free retention of the reference-device abnormal call (the honest two-regime distinction).
    sens_kind = "detection_retention" if glaucoma_labels is not None else "call_retention"
    quad_truth = envelope.sector_matrix(truth)[:, 1:]
    X_clean = envelope.to_features(clean_test_maps)

    def _metrics(model, X_shift) -> tuple[float, float, float]:
        """Per-eye recovery MAE (global, worst-quadrant) + the SYSTEMATIC read-drift (Bland-Altman bias).

        Recovery MAE is per-eye (noise-floored, catches the useless constant); systematic bias averages
        over eyes so the LoA noise cancels and only the recoverable a+b·x drift remains (the taxonomy read).
        """
        pred = model.predict_profile(X_shift)
        glob = float(np.nanmean(np.abs(pred - truth)))
        quad = envelope.sector_matrix(pred)[:, 1:]
        focal = float(np.nanmean(np.nanmax(np.abs(quad - quad_truth), axis=1)))
        sys_bias = float(np.nanmean(np.abs(np.nanmean(pred - truth, axis=0))))  # |mean_eyes(pred-truth)|
        return glob, focal, sys_bias

    const = envelope.build_interp_model("constant").fit(X_clean, truth).set_reference(ref)
    # CLEAN detection sensitivity per model — the reference for RETENTION under shift. The gate scores whether
    # the shift HIDES disease (sens drops) relative to the clean read, NOT the reader's absolute sensitivity:
    # a specific normative-percentile call is inherently low-sensitivity, and penalising a harmoniser for the
    # READER's insensitivity would be wrong. Retention decouples tolerance from absolute reader quality.
    sens_clean = {name: _sensitivity(model, sens_pos, X_clean) for name, model in arms.items()}
    clean_pos_call = {name: _sens_call_vector(model, sens_pos, X_clean) for name, model in arms.items()}
    # accumulate per-pair metrics, then average over the envelope
    acc = {name: {"mae": [], "focal": [], "sys": [], "flip": [], "sens": [], "shift_calls": []} for name in arms}
    const_mae_l, const_focal_l, dist_l = [], [], []
    for pair in pairs:
        seed = int(rng.integers(0, 2**31 - 1))
        shifted = np.stack([shift_fn(m, np.random.default_rng(seed + i), level, pair=pair)
                            for i, m in enumerate(clean_test_maps)])
        X_shift = envelope.to_features(shifted)
        dist_l.append(float(np.mean([transforms.format_distance(m, s)
                                     for m, s in zip(clean_test_maps, shifted)])))
        cm, cf, _ = _metrics(const, X_shift)
        const_mae_l.append(cm); const_focal_l.append(cf)
        for name, model in arms.items():
            mae, focal, sysb = _metrics(model, X_shift)
            acc[name]["mae"].append(mae); acc[name]["focal"].append(focal); acc[name]["sys"].append(sysb)
            # flip vs the TRUE reference-device call (robust for device-conditional harmonizers, whose own
            # clean read is not identity); on the synthetic axis the clean reference read is label-free.
            acc[name]["flip"].append(float(np.mean(model.predict_abn(X_shift) != abn)))
            acc[name]["sens"].append(_sensitivity(model, sens_pos, X_shift))
            acc[name]["shift_calls"].append(_sens_call_vector(model, sens_pos, X_shift))

    const_mae, const_focal = float(np.mean(const_mae_l)), float(np.mean(const_focal_l))
    mean_dist = float(np.mean(dist_l))
    baseline_mae = float(np.mean(acc[baseline_key]["mae"])) if baseline_key in arms else None

    boot_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1))) if n_boot > 0 else None
    scorecard = {}
    for name in arms:
        mae, focal_mae = float(np.mean(acc[name]["mae"])), float(np.mean(acc[name]["focal"]))
        sys_bias = float(np.mean(acc[name]["sys"]))
        flip_rate = float(np.mean(acc[name]["flip"]))
        sens_shift = float(np.mean(acc[name]["sens"]))  # detection under shift (vs label) OR call-retention
        # bootstrap the retention CI over the positive eyes (clean call tiled per pair vs the pooled shift calls)
        rci = None
        cp = clean_pos_call[name]
        if boot_rng is not None and len(cp):
            shift_pool = np.concatenate(acc[name]["shift_calls"]) if acc[name]["shift_calls"] else np.zeros(0, bool)
            rci = retention_ci(np.tile(cp, len(pairs)), shift_pool, boot_rng, n_boot=n_boot)
        earned = (baseline_mae is not None) and (mae <= baseline_mae * (1.0 - EARNED_MARGIN))
        cert = certify(mae=mae, focal_mae=focal_mae, const_mae=const_mae, const_focal=const_focal,
                       flip=flip_rate, sens_clean=sens_clean[name], sens_shift=sens_shift,
                       earned=earned, retention_ci=rci)
        scorecard[name] = {"rnfl_mae": mae, "focal_mae": focal_mae, "sys_bias_um": sys_bias,
                           "flip_rate": flip_rate, "distance": mean_dist, "sensitivity_kind": sens_kind,
                           "const_mae": const_mae, "const_focal": const_focal, "earned": bool(earned), **cert}
    return scorecard


def leaderboard(scorecard: dict) -> list[dict]:
    rows = [{"model": k, **v} for k, v in scorecard.items()]
    rows.sort(key=lambda r: (r["tier"], r["tolerance_score"]), reverse=True)
    return rows


def _score_under_knobs(row: dict, *, sens_floor: float, use_ratio: float, score_floor: float) -> tuple[int, float]:
    """Re-certify one scorecard row under swept certificate constants (no re-run) — for the knob sweep."""
    rg = _recovery_axis(row["rnfl_mae"], row["const_mae"])
    rf = _recovery_axis(row["focal_mae"], row["const_focal"])
    ret = row["sensitivity_retention"]
    stab = _clamp01(1.0 - row["flip_rate"]) if (np.isfinite(ret) and ret >= sens_floor) else 0.0
    vals = np.clip([rg, rf, stab], 1e-9, 1.0)
    score = float(np.exp(np.mean(np.log(vals))))
    sc = row["sensitivity_clean"]
    useful = (row["rnfl_mae"] <= use_ratio * row["const_mae"]) and np.isfinite(sc) and (sc > _SENS_EPS)
    if not useful:
        return 0, score
    return (2 if (row["earned"] and score >= score_floor) else 1), score


def knob_sweep(scorecard: dict, *, sens_floors=(0.5, 0.6, 0.7), use_ratios=(0.6, 0.7, 0.8),
               score_floors=(0.5, 0.55, 0.6)) -> dict:
    """Re-certify every model over the GRID of the 3 certificate constants → tier-stability + the robustness
    of the ORDERING (which model tops the board). The direction (best harmoniser wins) is expected knob-robust
    even though the discrete Tier is knob-sensitive (phase-review fix #8; pre-registered floors)."""
    from collections import Counter
    per_model_tiers: dict[str, list[int]] = {name: [] for name in scorecard}
    top1: Counter = Counter()
    n = 0
    for sf in sens_floors:
        for ur in use_ratios:
            for scf in score_floors:
                n += 1
                scored = {}
                for name, row in scorecard.items():
                    tier, score = _score_under_knobs(row, sens_floor=sf, use_ratio=ur, score_floor=scf)
                    per_model_tiers[name].append(tier)
                    scored[name] = score
                top1[max(scored, key=scored.get)] += 1
    stability = {name: {"tiers_seen": sorted(set(ts)),
                        "modal_tier": max(set(ts), key=ts.count),
                        "certified_fraction": float(np.mean([t == 2 for t in ts]))}
                 for name, ts in per_model_tiers.items()}
    winner, wins = top1.most_common(1)[0]
    return {"n_combos": n, "tier_stability": stability, "top1_counts": dict(top1),
            "ordering_top1": {"model": winner, "fraction": float(wins / n)}}


def _systematic_global_drift(model, shifted_maps: np.ndarray, truth_glob: np.ndarray) -> float:
    """|mean over eyes of (model's global read on shifted − true clean global)| — the recoverable bias.

    Averaging the per-eye read-shift over eyes cancels the LoA noise (Bland-Altman), leaving the
    SYSTEMATIC a+b·x device drift — the quantity the share/specialise TOST-vs-MDC rule acts on (not the
    noise-inflated mean-|Δ| the previous version used).
    """
    glob = np.array([canonical.profile_sectors(p)["global"] for p in model.predict_profile(
        envelope.to_features(shifted_maps))])
    return float(np.abs(np.mean(glob - truth_glob)))


def taxonomy_grid(model: envelope.InterpModel, clean_test_maps: np.ndarray, rng: np.random.Generator, *,
                  floor_um: float = MDC_GLOBAL_UM, tiers: tuple = TAXONOMY, harmonizer=None) -> list[dict]:
    """Per shift-tier SHARE / SHARE-AFTER-HARMONIZATION / SPECIALISE verdict — the deployment decision.

    For each device tier, apply the calibrated shift and measure the model's SYSTEMATIC global read-drift
    vs the true clean read (noise cancelled). Verdict:
      * SHARE                     — raw drift within the MDC floor (~4 um global): one model is safe as-is.
      * SHARE-AFTER-HARMONIZATION — raw drift exceeds MDC but a ``harmonizer`` (e.g. ComBat) brings it back
                                    within MDC: one model is safe IF the device is harmonized first.
      * SPECIALISE                — even harmonized drift exceeds MDC: use a device-specific model.
    (TOST-to-MDC on the systematic component; global here, extendable per-sector.)
    """
    truth_glob = np.array([canonical.profile_sectors(p)["global"]
                           for p in envelope.profile_true(clean_test_maps)])
    base_call = model.predict_abn(envelope.to_features(clean_test_maps))
    rows = []
    for name, pair, level in tiers:
        seed = int(rng.integers(0, 2**31 - 1))
        shifted = np.stack([transforms.apply_format(m, np.random.default_rng(seed + i), level, pair=pair)
                            for i, m in enumerate(clean_test_maps)])
        raw_drift = _systematic_global_drift(model, shifted, truth_glob)
        flip = float(np.mean(model.predict_abn(envelope.to_features(shifted)) != base_call))
        harm_drift = _systematic_global_drift(harmonizer, shifted, truth_glob) if harmonizer is not None else None
        if raw_drift <= floor_um:
            verdict = "share"
        elif harm_drift is not None and harm_drift <= floor_um:
            verdict = "share_after_harmonization"
        else:
            verdict = "specialize"
        rows.append({"tier": name, "pair": pair, "read_drift_um": raw_drift,
                     "harmonized_drift_um": harm_drift, "flip_rate": flip,
                     "within_floor": bool(raw_drift <= floor_um), "verdict": verdict})
    return rows
