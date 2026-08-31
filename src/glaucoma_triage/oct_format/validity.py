"""Benchmark-validity experiments for OCT-FormatShift (no same-eye pairs -> triangulate).

Two prongs from the methods review (docs/OCT_FORMATSHIFT_SCOPING.md §5):
  * V3 monotonicity  — the synthetic ``format_distance`` must ORDER real degradation: a clean model's
    read should drift / flip MORE at larger synthetic distance (Spearman rank corr). Validates the axis
    as a metric, not just binary transfer. Label-free, self-contained (Harvard-GF maps).
  * V2 transfer      — a de-confounded arm contrast on a REAL held-out shift (Harvard-GF -> Harvard-GDP,
    a within-Cirrus cross-institution/resolution shift we hold as maps): does DR-on-synthetic beat the
    clean baseline on the real OOD glaucoma-classification AUC, and — the MANDATORY control — does the
    literature-CALIBRATED augmenter beat an equal-strength GENERIC one? (Taori 2020 / BigAug: any strong
    aug helps generically, so realism is credited only by the calibrated>generic gap.)

Honest scope: GF->GDP is a SAME-VENDOR (Cirrus) shift, so a *vendor-calibrated* transform is not
expected to beat generic here — the calibrated advantage is for genuine vendor shifts, which need the
GLEAM/GOALS B-scans processed to thickness (Eyened, torch) — the deferred cross-vendor track. Reported
plainly.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import calibrated_offset, canonical, envelope, harmonize, transforms


def simulation_realism(maps: np.ndarray, rng: np.random.Generator, *,
                       pairs: tuple[str, ...] = transforms.CALIBRATED_PAIRS,
                       sector: str = "global") -> dict:
    """V1a (the SOUND realism anchor): the calibrated offset reproduces the PUBLISHED same-eye Bland-Altman
    bias + LoA on real RNFL values — TOST equivalence to the literature ground truth (Buchser 2012 / Leite
    2011 / Warner 2011). This validates the MAGNITUDE of what we inject against the paired-eye stats others
    measured — the µm-offset realism check (stronger than any image-level proxy).

    Uses PER-EYE global RNFL (as the published Bland-Altman was measured — one value per eye; the full
    0–268 µm pixel spread would spuriously inflate the proportional-term variance and the synthetic LoA)."""
    vals = np.array([canonical.rnfl_global(m) for m in maps])
    vals = vals[np.isfinite(vals)]
    return {pair: calibrated_offset.reproduces_published(vals, pair, np.random.default_rng(int(rng.integers(2**31))),
                                                         sector=sector) for pair in pairs}


def structural_loa(maps: np.ndarray, rng: np.random.Generator, *,
                   pair: str = transforms.DEFAULT_PAIR, bias_margin: float = 1.5,
                   loa_ratio_bounds: tuple[float, float] = (0.75, 1.3)) -> dict:
    """V1a-STRUCTURAL: the DEPLOYED per-eye MAP transform reproduces the published Bland-Altman — global
    BIAS + LoA AND the anatomy-coupled SECTOR structure (superior/inferior LoA wider than nasal/temporal;
    centration-driven, Cheung/Leung Eye 2008) — because the LoA spread is a PER-EYE ANATOMY-COUPLED draw
    (``calibrated_offset.eye_spread``), so it survives the eye-level averaging a map reader performs.

    This CLOSES the gap the old model had: the per-eye SCALAR round-trip (``simulation_realism``) showed
    σ≈3.92 µm, but the deployed map transform used i.i.d. PER-PIXEL noise that averaged to ≈0 at the eye
    level — so the maps the panel is actually scored on carried almost no LoA. Here we validate the transform
    AS DEPLOYED on real maps. Verdict pieces:
      * bias_reproduced  — TOST of the per-eye global diffs equivalent to the published bias (±bias_margin);
      * loa_reproduced   — global LoA half-width within ``loa_ratio_bounds`` of the published 1.96·sd;
      * sector_structure — min(superior, inferior) LoA half > max(nasal, temporal) LoA half (Cheung/Leung);
      * spread_survives  — the deployed global LoA half > 4 µm (the coherent per-eye spread did NOT average
                           to ~0; the structural fix over the old i.i.d.-per-pixel noise).
    Only the GLOBAL magnitude is calibrated (to Warner σ); the per-sector profile is mechanism-motivated
    (direction, not a precise per-sector calibration — no clean published per-sector cross-device LoA).
    """
    ba = calibrated_offset.map_difference_stats(
        maps, pair, np.random.default_rng(int(rng.integers(2**31))), severity=1.0)
    p = calibrated_offset.PUBLISHED[pair]
    x_mean = float(np.nanmean([canonical.rnfl_global(m) for m in maps]))
    pub_bias = calibrated_offset.expected_bias(x_mean, pair, severity=1.0)
    pub_loa_half = 1.96 * p.sd
    g = ba["global"]
    tost = calibrated_offset.tost_equivalence(g["diffs"], pub_bias, bias_margin)
    ratio = g["loa_half"] / pub_loa_half if pub_loa_half > 0 else float("nan")
    loa_ok = loa_ratio_bounds[0] <= ratio <= loa_ratio_bounds[1]
    sector_ok = (min(ba["s"]["loa_half"], ba["i"]["loa_half"])
                 > max(ba["n"]["loa_half"], ba["t"]["loa_half"]))
    survives = g["loa_half"] > 4.0
    return {
        "pair": pair,
        "reproduces": bool(tost["equivalent"] and loa_ok and sector_ok and survives),
        "bias_reproduced": bool(tost["equivalent"]),
        "loa_reproduced": bool(loa_ok),
        "sector_structure_superior_inferior_wider": bool(sector_ok),
        "spread_survives_eye_averaging": bool(survives),
        "published_bias_um": pub_bias, "deployed_global_bias_um": g["bias"],
        "published_loa_half_um": pub_loa_half, "deployed_global_loa_half_um": g["loa_half"],
        "loa_ratio": ratio, "tost_p": tost["p"], "extrapolated": p.extrapolated,
        "sector_loa_half_um": {s: ba[s]["loa_half"] for s in ("s", "i", "n", "t")},
        "note": "the DEPLOYED map transform's per-eye Bland-Altman vs published — extends the scalar V1a "
                "round-trip to the transform AS APPLIED to maps + validates the anatomy-coupled sector "
                "structure. Global magnitude calibrated (Warner σ); sector profile mechanism-motivated.",
    }


def provenance_report(maps: np.ndarray, *,
                      pairs: tuple[str, ...] = transforms.CALIBRATED_PAIRS) -> dict:
    """Per-pair calibration PROVENANCE — a plain honesty DISCLOSURE, NOT a validation.

    (The former ``out_of_sample_consistency`` "check" was CUT: for Cirrus↔Spectralis ``b=1.0`` so the induced
    bias is always the intercept, and the "independent range" was chosen to contain the calibration anchor →
    circular; RTVue was cohort-fragile.) This just reports, per pair, the calibration source, the induced
    bias at this cohort's mean RNFL, the independent literature biases for CONTEXT (not a pass/fail), and the
    extrapolated flag — so a reader sees exactly how well-sourced each pair is. No consistency verdict.
    """
    vals = np.array([canonical.rnfl_global(m) for m in maps])
    x_mean = float(np.nanmean(vals[np.isfinite(vals)]))
    out = {}
    for pair in pairs:
        prov = calibrated_offset.provenance(pair)
        out[pair] = {"calib_source": prov.calib_source, "status": prov.status,
                     "induced_bias_at_cohort_mean_um": calibrated_offset.expected_bias(x_mean, pair, severity=1.0),
                     "cohort_mean_rnfl_um": x_mean,
                     "independent_literature_biases_um_context": list(prov.independent_bias_um) if prov.independent_bias_um else None,
                     "independent_sources": prov.independent_sources, "note": prov.note}
    return out


def retouch_vendor_magnitude(by_vendor: dict[str, list[np.ndarray]]) -> dict:
    """V1b (image-level corroboration): do REAL Cirrus/Spectralis/Topcon B-scans differ measurably?

    Per-vendor image statistics (mean intensity, contrast=SD, speckle CV on tissue) + the cross-vendor
    spread. HONEST SCOPE: image-level (macular, MIRAGE-resampled, different-eyes) → corroborates the
    PREMISE that cross-vendor differences are real and non-trivial; it is NOT a disc µm-offset validation
    (that is V1a ``simulation_realism`` vs published Bland-Altman). Pure — takes the loaded arrays."""
    stats = {}
    for v, imgs in by_vendor.items():
        if not imgs:
            continue
        mi, ct, cv = [], [], []
        for a in imgs:
            fg = a[a > a.mean() * 0.5]  # rough tissue mask
            mi.append(float(a.mean())); ct.append(float(a.std()))
            cv.append(float(fg.std() / max(fg.mean(), 1e-6)))
        stats[v] = {"mean_intensity": float(np.mean(mi)), "contrast": float(np.mean(ct)),
                    "speckle_cv": float(np.mean(cv)), "n": len(imgs)}
    mis = [s["mean_intensity"] for s in stats.values()]
    spread = float(max(mis) - min(mis)) if len(mis) > 1 else 0.0
    return {"per_vendor": stats, "mean_intensity_spread": spread,
            "vendors_differ": bool(spread > 5.0),
            "note": "image-level only (macular, MIRAGE-resampled, different-eyes) — corroborates that "
                    "cross-vendor differences are real; the disc µm-offset magnitude is validated by V1a."}


def display_speckle_contrast(img: np.ndarray, win: int = 9) -> float:
    """DISPLAY speckle contrast = median local SD/mean in a ``win``×``win`` window over the tissue region.

    The committed estimator behind ``device_physics.DISPLAY_SPECKLE_CONTRAST``: a LOCAL (windowed) contrast on
    the log-compressed 8-bit B-scan display, tissue-masked at the 60th intensity percentile. It measures the
    per-vendor 'graininess' a reader sees — frame-averaging (Spectralis ART) lowers it, single-frame
    (Cirrus/Topcon) raises it — and is the log-compressed display analogue of the raw amplitude 0.52/√N.
    """
    from scipy.ndimage import uniform_filter
    a = np.asarray(img, dtype=float)
    mean = uniform_filter(a, win)
    sd = np.sqrt(np.clip(uniform_filter(a * a, win) - mean * mean, 0.0, None))
    m = a > np.percentile(a, 60)
    return float(np.median(sd[m] / np.maximum(mean[m], 1e-6)))


def retouch_display_speckle(by_vendor: dict[str, list[np.ndarray]], win: int = 9) -> dict[str, float]:
    """Per-vendor median ``display_speckle_contrast`` from REAL RETOUCH B-scans — the REPRODUCIBLE generator
    of ``device_physics.DISPLAY_SPECKLE_CONTRAST`` (Cirrus 0.365 > Topcon 0.184 > Spectralis 0.148, n=30 each;
    verified by ``tests/test_oct_retouch_speckle``). Vendor keys follow the loader ('Cirrus'/…, capitalised).

    ⚠ RETOUCH is MACULAR: the frame-averaging ORDERING is region-INDEPENDENT and transfers to the disc, but the
    ABSOLUTE level is a PROXY for the disc (disclosed in ``device_physics``). Pure — takes the loaded arrays
    (``loaders.load_retouch_sample``)."""
    out: dict[str, float] = {}
    for v, imgs in by_vendor.items():
        vals = [display_speckle_contrast(im) for im in imgs if getattr(im, "size", 0) > 100]
        if vals:
            out[v] = float(np.median(vals))
    return out


def _systematic_drift(pred: np.ndarray, ref: np.ndarray) -> float:
    """|mean over eyes of (pred - ref)| averaged over profile points — the LoA-noise-cancelled bias."""
    return float(np.nanmean(np.abs(np.nanmean(pred - ref, axis=0))))


def monotonicity(model: envelope.InterpModel, clean_maps: np.ndarray, levels: list[float],
                 rng: np.random.Generator, *, pair: str = transforms.DEFAULT_PAIR,
                 harmonizer=None) -> dict:
    """V3: does synthetic format_distance ORDER the clean model's read-drift + decision-flip (Spearman)?

    Reports the SYSTEMATIC drift (|mean_eyes(read(shifted) − read(clean))|, LoA-noise cancelled — the
    reframe metric) alongside the per-eye read-drift. If a ``harmonizer`` (e.g. ComBat) is passed, also
    reports its systematic drift per level — it should stay FLAT/low (the dose-response is flattened),
    the read-level analogue of the certificate result.
    """
    Xc = envelope.to_features(clean_maps)
    prof_clean = model.predict_profile(Xc)
    call_clean = model.predict_abn(Xc)
    rows = []
    for lv in levels:
        variants = np.stack([transforms.apply_format(m, rng, lv, pair=pair) for m in clean_maps])
        Xv = envelope.to_features(variants)
        pred = model.predict_profile(Xv)
        row = {"level": lv,
               "distance": float(np.mean([transforms.format_distance(m, v) for m, v in zip(clean_maps, variants)])),
               "read_drift": float(np.nanmean(np.abs(pred - prof_clean))),
               "sys_drift": _systematic_drift(pred, prof_clean),
               "flip_rate": float(np.mean(model.predict_abn(Xv) != call_clean))}
        if harmonizer is not None:
            row["harm_sys_drift"] = _systematic_drift(harmonizer.predict_profile(Xv), prof_clean)
        rows.append(row)
    dists = [r["distance"] for r in rows]
    out = {"rows": rows,
           "spearman_distance_vs_drift": float(spearmanr(dists, [r["sys_drift"] for r in rows]).statistic),
           "spearman_distance_vs_flip": float(spearmanr(dists, [r["flip_rate"] for r in rows]).statistic)}
    if harmonizer is not None:
        # the harmonizer flattens the curve iff its worst-level systematic drift stays well under the model's
        out["harmonizer_flattens"] = bool(max(r["harm_sys_drift"] for r in rows)
                                          < 0.5 * max(r["sys_drift"] for r in rows))
    return out


def read_gap(gf_maps: np.ndarray, gdp_maps: np.ndarray, rng: np.random.Generator) -> dict:
    """V2 (label-free, harmonization view): does ComBat close the REAL Harvard-GF → Harvard-GDP read gap?

    Both are Cirrus (same vendor, different institution + 200 vs 225 grid → a real cross-DATASET shift
    held as maps). A reader trained on GF reads GF-held-out and GDP; their global-RNFL read distributions
    differ. ComBat (fit on GF as the reference) harmonizes the GDP reads toward GF. Report the raw gap and
    the residual after harmonization — Wasserstein-1 (µm) + the systematic mean shift.

    HONEST CAVEAT: on a SAME-VENDOR shift the residual gap is largely POPULATION (case-mix), not a device
    offset — so ComBat closing it may be partly OVER-correction of a real cohort difference (Leek 2010),
    not device-bias removal. This is the corroboration's limit; a genuine vendor shift needs GLEAM/GOALS
    reads (the deferred Eyened track). The clean device result is on the synthetic axis + the certificate.
    """
    idx = rng.permutation(len(gf_maps)); k = int(0.7 * len(gf_maps))
    tr, gf_test = gf_maps[idx[:k]], gf_maps[idx[k:]]
    base = envelope.build_interp_model("linear").fit(envelope.to_features(tr), envelope.profile_true(tr))
    combat = harmonize.ComBatCorrector(base).fit(envelope.to_features(tr))

    def _global(model, maps):
        return np.array([canonical.profile_sectors(p)["global"] for p in model.predict_profile(envelope.to_features(maps))])

    gf = _global(base, gf_test)
    gdp = _global(base, gdp_maps)
    gdp_harm = _global(combat, gdp_maps)
    raw_w, harm_w = float(wasserstein_distance(gf, gdp)), float(wasserstein_distance(gf, gdp_harm))
    raw_shift = float(abs(gf.mean() - gdp.mean()))
    within_floor = raw_shift <= 4.0  # global RNFL MDC ~4 um (PMC6882427)
    return {"raw_wasserstein_um": raw_w, "harmonized_wasserstein_um": harm_w,
            "gap_closed_frac": float(1.0 - harm_w / raw_w) if raw_w > 1e-6 else float("nan"),
            "raw_mean_shift_um": raw_shift, "harmonized_mean_shift_um": float(abs(gf.mean() - gdp_harm.mean())),
            "raw_within_mdc_floor": bool(within_floor), "n_gf_test": int(len(gf)), "n_gdp": int(len(gdp)),
            "note": ("the real same-vendor (Cirrus) GF->GDP read gap is WITHIN the ~4 um MDC floor -> already "
                     "share-safe, no device offset to remove; forcing ComBat here OVER-CORRECTS a case-mix "
                     "difference (Leek 2010) — the two-regime caveat. The clean device result is on the "
                     "synthetic axis + certificate; a real vendor offset needs GLEAM/GOALS (Eyened track).")
            if within_floor else
            ("same-vendor gap exceeds the MDC floor (unexpected — likely case-mix); ComBat closure here is "
             "corroboration, not device-bias proof (Leek 2010 over-correction risk).")}


def _glaucoma_clf(train_maps: np.ndarray, train_y: np.ndarray, rng: np.random.Generator, *,
                  n_aug: int, augment_fn, levels: tuple[float, ...], pair: str) -> Pipeline:
    """Train a glaucoma-label classifier on map features (+ optional format augmentation, same label)."""
    Xs = [envelope.to_features(train_maps)]
    ys = [train_y]
    for _ in range(n_aug):
        lv = float(rng.choice(levels))
        variants = np.stack([augment_fn(m, rng, lv, pair=pair) for m in train_maps])
        Xs.append(envelope.to_features(variants))
        ys.append(train_y)
    clf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000))])
    clf.fit(np.vstack(Xs), np.concatenate(ys))
    return clf


def label_transfer(train_maps: np.ndarray, train_y: np.ndarray, ood_maps: np.ndarray, ood_y: np.ndarray,
                   rng: np.random.Generator, *, n_aug: int = 6,
                   levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                   pair: str = transforms.DEFAULT_PAIR) -> dict:
    """V2: glaucoma-AUC transfer to a REAL held-out shift for baseline / dr-calibrated / dr-generic arms."""
    idx = rng.permutation(len(train_maps))
    k = int(0.75 * len(train_maps))
    tr_m, tr_y = train_maps[idx[:k]], np.asarray(train_y)[idx[:k]]
    id_m, id_y = train_maps[idx[k:]], np.asarray(train_y)[idx[k:]]
    ood_y = np.asarray(ood_y)

    arms = {"baseline": (0, transforms.apply_format),
            "dr_calibrated": (n_aug, transforms.apply_format),
            "dr_generic": (n_aug, transforms.apply_generic_format)}
    X_id, X_ood = envelope.to_features(id_m), envelope.to_features(ood_maps)
    out = {}
    for name, (na, aug) in arms.items():
        clf = _glaucoma_clf(tr_m, tr_y, np.random.default_rng(abs(hash(name)) % 2**31),
                            n_aug=na, augment_fn=aug, levels=levels, pair=pair)
        id_auc = float(roc_auc_score(id_y, clf.predict_proba(X_id)[:, 1])) if len(np.unique(id_y)) > 1 else float("nan")
        ood_auc = float(roc_auc_score(ood_y, clf.predict_proba(X_ood)[:, 1])) if len(np.unique(ood_y)) > 1 else float("nan")
        out[name] = {"id_auc": id_auc, "ood_auc": ood_auc, "drop": id_auc - ood_auc}
    out["dr_closes_gap"] = bool(out["dr_calibrated"]["ood_auc"] > out["baseline"]["ood_auc"])
    out["calibrated_beats_generic"] = bool(out["dr_calibrated"]["ood_auc"] > out["dr_generic"]["ood_auc"])
    out["note"] = ("GF->GDP is same-vendor (Cirrus): a vendor-calibrated transform is not expected to beat "
                   "generic here; the calibrated>generic test needs a real vendor shift (GLEAM/GOALS via Eyened).")
    return out
