"""The input feature contract for the modular two-stage triage agent (proposal v0.45).

The two public arms stay **separate and modular** — we do NOT graft OCT onto PAPILA to fake a
complete multimodal set. Modular late fusion with per-modality calibration handles the
missing-modality reality; a missing modality drops its branch and the decision abstains below
the minimum set.

Stage 1 — Routing: two *separate* binary structural imaging classifiers — ``fundus`` (PAPILA)
and ``oct`` (Harvard-GF RNFLT). A referral uses whichever modality it carries; rules (not ML)
route the case. The rule is the **"Read & route" node (Step 3) of the Triage_Decision-Tree
v0.32** (companion to research brief v2.8; grounded in GIRFT Best Practice May 2026 and NICE
NG81 §1.1): suspicious structure → GECFS; normal structure with raised IOP (≥24) and/or an
abnormal/unreliable field → GRMS; normal structure with a *reliable* normal field and IOP < 24 →
clear-negative discharge; filing gap → return to referrer (see ``stage1_route``). The four routes
here are the Step-3 subset of the fuller v0.32 disposition set (:data:`DISPOSITIONS`).

Stage 2 — Triage: a metadata classifier → + imaging → + conversational, fused modularly.
Output is binary: discharge a confident clear-negative, else DAVRAP.

Race/ethnicity are **fairness-evaluation only — never model inputs**. ``axial_length`` and VF
total-deviations are kept as data, outside the metadata set (refraction is the myopia input;
the VF index used is MD). The matrix builders below are the interface the classifiers call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Stage 1: two separate, calibrated binary structural imaging classifiers ---------------
IMAGING_CLASSIFIERS: dict[str, dict[str, object]] = {
    # fundus disc read (PAPILA): vertical + area CDR, rim-to-disc area, and the ISNT-rule flag,
    # all derived from the expert cup/disc contours (datasets/disc_morphology.py).
    "fundus": {
        "source": "papila",
        "features": ["vcdr", "hcdr", "area_cdr", "rim_disc_area_ratio", "isnt_violation"],
        "ref": "fundus_path",
    },
    # OCT RNFL thickness read (Harvard-GF). Label is VF-derived (derived in the next step).
    # 3-feature arcuate panel — global mean + inferior + superior region means — mirroring the
    # progression OCT panel (`grape.OCT_FEATURES = [rnfl_mean, rnfl_i, rnfl_s]`) for consistency;
    # superior/inferior are laterality-invariant (Harvard-GF ships no laterality).
    "oct": {"source": "harvard_gf",
            "features": ["rnflt_mean", "rnflt_inf", "rnflt_sup"], "ref": "rnflt_path"},
}

# --- Fundus modality representations for the interaction decomposition (P1 progression plan) ----
# A modality's Shapley/interaction value is representation-dependent, so the fundus feature set is
# FIXED A PRIORI BY RULE — never selected/tuned. Headline rule: *all disc-morphometry derivable from
# the expert contours, de-duplicated for exact algebraic collinearity.* Drops the exact duplicates
# rim_disc_area_ratio (= 1 − area_cdr), cup_area, rim_area (recoverable from disc_area + area_cdr).
# disc_area is pixel-scaled (fine within single-camera PAPILA; a cross-cohort harmonization item).
FUNDUS_MORPHOMETRY_FULL: list[str] = [
    "vcdr", "hcdr", "area_cdr", "disc_area", "min_rim_disc_ratio",
    "isnt_i", "isnt_s", "isnt_n", "isnt_t", "isnt_violation",
]
# The pre-specified 3-feature CLINICAL PANEL — global cupping (vCDR) · area cupping (area-CDR) ·
# focal rim narrowing (min rim/disc, a BMO-MRW analog). This is the HEADLINE fundus representation,
# chosen to be identical to the progression study's panel (`grape.FUNDUS_FEATURES = [fundus_vcdr,
# fundus_area_cdr, fundus_min_rim]`) so structure is represented consistently across the paper.
# `FUNDUS_MORPHOMETRY_FULL` (10-feat) is retained as the representation-robustness comparator (Secondary B).
FUNDUS_CLINICAL_PANEL: list[str] = ["vcdr", "area_cdr", "min_rim_disc_ratio"]

# --- Stage 2: metadata (tabular) classifier ------------------------------------------------
METADATA_INPUTS: list[str] = ["iop", "vf_md", "cct", "age", "refraction", "sex"]
CONVERSATIONAL: list[str] = ["symptoms", "family_history"]  # declared; populated later

FAIRNESS_EVAL_ONLY: list[str] = ["race", "ethnicity"]  # subgroup metrics, never inputs

EXCLUDED_FROM_INPUT: dict[str, str] = {
    "race": "fairness evaluation only — never a feature",
    "ethnicity": "fairness evaluation only — never a feature",
    "axial_length": "kept as data (refraction is the myopia input)",
    "vf_tds": "kept as data (the VF index used is MD)",
    "diagnosis_raw": "target",
    "diagnosis": "target (structural label source)",
    "disposition": "target (triage label source)",
    "is_suspect": "target",
    "vf_md_leakage": "provenance flag — not a feature",
    "vf_md_simulated": "provenance flag — not a feature",
    "vcdr_e1": "per-expert label-construction metric — not a feature",
    "vcdr_e2": "per-expert label-construction metric — not a feature",
    "area_cdr_e1": "per-expert label-construction metric — not a feature",
    "area_cdr_e2": "per-expert label-construction metric — not a feature",
    "isnt_violation_e1": "per-expert label-construction metric — not a feature",
    "isnt_violation_e2": "per-expert label-construction metric — not a feature",
}

TRIAGE_TARGET: str = "disposition"
TRIAGE_POSITIVE: str = "refer"  # refer/DAVRAP -> 1, discharge -> 0


# --- labels --------------------------------------------------------------------------------
def fundus_structural_label(df: pd.DataFrame) -> pd.Series:
    """Binary **clinician glaucoma reference** (PAPILA-dx source): glaucoma = 1; healthy ∨ suspect = 0.
    Suspect is PAPILA's ocular-hypertension group — disc-normal by recruitment (Group 2 enrolled free of
    optic-nerve pathology) — so it is counted **negative**: a disc rule cannot/should not detect disc-normal
    OHT, and glaucoma is the safety-critical class. Rows outside {healthy, suspect, glaucoma} are <NA>."""
    d = df["diagnosis"]
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y[d.isin(["healthy", "suspect"])] = 0
    y[d == "glaucoma"] = 1
    return y


# Derived structural-label rule for the fundus arm. Every threshold is a recognised clinical
# standard (NOT fit to the clinician label — that would be circular):
#   * vcdr / area_cdr >= 0.70 — ISGEO/Foster glaucoma case definition, the 97.5th-percentile
#     population cut-off (Foster PJ et al., Br J Ophthalmol 2002;86:238).
#   * inter-eye VCDR asymmetry >= 0.20 — Soh ZD et al., Clin Exp Ophthalmol 2020;48:1210
#     (best-balanced with VCDR>=0.60: sens 95.1% / spec 90.9%); UK referral-refinement practice.
#   * ISNT-rule violation with VCDR >= 0.60 — the I>=S>=N>=T rim ordering (Jonas 1988; Harizman N
#     et al., Arch Ophthalmol 2006;124:1579, intact in 79% normal vs 28% glaucoma → specific but
#     insensitive, so used only as a support clause above the 0.60 suspicion floor, never alone).
# rim_disc_area_ratio is deliberately NOT a separate clause: it equals 1 - area_cdr, so a
# "thin rim" cut-off would be algebraically identical to the area_cdr clause.
DERIVED_LABEL_THRESHOLDS: dict[str, float] = {
    "vcdr": 0.7,
    "area_cdr": 0.7,
    "vcdr_asymmetry": 0.20,
}


def _read_name(df: pd.DataFrame, base: str, read: str) -> str:
    """Column name for ``base`` under ``read``: the per-expert ``{base}_{read}`` if that column exists,
    else the averaged ``base``. Lets ``combine="or"`` use the persisted per-expert morphometry on real
    data while degrading gracefully to the consensus read on frames without per-expert columns."""
    if read != "mean":
        col = f"{base}_{read}"
        if col in df.columns:
            return col
    return base


def _vcdr_asymmetry(df: pd.DataFrame, vcdr_col: str = "vcdr") -> pd.Series:
    """Per-patient inter-eye VCDR spread ``|vcdr_OD - vcdr_OS|`` broadcast to each eye's row.

    Uses ``vcdr_col`` (``"vcdr"`` for the averaged read, ``"vcdr_e1"``/``"vcdr_e2"`` per expert).
    ``NaN`` where a patient has fewer than two eyes with a measured vCDR (clause then skipped)."""
    vcdr = pd.to_numeric(pd.Series(df.get(vcdr_col), index=df.index), errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if "patient_id" not in df.columns:
        return out
    for idx in df.groupby("patient_id").groups.values():
        vals = vcdr.loc[idx].dropna()
        if len(vals) >= 2:
            out.loc[idx] = float(vals.max() - vals.min())
    return out


def _derived_positive(
    vcdr: pd.Series, area_cdr: pd.Series, asym: pd.Series, t: dict[str, float],
) -> pd.Series:
    """Boolean glaucomatous-structure test for one metric set — the ISGEO Category I case definition
    (VCDR/area-CDR ≥ 0.70 or inter-eye asymmetry ≥ 0.20). (Strict-ISNT was dropped: not part of
    ISGEO/NICE/GIRFT — a research heuristic — see the sensitive read's IST clause instead.)"""
    vcdr = pd.to_numeric(vcdr, errors="coerce")
    area_cdr = pd.to_numeric(area_cdr, errors="coerce")
    return (
        (vcdr >= t["vcdr"])
        | (area_cdr >= t["area_cdr"])
        | (asym >= t["vcdr_asymmetry"])
    )


def _mean_terms(df: pd.DataFrame, t: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """``(positive, scorable)`` from the expert-averaged metric columns."""
    vcdr = pd.to_numeric(df.get("vcdr"), errors="coerce")
    positive = _derived_positive(vcdr, df.get("area_cdr"), _vcdr_asymmetry(df, "vcdr"), t)
    return positive, vcdr.notna()


def _expert_terms(df: pd.DataFrame, e: str, t: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """``(positive, scorable)`` from one expert's columns (``e`` is ``"e1"``/``"e2"``), using
    that expert's own single-eye metrics and its own inter-eye VCDR asymmetry."""
    vcdr_e = pd.to_numeric(df.get(f"vcdr_{e}"), errors="coerce")
    positive = _derived_positive(vcdr_e, df.get(f"area_cdr_{e}"), _vcdr_asymmetry(df, f"vcdr_{e}"), t)
    return positive, vcdr_e.notna()


DERIVED_COMBINE_MODES = ("mean", "or", "e1", "e2")


def fundus_structural_label_derived(
    df: pd.DataFrame, thresholds: dict[str, float] | None = None, combine: str = "mean"
) -> pd.Series:
    """Binary imaging-morphology structural label for the fundus arm (the compared source).

    Positive (1) if disc morphology meets ANY glaucomatous-structure criterion in
    :data:`DERIVED_LABEL_THRESHOLDS`; 0 if scorable and none met; ``<NA>`` if the eye has no
    morphometry. Independent of the clinician diagnosis, so it can be compared against
    :func:`fundus_structural_label` without circularity.

    ``combine`` chooses how the two expert segmentations are combined:
      * ``"mean"`` (default) — apply the rule to the expert-*averaged* metrics (consensus).
      * ``"e1"`` / ``"e2"`` — apply the rule to a *single* expert's segmentation.
      * ``"or"`` — the union of ``"e1"`` and ``"e2"`` (positive if either expert's disc read
        fires), a sensitivity-maximising / safety-first labeling.
    """
    t = DERIVED_LABEL_THRESHOLDS if thresholds is None else thresholds

    if combine == "mean":
        positive, scorable = _mean_terms(df, t)
    elif combine in ("e1", "e2"):
        positive, scorable = _expert_terms(df, combine, t)
    elif combine == "or":
        p1, s1 = _expert_terms(df, "e1", t)
        p2, s2 = _expert_terms(df, "e2", t)
        positive, scorable = (p1 | p2), (s1 | s2)
    else:
        raise ValueError(f"combine must be one of {DERIVED_COMBINE_MODES}, got {combine!r}")

    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y[scorable] = 0
    y[scorable & positive] = 1
    return y


def gold_clauses(df: pd.DataFrame, read: str = "mean") -> dict[str, pd.Series]:
    """The individual OR-clauses of the *gold* derived label as named boolean masks (NA→False),
    mirroring :func:`sensitive_clauses` so both rule families share one interface for the
    label-combination search (:func:`eda.fundus_label.label_combination_search`). ``read`` selects
    the expert segmentation: ``"mean"`` (averaged), ``"e1"`` / ``"e2"`` (single expert — enables the
    inter-expert reproducibility of gold-only candidate labels). All thresholds are external, recognised
    standards — VCDR/area ≥ 0.70 AND inter-eye asymmetry ≥ 0.20 are both ISGEO Category I (Foster 2002),
    no diagnosis, no in-sample normal limits. (Strict-ISNT was dropped — not part of ISGEO/NICE/GIRFT.)"""
    t = DERIVED_LABEL_THRESHOLDS
    vcol = _read_name(df, "vcdr", read)
    acol = _read_name(df, "area_cdr", read)
    vcdr = pd.to_numeric(pd.Series(df.get(vcol), index=df.index), errors="coerce")
    area = pd.to_numeric(pd.Series(df.get(acol), index=df.index), errors="coerce")
    asym = _vcdr_asymmetry(df, vcol)
    return {
        "vcdr>=0.70": (vcdr >= t["vcdr"]).fillna(False),
        "area-cdr>=0.70": (area >= t["area_cdr"]).fillna(False),
        "asymmetry>=0.20 (gold)": (asym >= t["vcdr_asymmetry"]).fillna(False),
    }


# --- high-sensitivity (low-FNR) profile ----------------------------------------------------
# A deliberately more sensitive rule to lower the suspect/glaucoma false-negative rate, using
# literature-supported cut-offs that deviate from the conservative gold standard. Every criterion
# is an OR-clause (union maximises sensitivity).
#
# OPERATIVE cup criterion = the disc-size-adjusted per-tertile healthy normal limit below
# (`_disc_size_adjusted_positive`, cuts ~0.494-0.540 — EXTERNAL, from Chákṣu; see below). The flat
# `vcdr`/`area_cdr` >= 0.60 clause below is REDUNDANT (every tertile cut < 0.60, so anything clearing
# 0.60 is already flagged) — it is kept only as a transparent disc-size-independent floor, consistent
# with the *direction* of Soh 2020's >=0.60 screening cut-off, NOT the binding threshold. Real cut ~0.49-0.54.
#
# Achieved sensitivity on PAPILA stays below the source studies (mild cohort + contour-derived CDR)
# — reported, not assumed. A single structural rule cannot reach ~0 FNR (Cochrane/Michelessi 2015;
# Mowatt 2008): glaucoma/suspect cups overlap healthy, so near-zero FNR needs the multi-signal
# routing (structure OR IOP OR field) + abstain.
#
# NORMAL LIMITS — now EXTERNAL (non-circular): the disc-size tertile cuts (the only adopted normal
# limit) are the healthy 97.5th-pct of Chákṣu, an INDEPENDENT dataset (see DISC_SIZE_ADJUSTED_VCDR),
# not fit to PAPILA — so the reported healthy SPECIFICITY of the adopted read is a fair held-out
# estimate, not the optimistic in-sample figure it was before. `IN_SAMPLE_SENSITIVE_CLAUSES` is now
# empty. Reassuringly the external cut reproduces the retired PAPILA in-sample 97.5th within 0.02 on
# every tertile (mutual corroboration). (The candidate-library `min_rim_disc_ratio` p5 remains an
# in-sample limit, but it is NOT part of the adopted union.) `eda.fundus_label.in_sample_contribution`
# still reports the full-vs-external split; with no in-sample adopted clause the two now coincide.
DERIVED_LABEL_THRESHOLDS_SENSITIVE: dict[str, float] = {
    "isnt_support_vcdr": 0.50,  # low cup floor gating the IST rim-rule clause
}
# Disc-size-adjusted cup cut-offs — the OPERATIVE cup criterion (larger discs tolerate a larger
# physiological cup; small discs with glaucoma have small cups a fixed cut-off misses — Blue
# Mountains/Crowston 2004; Jonas 1990). Per-tertile cuts are the EXTERNAL healthy 97.5th-pct vcdr from
# the Chákṣu database (Kumar et al. 2023: 1,345 fundus images, 3 cameras, 5 expert graders; same 2-D
# cup/disc contour vcdr as PAPILA), Train+Test, deduplicated. Healthy = unanimous NORMAL (all 5 experts,
# n=739); impossible segmentations (any expert vcdr >= 1.0) excluded; disc size binned by WITHIN-CAMERA
# tertile (raw-pixel disc area is not comparable across the 3 cameras); per-tertile 97.5th-pct of the
# Mean-fusion vcdr. Applied to PAPILA by PAPILA's OWN disc-area tertiles (DISC_AREA_TERTILE_EDGES below —
# the disc-size axis for PAPILA eyes; disc_area is pixel-scaled, no mm). External + non-circular; agrees
# with the retired PAPILA in-sample cut {0.505, 0.528, 0.558} within 0.02 per tertile. See FEATURES.md.
DISC_AREA_TERTILE_EDGES: tuple[float, float] = (152797.0, 176587.0)
DISC_SIZE_ADJUSTED_VCDR: dict[str, float] = {"small": 0.494, "med": 0.527, "large": 0.540}
# vCDR floor gating the inter-eye ASYMMETRY clause in the adopted (sensitive) read: asymmetry counts
# only on an eye with at least moderate cupping. Without it the asymmetry clause flags the small-cup
# NORMAL fellow of a big-cup eye (a physiological-cup false positive). 0.50 = the conventional "glaucoma
# suspect" CDR cut (population mean vCDR ~0.4) — the more clinically defensible floor. A uniqueness check
# (patient-level, out-of-fold) showed 0.45 is NOT uniquely helpful: 0.45 vs 0.50 (and even dropping
# asymmetry) give the SAME patient-level glaucoma FN and IDENTICAL 3-class classifier confusion — 0.45
# only uniquely flagged 2 glaucoma EYES already field/bilateral-covered + 1 healthy FP. So the earlier
# 0.45 (keep 0.45-0.50 cups in disc-GECFS vs GRMS) is an eye-level/routing distinction only. (The
# `asymmetry>=0.20 & vcdr>=0.45` candidate variant was dropped from `candidate_clauses`.)
ASYMMETRY_VCDR_FLOOR: float = 0.50


def _disc_size_adjusted_positive(df: pd.DataFrame, read: str = "mean") -> pd.Series:
    """``vcdr`` above the healthy 97.5th-pct for its disc-area tertile (size-adjusted cup). ``read`` picks
    the expert reading: ``"mean"`` (averaged) or ``"e1"``/``"e2"`` (single expert, for the per-expert
    union). The per-tertile cuts are the EXTERNAL Chákṣu healthy 97.5th-pct (see DISC_SIZE_ADJUSTED_VCDR);
    PAPILA eyes are binned by PAPILA's own disc-area tertile edges, applied to whichever read."""
    da = pd.to_numeric(pd.Series(df.get(_read_name(df, "disc_area", read)), index=df.index), errors="coerce")
    vcdr = pd.to_numeric(pd.Series(df.get(_read_name(df, "vcdr", read)), index=df.index), errors="coerce")
    lo, hi = DISC_AREA_TERTILE_EDGES
    cut = pd.Series(np.nan, index=df.index, dtype=float)
    cut[da <= lo] = DISC_SIZE_ADJUSTED_VCDR["small"]
    cut[(da > lo) & (da <= hi)] = DISC_SIZE_ADJUSTED_VCDR["med"]
    cut[da > hi] = DISC_SIZE_ADJUSTED_VCDR["large"]
    return (vcdr >= cut).fillna(False)


def _ist_violation(df: pd.DataFrame, rel_tol: float = 0.02, read: str = "mean") -> pd.Series:
    """IST-rule violation: NOT (inferior ≥ superior ≥ temporal) on the sector rim widths, with a
    small relative tolerance. The **nasal** sector is dropped — the full ISNT's S≥N / N≥T steps
    fail in ~1/3 of normal eyes (nasal ≈ superior), so IST roughly doubles specificity over strict
    ISNT (Higashide 2022; Poon 2017; Maupin 2020). ``False`` where any sector is missing. ``read`` picks
    the averaged (``"mean"``) or single-expert (``"e1"``/``"e2"``) sector widths for the per-expert union."""
    def _col(name: str) -> pd.Series:
        return pd.to_numeric(pd.Series(df.get(_read_name(df, name, read)), index=df.index), errors="coerce")

    i, s, t = _col("isnt_i"), _col("isnt_s"), _col("isnt_t")
    tol = rel_tol * ((i + s + t) / 3.0)
    ordered = (i >= s - tol) & (s >= t - tol)
    return (~ordered).fillna(False)


def sensitive_clauses(
    df: pd.DataFrame, thresholds: dict[str, float] | None = None, read: str = "mean"
) -> dict[str, pd.Series]:
    """The OR-clauses of the **adopted** low-FNR structural read as named boolean masks (NA→False) — their
    **per-expert union** (fire if either grader's contours trip a clause) is :func:`fundus_suspicious_disc_
    label`, the fundus classifier's target. **Two** clinically-standard clauses: the **disc-size-adjusted
    cup** (Blue Mountains/Jonas, with Chákṣu-external normal limits) and the **IST rim rule** (I≥S≥T, nasal
    step dropped; Higashide 2022), the latter gated to a meaningfully-cupped eye (vCDR ≥ 0.50,
    :data:`DERIVED_LABEL_THRESHOLDS_SENSITIVE`). It differs from `gold` by dropping the flat VCDR/area ≥ 0.70
    and strict I≥S≥N≥T cuts and adding the disc-size cut + the IST variant. ``read`` selects the expert
    reading (``"e1"``/``"e2"`` for the adopted per-expert union, or ``"mean"`` averaged). (The flat cup ≥
    0.60 floor, ISGEO inter-eye asymmetry ≥ 0.20, VCDR/area ≥ 0.70, in-sample focal min-rim and the
    larger-eye asymmetry variant were evaluated and dropped — subsumed / redundant / no benefit /
    non-standard; they remain search candidates.)"""
    t = DERIVED_LABEL_THRESHOLDS_SENSITIVE if thresholds is None else thresholds
    vcdr = pd.to_numeric(pd.Series(df.get(_read_name(df, "vcdr", read)), index=df.index), errors="coerce")
    return {
        "disc-size-adjusted cup": _disc_size_adjusted_positive(df, read=read),
        # IST rim rule (I>=S>=T, nasal step dropped; Higashide 2022) gated to a meaningfully-cupped eye
        # (vcdr >= 0.50) — the literature ISNT variant that ~doubles specificity vs the strict I>=S>=N>=T.
        "IST rule (I>=S>=T) & vcdr>=0.5": (_ist_violation(df, read=read)
                                           & (vcdr >= t["isnt_support_vcdr"])).fillna(False),
    }


# Adopted sensitive clauses whose threshold is an in-sample (PAPILA-healthy-derived) normal limit.
# NOW EMPTY: the disc-size cut moved to an EXTERNAL Chákṣu normative (see DISC_SIZE_ADJUSTED_VCDR), so
# every adopted clause uses an external/literature threshold and the reported specificity is non-circular.
# Kept (typed) for `eda.fundus_label.in_sample_contribution`, which then reports the full and
# external-only labels as identical.
IN_SAMPLE_SENSITIVE_CLAUSES: tuple[str, ...] = ()


# ============================================================================================
# CANDIDATE CLAUSE LIBRARY — for the literature-grounded combination search
# --------------------------------------------------------------------------------------------
# Every recognised suspicious-disc criterion DERIVABLE from the two-expert 2-D cup/disc contours,
# each with an EXTERNAL fixed threshold (no fitting) unless marked in-sample. This is the searchable
# pool used by :func:`eda.fundus_label.label_combination_search`; it is deliberately a SUPERSET of
# :func:`gold_clauses`/:func:`sensitive_clauses`. It is NOT the adopted label — the frozen
# :data:`FUNDUS_SUSPICIOUS_DISC_CLAUSES` is unchanged. Signals NOT derivable from 2-D contours (disc
# haemorrhage, RNFL wedge defect, peripapillary atrophy, rim pallor, 3-D cup depth) are out of scope
# — the fundus-only structural ceiling (documented in the artifact limitations).
VCDR_SUSPECT: float = 0.60          # glaucomatous-appearing/suspect cup (Soh 2020; eophtha disc review)
HCDR_CUT: float = 0.70              # horizontal CDR, ISGEO-parallel to the vertical cut
VCDR_ASYMMETRY_SUSPECT_FLOOR: float = 0.50  # asymmetry floor at the CONVENTIONAL glaucoma-suspect CDR
                                    # cut — the standard, clinically-defensible threshold; == ASYMMETRY_VCDR_FLOOR
                                    # (the adopted floor is now 0.50 too); population mean vCDR ~0.4, CDR>=0.5 = suspect
FOCAL_POLE_RIM_DD: float = 0.10     # ISGEO Cat I focal rim < 0.1 disc-diameter at the sup/inf pole
                                    # (Foster 2002; rim width ÷ equivalent disc diameter)
# EXTERNAL healthy normal limits — Chákṣu database (Kumar et al. 2023), NOT PAPILA in-sample. Same
# derivation as DISC_SIZE_ADJUSTED_VCDR: Train+Test, deduplicated, healthy = unanimous NORMAL (all 5
# expert graders, n=739), impossible segmentations excluded (any expert vCDR >= 1.0), Mean fusion.
# rim/disc RATIOS are camera-scale-independent → pooled across the 3 cameras; the DDLS disc-area
# tertiles are WITHIN-CAMERA. min_rim_disc_ratio is computed by a pure-numpy raster angular-max on
# Chákṣu's Cup+Disc masks (BMO-MRW analog; validated vs PAPILA's contour method — healthy 5th-pct
# within 0.005). Each reproduces the retired PAPILA in-sample limit within ~0.01 (mutual corroboration);
# so the whole clause library is now external/non-circular (IN_SAMPLE_CANDIDATE_CLAUSES empty). See FEATURES.md.
MIN_RIM_DISC_RATIO_P5: float = 0.321        # narrowest rim/disc, Chákṣu healthy 5th pct (DDLS unit)
RIM_DISC_AREA_RATIO_P2_5: float = 0.707     # neuroretinal rim/disc AREA, Chákṣu healthy 2.5th pct (= 1 - ACDR)
# disc-size-adjusted narrowest rim-to-disc (Spaeth DDLS): larger discs tolerate a narrower rim. Per
# disc-area tertile (:data:`DISC_AREA_TERTILE_EDGES`) — Chákṣu healthy WITHIN-CAMERA 2.5th pct of ``min_rim_disc_ratio``.
DDLS_MIN_RIM_BY_TERTILE: dict[str, float] = {"small": 0.341, "med": 0.300, "large": 0.282}


def _read_series(df: pd.DataFrame, base: str, read: str) -> pd.Series:
    """Numeric Series for ``base`` under the chosen expert ``read`` (``_read_name`` degrades to the
    averaged column when a per-expert one is absent — e.g. ``hcdr``/``min_rim_disc_ratio``/``isnt_n``
    have no per-expert variants, so ``e1``/``e2`` fall back to the consensus for those clauses)."""
    return pd.to_numeric(pd.Series(df.get(_read_name(df, base, read)), index=df.index), errors="coerce")


def _disc_equiv_diameter(df: pd.DataFrame, read: str = "mean") -> pd.Series:
    """Equivalent-circle disc diameter ``2·sqrt(disc_area/π)`` (pixel-scaled) — the denominator for the
    focal pole rim-to-disc-diameter clause (no direct disc-diameter column is persisted)."""
    return 2.0 * np.sqrt(_read_series(df, "disc_area", read) / np.pi)


def _isnt_ordered(df: pd.DataFrame, names: tuple[str, ...], rel_tol: float = 0.02,
                  read: str = "mean") -> pd.Series:
    """Whether the sector rim widths in ``names`` are in non-increasing order (each ≥ the next, within a
    small relative tolerance) — the general ISNT-family ordering test. ``names`` is largest-first, e.g.
    ``("isnt_i","isnt_s","isnt_n","isnt_t")`` for the full ISNT rule (I≥S≥N≥T)."""
    cols = [_read_series(df, n, read) for n in names]
    tol = rel_tol * (sum(cols) / len(cols))
    ordered = pd.Series(True, index=df.index)
    for a, b in zip(cols, cols[1:]):
        ordered = ordered & (a >= b - tol)
    return ordered


def _below_limit(df: pd.DataFrame, base: str, limit: float, read: str = "mean") -> pd.Series:
    """``base ≤ limit`` (NA→False) — the lower-tail normal-limit test for rim-ratio clauses."""
    return (_read_series(df, base, read) <= limit).fillna(False)


def _focal_pole_rim_positive(df: pd.DataFrame, read: str = "mean") -> pd.Series:
    """ISGEO focal rim-notch criterion: superior OR inferior pole rim width < 0.1 disc diameter — the
    at-risk-pole thinning of Foster 2002 Category I (rim width ÷ equivalent disc diameter)."""
    dd = _disc_equiv_diameter(df, read)
    s, i = _read_series(df, "isnt_s", read), _read_series(df, "isnt_i", read)
    return (((s / dd) < FOCAL_POLE_RIM_DD) | ((i / dd) < FOCAL_POLE_RIM_DD)).fillna(False)


def _ddls_adjusted_positive(df: pd.DataFrame, read: str = "mean") -> pd.Series:
    """DDLS-style disc-size-adjusted narrowest-rim clause: ``min_rim_disc_ratio`` below its healthy
    per-disc-area-tertile 2.5th-pct limit (:data:`DDLS_MIN_RIM_BY_TERTILE`). In-sample normal limit."""
    da = _read_series(df, "disc_area", read)
    mr = _read_series(df, "min_rim_disc_ratio", read)
    lo, hi = DISC_AREA_TERTILE_EDGES
    cut = pd.Series(np.nan, index=df.index, dtype=float)
    cut[da <= lo] = DDLS_MIN_RIM_BY_TERTILE["small"]
    cut[(da > lo) & (da <= hi)] = DDLS_MIN_RIM_BY_TERTILE["med"]
    cut[da > hi] = DDLS_MIN_RIM_BY_TERTILE["large"]
    return (mr <= cut).fillna(False)


def candidate_clauses(df: pd.DataFrame, read: str = "mean") -> dict[str, pd.Series]:
    """The full literature-grounded candidate pool as named boolean masks (NA→False) for the combination
    search — a SUPERSET of :func:`gold_clauses`/:func:`sensitive_clauses`. All thresholds are external
    recognised standards except the four in-sample normal limits in :data:`IN_SAMPLE_CANDIDATE_CLAUSES`
    (re-derived out-of-fold by the search). ``read`` selects the expert segmentation (``"mean"`` averaged,
    ``"e1"``/``"e2"`` single-expert — the per-expert union is ``combine="or"`` in the search)."""
    t = DERIVED_LABEL_THRESHOLDS
    vcdr = _read_series(df, "vcdr", read)
    area = _read_series(df, "area_cdr", read)
    hcdr = _read_series(df, "hcdr", read)
    asym = _vcdr_asymmetry(df, _read_name(df, "vcdr", read))
    return {
        # --- CDR family (fixed literature cuts) ---
        "vcdr>=0.70": (vcdr >= t["vcdr"]).fillna(False),                      # ISGEO Cat I (Foster 2002)
        "vcdr>=0.60": (vcdr >= VCDR_SUSPECT).fillna(False),                   # suspect/glaucomatous cup
        "area-cdr>=0.70": (area >= t["area_cdr"]).fillna(False),              # ISGEO Cat I
        "hcdr>=0.70": (hcdr >= HCDR_CUT).fillna(False),                       # horizontal CDR
        "disc-size-adjusted cup": _disc_size_adjusted_positive(df, read=read),  # Chákṣu external (fixed)
        # --- inter-eye asymmetry (fixed) ---
        "asymmetry>=0.20": (asym >= t["vcdr_asymmetry"]).fillna(False),       # ISGEO Cat I, unfloored
        # asymmetry floored at the CONVENTIONAL glaucoma-suspect CDR cut (0.50) = the ADOPTED floor
        # (ASYMMETRY_VCDR_FLOOR): pop. mean vCDR ~0.4, CDR>=0.5 = standard "suspect"; 0 healthy FP standalone.
        # The 0.45-floored variant was dropped — not uniquely helpful (same patient-level glaucoma FN as 0.50).
        "asymmetry>=0.20 & vcdr>=0.5": ((asym >= t["vcdr_asymmetry"])
                                        & (vcdr >= VCDR_ASYMMETRY_SUSPECT_FLOOR)).fillna(False),
        # (asymmetry>=0.30 dropped — an uncited "higher-specificity" ad-hoc cut; ISGEO's recognised
        #  inter-eye asymmetry threshold is >=0.20, so the candidate pool stays literature-grounded.)
        # --- neuroretinal rim ISNT-family rules (fixed) ---
        "full-ISNT violation": (~_isnt_ordered(df, ("isnt_i", "isnt_s", "isnt_n", "isnt_t"),
                                               read=read)).fillna(False),     # Jonas/Harizman I>=S>=N>=T
        "IST violation & vcdr>=0.5": (_ist_violation(df, read=read)
                                      & (vcdr >= DERIVED_LABEL_THRESHOLDS_SENSITIVE["isnt_support_vcdr"])
                                      ).fillna(False),                        # Higashide 2022 (nasal dropped)
        "IS violation": (~_isnt_ordered(df, ("isnt_i", "isnt_s"), read=read)).fillna(False),  # Poon 2017
        # --- rim-to-disc / DDLS / focal ---
        "rim-disc-area<=p2.5": _below_limit(df, "rim_disc_area_ratio", RIM_DISC_AREA_RATIO_P2_5, read),  # Chákṣu ext
        "min-rim-disc<=p5": _below_limit(df, "min_rim_disc_ratio", MIN_RIM_DISC_RATIO_P5, read),         # Chákṣu ext
        "DDLS-adjusted narrowest rim": _ddls_adjusted_positive(df, read=read),   # Spaeth DDLS; Chákṣu ext
        "focal pole rim<0.1DD": _focal_pole_rim_positive(df, read=read),         # ISGEO focal notch
    }


# Candidate clauses whose thresholds are in-sample (PAPILA-healthy) normal limits re-derived out-of-fold
# by the search. NOW EMPTY: every candidate clause uses a fixed EXTERNAL threshold — the disc-size cup,
# rim-disc-area, min-rim and DDLS limits are all Chákṣu-derived (see DISC_SIZE_ADJUSTED_VCDR,
# RIM_DISC_AREA_RATIO_P2_5, MIN_RIM_DISC_RATIO_P5, DDLS_MIN_RIM_BY_TERTILE), so the search scores them all
# with their fixed external cuts and `_healthy_limits`/`_insample_masks` are no longer exercised.
IN_SAMPLE_CANDIDATE_CLAUSES: tuple[str, ...] = ()


def _sensitive_positive(df: pd.DataFrame, thresholds: dict[str, float] | None, combine: str) -> pd.Series:
    """Boolean positive mask for the sensitive read. ``combine="mean"`` = union of the averaged clauses;
    ``combine="or"`` = the **per-expert union** (fire if EITHER expert's contours trip any clause)."""
    reads = ("e1", "e2") if combine == "or" else ("mean",)
    positive = pd.Series(False, index=df.index)
    for r in reads:
        for mask in sensitive_clauses(df, thresholds, read=r).values():
            positive = positive | mask
    return positive


def fundus_structural_label_sensitive(
    df: pd.DataFrame, thresholds: dict[str, float] | None = None, combine: str = "or"
) -> pd.Series:
    """The adopted low-FNR fundus structural read — **identical to** :func:`fundus_suspicious_disc_label`,
    the classifier's target: the union of the **disc-size-adjusted cup** (Blue Mountains/Jonas) and the
    **IST rim rule** (I≥S≥T, Higashide 2022) — see :func:`sensitive_clauses`. ``combine="or"`` (default,
    adopted) takes the **per-expert union** — fire if EITHER grader's contours trip a clause, maximising
    sensitivity to a single grader's concern (and recovering RET042); ``combine="mean"`` uses the
    expert-averaged metrics (the more conservative, lower-sensitivity consensus read). ``<NA>`` if the eye
    has no vcdr. All clause thresholds are external/literature, so the reported specificity is non-circular."""
    vcdr = pd.to_numeric(df.get("vcdr"), errors="coerce")
    positive = _sensitive_positive(df, thresholds, combine)
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    scorable = vcdr.notna()
    y[scorable] = 0
    y[scorable & positive] = 1
    return y


# --- the fundus IMAGE classifier's train/test target = the adopted "sensitive" structural read --------
# These are exactly the :func:`sensitive_clauses` keys — `fundus_suspicious_disc_label` ==
# `fundus_structural_label_sensitive(combine="or")`. Selected by eda.fundus_label.label_combination_search
# (patient-level 5-fold CV vs the clinician diagnosis): the disc-size-adjusted cup (Blue Mountains / Jonas,
# Chákṣu-external normal limits) unioned with the IST rim rule (I>=S>=T & vcdr>=0.50; Higashide 2022),
# evaluated as a PER-EXPERT UNION (fire if either grader trips a clause). The union read maximises sensitivity
# to a single grader's borderline trace — it recovers one normal-tension patient (RET042) the averaged read
# hides. The flat VCDR/area >= 0.70 cuts are subsumed by the disc-size cut; ISGEO inter-eye asymmetry, focal
# min-rim, flat cup>=0.60 and larger-eye asymmetry were evaluated and dropped (they remain search candidates).
#
# WHY THIS IS THE TRAIN/EVAL LABEL: (a) DIAGNOSIS-INDEPENDENT & image-derivable — a fixed rule on the disc
# morphometry, so a fundus-image model learns it with NO diagnosis leakage; (b) built entirely from
# RECOGNISED CLINICAL CRITERIA (above); (c) the LOW-FNR ("sensitive") read — it lowers the bar below the
# strict 0.70 gold cut to catch borderline disc suspects, matching the triage priority that a missed
# suspicious disc is the dangerous error; (d) chosen EMPIRICALLY as the best clause union vs diagnosis.
FUNDUS_SUSPICIOUS_DISC_CLAUSES: tuple[str, ...] = (
    "disc-size-adjusted cup", "IST rule (I>=S>=T) & vcdr>=0.5",   # disc-size cup ∪ IST rim rule (per-expert union)
)


def fundus_suspicious_disc_label(df: pd.DataFrame) -> pd.Series:
    """The adopted **diagnosis-independent** binary suspicious-disc label — the fundus image classifier's
    train/test target and the project's "sensitive" structural read (== :func:`fundus_structural_label_sensitive`,
    ``combine="or"``). Positive (1) if either :data:`FUNDUS_SUSPICIOUS_DISC_CLAUSES` clause (disc-size-adjusted
    cup or IST rim rule) fires on **either** expert's contours (the per-expert union); 0 if scorable and none;
    ``<NA>`` without morphometry. No diagnosis, no override — image-derivable, no leakage (see the module
    comment for why this is the label)."""
    return fundus_structural_label_sensitive(df, combine="or")


_NON_HEALTHY = ("suspect", "glaucoma")  # clinician non-healthy classes (for the concordance tag)


def structure_concordance(df: pd.DataFrame) -> pd.Series:
    """Four-way **validation** tag of the dx-independent `fundus_suspicious_disc_label` vs the clinician
    diagnosis: ``glaucomatous`` (suspicious & non-healthy), ``physiological`` (suspicious & healthy),
    ``disc_invisible`` (not-suspicious & non-healthy), ``normal`` (not-suspicious & healthy). ``<NA>``
    where the label or the diagnosis is missing."""
    gt = fundus_suspicious_disc_label(df)
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    if "diagnosis" not in df.columns:
        return out
    nh, healthy = df["diagnosis"].isin(_NON_HEALTHY), df["diagnosis"] == "healthy"
    sus, notsus = (gt == 1).fillna(False), (gt == 0).fillna(False)  # NA gt -> no tag
    out[sus & nh] = "glaucomatous"
    out[sus & healthy] = "physiological"
    out[notsus & nh] = "disc_invisible"
    out[notsus & healthy] = "normal"
    return out


def assign_fundus_suspicious_disc(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the ``fundus_suspicious_disc`` label + its ``structure_concordance``
    validation tag added."""
    out = df.copy()
    out["fundus_suspicious_disc"] = fundus_suspicious_disc_label(df)
    out["structure_concordance"] = structure_concordance(df)
    return out


def triage_label(df: pd.DataFrame) -> pd.Series:
    """Binary Stage-2 triage label: discharge = 0, refer/DAVRAP = 1 (<NA> if missing)."""
    disp = df[TRIAGE_TARGET]
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y[disp == TRIAGE_POSITIVE] = 1
    y[(disp.notna()) & (disp != TRIAGE_POSITIVE)] = 0
    return y


def clinical_class(df: pd.DataFrame) -> pd.Series:
    """Three-class **benchmark** grouping — ``healthy`` / ``OHT`` / ``glaucoma`` — that re-splits PAPILA's
    non-glaucomatous *Group 2* by intraocular pressure. PAPILA's paper split Group 2 into healthy (IOP ≤ 22)
    vs suspect (ocular-hypertensive, IOP 22–28); here we re-split the **same Group 2** into **healthy vs OHT
    at IOP ≥ 24** (:data:`IOP_RAISED_THRESHOLD` — the NICE NG81 / GIRFT action threshold). Group 1 (glaucoma)
    is unchanged: a glaucoma eye stays ``glaucoma`` regardless of IOP (OHT = raised IOP *without* glaucomatous
    damage).

    ``"glaucoma"`` where the diagnosis is glaucoma; ``"OHT"`` for a non-glaucoma eye with ``iop`` ≥ 24;
    ``"healthy"`` otherwise (non-glaucoma with ``iop`` < 24 or IOP missing); ``<NA>`` where the diagnosis is
    missing (e.g. Harvard-GF). Reuses :func:`iop_raised_label` for the ≥ 24 test.

    IOP source (**internal note**): uses the ``iop`` column, which is **non-contact-first**
    (``Pneumatic.fillna(Perkins)``) — the paper names only the Nidek NT-2000 non-contact tonometer, so we
    trust NCT. This is a **benchmark** label meant to supply OHT cases for testing other methods, **not** a
    clinical OHT diagnosis: the recorded single IOP cannot reproduce PAPILA's own suspect band (see
    docs/FEATURES.md). The *published* caveat is that we treat this IOP as Goldmann-equivalent for the ≥ 24
    threshold."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    if "diagnosis" not in df.columns:
        return out
    dx = df["diagnosis"]
    raised = (iop_raised_label(df) == 1).fillna(False)     # iop >= 24 (NICE NG81), NA-safe
    out[dx.notna()] = "healthy"                            # default for any scored diagnosis
    out[dx.notna() & (dx != "glaucoma") & raised] = "OHT"  # non-glaucoma & raised IOP
    out[dx == "glaucoma"] = "glaucoma"                     # last: IOP-independent, overrides
    return out


# oct_structural_label(df): VF-derived binary label (functional-follows-structural). Derived
# in the next step from Harvard-GF's visual field — not implemented here.


# --- Stage 1 routing (rules over the structural read + IOP + field) ------------------------
# The Step-3 "Read & route" rule of the Triage_Decision-Tree v0.32 (docs/FEATURES.md; grounded in
# GIRFT Best Practice May 2026 + NICE NG81 §1.1): suspicious structure → GECFS; normal structure
# with raised IOP and/or an abnormal/unreliable field → GRMS; normal structure with a *reliable*
# normal field and IOP < 24 → clear-negative discharge; filing gap → return to referrer. Only the
# imaging read is ML; the routing is rules. PAPILA carries no VF reliability indices, so the
# "reliable"/"unreliable" field qualifier is unmodeled here (the field term is an MD-only proxy).
IOP_RAISED_THRESHOLD: float = 24.0  # mmHg (non-contact) — NICE NG81; the v0.32 tree (§8 confirm-item)
#                                     pins the "raised" / discharge boundary at IOP < 24

# "Abnormal field" MD proxy. The gold-standard abnormal-field definition is PSD p<5% or GHT outside
# normal limits (perimetric glaucoma); PAPILA carries ONLY MD (no PSD/GHT/reliability), so this is an
# MD-only proxy. It is NOT a leak: the raw missingness leak (MD present only for non-healthy) was
# removed by the validated VF-MD simulation (healthy MD imputed from a faithful normal distribution —
# see eda.data_status.vf_md_distribution_check), so MD is a legitimate functional feature.
FIELD_ABNORMAL_MD: dict[str, float] = {
    "normal_limit": -2.0,  # lower limit of the normal MD range (~95% normal band; StatPearls NBK585112)
    "hpa_moderate": -6.0,  # Hodapp-Parrish-Anderson mild/moderate boundary (Hodapp et al. 1993;
    #                        mild ≥ -6, moderate -6..-12, severe < -12)
}
FIELD_ABNORMAL_MD_DEFAULT: float = FIELD_ABNORMAL_MD["normal_limit"]

ROUTES = ("GECFS", "GRMS", "discharge", "referrer")

# The full SPOA disposition set of the Triage_Decision-Tree v0.32 (§2). Stage-1 routing (Step 3)
# emits only the four ``ROUTES`` above; the remaining dispositions belong to steps NOT built here —
# ``escalate`` (Step 1 urgency screen: IOP > 35 / ≥ 40, acute angle closure, advanced-late),
# ``DAVRAP`` (Step 2 valid pre-filtered referral, and the Step-4 no-capacity "safe default"), and
# ``abstain`` (low confidence / incomplete → repeat measures or human review). Declared now,
# logic later — the same declared-ahead-of-use pattern as :data:`CONVERSATIONAL`.
DISPOSITIONS = ("escalate", "GECFS", "GRMS", "discharge", "referrer", "DAVRAP", "abstain")

# Human-readable v0.32 disposition label for each Stage-1 (Step-3) route. Note the Step-3
# ``discharge`` is the *rules* clear-negative ("met no trigger"), distinct from the Step-4
# multimodal *paper-based-triage* discharge (a separate, not-yet-built decision).
# The two community routes, per v0.32 (and the LOCSU CMGs): GECFS = Glaucoma Enhanced Case-Finding,
# a full work-up incl. OCT for a disc suspect (LOCSU Glaucoma ECF, Jul 2025); GRMS = Glaucoma Repeat
# Measures, a repeat Goldmann IOP + repeat field to confirm a single/unreliable reading on a normal
# disc (OHT-type; LOCSU Glaucoma Repeat Readings, Feb 2019). NICE NG81 §1.1.2 ("do not base a decision
# to refer solely on non-contact-tonometry IOP" — NCT overestimates) is exactly why GRMS exists.
ROUTE_DISPOSITION_LABELS: dict[str, str] = {
    "GECFS": "Route to GECFS (enhanced case-finding)",
    "GRMS": "Route to GRMS (repeat measures)",
    "discharge": "Discharge — Step-3 clear-negative (met no trigger)",
    "referrer": "Return to referrer (filing gap)",
}


def iop_raised_label(df: pd.DataFrame, threshold: float = IOP_RAISED_THRESHOLD) -> pd.Series:
    """Binary raised-IOP label: ``iop >= threshold`` → 1, else 0; ``<NA>`` if IOP missing."""
    iop = pd.to_numeric(df.get("iop"), errors="coerce")
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y[iop.notna()] = 0
    y[iop >= threshold] = 1
    return y


def field_abnormal_label(df: pd.DataFrame, md_threshold: float = FIELD_ABNORMAL_MD_DEFAULT) -> pd.Series:
    """Binary abnormal-field label from VF-MD: ``vf_md < md_threshold`` → 1, else 0; ``<NA>`` if MD
    missing. MD-only proxy for the PSD/GHT gold standard; healthy MD is simulated-but-validated, so
    it is a legitimate functional feature, not a leak (see module notes)."""
    md = pd.to_numeric(df.get("vf_md"), errors="coerce")
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y[md.notna()] = 0
    y[md < md_threshold] = 1
    return y


def stage1_route(
    df: pd.DataFrame,
    combine: str = "mean",
    profile: str = "gold",
    iop_threshold: float | None = None,
    field_md_threshold: float = FIELD_ABNORMAL_MD_DEFAULT,
    use_field: bool = True,
    bilateral: bool = False,
) -> pd.Series:
    """Rules-based Stage-1 route per the v0.32 decision tree (Step 3, "Read & route") — an object
    Series in :data:`ROUTES` (the Step-3 subset of :data:`DISPOSITIONS`).

    IOP uses the **NICE NG81 gold standard ≥ 24 mmHg** for raised in every profile (discharge iff
    IOP < 24). The only difference is the structural read: ``profile="gold"`` →
    :func:`fundus_structural_label_derived` (``combine``); ``profile="sensitive"`` →
    :func:`fundus_structural_label_sensitive` (low-FNR); ``profile="disc"`` →
    :func:`fundus_suspicious_disc_label`, the chosen **dx-independent** suspicious-disc gate — the
    proposed Stage-1 structural read (no diagnosis, no leakage). ``use_field``
    adds the MD-proxy abnormal-field term to the GRMS/discharge split (VF-MD is a validated functional
    feature — the missingness leak was removed by simulation); ``use_field=False`` is the IOP-only PoC
    discharge (illustrative, not deployable — NICE 1.1.2). ``bilateral=True`` applies a patient-level
    rule (uses ``patient_id``): a ``discharge`` eye is escalated to its fellow's attention route
    (GECFS, else GRMS) when the fellow is flagged — glaucoma is asymmetric, so the patient is worked up
    together."""
    if profile == "sensitive":
        struct = fundus_structural_label_sensitive(df)
    elif profile == "disc":
        struct = fundus_suspicious_disc_label(df)
    else:
        struct = fundus_structural_label_derived(df, combine=combine)
    iop_threshold = IOP_RAISED_THRESHOLD if iop_threshold is None else iop_threshold
    pos = (struct == 1).fillna(False).astype(bool)    # NA-safe masks
    neg = (struct == 0).fillna(False).astype(bool)
    iop = pd.to_numeric(df.get("iop"), errors="coerce")
    md = pd.to_numeric(df.get("vf_md"), errors="coerce")

    raised_iop = iop >= iop_threshold                 # NICE NG81 ≥24 (NaN → False → referrer)
    normal_iop = iop < iop_threshold
    if use_field:
        abnormal_field = md < field_md_threshold
        normal_field = md >= field_md_threshold       # discharge needs a *confirmed* normal field
    else:
        abnormal_field = pd.Series(False, index=df.index)
        normal_field = pd.Series(True, index=df.index)

    route = pd.Series("referrer", index=df.index, dtype=object)  # default = filing gap / abstain
    route[pos] = "GECFS"
    route[neg & (raised_iop | abnormal_field)] = "GRMS"
    route[neg & normal_iop & normal_field] = "discharge"
    if bilateral and "patient_id" in df.columns:
        # patient-level (bilateral) rule: glaucoma is asymmetric, so a discharge eye is escalated to
        # accompany a flagged fellow — the patient is worked up together (NICE 'refer when in doubt').
        gecfs = route.eq("GECFS").groupby(df["patient_id"]).transform("any")
        grms = route.eq("GRMS").groupby(df["patient_id"]).transform("any")
        disc = route.eq("discharge")
        route = route.mask(disc & gecfs, "GECFS").mask(disc & grms & ~gecfs, "GRMS")
    return route


def clinical_class_rule(
    df: pd.DataFrame,
    combine: str = "mean",
    profile: str = "disc",
    iop_threshold: float | None = None,
    field_md_threshold: float = FIELD_ABNORMAL_MD_DEFAULT,
    use_field: bool = True,
    bilateral: bool = False,
) -> pd.Series:
    """Stage-1 **clinical-class classifier** — predicts :func:`clinical_class` (``healthy`` / ``OHT`` /
    ``glaucoma``) from the image + measurement signals, the direct-classification replacement for the
    community-pathway :func:`stage1_route`. A **priority cascade**: **damage** (suspicious disc OR abnormal
    field) → ``glaucoma``; else **raised IOP** (≥ 24) → ``OHT``; else ``healthy``.

    Glaucoma is **IOP-independent** (≈86% of PAPILA glaucoma is normal-tension), and it is either structural
    (a suspicious disc) or functional (an abnormal field) damage — so ``disc OR field``, not ``AND`` (an
    ``AND`` gate misses the disc-quiet and field-quiet glaucoma). OHT is raised IOP with a *normal* disc and
    field. Structural read by ``profile`` as in :func:`stage1_route` (``"disc"``/``"sensitive"`` →
    :func:`fundus_suspicious_disc_label`; ``"gold"`` → :func:`fundus_structural_label_derived`); the
    abnormal-field term is the MD-only proxy (dropped by ``use_field=False``). ``bilateral=True`` escalates an
    eye to ``glaucoma`` when its fellow is glaucoma (glaucoma is asymmetric — trims the disc+field-silent
    miss toward the RET026 floor). All masks NA-safe (missing signal → not fired)."""
    if profile == "sensitive":
        struct = fundus_structural_label_sensitive(df)
    elif profile == "disc":
        struct = fundus_suspicious_disc_label(df)
    else:
        struct = fundus_structural_label_derived(df, combine=combine)
    disc = (struct == 1).fillna(False).astype(bool)          # suspicious disc (NA → False)
    return _clinical_class_from_disc(df, disc, iop_threshold=iop_threshold,
                                     field_md_threshold=field_md_threshold, use_field=use_field,
                                     bilateral=bilateral)


def _clinical_class_from_disc(
    df: pd.DataFrame,
    disc: pd.Series,
    iop_threshold: float | None = None,
    field_md_threshold: float = FIELD_ABNORMAL_MD_DEFAULT,
    use_field: bool = True,
    bilateral: bool = False,
) -> pd.Series:
    """The clinical-class cascade over an **arbitrary** suspicious-disc mask ``disc`` (boolean) — the body
    of :func:`clinical_class_rule` factored out so the combination search can push a *candidate* clause
    union through the exact same damage→OHT→healthy rule. **damage** (``disc`` OR abnormal field) →
    ``glaucoma``; else **raised IOP** (≥ ``iop_threshold``) → ``OHT``; else ``healthy``. ``bilateral=True``
    escalates an eye to ``glaucoma`` when its fellow is glaucoma (glaucoma is asymmetric)."""
    iop_threshold = IOP_RAISED_THRESHOLD if iop_threshold is None else iop_threshold
    disc = pd.Series(disc, index=df.index).fillna(False).astype(bool)
    iop = pd.to_numeric(df.get("iop"), errors="coerce")
    md = pd.to_numeric(df.get("vf_md"), errors="coerce")
    raised_iop = iop >= iop_threshold                        # NICE NG81 ≥24 (NaN → False)
    abnormal_field = (md < field_md_threshold) if use_field else pd.Series(False, index=df.index)

    out = pd.Series("healthy", index=df.index, dtype=object)
    out[raised_iop] = "OHT"                                   # raised IOP, no damage
    out[disc | abnormal_field] = "glaucoma"                  # structural or functional damage — overrides
    if bilateral and "patient_id" in df.columns:
        fellow_glaucoma = out.eq("glaucoma").groupby(df["patient_id"]).transform("any")
        out = out.mask(out.ne("glaucoma") & fellow_glaucoma, "glaucoma")
    return out


def patient_referred(df: pd.DataFrame, **route_kwargs) -> pd.Series:
    """Patient-level (bilateral) aggregation of :func:`stage1_route`: per-eye boolean, ``True`` if
    **any** eye of the patient is routed to GECFS or GRMS (flagged for attention). Glaucoma is a
    patient-level, often-asymmetric disease, so a structure-silent eye is still worked up when its
    fellow is flagged. Deployable — uses ``patient_id`` (not the diagnosis). Defaults to the
    ``disc`` (fundus_suspicious_disc) structural gate; falls back to the eye's own attention route if
    there is no ``patient_id``."""
    route_kwargs.setdefault("profile", "disc")
    attention = stage1_route(df, **route_kwargs).isin(("GECFS", "GRMS"))
    if "patient_id" not in df.columns:
        return attention
    return attention.groupby(df["patient_id"]).transform("any")


# --- matrix builders (the interface downstream classifiers call) ---------------------------
def _encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in cols:
        if col == "sex":
            X["sex_male"] = df["sex"].map({"male": 1, "female": 0}).astype("Int64")
        elif col in CONVERSATIONAL:
            X[col] = df.get(col)  # object placeholder until populated
        else:
            X[col] = pd.to_numeric(df.get(col), errors="coerce")
    return X


def imaging_columns(modality: str) -> list[str]:
    if modality not in IMAGING_CLASSIFIERS:
        raise ValueError(f"modality must be one of {list(IMAGING_CLASSIFIERS)}, got {modality!r}")
    return list(IMAGING_CLASSIFIERS[modality]["features"])


def metadata_columns() -> list[str]:
    return list(METADATA_INPUTS)


def imaging_matrix(df: pd.DataFrame, modality: str) -> tuple[pd.DataFrame, pd.Series]:
    """Stage-1 imaging classifier ``(X, y)`` over that modality's source arm only.

    ``fundus`` → PAPILA rows with the clinician-dx structural label; ``oct`` → Harvard-GF rows
    (label is VF-derived → <NA> here, filled in the next step)."""
    spec = IMAGING_CLASSIFIERS[modality] if modality in IMAGING_CLASSIFIERS else None
    if spec is None:
        raise ValueError(f"modality must be one of {list(IMAGING_CLASSIFIERS)}, got {modality!r}")
    sub = df[df["source"] == spec["source"]]
    X = _encode(sub, list(spec["features"]))
    y = fundus_structural_label(sub) if modality == "fundus" else pd.Series(
        pd.NA, index=sub.index, dtype="Int64"
    )
    return X, y


def metadata_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Stage-2 metadata classifier ``(X, y_triage)``. Meant for the arm that carries metadata
    + disposition (PAPILA); ``sex`` is binary-encoded as ``sex_male``, missing → ``NaN``."""
    return _encode(df, METADATA_INPUTS), triage_label(df)


def stage2_matrix(
    df: pd.DataFrame, modality: str = "fundus", include_conversational: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage-2 fused ``(X, y_triage)``: metadata + one imaging read (+ conversational).

    Modular late fusion is per-modality in practice; this concatenates the columns so a single
    fused model can also be trained/ablated. Missing values stay ``NaN`` (branch drops)."""
    cols = list(METADATA_INPUTS) + imaging_columns(modality)
    if include_conversational:
        cols += CONVERSATIONAL
    return _encode(df, cols), triage_label(df)


def minimum_set(modality: str = "fundus") -> list[str]:
    """Provisional minimum set for an autonomous Stage-2 decision (abstain below it): the
    chosen imaging read + core metadata. The exact set + cut-offs are a clinical confirm-item."""
    return imaging_columns(modality) + ["iop", "age"]


def has_minimum_set(df: pd.DataFrame, modality: str = "fundus") -> pd.Series:
    """Boolean mask of rows carrying the (provisional) minimum set for the chosen modality."""
    present = pd.Series(True, index=df.index)
    for col in minimum_set(modality):
        present &= pd.to_numeric(df[col], errors="coerce").notna()
    return present
