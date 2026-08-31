"""Literature-calibrated cross-device RNFL offset transform — the novel core of OCT-FormatShift.

The device-tolerance headline is unmeasurable from our data directly: every public disc-OCT source is
single-vendor, so *device is aliased with cohort* and the real inter-vendor effect is not identified
(see docs/OCT_FORMATSHIFT_SCOPING.md §0). We import identification from OUTSIDE: the synthetic offset
transform is the manipulation we control (identified by construction), and its parameters are read off
**published same-eye cross-device RNFL agreement** — the paired-eye ground truth others already measured.

The transform has TWO parts, matching how the measurement-science literature reports cross-device
agreement (a Bland–Altman BIAS + LIMITS-OF-AGREEMENT):

  * a deterministic per-sector **BIAS** (the mean shift) — ``sev·a_s + (1 + sev·(b_s − 1))·x``. A
    pure-additive offset is WRONG: real cross-vendor bias is fixed **and** proportional (slope 0.83–1.27),
    so a single global offset understates real disagreement 2–3× and mis-specifies the thick S/I bundles.
  * a **PER-EYE, ANATOMY-COUPLED spread** (the LoA) — the published LoA is a BETWEEN-EYE quantity: two
    devices scanning the SAME eye disagree by a per-eye amount, not by i.i.d. per-pixel noise. So the
    spread is drawn ONCE per eye (coherent across the map) and decomposed into three literature-anchored
    mechanisms (segmentation ⊕ centration ⊕ magnification) added in quadrature, calibrated so the global
    difference SD reproduces the published 95% LoA half-width (= 1.96·sd). See the mechanism block below.

``severity`` in [0, 2] is the dose knob: 0 → identity, 1 → the published gap, 2 → double (the sweep x-axis).

Published anchors (all SAME-EYE / same-visit; input assumed **Cirrus**, our on-disk primary):
  * Cirrus->Spectralis  — Warner et al. 2011 (PMC3154907, n=95): Spectralis-Cirrus mean +1.94 um,
    95% LoA -5.74..+9.62 (slope ~1.0, near-additive; corroborated Buchser 2014 PMC4074471 +4.67+/-6.55).
  * Cirrus->Topcon-3D   — Buchser et al. 2012 (IOVS, PMC3390182): Cirrus = -15.521 + 1.056*Topcon3D
    => Topcon = 0.947*Cirrus + 14.70 (Topcon reads ~11 um thicker); LoA ~ +/-20 um (PMID 23225090).
  * Cirrus->RTVue       — Buchser 2012: Cirrus = 2.136 + 0.831*RTVue => RTVue = 1.203*Cirrus - 2.57
    (RTVue thickest; scale bias 1.204, 95% CI 1.094-1.317).

Calibration to published summary statistics (not raw pairs) is the simulation-based-inference / ABC
paradigm (Cranmer et al. PNAS 2020; microsimulation calibration Rutter et al. 2019); reproduction is
checked by TOST equivalence to the published bias (Schuirmann 1987; Lakens 2017). The Topcon/ONH arm
is thinly sourced -> flagged ``extrapolated`` (widen or treat as uncalibrated).

This module is representation-agnostic: ``apply`` transforms a per-eye thickness array with one sector's
BIAS + per-eye spread; ``apply_sectored_map`` applies per-quadrant BIAS + the per-eye anatomy-coupled
spread to a disc-centred RNFLT map by angular wedge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter

from . import canonical

# Within-eye spatial texture as a fraction of the pair LoA σ. MEAN-ZERO over the disc, so the calibrated
# BETWEEN-EYE LoA (what Warner measured) is preserved, while the deployed map gains realistic spatially-
# correlated within-sector variation (segmentation differences are spatially correlated, not hard-quadrant
# blocks; Patel 2012) instead of the old piecewise-constant offset.
WITHIN_SECTOR_TEX_FRAC: float = 0.4

# RNFL sectors (global + Superior/Nasal/Inferior/Temporal quadrants; matches datasets.grape.OCT_FEATURES_FULL).
SECTORS = ("global", "s", "n", "i", "t")


# --------------------------------------------------------------------------------------------------
# Per-eye anatomy-coupled LoA mechanisms — the STRUCTURAL spread model (the v3 upgrade).
#
# The published cross-device LoA (Warner σ≈3.92 µm global) is a BETWEEN-EYE quantity: two devices scanning
# the SAME eye disagree by a per-eye amount. A deterministic appearance/offset transform reproduces only the
# BIAS (the mean shift); the LoA SPREAD needs PER-EYE, ANATOMY-COUPLED perturbations drawn ONCE per eye
# (coherent across the map) — an i.i.d. per-pixel noise term AVERAGES OUT at the eye level, so the deployed
# map transform would carry almost no LoA (the bug this replaces). We decompose the spread into three
# literature-anchored mechanisms, added in quadrature:
#   seg  — per-eye RNFL boundary-placement difference. The LARGEST term: Patel 2012 (PMC3299866) shows
#          forcing a COMMON segmentation collapses the Cirrus↔Spectralis gap (difference σ 4.7→3.0 µm), i.e.
#          segmentation is the dominant cross-device variance component. ≈uniform in µm across sectors.
#   cent — per-eye scan-circle decentration (~0.1 mm). Cheung/Leung Eye 2008 (PMID 18806767): 2.3 µm/0.1 mm
#          GLOBAL but 5–6 µm/0.1 mm SUPERIOR/INFERIOR — the thick, steep bundles shift most → sup/inf-loaded.
#   mag  — per-eye ocular-magnification / axial-length mismatch scaling the sampled circle radius (axial
#          length negatively correlates with measured RNFL via magnification, ~−2 to −3 µm/mm in the
#          literature; Littmann/Bennett correction removes it — PMC4162641 gives the axial-length THRESHOLDS
#          that need correcting, not this slope). Mild sup/inf loading (radial RNFL gradient); smallest term.
# Only the GLOBAL magnitude is CALIBRATED (the quadrature = the published σ); the per-sector PROFILE is
# mechanism-MOTIVATED (direction, not a precise per-sector calibration — there is no clean published
# per-sector cross-device LoA). ``SECTOR_SD_SCALE`` is DERIVED from this model so the per-eye SCALAR path
# (``apply`` / ``difference_stats``) and the per-eye MAP path (``apply_sectored_map``) agree by construction.
# NOTE (honest): the per-sector pattern here is the cross-device AGREEMENT pattern (superior/inferior wider,
# centration-driven), which is DIFFERENT from the same-device REPRODUCIBILITY pattern (nasal noisiest;
# Soltani-Moghadam 2015 / Mwanza 2010). This axis models cross-device AGREEMENT, so it follows Cheung/Leung.
_MECH_SD_UM: dict[str, float] = {"seg": 2.8, "cent": 2.3, "mag": 1.5}   # per-mechanism between-eye SD (µm), sev 1
_MECH_SECTOR_GAIN: dict[str, dict[str, float]] = {
    "seg":  {"s": 1.0, "n": 1.0, "i": 1.0, "t": 1.0},   # boundary bias ≈ uniform in µm
    "cent": {"s": 1.6, "n": 0.4, "i": 1.6, "t": 0.4},   # superior/inferior-loaded (Cheung/Leung); mean 1.0
    "mag":  {"s": 1.3, "n": 0.7, "i": 1.3, "t": 0.7},   # mild sup/inf (radial gradient); mean 1.0
}
# quadrature global σ (the Warner anchor) — ≈3.92 µm by construction: sqrt(2.8²+2.3²+1.5²).
_MECH_GLOBAL_SD: float = float(np.sqrt(sum(s * s for s in _MECH_SD_UM.values())))


def _mech_sector_sigma(sector: str) -> float:
    """Between-eye σ (µm, before the per-pair scale) of one quadrant from the quadrature of mechanisms."""
    return float(np.sqrt(sum((_MECH_SD_UM[k] * _MECH_SECTOR_GAIN[k][sector]) ** 2 for k in _MECH_SD_UM)))


# Per-sector SD multiplier (vs global) — DERIVED from the mechanism model (superior/inferior wider than
# nasal/temporal: the cross-device AGREEMENT pattern, centration-driven). Both the scalar and map paths
# read their per-sector σ from this, so they are consistent by construction.
SECTOR_SD_SCALE: dict[str, float] = {
    "global": 1.0,
    **{s: _mech_sector_sigma(s) / _MECH_GLOBAL_SD for s in ("s", "n", "i", "t")},
}


@dataclass(frozen=True)
class BAParams:
    """Bland-Altman offset params for one vendor pair at one sector (severity 1.0).

    ``a`` intercept (um), ``b`` slope (proportional bias), ``sd`` SD of the (B-A) difference (um);
    the published 95% LoA half-width is ``1.96 * sd``. ``extrapolated`` flags a thinly-sourced arm.
    """

    a: float
    b: float
    sd: float
    source: str
    extrapolated: bool = False


# Global-sector calibration, input = Cirrus. Per-quadrant params derive from these via SECTOR_SD_SCALE
# unless a citable quadrant regression is added.
PUBLISHED: dict[str, BAParams] = {
    "cirrus_to_spectralis": BAParams(
        a=1.94, b=1.0, sd=3.92,
        source="Warner 2011 PMC3154907 (LoA -5.74..+9.62); Leite 2011 PMC3102819; Buchser 2014 PMC4074471",
    ),
    "cirrus_to_topcon": BAParams(
        a=14.70, b=0.947, sd=11.0, source="Buchser 2012 PMC3390182 (Cirrus=-15.5+1.056*Topcon3D); PMID 23225090 LoA~+/-20",
        extrapolated=True,  # Topcon-at-ONH paired data is thin -> widen / treat as uncalibrated
    ),
    "cirrus_to_rtvue": BAParams(
        a=-2.57, b=1.203, sd=9.0, source="Buchser 2012 PMC3390182 (Cirrus=2.136+0.831*RTVue; scale bias 1.204)"
    ),
}

PAIRS = tuple(PUBLISHED)


@dataclass(frozen=True)
class Provenance:
    """Calibration provenance for one pair (the audit's per-pair disclosure — not all pairs are equal).

    ``calib_source`` = the study the (a,b,sd) were read from; ``independent_bias_um`` = the bias RANGE
    reported by OTHER (independent) studies for the same pair, kept as CONTEXT for the provenance
    disclosure (not a pass/fail check); ``status`` is the honest one-word grade.
    """

    status: str                              # "multi-source" | "single-source-extrapolated"
    calib_source: str
    independent_bias_um: tuple[float, float] | None
    independent_sources: str
    note: str = ""


# Independent published cross-vendor RNFL biases (studies OTHER than the calibration source) → the
# provenance CONTEXT. Ranges (not point values) because cross-vendor bias is genuinely cohort/method-
# dependent; disclosed so a reader sees how well-sourced each pair is (no consistency verdict is claimed).
PROVENANCE: dict[str, Provenance] = {
    "cirrus_to_spectralis": Provenance(
        status="multi-source", calib_source="Warner 2011 (+1.94)",
        independent_bias_um=(1.9, 7.2),
        independent_sources="Mwanza 2010 +1.9 (concordant); Buchser 2014 +4.67; Patel 2012 +6.7-7.2 "
                            "(centration-regime, collapses to ~0 with recentring + common segmentation, PMC3299866)",
        note="best-sourced pair; b≈1.0 additive; the high end is centration/segmentation-driven, disclosed."),
    "cirrus_to_rtvue": Provenance(
        status="multi-source", calib_source="Buchser 2012 slope 1.203",
        independent_bias_um=(9.0, 13.0),
        independent_sources="Leite 2011 RTVue +9.77 vs Cirrus (330 same eyes, DIGS)",
        note="proportional bias; slope from Buchser, magnitude corroborated by Leite; sd itself uncited."),
    "cirrus_to_topcon": Provenance(
        status="single-source-extrapolated", calib_source="Buchser 2012 regression",
        independent_bias_um=None,
        independent_sources="PMID 23225090 LoA~±20 (a clean independent BIAS is not isolable)",
        note="Topcon-at-ONH paired data thin → EXTRAPOLATED; NO clean out-of-sample validation available."),
}


def provenance(pair: str) -> Provenance:
    """Per-pair calibration provenance + independent-literature context (the audit's honesty disclosure)."""
    if pair not in PROVENANCE:
        raise KeyError(f"unknown vendor pair {pair!r}; known: {PAIRS}")
    return PROVENANCE[pair]


def params_for(pair: str, sector: str = "global") -> BAParams:
    """Published params for ``pair`` at ``sector`` (global regression + per-sector SD scaling)."""
    if pair not in PUBLISHED:
        raise KeyError(f"unknown vendor pair {pair!r}; known: {PAIRS}")
    if sector not in SECTOR_SD_SCALE:
        raise KeyError(f"unknown sector {sector!r}; known: {tuple(SECTOR_SD_SCALE)}")
    base = PUBLISHED[pair]
    return replace(base, sd=base.sd * SECTOR_SD_SCALE[sector])


def apply(
    x: np.ndarray,
    pair: str,
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    sector: str = "global",
) -> np.ndarray:
    """Apply the calibrated offset to a PER-EYE thickness array (one value per eye).

    ``T(x) = sev*a + (1 + sev*(b-1))*x + N(0, (sev*sd)^2)`` elementwise. On a per-eye array each element is a
    different eye, so the i.i.d. draw IS the between-eye LoA (per-eye is the level the published Bland–Altman
    was measured at). For a 2-D MAP use ``apply_sectored_map`` (the anatomy-coupled per-eye spread) — do NOT
    pass a map's flat pixels here, as the per-pixel i.i.d. noise would average out at the eye level.
    ``severity`` 0 -> identity, 1 -> the published gap, 2 -> double. NaNs pass through untouched.
    """
    if severity < 0:
        raise ValueError("severity must be >= 0")
    p = params_for(pair, sector)
    x = np.asarray(x, dtype=float)
    out = severity * p.a + (1.0 + severity * (p.b - 1.0)) * x
    noise_sd = severity * p.sd
    if noise_sd > 0:
        out = out + rng.normal(0.0, noise_sd, size=out.shape)
    return np.where(np.isfinite(x), out, x)


def _bias(x: np.ndarray, pair: str, *, severity: float, sector: str) -> np.ndarray:
    """Deterministic per-sector mean shift (the published BIAS): ``sev*a + (1 + sev*(b-1))*x``. No noise."""
    p = params_for(pair, sector)
    x = np.asarray(x, dtype=float)
    return severity * p.a + (1.0 + severity * (p.b - 1.0)) * x


def eye_spread(pair: str, rng: np.random.Generator, *, severity: float = 1.0) -> dict[str, float]:
    """Per-eye anatomy-coupled LoA spread (µm) per quadrant — the BETWEEN-EYE device-agreement term.

    Draws the three mechanism latents ONCE (so the perturbation is COHERENT across the eye) and returns the
    per-quadrant offset ``sev * scale * Σ_k MECH_SD[k]·u_k·gain_k[sector]``, where ``scale`` matches this
    pair's published global σ. Mean-zero; each mechanism's quadrant gains average to 1, so the per-eye GLOBAL
    offset has SD = the published σ (Warner for Spectralis), while superior/inferior quadrants carry a wider
    LoA than nasal/temporal (centration-driven). Replaces the old i.i.d. per-pixel noise (which averaged out
    at the eye level, leaving the deployed map transform with almost no LoA).
    """
    if pair not in PUBLISHED:
        raise KeyError(f"unknown vendor pair {pair!r}; known: {PAIRS}")
    if severity < 0:
        raise ValueError("severity must be >= 0")
    scale = severity * PUBLISHED[pair].sd / _MECH_GLOBAL_SD
    u = {k: float(rng.standard_normal()) for k in _MECH_SD_UM}
    return {sec: scale * sum(_MECH_SD_UM[k] * u[k] * _MECH_SECTOR_GAIN[k][sec] for k in _MECH_SD_UM)
            for sec in ("s", "n", "i", "t")}


def _nan_aware_blur(field: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur ``field`` over the valid (disc) region only, so quadrant boundaries feather smoothly
    without the zero background bleeding inward (normalise by the blurred validity mask)."""
    num = gaussian_filter(np.where(valid, field, 0.0), sigma=sigma)
    den = gaussian_filter(valid.astype(float), sigma=sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def apply_sectored_map(
    m: np.ndarray,
    pair: str,
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply the per-quadrant calibrated BIAS + a PER-EYE ANATOMY-COUPLED spread to a 2-D RNFLT map (um).

    The bias is the deterministic per-sector mean shift; the spread is drawn ONCE per eye (``eye_spread``),
    so the between-eye LoA survives the eye-level averaging a map reader performs (an i.i.d. per-pixel noise
    would average to ≈0). Superior/inferior quadrants carry a wider spread than nasal/temporal (centration-
    driven). The per-quadrant offset is then made SPATIALLY REALISTIC — the hard quadrant boundaries are
    FEATHERED (a NaN-aware blur) and a low-frequency, MEAN-ZERO within-sector texture is added — because two
    of the three LoA mechanisms are inherently smooth (centration = geometric circle-shift, Cheung–Leung 2008;
    magnification = radial rescale, PMC4162641) and segmentation differences are spatially CORRELATED, not
    hard blocks (Patel 2012). Because the texture is mean-zero over the disc, the calibrated BETWEEN-EYE LoA
    (Warner) is preserved; only the WITHIN-eye spatial appearance changes (validated by ``validity``).
    """
    m = np.asarray(m, dtype=float)
    if m.ndim != 2:
        raise ValueError("apply_sectored_map expects a 2-D map")
    valid = np.isfinite(m)
    spread = eye_spread(pair, rng, severity=severity)
    # per-pixel offset = bias delta + per-eye per-quadrant spread (hard, then feathered)
    delta = np.zeros_like(m)
    for sector, mask in canonical.quadrant_masks(m.shape, center).items():
        delta[mask] = (_bias(m[mask], pair, severity=severity, sector=sector) - m[mask]) + spread[sector]
    delta = _nan_aware_blur(delta, valid, sigma=max(m.shape) / 14.0)     # feather the quadrant boundaries
    if severity > 0:                                                      # low-frequency within-sector texture
        tex = gaussian_filter(rng.standard_normal(m.shape), sigma=max(m.shape) / 10.0)
        tv = tex[valid]
        if tv.std() > 1e-6:
            tex = (tex - tv.mean()) / tv.std() * (WITHIN_SECTOR_TEX_FRAC * severity * PUBLISHED[pair].sd)
            delta = delta + np.where(valid, tex, 0.0)
    return np.where(valid, m + delta, m)


def difference_stats(
    x: np.ndarray,
    pair: str,
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    sector: str = "global",
) -> dict[str, float]:
    """Bland-Altman stats of the SYNTHETIC PER-EYE difference (T(x) - x): bias + 95% LoA + SD."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    diff = apply(x, pair, rng, severity=severity, sector=sector) - x
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return {"bias": bias, "sd": sd, "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd, "n": int(x.size)}


def map_difference_stats(
    maps: np.ndarray,
    pair: str,
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    center: tuple[float, float] | None = None,
) -> dict[str, dict]:
    """Bland-Altman stats of the DEPLOYED per-eye map transform (``apply_sectored_map``): per-eye GLOBAL +
    per-quadrant (shifted − clean) bias, SD, 95% LoA half-width.

    Unlike ``difference_stats`` (a per-eye SCALAR round-trip), this measures the transform AS DEPLOYED on
    real maps — the check that the per-eye coherent spread survives eye-level averaging (the old per-pixel
    noise did not). The ``global`` entry additionally carries the raw per-eye ``diffs`` array (for TOST).
    """
    secs = ("s", "n", "i", "t")
    g_clean: list[float] = []
    g_shift: list[float] = []
    sc = {s: [] for s in secs}
    ss = {s: [] for s in secs}
    for m in maps:
        mc = canonical.clean(m)
        msh = apply_sectored_map(mc, pair, rng, severity=severity, center=center)
        g_clean.append(canonical.rnfl_global(mc))
        g_shift.append(canonical.rnfl_global(msh))
        rc = canonical.rnfl_sectors(mc, center)
        rs = canonical.rnfl_sectors(msh, center)
        for s in secs:
            sc[s].append(rc[s])
            ss[s].append(rs[s])

    def _ba(clean_v: list[float], shift_v: list[float], *, keep: bool = False) -> dict:
        d = np.asarray(shift_v, float) - np.asarray(clean_v, float)
        d = d[np.isfinite(d)]
        sd = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
        out = {"bias": float(np.mean(d)) if d.size else float("nan"), "sd": sd,
               "loa_half": 1.96 * sd, "n": int(d.size)}
        if keep:
            out["diffs"] = d
        return out

    result = {"global": _ba(g_clean, g_shift, keep=True)}
    for s in secs:
        result[s] = _ba(sc[s], ss[s])
    return result


def expected_bias(x_mean: float, pair: str, *, severity: float = 1.0, sector: str = "global") -> float:
    """Analytic expected difference bias E[T(x)-x] = sev*a + sev*(b-1)*mean(x) (proportional term)."""
    p = params_for(pair, sector)
    return severity * p.a + severity * (p.b - 1.0) * x_mean


def tost_equivalence(
    diffs: np.ndarray, target_bias: float, margin: float, *, alpha: float = 0.05
) -> dict[str, float | bool]:
    """TOST (two one-sided tests): is the mean of ``diffs`` equivalent to ``target_bias`` within +/-margin?

    Schuirmann 1987; Lakens 2017. Equivalent iff both one-sided tests reject at ``alpha`` (i.e. the
    combined p = max(p_lower, p_upper) < alpha).
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = diffs.size
    if n < 2:
        raise ValueError("need >= 2 differences for TOST")
    mean = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1)) / np.sqrt(n)
    df = n - 1
    if se == 0:
        equivalent = abs(mean - target_bias) < margin
        return {"equivalent": bool(equivalent), "p": 0.0 if equivalent else 1.0, "mean": mean, "n": n}
    t_lower = (mean - (target_bias - margin)) / se  # H0: mean <= target-margin ; reject if t_lower large
    t_upper = (mean - (target_bias + margin)) / se  # H0: mean >= target+margin ; reject if t_upper small
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p = max(p_lower, p_upper)
    return {"equivalent": bool(p < alpha), "p": p, "mean": mean, "n": n}


def reproduces_published(
    x: np.ndarray,
    pair: str,
    rng: np.random.Generator,
    *,
    sector: str = "global",
    bias_margin: float = 2.0,
    loa_ratio_bounds: tuple[float, float] = (0.8, 1.25),
    alpha: float = 0.05,
) -> dict[str, object]:
    """Validate that the calibrated transform at severity 1 reproduces the published bias + LoA (PER-EYE).

    (a) TOST that the synthetic per-eye mean difference is equivalent (within +/-``bias_margin`` um) to the
        analytic published bias for this cohort's value range;
    (b) the synthetic LoA half-width is within ``loa_ratio_bounds`` of the published 1.96*sd.
    This is the per-eye SCALAR round-trip; the DEPLOYED map transform is checked separately by
    ``validity.structural_loa`` (via ``map_difference_stats``), which also validates the sector structure.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    diff = apply(x, pair, rng, severity=1.0, sector=sector) - x
    p = params_for(pair, sector)
    target_bias = expected_bias(float(np.mean(x)), pair, sector=sector)
    tost = tost_equivalence(diff, target_bias, bias_margin, alpha=alpha)
    syn_loa_half = 1.96 * float(np.std(diff, ddof=1))
    pub_loa_half = 1.96 * p.sd
    ratio = syn_loa_half / pub_loa_half if pub_loa_half > 0 else float("nan")
    loa_ok = loa_ratio_bounds[0] <= ratio <= loa_ratio_bounds[1]
    return {
        "pair": pair,
        "sector": sector,
        "reproduces": bool(tost["equivalent"] and loa_ok),
        "bias_equivalent": bool(tost["equivalent"]),
        "loa_ok": bool(loa_ok),
        "target_bias": target_bias,
        "synthetic_bias": tost["mean"],
        "tost_p": tost["p"],
        "synthetic_loa_half": syn_loa_half,
        "published_loa_half": pub_loa_half,
        "loa_ratio": ratio,
        "extrapolated": p.extrapolated,
    }
