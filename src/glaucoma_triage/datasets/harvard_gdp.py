"""Harvard-GDP1000 loader for the progression go/no-go probe (baseline-only, leakage-safe).

The CSV manifest (`data_summary.csv`) carries everything the VF-only Stage-0 probe needs — **baseline**
visual field (`md` + `td1..td54`, blind-spot points td25/td34 absent) + the six progression labels +
demographics + splits — for the 500 progression-labelled rows (of 1000). OCT RNFLT-map features
(:func:`rnflt_features`) are read from the per-eye `.npz` only when the 9.5 GB zip has been extracted
(Stage 1).

**Leakage discipline (critical):** inputs are the **single baseline** VF (and baseline OCT); the labels
are FUTURE progression flags computed from a follow-up VF series that this dataset does not even release.
So baseline-VF → future-progression is legitimate (not the AUC≈1 trivial case) but **VF-anchored** — the
VF modality has a built-in edge, so weight the OCT-given-VF direction and the interaction, not VF-given-OCT
alone, when reading the probe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "harvard_gdp"
# The six released progression labels (all VF-derived). td_pointwise has the most events (47/500);
# md_fast is unusable (10/500). Values are 0/1 for labelled rows, "NA" otherwise.
PROGRESSION_LABELS = ("td_pointwise", "md", "vfi", "md_fast", "md_fast_no_p_cut", "td_pointwise_no_p_cut")


def _td_columns(df: pd.DataFrame) -> list[str]:
    """The baseline total-deviation columns present in the manifest (td1..td54 minus blind spot)."""
    return [c for c in df.columns if c.startswith("td") and c[2:].isdigit()]


def _vf_summary(df: pd.DataFrame, td_cols: list[str]) -> pd.DataFrame:
    """Compact, low-dimensional baseline-VF features (kept small for ~45 events): the global MD plus
    lower-tail total-deviation summaries (focal/scotoma burden)."""
    td = df[td_cols].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["vf_md"] = pd.to_numeric(df["md"], errors="coerce")
    out["vf_td_mean"] = td.mean(axis=1)
    out["vf_td_p10"] = td.quantile(0.10, axis=1)
    out["vf_td_p25"] = td.quantile(0.25, axis=1)
    out["vf_frac_td_below_m5"] = (td < -5).mean(axis=1)
    return out


# compact feature name lists (the probe's modality specs)
VF_FEATURES = ["vf_md", "vf_td_mean", "vf_td_p10", "vf_td_p25", "vf_frac_td_below_m5"]


def load_harvard_gdp(raw_dir: Path, summary_csv: str = "data_summary.csv",
                     labeled_only: bool = True) -> pd.DataFrame:
    """One row per eye: keys + compact baseline-VF features + progression labels + demographics + split.

    ``labeled_only`` keeps the 500 rows with a progression label (``progression_forecasting_use != NA``)."""
    csv = Path(raw_dir) / summary_csv
    raw = pd.read_csv(csv)
    if labeled_only:
        # pandas parses the manifest's "NA" split as NaN — drop both NaN and any literal "NA".
        use = raw["progression_forecasting_use"]
        raw = raw[use.notna() & (use.astype(str).str.upper() != "NA")].reset_index(drop=True)

    df = pd.DataFrame(index=raw.index)
    df["eye_id"] = raw["filename"].astype(str)
    df["patient_id"] = raw["filename"].astype(str)  # one eye per subject in Harvard-GDP
    df["source"] = SOURCE
    df["split"] = raw["progression_forecasting_use"].astype(str)
    for demo in ("age", "gender", "race", "hispanic"):
        if demo in raw.columns:
            df[demo] = raw[demo]

    df = pd.concat([df, _vf_summary(raw, _td_columns(raw))], axis=1)

    for lab in PROGRESSION_LABELS:
        col = f"progression.{lab}"
        if col in raw.columns:
            v = pd.to_numeric(raw[col], errors="coerce")  # "NA" -> NaN
            df[f"prog_{lab}"] = v.astype("Int64")
    return df


_GRID = 3  # 3x3 spatial grid over the RNFLT map


def rnflt_features(npz_path: Path) -> dict[str, float] | None:
    """OCT features from an eye's 225x225 RNFLT thickness map (Stage 1) — a fair structural
    representation: global distribution (mean/std/percentiles + thinning fraction) **plus a 3x3 spatial
    grid** of cell means (16 features) so regional RNFL loss (not just global thinning) can register.
    (Clinical superior/inferior sectorisation isn't reliable here — the maps aren't co-registered
    OD/OS — so a data-driven spatial grid is the honest choice.) ``None`` if the map is missing."""
    p = Path(npz_path)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        if "rnflt" not in z.files:
            return None
        m = np.asarray(z["rnflt"], dtype=float)
    m = np.where(np.isfinite(m) & (m >= 0), m, np.nan)  # mask the ~-2 outside-scan sentinel
    nm = np.nanmean
    feats: dict[str, float] = {
        "rnflt_mean": float(nm(m)), "rnflt_std": float(np.nanstd(m)),
        "rnflt_p5": float(np.nanpercentile(m, 5)), "rnflt_p10": float(np.nanpercentile(m, 10)),
        "rnflt_p25": float(np.nanpercentile(m, 25)), "rnflt_p50": float(np.nanpercentile(m, 50)),
        "rnflt_frac_lt50": float(np.nanmean(m < 50)),  # thinning burden
    }
    h, w = m.shape[-2], m.shape[-1]
    ys = np.linspace(0, h, _GRID + 1, dtype=int)
    xs = np.linspace(0, w, _GRID + 1, dtype=int)
    for i in range(_GRID):
        for j in range(_GRID):
            feats[f"rnflt_g{i}{j}"] = float(nm(m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]))
    return feats


OCT_FEATURES = (["rnflt_mean", "rnflt_std", "rnflt_p5", "rnflt_p10", "rnflt_p25", "rnflt_p50",
                 "rnflt_frac_lt50"] + [f"rnflt_g{i}{j}" for i in range(_GRID) for j in range(_GRID)])
# Compact, a-priori set (3) for the low-event probe: global thickness + focal thinning + thinning
# burden — matched to VF_FEATURES-compact so the 2-modality model stays at healthy events/predictor.
OCT_FEATURES_COMPACT = ["rnflt_mean", "rnflt_p5", "rnflt_frac_lt50"]
VF_FEATURES_COMPACT = ["vf_md", "vf_frac_td_below_m5"]
