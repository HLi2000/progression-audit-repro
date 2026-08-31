"""The harmonized per-eye schema shared by both arms.

One row = one eye. The two public arms stay **separate and modular** (proposal v0.45 — no
grafting); this schema just gives both a common column vocabulary. Columns are grouped by the
role each signal plays in the two-stage agent (Stage 1 structural imaging classifiers; Stage 2
metadata → +imaging → +conversational triage). A signal an arm does not carry is left missing
(``NaN``/``None``) so a modality branch can drop and the decision abstains below the minimum
set. Single source of truth for column names, order, and validation.
"""

from __future__ import annotations

import pandas as pd

# Canonical columns grouped by proposal role. Order here is the on-disk column order.
COLUMN_ROLES: dict[str, list[str]] = {
    "keys": ["eye_id", "patient_id", "source", "laterality", "split"],
    "structural": [
        "vcdr", "hcdr", "area_cdr", "disc_area", "cup_area", "rim_area",
        "rim_disc_area_ratio", "isnt_s", "isnt_i", "isnt_n", "isnt_t", "isnt_violation",
        "min_rim_disc_ratio",  # focal minimum rim fraction (BMO-MRW analog)
        # per-expert label metrics (exp1/exp2) — feed the per-expert-union derived labels
        # (gold OR-of-experts + the sensitive/suspicious-disc union: disc-size cup & IST per expert)
        "vcdr_e1", "vcdr_e2", "area_cdr_e1", "area_cdr_e2",
        "isnt_violation_e1", "isnt_violation_e2",
        "disc_area_e1", "disc_area_e2",
        "isnt_i_e1", "isnt_i_e2", "isnt_s_e1", "isnt_s_e2", "isnt_t_e1", "isnt_t_e2",
        "rnflt_path", "rnflt_mean", "rnflt_p5", "rnflt_sup", "rnflt_inf",
    ],
    "functional": ["vf_md", "vf_tds"],
    "risk": ["iop", "cct", "axial_length", "refraction", "age", "sex"],
    "conversational": ["symptoms", "family_history"],  # declared now, populated later
    "label": ["diagnosis_raw", "diagnosis", "disposition", "is_suspect", "clinical_class"],
    # the dx-independent fundus suspicious-disc label (the image classifier's target) + its
    # label-vs-diagnosis validation tag (see features.py). PAPILA-only; Harvard-GF leaves it NA.
    "target": ["fundus_suspicious_disc", "structure_concordance"],
    "fairness": ["race", "ethnicity"],
    "image": ["fundus_path"],
    # Provenance flags (metadata, never model inputs): raw VF-MD missingness leak, and
    # whether a row's VF-MD was synthesised for a healthy eye (see datasets/simulate.py).
    "provenance": ["vf_md_leakage", "vf_md_simulated"],
}

SCHEMA_COLUMNS: list[str] = [col for cols in COLUMN_ROLES.values() for col in cols]

# Signals reported in the proposal's coverage table, mapped to the schema column whose
# presence indicates the signal is available for a row.
SIGNAL_COLUMNS: dict[str, str] = {
    "fundus disc structure (vCDR)": "vcdr",
    "optic-disc OCT (RNFLT)": "rnflt_path",
    "visual field (MD)": "vf_md",
    "visual field (TDs)": "vf_tds",
    "IOP": "iop",
    "CCT": "cct",
    "axial length / refraction": "axial_length",
    "age / sex": "age",
    "race / ethnicity": "race",
    "disposition label (incl. suspect)": "disposition",
    "fundus image": "fundus_path",
}

VALID_DISPOSITIONS = {"refer", "discharge"}
VALID_SOURCES = {"papila", "harvard_gf"}
VALID_CONCORDANCE = {"glaucomatous", "physiological", "disc_invisible", "normal"}
VALID_CLINICAL_CLASS = {"healthy", "OHT", "glaucoma"}


def empty_frame() -> pd.DataFrame:
    """An empty DataFrame with every schema column present."""
    return pd.DataFrame({col: pd.Series(dtype="object") for col in SCHEMA_COLUMNS})


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing schema columns (as NA) and return columns in canonical order."""
    out = df.copy()
    for col in SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    extra = [c for c in out.columns if c not in SCHEMA_COLUMNS]
    return out[SCHEMA_COLUMNS + extra]


def validate(df: pd.DataFrame) -> None:
    """Raise ``AssertionError`` if the frame violates the schema's invariants."""
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    assert not missing, f"missing schema columns: {missing}"
    assert df["eye_id"].is_unique, "eye_id must be unique"
    assert df["source"].dropna().isin(VALID_SOURCES).all(), "invalid source value"
    disp = df["disposition"].dropna()
    assert disp.isin(VALID_DISPOSITIONS).all(), "disposition must be refer/discharge"
    lat = df["laterality"].dropna()
    assert lat.isin({"OD", "OS"}).all(), "laterality must be OD/OS"
    for ratio in ("vcdr", "hcdr", "area_cdr", "vcdr_e1", "vcdr_e2", "area_cdr_e1", "area_cdr_e2"):
        vals = pd.to_numeric(df[ratio], errors="coerce").dropna()
        assert ((vals > 0) & (vals <= 1.5)).all(), f"{ratio} out of plausible range"
    for frac in ("rim_disc_area_ratio", "min_rim_disc_ratio"):
        vals = pd.to_numeric(df[frac], errors="coerce").dropna()
        assert ((vals >= 0) & (vals <= 1)).all(), f"{frac} out of [0, 1]"
    for area in ("disc_area", "cup_area", "rim_area", "isnt_s", "isnt_i", "isnt_n", "isnt_t",
                 "disc_area_e1", "disc_area_e2", "isnt_i_e1", "isnt_i_e2",
                 "isnt_s_e1", "isnt_s_e2", "isnt_t_e1", "isnt_t_e2"):
        vals = pd.to_numeric(df[area], errors="coerce").dropna()
        assert (vals >= 0).all(), f"{area} must be non-negative"
    sus = pd.to_numeric(df["fundus_suspicious_disc"], errors="coerce").dropna()
    assert sus.isin((0, 1)).all(), "fundus_suspicious_disc must be 0/1"
    conc = df["structure_concordance"].dropna()
    assert conc.isin(VALID_CONCORDANCE).all(), "structure_concordance has an invalid tag"
    cc = df["clinical_class"].dropna()
    assert cc.isin(VALID_CLINICAL_CLASS).all(), "clinical_class must be healthy/OHT/glaucoma"
