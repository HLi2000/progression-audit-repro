"""Load PAPILA into the harmonized per-eye schema.

PAPILA ships (CC BY 4.0, figshare 14798004):
  * ``ClinicalData/`` — two spreadsheets (right eye / left eye) with age, gender,
    diagnosis (0=healthy, 1=glaucoma, 2=suspect), refraction, IOP (Pneumatic +
    Perkins), pachymetry, axial length, and VF mean deviation.
  * ``FundusImages/`` — ``RET###{OD,OS}.jpg`` colour fundus photos.
  * ``ExpertsSegmentations/`` — optic cup/disc contours from two ophthalmologists,
    from which we derive the vertical cup-to-disc ratio (vCDR), the demonstrator
    structural read that stands in for OCT.

Leakage note: PAPILA's VF MD is recorded only for non-healthy eyes, so the *presence*
of ``vf_md`` correlates with the label. We keep the value but stamp ``vf_md_leakage`` so
it is documented and excluded from any downstream feature set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import disc_morphology, schema

# PAPILA clinical spreadsheets use a fixed 13-column layout with a merged two-row header
# (dioptre_1/2/astigmatism and Pneumatic/Perkins are second-row sub-headers; real data
# begins at the first row whose id cell looks like "#NNN"). We therefore read by position.
# These names are what ``_finalize_clinical`` consumes (``_``-prefixed ones are temporary).
PAPILA_COLUMNS: list[str] = [
    "_id",              # 0  patient id, e.g. "#002"
    "age",              # 1
    "_gender",          # 2  0=male, 1=female
    "diagnosis_raw",    # 3  0=healthy, 1=glaucoma, 2=suspect
    "_dioptre_1",       # 4  refraction, principal meridian 1
    "_dioptre_2",       # 5  refraction, principal meridian 2
    "_astigmatism",     # 6
    "_phakic",          # 7  0=phakic, 1=pseudophakic
    "_iop_pneumatic",   # 8  non-contact IOP
    "_iop_perkins",     # 9  applanation IOP (reference)
    "cct",              # 10 pachymetry
    "axial_length",     # 11
    "vf_md",            # 12 visual field mean deviation
]


def find_dir(root: Path, name: str) -> Path:
    """Locate a subdirectory named ``name`` anywhere under ``root``."""
    root = Path(root)
    if (root / name).is_dir():
        return root / name
    for cand in root.rglob(name):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"could not find '{name}' under {root}")


# ----------------------------------------------------------------------------- vCDR
def load_contour(path: Path) -> np.ndarray:
    """Parse a segmentation contour file into an (N, 2) array of (x, y) points."""
    pts: list[tuple[float, float]] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().replace(",", " ")
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # skip header/comment lines
    return np.asarray(pts, dtype=float)


def _vertical_extent(contour: np.ndarray) -> float:
    if contour.size == 0:
        return float("nan")
    return float(contour[:, 1].max() - contour[:, 1].min())


def compute_vcdr(cup: np.ndarray, disc: np.ndarray) -> float:
    """Vertical cup-to-disc ratio = vertical cup extent / vertical disc extent."""
    disc_v = _vertical_extent(disc)
    cup_v = _vertical_extent(cup)
    if not np.isfinite(disc_v) or disc_v <= 0:
        return float("nan")
    return cup_v / disc_v


def eye_vcdr(seg_dir: Path, patient_tag: str, eye: str) -> float:
    """Mean vCDR across all experts for one eye (``patient_tag`` like ``RET001``)."""
    vals: list[float] = []
    cups = sorted(seg_dir.rglob(f"{patient_tag}{eye}*cup*.txt"))  # contours live in Contours/
    for cup_path in cups:
        disc_path = Path(str(cup_path).replace("cup", "disc"))
        if not disc_path.exists():
            continue
        v = compute_vcdr(load_contour(cup_path), load_contour(disc_path))
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------------- clinical
def _read_clinical_sheet(path: Path) -> pd.DataFrame:
    """Read one PAPILA clinical spreadsheet by position (skips the merged header rows)."""
    raw = pd.read_excel(path, header=None)
    col0 = raw.iloc[:, 0].astype(str).str.strip()
    is_data = col0.str.match(r"^#?\d+$")  # id cells like "#002" or "2"; skips header rows
    if not is_data.any():
        raise ValueError(f"could not locate data rows in {path.name}")
    data = raw.iloc[is_data.to_numpy().argmax():].reset_index(drop=True)
    if data.shape[1] != len(PAPILA_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(PAPILA_COLUMNS)} columns, got {data.shape[1]}"
        )
    data.columns = PAPILA_COLUMNS
    return data


def _finalize_clinical(df: pd.DataFrame, laterality: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["patient_num"] = df.get("_id")  # non-underscore name survives the temp-col drop
    for num_col in ["age", "cct", "axial_length", "vf_md", "diagnosis_raw",
                    "_dioptre_1", "_dioptre_2", "_iop_pneumatic", "_iop_perkins", "_gender"]:
        if num_col in df:
            out[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # IOP: use non-contact Pneumatic (near-complete, the method the proposal references),
    # falling back to the sparse Perkins applanation only when Pneumatic is missing.
    out["iop"] = out.get("_iop_pneumatic")
    if "_iop_perkins" in out:
        out["iop"] = out["iop"].fillna(out["_iop_perkins"])

    # Refraction: spherical equivalent as the mean of the two principal-meridian dioptres.
    if "_dioptre_1" in out and "_dioptre_2" in out:
        out["refraction"] = out[["_dioptre_1", "_dioptre_2"]].mean(axis=1)

    # Sex: PAPILA codes 0=male, 1=female.
    out["sex"] = out.get("_gender").map({0: "male", 1: "female"}) if "_gender" in out else pd.NA

    out["laterality"] = laterality
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def _patient_tag(raw_id) -> str:
    """Normalize a clinical ID to the ``RET###`` tag used in file names."""
    s = str(raw_id).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"RET{int(digits):03d}" if digits else s


def load_papila(raw_dir: Path) -> pd.DataFrame:
    """Build the PAPILA per-eye frame in the harmonized schema."""
    raw_dir = Path(raw_dir)
    clinical_dir = find_dir(raw_dir, "ClinicalData")
    fundus_dir = find_dir(raw_dir, "FundusImages")
    try:
        seg_dir = find_dir(raw_dir, "ExpertsSegmentations")
    except FileNotFoundError:
        seg_dir = None

    xlsx = sorted(clinical_dir.glob("*.xls*"))
    if len(xlsx) < 2:
        raise FileNotFoundError(f"expected two clinical spreadsheets in {clinical_dir}")

    frames = []
    for path in xlsx:
        stem = path.stem.lower()
        laterality = "OS" if ("os" in stem or "left" in stem) else "OD"
        frames.append(_finalize_clinical(_read_clinical_sheet(path), laterality))
    clin = pd.concat(frames, ignore_index=True)

    # Build id strings with f-strings (robust to arrow-backed vs object string dtypes).
    clin["patient_tag"] = [_patient_tag(x) for x in clin["patient_num"]]
    clin["patient_id"] = [f"{t}-papila" for t in clin["patient_tag"]]
    clin["eye_id"] = [
        f"{tag}{lat}-papila" for tag, lat in zip(clin["patient_tag"], clin["laterality"])
    ]
    clin["source"] = "papila"
    clin["split"] = "all"

    # Derived structural read (vCDR + rim / ISNT morphometry) + image path per eye. The
    # expert cup/disc contours give vertical & area cup-to-disc ratios, neuroretinal-rim area,
    # and per-sector rim widths for the ISNT rule (see datasets/disc_morphology.py).
    morpho_keys = ["vcdr", "hcdr", "area_cdr", "disc_area", "cup_area", "rim_area",
                   "rim_disc_area_ratio", "isnt_s", "isnt_i", "isnt_n", "isnt_t",
                   "min_rim_disc_ratio"]
    metrics, fundus_paths = [], []
    for _, row in clin.iterrows():
        tag, eye = row["patient_tag"], row["laterality"]
        metrics.append(disc_morphology.eye_disc_metrics(seg_dir, tag, eye, eye))
        img = fundus_dir / f"{tag}{eye}.jpg"
        fundus_paths.append(str(img.relative_to(raw_dir)) if img.exists() else pd.NA)
    for key in morpho_keys:
        clin[key] = [m[key] for m in metrics]
    clin["isnt_violation"] = pd.array([m["isnt_violation"] for m in metrics], dtype="boolean")

    # Per-expert label metrics (exp1, exp2), so the per-expert-union derived labels and the
    # inter-expert reliability check can be computed straight from the table. Persist the metrics
    # that feed the derived-label rules per expert: the gold trio (vcdr, area_cdr, isnt_violation)
    # plus disc_area and the IST sectors (isnt_i/s/t) the sensitive union's disc-size-cup & IST
    # clauses need per expert.
    def _expert_val(m, idx, key):
        pe = m["per_expert"]
        default = None if key == "isnt_violation" else float("nan")
        return pe[idx][key] if idx < len(pe) else default

    for e in (1, 2):
        clin[f"vcdr_e{e}"] = [_expert_val(m, e - 1, "vcdr") for m in metrics]
        clin[f"area_cdr_e{e}"] = [_expert_val(m, e - 1, "area_cdr") for m in metrics]
        clin[f"isnt_violation_e{e}"] = pd.array(
            [_expert_val(m, e - 1, "isnt_violation") for m in metrics], dtype="boolean")
        clin[f"disc_area_e{e}"] = [_expert_val(m, e - 1, "disc_area") for m in metrics]
        for sec in ("i", "s", "t"):
            clin[f"isnt_{sec}_e{e}"] = [_expert_val(m, e - 1, f"isnt_{sec}") for m in metrics]
    clin["fundus_path"] = fundus_paths

    # Label mapping: glaucoma (1) or suspect (2) -> refer; healthy (0) -> discharge.
    clin["is_suspect"] = clin["diagnosis_raw"] == 2
    clin["diagnosis"] = clin["diagnosis_raw"].map({0: "healthy", 1: "glaucoma", 2: "suspect"})
    clin["disposition"] = np.where(clin["diagnosis_raw"].isin([1, 2]), "refer", "discharge")
    clin.loc[clin["diagnosis_raw"].isna(), "disposition"] = pd.NA

    # PAPILA VF MD leaks the label: it is recorded for ~100% of refer eyes but only ~3%
    # of discharge eyes, so its very presence encodes the label. Flag the whole arm's MD
    # as unusable as a feature (the leak is the missingness pattern, not any single row).
    clin["vf_md_leakage"] = True

    return schema.coerce(clin.drop(columns=["patient_tag", "patient_num"], errors="ignore"))
