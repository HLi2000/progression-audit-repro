"""GRAPE loader for the P1 pilot atlas (baseline modalities + longitudinal follow-up).

GRAPE (Huang et al., Scientific Data 2023) is an **all-glaucoma** longitudinal cohort — 263 eyes / 144
patients, 3–9 visits/eye — so it supports **progression only** (no diagnosis label). The core tabular
file `vf_clinical.xlsx` has two sheets with 2-row merged headers (data from row 3):

* **Baseline** (263 rows/eye): Subject#, Laterality, Age, Gender, IOP, CCT, Total-Visits, the three
  per-eye progression labels (PLR2/PLR3/MD, 0/1), OAG/ACG, sectoral OCT-RNFL (Mean/S/N/I/T, baseline
  only), and the baseline VF (61 raw Octopus sensitivity points, blind-spot/non-perceivable = −1).
* **Follow-up** (1,115 rows/visit): Subject#, Laterality, Visit#, Interval-Years, IOP, and the per-visit
  VF (61 points). CCT/RNFL do NOT repeat.

Eye identity = (Subject#, Laterality). Labels are VF-derived (PLR/MD-slope over the VF series) → same
**anchoring discipline** as Harvard-GDP (weight structure|VF and the interaction, not VF|structure).
Octopus VF: higher sensitivity = better (opposite sign to Humphrey TD).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from .disc_morphology import expert_metrics

SOURCE = "grape"

# baseline-sheet column indices (0-based), verified from the workbook header
_B = {
    "subject": 0, "lat": 1, "age": 2, "gender": 3, "iop": 4, "cct": 5, "total_visits": 6,
    "plr2": 7, "plr3": 8, "md": 9, "category": 10,
    "rnfl_mean": 11, "rnfl_s": 12, "rnfl_n": 13, "rnfl_i": 14, "rnfl_t": 15, "cfp": 16,
}
_B_VF = slice(19, 80)   # 61 VF points (cols 19..79)
_F = {"subject": 0, "lat": 1, "visit": 2, "interval_years": 3, "iop": 4, "cfp": 5}
_F_VF = slice(8, 69)    # 61 VF points on the follow-up sheet (cols 5=CFP,6=device,7=resolution)

# OCT PRIMARY clinical panel: global mean RNFL + inferior & superior quadrants — the arcuate sectors where
# glaucomatous RNFL loss concentrates (ISNT vulnerability); nasal/temporal are least glaucoma-informative.
OCT_FEATURES = ["rnfl_mean", "rnfl_i", "rnfl_s"]
OCT_FEATURES_FULL = ["rnfl_mean", "rnfl_s", "rnfl_n", "rnfl_i", "rnfl_t"]   # all quadrants (rep. sensitivity)
OCT_FEATURES_SINGLE = ["rnfl_mean"]                                        # single global metric (sensitivity)
# VF panel: GLOBAL mean sensitivity (an MD-analog) + FOCAL/pattern SD of the pointwise sensitivities (a
# PSD / loss-variance analog, the standard focal VF index). Octopus VF ships raw 61 points, not MD/PSD, so
# these are derived *analogs*. `vf_p10` (low-tail decile) is kept for the representation sensitivity only.
VF_FEATURES = ["vf_mean_sens", "vf_sd"]              # global (MD-analog) + focal/pattern (PSD-analog)
PROGRESSION_LABELS = ("plr2", "plr3", "md")


def _vf_summary(points: list) -> tuple[float, float, float]:
    """(mean sensitivity [MD-analog], pointwise SD [PSD/loss-variance analog], 10th-pct low-tail) over the
    valid VF points (−1 blind-spot sentinels masked)."""
    v = pd.to_numeric(pd.Series(points), errors="coerce")
    v = v[v >= 0]                                     # drop the −1 sentinels
    if v.empty:
        return (np.nan, np.nan, np.nan)
    return (float(v.mean()), float(v.std(ddof=0)), float(np.percentile(v, 10)))


def load_grape(raw_dir: Path, xlsx: str = "vf_clinical.xlsx") -> pd.DataFrame:
    """One row per eye from the Baseline sheet: keys + compact baseline modalities + progression labels."""
    wb = openpyxl.load_workbook(Path(raw_dir) / xlsx, read_only=True, data_only=True)
    ws = wb["Baseline"]
    recs = []
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[_B["subject"]] is None:
            continue
        subj, lat = row[_B["subject"]], row[_B["lat"]]
        mean_sens, vf_sd, p10 = _vf_summary(list(row[_B_VF]))
        recs.append({
            "eye_id": f"{subj}_{lat}", "patient_id": str(subj), "source": SOURCE,
            "laterality": lat, "age": row[_B["age"]],
            "sex_male": 1 if str(row[_B["gender"]]).strip().upper().startswith("M") else 0,
            "iop": row[_B["iop"]], "cct": row[_B["cct"]], "category": row[_B["category"]],
            "total_visits": row[_B["total_visits"]], "cfp": row[_B["cfp"]],
            "rnfl_mean": row[_B["rnfl_mean"]], "rnfl_s": row[_B["rnfl_s"]],
            "rnfl_n": row[_B["rnfl_n"]], "rnfl_i": row[_B["rnfl_i"]], "rnfl_t": row[_B["rnfl_t"]],
            "vf_mean_sens": mean_sens, "vf_sd": vf_sd, "vf_p10": p10,
            "prog_plr2": row[_B["plr2"]], "prog_plr3": row[_B["plr3"]], "prog_md": row[_B["md"]],
        })
    df = pd.DataFrame(recs)
    for c in ("prog_plr2", "prog_plr3", "prog_md"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


# Fundus-disc morphometry (from GRAPE's disc/cup JSON contours — one per visit's CFP — reusing
# disc_morphology). Only SCALE-INVARIANT metrics are exported: GRAPE uses ≥3 cameras at different
# resolutions, so pixel areas/widths are not comparable — but vCDR/area-CDR/rim & sector *ratios* are.
# Full scale-invariant disc-morphometry set (mirrors step-0's full-morphometry, minus pixel-scaled metrics):
FUNDUS_FEATURES_FULL = [                        # full scale-invariant morphometry (representation sensitivity)
    "fundus_vcdr", "fundus_hcdr", "fundus_area_cdr", "fundus_min_rim",
    "fundus_isnt_i", "fundus_isnt_s", "fundus_isnt_n", "fundus_isnt_t", "fundus_isnt_violation",
]
# PRIMARY clinical panel: global cupping (vertical CDR) · area cupping (area CDR) · focal rim narrowing
# (min rim/disc, a fundus BMO-MRW analog) — the three complementary axes of guideline disc assessment.
FUNDUS_FEATURES = ["fundus_vcdr", "fundus_area_cdr", "fundus_min_rim"]
FUNDUS_FEATURES_COMPACT = ["fundus_vcdr"]      # single guideline-standard metric (representation sensitivity)
FUNDUS_FEATURES_ALL = FUNDUS_FEATURES_FULL


def _load_shapes(jdir: Path, cfp) -> dict | None:
    """Load a CFP's disc/cup contour polygons ({'OD': disc, 'OC': cup}) from its `<stem>.json`."""
    if not cfp or str(cfp) == "None":
        return None
    jpath = jdir / f"{Path(str(cfp)).stem}.json"
    if not jpath.exists():
        return None
    with open(jpath, encoding="utf-8") as fh:
        return {s["label"]: np.asarray(s["points"], dtype=float)
                for s in json.load(fh).get("shapes", [])}


def _fundus_metrics(shapes: dict | None, laterality) -> dict | None:
    """Full scale-invariant disc morphometry (fundus_* keys) from contour shapes, or None if incomplete."""
    if not shapes or "OD" not in shapes or "OC" not in shapes:      # OD=disc, OC=cup (NOT laterality)
        return None
    m = expert_metrics(cup=shapes["OC"], disc=shapes["OD"], laterality=str(laterality))
    viol = m["isnt_violation"]
    return {
        "fundus_vcdr": m["vcdr"], "fundus_hcdr": m["hcdr"], "fundus_area_cdr": m["area_cdr"],
        "fundus_min_rim": m["min_rim_disc_ratio"],
        "fundus_isnt_i": m["isnt_i_ratio"], "fundus_isnt_s": m["isnt_s_ratio"],
        "fundus_isnt_n": m["isnt_n_ratio"], "fundus_isnt_t": m["isnt_t_ratio"],
        "fundus_isnt_violation": float(viol) if viol is not None else float("nan"),
    }


def fundus_features(raw_dir: Path, baseline_df: pd.DataFrame, json_subdir: str = "json") -> pd.DataFrame:
    """Per-eye BASELINE disc morphometry (full scale-invariant set) from the labelme JSON contours.

    Matches each eye's baseline `cfp` to its `<stem>.json`; laterality drives the ISNT frame. Returns
    `eye_id` + the `FUNDUS_FEATURES` columns (rows dropped where the contour is absent/incomplete)."""
    jdir = Path(raw_dir) / json_subdir
    recs = []
    for _, r in baseline_df.iterrows():
        fm = _fundus_metrics(_load_shapes(jdir, r.get("cfp")), r["laterality"])
        if fm is not None:
            recs.append({"eye_id": r["eye_id"], **fm})
    return pd.DataFrame(recs)


def followup_fundus_features(raw_dir: Path, xlsx: str = "vf_clinical.xlsx",
                             json_subdir: str = "json") -> pd.DataFrame:
    """Per-VISIT disc morphometry over the follow-up series → longitudinal fundus trajectory.

    One row per (eye, visit) with the `FUNDUS_FEATURES` columns, from that visit's `Corresponding CFP`
    (follow-up col 5) → `<stem>.json`. Enables fundus-history / serial-vCDR reliability / a structural
    (vCDR-slope) progression endpoint. NaN/absent contours are skipped."""
    jdir = Path(raw_dir) / json_subdir
    wb = openpyxl.load_workbook(Path(raw_dir) / xlsx, read_only=True, data_only=True)
    ws = wb["Follow-up"]
    recs = []
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[_F["subject"]] is None:
            continue
        lat = row[_F["lat"]]
        fm = _fundus_metrics(_load_shapes(jdir, row[_F["cfp"]]), lat)
        if fm is None:
            continue
        recs.append({"eye_id": f"{row[_F['subject']]}_{lat}", "patient_id": str(row[_F["subject"]]),
                     "visit": row[_F["visit"]], "interval_years": row[_F["interval_years"]], **fm})
    return pd.DataFrame(recs)


def load_grape_followup(raw_dir: Path, xlsx: str = "vf_clinical.xlsx") -> pd.DataFrame:
    """Long per-visit frame (for the time-axis sweep): eye_id, visit#, interval_years, iop, VF summary."""
    wb = openpyxl.load_workbook(Path(raw_dir) / xlsx, read_only=True, data_only=True)
    ws = wb["Follow-up"]
    recs = []
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[_F["subject"]] is None:
            continue
        mean_sens, vf_sd, p10 = _vf_summary(list(row[_F_VF]))
        recs.append({
            "eye_id": f"{row[_F['subject']]}_{row[_F['lat']]}",
            "patient_id": str(row[_F["subject"]]), "source": SOURCE,
            "visit": row[_F["visit"]], "interval_years": row[_F["interval_years"]],
            "iop": row[_F["iop"]], "cfp": row[_F["cfp"]], "vf_mean_sens": mean_sens, "vf_sd": vf_sd,
            "vf_p10": p10,
        })
    return pd.DataFrame(recs)


def followup_vf_points(raw_dir: Path, xlsx: str = "vf_clinical.xlsx") -> dict[str, tuple]:
    """Per-eye longitudinal **pointwise** VF for time-to-event derivation (pointwise linear regression).

    Returns ``{eye_id: (t, P)}`` where ``t`` = visit times (interval-years, sorted ascending) and ``P`` =
    the ``(n_visit, 61)`` Octopus sensitivity matrix (−1 blind-spot sentinels preserved; higher = better).
    Used by :func:`glaucoma_triage.progression.survival` to recompute a transparent PLR progression event.
    """
    wb = openpyxl.load_workbook(Path(raw_dir) / xlsx, read_only=True, data_only=True)
    ws = wb["Follow-up"]
    rows: dict[str, list] = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[_F["subject"]] is None:
            continue
        eid = f"{row[_F['subject']]}_{row[_F['lat']]}"
        rows.setdefault(eid, []).append(
            (row[_F["interval_years"]], np.asarray(row[_F_VF], dtype=float)))
    out: dict[str, tuple] = {}
    for eid, visits in rows.items():
        visits = sorted(visits, key=lambda v: (v[0] if v[0] is not None else 0.0))
        t = np.array([v[0] for v in visits], dtype=float)
        P = np.vstack([v[1] for v in visits])
        out[eid] = (t, P)
    return out
