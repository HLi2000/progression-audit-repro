"""UWHVF loader (+ thin adapters exposing other cohorts' pointwise VF as canonical fields).

UWHVF (Montesano et al., TVST 2022) is the training anchor + format-variant source for the VF
format-tolerance study (P4-reframed). One JSON file, ``alldata.json``::

    {pts, eyes, hvfs,
     coords: [{x, y, index}, ...],                       # the 24-2 spatial layout (seq order)
     data: {"<pid>": {gender, year, "R": [rec, ...], "L": [rec, ...]}}}

    rec = {age, hvf, hvf_seq, td, td_seq}                # 100.0 marks an untested location

**One row = one HVF test** (not one eye) — a bespoke frame off the eye-keyed harmonized ``schema``
(the Harvard-GDP / GRAPE precedent). UWHVF is unlabeled, so the interpretation target is VF-intrinsic:
``md`` = mean total deviation (the repo's MD-analog convention, cf. ``grape.vf_mean_sens``).

Sign convention: total deviation, loss = negative (Humphrey), matching the rest of the repo. The
per-test ``td_seq``/``sens_seq`` are stored with untested locations as ``NaN`` and align by index to
:func:`coords`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "uwhvf"
_UNTESTED = 100.0
_LAT = {"R": "OD", "L": "OS"}


def _clean_seq(seq: list) -> np.ndarray:
    """A pointwise sequence with the 100.0 untested sentinel mapped to NaN."""
    v = np.asarray(seq, dtype=float)
    v[v == _UNTESTED] = np.nan
    return v


def coords(raw_dir: Path, json_name: str = "alldata.json") -> np.ndarray:
    """The shared 24-2 layout as an ``(n_points, 2)`` array of ``(x, y)`` in **degrees**, ordered by
    the ``index`` field so it aligns to each test's ``td_seq``/``sens_seq``.

    UWHVF ships ``coords`` as 8x9 matrix indices, not degrees; they are converted to a centered
    6-degree grid (``x=(col-nx/2)*6``, ``y=(ny/2-row)*6``). **NOTE (Phase 1r §0):** this legacy conversion is a
    *transpose + y-mis-phase* of the true anatomical 24-2 and is **no longer the pipeline grid** — the drivers now
    use :data:`vf_format.canonical.CANON_COORDS` (the corrected anatomical 24-2, same UWHVF seq-index order, so a
    ``td_seq`` still aligns by index → identity resample). This function is retained only for reference/tests."""
    with open(Path(raw_dir) / json_name, encoding="utf-8") as fh:
        blob = json.load(fh)
    return _coords_from_blob(blob)


def _coords_from_blob(blob: dict) -> np.ndarray:
    cs = sorted(blob["coords"], key=lambda c: c["index"])
    xi = np.array([c["x"] for c in cs], dtype=float)
    yi = np.array([c["y"] for c in cs], dtype=float)
    xdeg = (xi - xi.max() / 2.0) * 6.0
    ydeg = (yi.max() / 2.0 - yi) * 6.0
    return np.column_stack([xdeg, ydeg])


def load_uwhvf(raw_dir: Path, json_name: str = "alldata.json") -> pd.DataFrame:
    """One row per HVF test: keys + age/sex + pointwise ``td_seq``/``sens_seq`` + ``md`` (mean TD)."""
    with open(Path(raw_dir) / json_name, encoding="utf-8") as fh:
        blob = json.load(fh)
    recs = []
    for pid, pdata in blob["data"].items():
        gender = str(pdata.get("gender", "")).strip().upper()
        sex_male = 1 if gender.startswith("M") else 0
        for eye_key, lat in _LAT.items():
            for i, rec in enumerate(pdata.get(eye_key, []) or []):
                td = _clean_seq(rec["td_seq"])
                sens = _clean_seq(rec["hvf_seq"])
                valid = td[~np.isnan(td)]
                recs.append({
                    "vf_test_id": f"{pid}_{lat}_{i}",
                    "patient_id": str(pid), "source": SOURCE, "laterality": lat,
                    "age": rec.get("age"), "sex_male": sex_male, "year": pdata.get("year"),
                    "md": float(valid.mean()) if valid.size else np.nan,
                    "n_valid": int(valid.size),
                    "td_seq": td, "sens_seq": sens,
                })
    return pd.DataFrame(recs)


# --- thin adapters: other cohorts' pointwise Humphrey VF as (values, coords, md) fields ----------
# Each returns a list of {"values": td-vector (NaN=missing), "coords": (n,2), "md": mean-TD,
# "patient_id": str} — the same field shape the vf_format layer consumes. These are the real
# held-out ladder rungs; they are best-effort (wrapped by the driver) since they depend on the
# other cohorts being downloaded.

def _field(values: np.ndarray, coords_xy: np.ndarray, patient_id: str) -> dict:
    v = np.asarray(values, dtype=float)
    valid = v[~np.isnan(v)]
    return {"values": v, "coords": np.asarray(coords_xy, dtype=float),
            "md": float(valid.mean()) if valid.size else np.nan, "patient_id": str(patient_id)}


def harvard_gf_fields(raw_dir: Path, target_coords: np.ndarray, *,
                      with_label: bool = False) -> list[dict]:
    """Harvard-GF Humphrey 24-2 total-deviation fields (list-valued ``vf_tds`` + ``vf_md``).

    Harvard-GF ships the TD vector but not its per-point coordinates; we assume it aligns to the
    canonical 24-2 layout (``target_coords``) truncated/padded to the TD length. Institution rung.
    ``with_label=True`` attaches the binary glaucoma label (``diagnosis == 'glaucoma'`` → 1) as
    ``field['label']`` — used as the **clinical** interpretation target (not the MD-derived screening flag)."""
    from . import harvard_gf as hgf
    df = hgf.load_harvard_gf(raw_dir)
    out = []
    for _, r in df.iterrows():
        tds = r.get("vf_tds")
        if tds is None or (isinstance(tds, float) and np.isnan(tds)):
            continue
        v = np.asarray(tds, dtype=float)
        n = min(len(v), len(target_coords))
        fld = _field(v[:n], target_coords[:n], r["patient_id"])
        nm = r.get("vf_md")                                  # shipped real HFA MD → native GT for R2
        if nm is not None and np.isfinite(nm):
            fld["native_md"] = float(nm)
        if with_label:
            fld["label"] = int(str(r.get("diagnosis", "")).strip().lower() == "glaucoma")
        out.append(fld)
    return out


def harvard_gdp_fields(raw_dir: Path, target_coords: np.ndarray,
                       summary_csv: str = "data_summary.csv",
                       progressors_only: str | None = None) -> list[dict]:
    """Harvard-GDP1000 Humphrey 24-2 total-deviation fields from the CSV manifest (``td1..td54`` +
    ``md``; blind-spot td25/td34 absent → 52 points). CSV-only (no 9.5 GB OCT zip needed) — the
    algorithm/institution rung. Same canonical-alignment caveat as :func:`harvard_gf_fields`.

    ``progressors_only``: a progression-label name (e.g. ``"td_pointwise"``) → return ONLY the baseline fields of
    eyes flagged as future progressors (``progression.<label> == 1``), tagged ``label=1``. These are the
    most-certain glaucoma (their VF worsened over time), used as a high-confidence **recall** cohort — a stronger
    positive than a recruited dx or a same-field MD threshold (both VF-derived/circular)."""
    import pandas as pd
    df = pd.read_csv(Path(raw_dir) / summary_csv)
    td_cols = sorted((c for c in df.columns if c.startswith("td") and c[2:].isdigit()),
                     key=lambda c: int(c[2:]))  # numeric order (td1..td54), not lexical
    if not td_cols:
        return []
    prog = None
    if progressors_only is not None:
        col = f"progression.{progressors_only}"
        if col not in df.columns:
            return []
        prog = pd.to_numeric(df[col], errors="coerce")   # "NA" → NaN; progressors == 1
    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        if prog is not None and prog.iloc[i] != 1:
            continue
        v = pd.to_numeric(r[td_cols], errors="coerce").to_numpy(dtype=float)
        if np.isnan(v).all():
            continue
        n = min(len(v), len(target_coords))
        pid = str(r.get("filename", r.name))
        fld = _field(v[:n], target_coords[:n], pid)
        nm = pd.to_numeric(r.get("md"), errors="coerce")     # shipped real HFA MD → native GT for R2
        if pd.notna(nm):
            fld["native_md"] = float(nm)
        if prog is not None:
            fld["label"] = 1
        out.append(fld)
    return out


def rotterdam_fields(raw_dir: Path, target_coords: np.ndarray) -> list[dict]:
    """Rotterdam Humphrey 24-2 **Full-Threshold** total-deviation fields — the algorithm rung.

    Joins ``VFPoints.csv`` (per-point ``TOTAL_DEVIATION`` at real ``(X, Y)`` degrees) to
    ``VisualFields.csv`` (``STUDY_ID`` per ``FIELD_ID``). Unlike the Harvard adapters, Rotterdam ships
    **real** point coordinates, so each field carries its own ``(X, Y)`` (reoriented to the canonical
    frame via :func:`vf_format.canonical.best_orientation`) and is placed spatially by resampling — a
    **spatially valid** rung (not just level-valid)."""
    import pandas as pd

    from ..vf_format.canonical import best_orientation
    raw = Path(raw_dir)
    vf = pd.read_csv(raw / "VisualFields.csv")
    pts = pd.read_csv(raw / "VFPoints.csv")
    field_to_subj = dict(zip(vf["FIELD_ID"], vf["STUDY_ID"]))
    field_to_md = dict(zip(vf["FIELD_ID"], pd.to_numeric(vf.get("MD"), errors="coerce"))) if "MD" in vf else {}
    out = []
    for fid, g in pts.groupby("FIELD_ID"):
        v = pd.to_numeric(g["TOTAL_DEVIATION"], errors="coerce").to_numpy(dtype=float)
        xy = g[["X", "Y"]].to_numpy(dtype=float)
        if np.isnan(v).all():
            continue
        fld = _field(v, best_orientation(xy), field_to_subj.get(fid, fid))
        fld["aligned"] = True                                  # real (X,Y) coords → HLO-weightable rung
        nm = field_to_md.get(fid)                              # shipped Full-Threshold MD → native GT for R2
        if nm is not None and np.isfinite(nm):
            fld["native_md"] = float(nm)
        out.append(fld)
    return out


def grape_octopus_fields(raw_dir: Path, *, harmonize: bool = False,
                         normative: dict | None = None) -> list[dict]:
    """GRAPE baseline Octopus fields (61 raw sensitivities, higher=better) — the cross-instrument rung.

    ``harmonize=False`` (default): native sensitivity units → the extreme-OOD probe (breaks: wrong
    units/sign, MD ≈ +21). ``harmonize=True``: convert to an HFA total-deviation analog via
    :mod:`vf_format.harmonize` (sign flipped, age-normal subtracted) so it is commensurable with the
    UWHVF-trained layer — the "can we bridge Octopus?" test. Coords are an approximate radial Octopus-G
    layout; no paired ground truth (see study notes)."""
    from ..vf_format.harmonize import sensitivity_to_td
    from . import grape
    base = grape.load_grape(raw_dir)
    pts = grape.followup_vf_points(raw_dir)  # {eye_id: (t, P[n_visit,61])}
    ozc = _octopus_g_coords()
    out = []
    for _, r in base.iterrows():
        eid = r["eye_id"]
        if eid not in pts:
            continue
        _, P = pts[eid]
        v = np.array(P[0], dtype=float)    # baseline visit (copy — don't mutate the loader's array)
        v[v < 0] = np.nan                  # -1 blind-spot sentinel
        n = min(len(v), len(ozc))
        vals, cds = v[:n], ozc[:n]
        if harmonize:
            vals = sensitivity_to_td(vals, cds, r.get("age"), normative=normative)
        out.append(_field(vals, cds, r["patient_id"]))
    return out


def _octopus_g_coords(n: int = 61) -> np.ndarray:
    """Approximate radial Octopus-G test layout (concentric rings) — used only to place the OOD
    rung on a coordinate grid for resampling; exact G-pattern positions are not published per-point."""
    rings = [(0, 1), (5, 6), (10, 12), (15, 14), (22, 16), (28, 12)]
    pts = [(0.0, 0.0)]
    for radius, count in rings[1:]:
        for k in range(count):
            ang = 2.0 * np.pi * k / count
            pts.append((radius * np.cos(ang), radius * np.sin(ang)))
    arr = np.array(pts, dtype=float)
    if len(arr) >= n:
        return arr[:n]
    pad = np.zeros((n - len(arr), 2), dtype=float)
    return np.vstack([arr, pad])
