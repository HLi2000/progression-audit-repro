"""Load Harvard-GF into the harmonized per-eye schema.

Primary path uses the ``data_summary.csv`` manifest shipped with the HF mirror — it carries
decoded demographics (gender/race/ethnicity), the glaucoma label, and the train/val/test
split for all 3,300 eyes — and reads only ``rnflt`` + VF (``md``, ``tds``) from each
``.npz``. If the manifest is absent, it falls back to scanning ``.npz`` files and decoding
their in-file integer fields.

The 200x200x200 ``oct_bscans`` volume is never read (and is dropped on download by the
slim step), keeping the arm to an RNFLT+VF footprint. Harvard-GF is a binary detection
dataset (no suspect class): ``glaucoma`` -> ``refer``, else ``discharge``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

_RACE = {0: "Asian", 1: "Black", 2: "White"}
_ETHNICITY = {0: "non-hispanic", 1: "hispanic", -1: "unknown"}


def _scalar(z, key):
    """Read a scalar-ish npz entry as a Python float, or None if absent."""
    if key not in z.files:
        return None
    try:
        return float(np.asarray(z[key]).reshape(-1)[0])
    except (ValueError, IndexError):
        return None


def _rnflt_summary(z) -> dict[str, float]:
    """Coarse RNFLT summaries over valid (positive, finite) pixels of the 200x200 map:
    global mean, 5th percentile (a focal-thinning proxy), and **superior / inferior** region
    means (top-half vs bottom-half rows). The superior/inferior split is a coarse *arcuate
    analog* — the two sectors where glaucomatous RNFL loss concentrates — and is
    **laterality-invariant** (rows, not columns), which matters because Harvard-GF ships no
    laterality. These are NOT the clinical disc-centred sector cpRNFL; the fusion model consumes
    the full map, these scalars are the tabular modality summaries."""
    nan = float("nan")
    if "rnflt" not in z.files:
        return {"rnflt_mean": nan, "rnflt_p5": nan, "rnflt_sup": nan, "rnflt_inf": nan}
    arr = np.asarray(z["rnflt"], dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    valid = arr[mask]
    if valid.size == 0:
        return {"rnflt_mean": nan, "rnflt_p5": nan, "rnflt_sup": nan, "rnflt_inf": nan}

    def _region_mean(rows: slice) -> float:
        sub = arr[rows]
        m = np.isfinite(sub) & (sub > 0)
        return float(sub[m].mean()) if m.any() else nan

    half = arr.shape[0] // 2                       # rows: top half = superior, bottom half = inferior
    return {
        "rnflt_mean": float(valid.mean()),
        "rnflt_p5": float(np.percentile(valid, 5)),
        "rnflt_sup": _region_mean(slice(0, half)),
        "rnflt_inf": _region_mean(slice(half, None)),
    }


def _imaging_fields(z) -> dict:
    """The imaging/functional signals we keep: RNFLT summaries + VF (md, tds)."""
    tds = z["tds"].astype(float).tolist() if "tds" in z.files else None
    return {**_rnflt_summary(z), "vf_md": _scalar(z, "md"), "vf_tds": tds}


def _npz_imaging(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return _imaging_fields(z)


def load_sample(path: Path) -> dict:
    """Load one npz fully from its in-file fields (scan-fallback path)."""
    with np.load(path, allow_pickle=True) as z:
        rec = _imaging_fields(z)
        male = _scalar(z, "male")
        race = _scalar(z, "race")
        # Real npz names this field "hispanic" (0/1); older docs call it "ethnicity".
        eth = _scalar(z, "ethnicity")
        if eth is None:
            eth = _scalar(z, "hispanic")
        rec.update(
            age=_scalar(z, "age"),
            sex={0: "female", 1: "male"}.get(male) if male is not None else pd.NA,
            race=_RACE.get(int(race)) if race is not None else pd.NA,
            ethnicity=_ETHNICITY.get(int(eth)) if eth is not None else pd.NA,
            _glaucoma=_scalar(z, "glaucoma"),
        )
        return rec


def _finalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    g = df.pop("_glaucoma") if "_glaucoma" in df else df["diagnosis_raw"]
    df["diagnosis_raw"] = g
    df["diagnosis"] = pd.Series(g).map({0: "non-glaucoma", 1: "glaucoma"})
    df["disposition"] = np.where(pd.Series(g) == 1, "refer", "discharge")
    df.loc[pd.Series(g).isna(), "disposition"] = pd.NA
    df["is_suspect"] = False       # binary detection dataset: no suspect class
    df["vf_md_leakage"] = False    # Harvard-GF VF is complete, not label-conditional
    return schema.coerce(df)


def _load_from_manifest(raw_dir: Path, csv_path: Path, splits_map: dict[str, str]) -> pd.DataFrame:
    manifest = pd.read_csv(csv_path)
    index = {p.name: p for p in raw_dir.rglob("*.npz")}
    rows = []
    for _, m in manifest.iterrows():
        fn = str(m["filename"]).strip()
        path = index.get(fn)
        if path is None:
            continue
        stem = Path(fn).stem
        use = str(m.get("use", "")).strip()
        g = str(m.get("glaucoma", "")).strip().lower()
        rec = _npz_imaging(path)
        rec.update(
            eye_id=f"{stem}-harvardgf",
            patient_id=f"{stem}-harvardgf",
            source="harvard_gf",
            laterality=pd.NA,
            split=splits_map.get(use, use or "unknown"),
            rnflt_path=str(path.relative_to(raw_dir)),
            age=float(m["age"]) if pd.notna(m.get("age")) else pd.NA,
            sex=str(m["gender"]).strip().lower() if pd.notna(m.get("gender")) else pd.NA,
            race=str(m["race"]).strip().title() if pd.notna(m.get("race")) else pd.NA,
            ethnicity=str(m["ethnicity"]).strip().lower() if pd.notna(m.get("ethnicity")) else pd.NA,
            diagnosis_raw=1 if g in ("yes", "1", "glaucoma", "true") else 0,
        )
        rows.append(rec)
    if not rows:
        raise FileNotFoundError(
            f"manifest {csv_path.name} lists no .npz files found under {raw_dir}"
        )
    return _finalize_labels(pd.DataFrame(rows))


def _load_from_scan(raw_dir: Path, splits_map: dict[str, str]) -> pd.DataFrame:
    npz_files = sorted(raw_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"no .npz files under {raw_dir} — download Harvard-GF first "
            "(uv run scripts/download_harvard_gf.py)"
        )
    rows = []
    for path in npz_files:
        rec = load_sample(path)
        stem = path.stem
        split = "unknown"
        for part in path.relative_to(raw_dir).parts[:-1]:
            if part in splits_map:
                split = splits_map[part]
        rec.update(
            eye_id=f"{stem}-harvardgf",
            patient_id=f"{stem}-harvardgf",
            source="harvard_gf",
            laterality=pd.NA,
            split=split,
            rnflt_path=str(path.relative_to(raw_dir)),
        )
        rows.append(rec)
    return _finalize_labels(pd.DataFrame(rows))


def load_harvard_gf(
    raw_dir: Path,
    splits_map: dict[str, str] | None = None,
    summary_csv: str = "data_summary.csv",
) -> pd.DataFrame:
    """Build the Harvard-GF per-eye frame; prefers the CSV manifest, else scans npz."""
    raw_dir = Path(raw_dir)
    splits_map = splits_map or {}
    csv_path = raw_dir / summary_csv
    if not csv_path.exists():
        found = list(raw_dir.rglob(summary_csv))
        csv_path = found[0] if found else None
    if csv_path and csv_path.exists():
        return _load_from_manifest(raw_dir, csv_path, splits_map)
    return _load_from_scan(raw_dir, splits_map)
