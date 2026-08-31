"""GRAPE baseline colour-fundus-photo (CFP) loader — pixel paths + camera, for FM embedding.

The core :func:`grape.load_grape` frame carries the baseline ``cfp`` FILENAME per eye; this resolves it to an
on-disk path under ``CFPs/`` (extracted from ``CFPs.rar``, Figshare article 23575926), tags the acquisition
camera + resolution (baseline-sheet cols 17/18), and returns one row per baseline eye. It feeds
``scripts/run_grape_cfp_embeddings.py`` (frozen foundation-model embeddings) → the ``fundus_emb`` modality of
the progression engine.

263 baseline eyes; each has its visit-1 CFP ``{subject}_{laterality}_1.jpg``. Two cameras (a real
distribution-shift axis, retained for a deferred robustness extension): TOPCON TRC-NW8 (1556²) and Canon CR-2
(2136²). The CFP is read as RGB (unlike the OCT B-scan reader, which is grayscale).
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from . import grape

# baseline-sheet columns just past the CFP filename (verified from the workbook header)
_B_DEVICE = 17   # "Acquisition Device"
_B_RESOLUTION = 18   # "Resolusion" [sic]

# normalise the verbose device strings to a compact camera tag
_CAMERA = {"TRC-NW8": "topcon", "CR-2": "canon"}


def _camera_of(device: object) -> str:
    s = str(device)
    for key, tag in _CAMERA.items():
        if key in s:
            return tag
    return "other"


def _camera_map(raw_dir: Path, xlsx: str) -> dict[str, tuple[str, str]]:
    """``eye_id -> (camera_tag, resolution)`` from the Baseline sheet's device/resolution columns."""
    wb = openpyxl.load_workbook(Path(raw_dir) / xlsx, read_only=True, data_only=True)
    ws = wb["Baseline"]
    out: dict[str, tuple[str, str]] = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[grape._B["subject"]] is None:
            continue
        eid = f"{row[grape._B['subject']]}_{row[grape._B['lat']]}"
        out[eid] = (_camera_of(row[_B_DEVICE]), str(row[_B_RESOLUTION]))
    return out


def load_grape_cfp(raw_dir: Path, cfp_subdir: str = "CFPs", xlsx: str = "vf_clinical.xlsx") -> pd.DataFrame:
    """One row per baseline eye: keys + resolved CFP path + camera + the three progression labels.

    Columns: ``eye_id``, ``patient_id``, ``cfp`` (filename), ``cfp_path`` (absolute), ``exists`` (bool),
    ``camera`` (topcon/canon/other), ``resolution``, and ``prog_plr2``/``prog_plr3``/``prog_md``. Rows are
    kept even if the image is missing (``exists=False``) so the caller can report coverage.
    """
    base = grape.load_grape(raw_dir, xlsx)
    cams = _camera_map(raw_dir, xlsx)
    cdir = Path(raw_dir) / cfp_subdir
    recs = []
    for r in base.itertuples(index=False):
        cfp = getattr(r, "cfp")
        path = cdir / str(cfp) if cfp is not None else None
        cam, res = cams.get(r.eye_id, ("other", ""))
        recs.append({
            "eye_id": r.eye_id, "patient_id": r.patient_id, "cfp": cfp,
            "cfp_path": str(path) if path is not None else None,
            "exists": bool(path is not None and path.exists()),
            "camera": cam, "resolution": res,
            "prog_plr2": r.prog_plr2, "prog_plr3": r.prog_plr3, "prog_md": r.prog_md,
        })
    return pd.DataFrame(recs)
