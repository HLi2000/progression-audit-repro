"""On-disk RNFLT-map loaders for OCT-FormatShift (the two local map sources, both Cirrus).

Harvard-GF (200x200) is the primary map workhorse; Harvard-GDP (225x225) is the second institution and
the clean within-Cirrus grid/resolution rung. Both are read straight to the canonical disc grid. The
cross-vendor sources (GLEAM Spectralis / GOALS Topcon) are raw circumpapillary B-scans -> they need a
B-scan->thickness step (Eyened/OCT-ONH, torch) before they can enter the *map* benchmark, so they are
not loaded here (they enter as a separate B-scan/harmonizer-contender track).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..config import data_dir, raw_dir
from . import canonical


def _load_npz_maps(files: list[Path], map_key: str, label_key: str,
                   target: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    maps, labels = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if map_key not in z.files:
            continue
        maps.append(z[map_key])
        labels.append(int(z[label_key]) if label_key in z.files else -1)
    X = canonical.map_matrix(maps, target=target)
    return X, np.asarray(labels, dtype=int)


def load_harvard_gf_maps(*, limit: int | None = None, raw: Path | None = None,
                         target: tuple[int, int] = canonical.CANON_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """Harvard-GF 200x200 RNFLT maps -> (N,H,W) canonical + glaucoma labels. The primary map source."""
    d = Path(raw) if raw is not None else raw_dir("harvard_gf")
    files = sorted(d.rglob("*.npz"))
    if limit is not None:
        files = files[:limit]
    return _load_npz_maps(files, "rnflt", "glaucoma", target)


def load_harvard_gdp_maps(*, limit: int | None = None, raw: Path | None = None,
                          target: tuple[int, int] = canonical.CANON_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """Harvard-GDP 225x225 RNFLT maps (RNFLT/*.npz) -> canonical + glaucoma labels. 2nd institution."""
    d = Path(raw) if raw is not None else raw_dir("harvard_gdp")
    rnflt_dir = d / "RNFLT"
    files = sorted(rnflt_dir.glob("*.npz")) if rnflt_dir.is_dir() else sorted(d.rglob("*.npz"))
    if limit is not None:
        files = files[:limit]
    return _load_npz_maps(files, "rnflt", "glaucoma", target)


def load_gleam(*, limit: int | None = None, target: tuple[int, int] = (224, 224),
               raw: Path | None = None, balanced: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """GLEAM Spectralis circumpapillary ``Thickness.jpg`` -> (N,H,W) grayscale images + BINARY glaucoma labels.

    The reusable loader Agent-2 flagged as missing: GLEAM's 4-class ``all_samples_label.csv`` (0=healthy,
    1/2/3=glaucoma severity) is ON DISK but was previously read only inside the FM producers, and the Eyened
    extractor discarded it. Binarises non-glaucoma(0) vs glaucoma(>0) -> a real Spectralis detection rung
    (Zhang et al. IEEE TMI 2026). ``balanced`` sub-samples the majority class (by striding) to ~equal counts.

    ⚠ HONEST SCOPE (§0): GLEAM is single-center (Shenyang) so device≡cohort is confounded and it is
    underpowered (~200 glaucoma) — a NON-ranked cross-vendor corroboration rung, NEVER pooled with the
    identified synthetic-axis AUC. NOTE: despite the ``Thickness.jpg`` filename, each image is a RAW
    circumpapillary **B-scan** (verified by opening one — a layered cross-section with speckle + a 200 µm depth
    scale bar), read grayscale — NOT a derived en-face thickness map. So B-scan FMs read their native modality.
    """
    from PIL import Image
    d = Path(raw) if raw is not None else data_dir() / "raw" / "gleam" / "GLEAM"
    csv_path = d / "all_samples_label.csv"
    if not csv_path.exists():
        return np.empty((0, *target), dtype=float), np.empty(0, dtype=int)
    with open(csv_path) as fh:
        rows = {int(r["ID"]): int(r["Label"]) for r in csv.DictReader(fh)}
    items = [(d / "all_samples" / f"{cid:04d}" / "Thickness.jpg", 0 if lab == 0 else 1)
             for cid, lab in sorted(rows.items())]
    items = [(f, y) for f, y in items if f.exists()]
    if balanced and items:
        healthy = [x for x in items if x[1] == 0]
        glauc = [x for x in items if x[1] == 1]
        k = min(len(healthy), len(glauc))
        if k:
            h = healthy[:: max(1, len(healthy) // k)][:k]
            g = glauc[:: max(1, len(glauc) // k)][:k]
            items = [x for pair in zip(h, g) for x in pair]   # interleave so a `limit` stays class-balanced
    if limit is not None:
        items = items[:limit]
    imgs, labs = [], []
    for f, y in items:
        imgs.append(np.asarray(Image.open(f).convert("L").resize(target[::-1]), dtype=np.float32))
        labs.append(y)
    X = np.stack(imgs) if imgs else np.empty((0, *target), dtype=float)
    return X, np.asarray(labs, dtype=int)


def load_goals_topcon_profiles(*, n: int = canonical.PROFILE_N, um_per_px: float = 3.25,
                               raw: Path | None = None) -> np.ndarray:
    """Real TOPCON circumpapillary RNFL profiles from the GOALS layer masks — TORCH-FREE (no Eyened needed).

    GOALS (Fang 2022, MICCAI; arXiv 2207.14447) ships Topcon **DRI OCT Triton** circumpapillary B-scans
    WITH RNFL/GCIPL/choroid masks, so the RNFL thickness is read directly: RNFL = mask value 80 (verified
    empirically as the topmost thin layer — the MIRAGE release remapped the paper's original values, its
    INFO.json is authoritative), and the per-A-scan-column RNFL-pixel count is the thickness along the
    peripapillary circle -> resampled to ``n`` points (a real Topcon TSNIT profile).

    ⚠ TWO honest confounds (docs §6): (1) SCALE — ``um_per_px`` is the Triton axial sampling; the challenge
    PNG (1100x800) was RESIZED from native (1024x247 / 1270x763), so it is uncertain. Default 3.25 = the
    Triton ~2.6 mm imaging depth / 800 px; the absolute µm are approximate → use the SHAPE (scale-invariant)
    for the robust read. (2) COHORT — the seg release is ~22% glaucoma (mostly normal per the paper) with NO
    class labels on disk (MIRAGE dropped them), so device is aliased with cohort. Returns (N, n) profiles;
    the angular origin is aligned to a reference downstream (``cross_vendor``).
    """
    from PIL import Image
    d = Path(raw) if raw is not None else Path("data/raw/goals/GOALS")
    masks = sorted(d.glob("*/semseg/*.png"))
    profs = []
    for f in masks:
        t = (np.asarray(Image.open(f)) == 80).sum(axis=0).astype(float) * um_per_px
        profs.append(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(t)), t))
    return np.array(profs) if profs else np.zeros((0, n))


def load_retouch_sample(*, per_vendor: int = 20, tar_path: Path | None = None,
                        vendors: tuple[str, ...] = ("Cirrus", "Spectralis", "Topcon")
                        ) -> dict[str, list[np.ndarray]]:
    """A per-vendor sample of REAL RETOUCH B-scans (Cirrus/Spectralis/Topcon) for the V1b realism check.

    ⚠ IMAGE-LEVEL ONLY — RETOUCH is MACULAR fluid, MIRAGE-resampled to 1024x512, different eyes. It
    corroborates that cross-vendor differences are REAL/non-trivial, NOT the disc RNFL µm offset (that is
    validated by ``validity.simulation_realism`` / V1a against published Bland-Altman). Uses the system
    ``tar`` (Python ``tarfile`` chokes on the concatenated part_aa+ab archive) + PIL; returns {} if absent.
    """
    import glob
    import os
    import subprocess
    import tempfile

    tar = Path(tar_path) if tar_path is not None else Path("data/raw/retouch/RETOUCH.tar")
    if not tar.exists():
        return {}
    from PIL import Image
    listing = subprocess.run(["tar", "tf", str(tar)], capture_output=True, text=True)
    bscan = [n for n in listing.stdout.splitlines() if n.endswith(".png") and "/bscan/" in n]
    out: dict[str, list[np.ndarray]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        pick: list[str] = []
        for v in vendors:
            vn = [n for n in bscan if os.path.basename(n).startswith(v + "-")]
            pick += vn[:: max(1, len(vn) // per_vendor)][:per_vendor]  # even sample across the vendor
        if pick:
            names_file = Path(tmp) / "names.txt"
            names_file.write_text("\n".join(pick))
            subprocess.run(["tar", "xf", str(tar), "-C", tmp, "-T", str(names_file)], capture_output=True)
        for f in glob.glob(f"{tmp}/**/*.png", recursive=True):
            v = os.path.basename(f).split("-")[0]
            if v in vendors:
                out.setdefault(v, []).append(np.asarray(Image.open(f).convert("L"), dtype=float))
    return out
