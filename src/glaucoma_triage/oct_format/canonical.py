"""Canonical disc grid + dense-2-D resampler + RNFL read metrics for OCT-FormatShift.

The structural analog of ``vf_format.canonical``. The one real infra difference from the VF layer:
VF fields are point-lists scattered on a 54-location grid, so VF ``resample`` is nearest-neighbour on
coordinates; OCT RNFLT are **dense 2-D arrays at differing native resolution** (Harvard-GF 200x200,
Harvard-GDP 225x225), so the OCT ``resample`` interpolates a whole map onto a common H x W disc grid
(NaN-aware, sentinel-safe). Every dataset yields a canonical-grid map; models read a global/sectoral
RNFL thickness + an abnormal call (the structural analog of VF MD/PSD + abnormal-field staging).

Sector convention (image coords, row 0 = top = superior); nasal/temporal need laterality to disambiguate
and are assigned by OD convention (right half = temporal) unless the caller remaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, zoom

# Canonical disc grid: a small en-face RNFLT raster all sources resample onto (CPU-light, no 3D CNN).
CANON_H: int = 64
CANON_W: int = 64
CANON_SHAPE: tuple[int, int] = (CANON_H, CANON_W)

# Harvard RNFLT maps mark outside-scan / masked pixels with these sentinels; also drop non-positive um.
OUTSIDE_SENTINELS: tuple[float, ...] = (-1.0, -2.0)

# Global RNFL abnormal cut (um): an untuned literature round number (mean RNFL < 80 um flags glaucoma),
# the structural analog of the VF -2 dB normal limit. Opportunistic, disclosed (see scoping doc §11).
RNFL_ABNORMAL: float = 80.0

SECTORS = ("global", "s", "n", "i", "t")


def clean(m: np.ndarray) -> np.ndarray:
    """Map sentinels / non-positive thickness to NaN (float copy)."""
    m = np.asarray(m, dtype=float).copy()
    bad = ~np.isfinite(m) | (m <= 0)
    for s in OUTSIDE_SENTINELS:
        bad |= np.isclose(m, s)
    m[bad] = np.nan
    return m


def render_map_rgb(m: np.ndarray, *, vmin: float = 0.0, vmax: float = 175.0,
                   cmap: str = "viridis", size: int = 224) -> np.ndarray:
    """Render an RNFLT thickness map (µm; NaN/sentinel = invalid) as a clinical colour-coded RGB image (uint8) —
    the view a clinician (and a vision model / VLM) actually reads. FIXED ``vmin``/``vmax`` so every eye shares the
    colour scale (glaucomatous thinning → a consistent colour shift); invalid pixels → black. Returned (size,size,3).
    Lazy matplotlib import so the torch-free harness stays light."""
    import matplotlib
    m = clean(m)
    finite = np.isfinite(m)
    norm = np.clip((np.where(finite, m, vmin) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (matplotlib.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    rgb[~finite] = 0
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((size, size)))


def resample(m: np.ndarray, target: tuple[int, int] = CANON_SHAPE, *, min_coverage: float = 0.3) -> np.ndarray:
    """NaN-aware bilinear resample of a 2-D RNFLT map onto the canonical grid.

    Sentinels/holes are cleaned to NaN first; a mask-weighted zoom then interpolates only over valid
    pixels, and target pixels whose interpolated validity weight falls below ``min_coverage`` are NaN.
    """
    m = clean(m)
    if m.ndim != 2:
        raise ValueError("resample expects a 2-D map")
    valid = np.isfinite(m).astype(float)
    filled = np.where(valid > 0, np.nan_to_num(m, nan=0.0), 0.0)
    zy, zx = target[0] / m.shape[0], target[1] / m.shape[1]
    num = zoom(filled, (zy, zx), order=1, mode="nearest")
    den = zoom(valid, (zy, zx), order=1, mode="nearest")
    out = np.full(target, np.nan, dtype=float)
    ok = den >= min_coverage
    out[ok] = num[ok] / np.maximum(den[ok], 1e-6)
    return out


def quadrant_masks(shape: tuple[int, int], center: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
    """Boolean s/n/i/t quadrant masks (90-degree wedges) over a disc-centred map.

    Angle 0 = right (temporal, OD convention), 90 = up (superior). Callers with laterality should
    remap nasal<->temporal for OS.
    """
    h, w = shape
    cy, cx = (h / 2.0, w / 2.0) if center is None else center
    yy, xx = np.mgrid[0:h, 0:w]
    dy = cy - yy  # up positive (superior)
    dx = xx - cx  # right positive (temporal, OD)
    ang = np.degrees(np.arctan2(dy, dx))
    return {
        "s": (ang >= 45) & (ang < 135),
        "n": (ang >= 135) | (ang < -135),
        "i": (ang >= -135) & (ang < -45),
        "t": (ang >= -45) & (ang < 45),
    }


def rnfl_global(m: np.ndarray) -> float:
    """Global mean RNFL thickness (um) over valid pixels."""
    m = clean(m)
    v = m[np.isfinite(m)]
    return float(np.mean(v)) if v.size else float("nan")


def rnfl_sectors(m: np.ndarray, center: tuple[float, float] | None = None) -> dict[str, float]:
    """Global + per-quadrant (S/N/I/T) mean RNFL thickness (um)."""
    m = clean(m)
    out: dict[str, float] = {"global": rnfl_global(m)}
    for sector, mask in quadrant_masks(m.shape, center).items():
        v = m[mask & np.isfinite(m)]
        out[sector] = float(np.mean(v)) if v.size else float("nan")
    return out


def abnormal_binary(rnfl_global_um: float, threshold: float = RNFL_ABNORMAL) -> int:
    """Abnormal call: global RNFL below ``threshold`` um (1 = abnormal)."""
    if not np.isfinite(rnfl_global_um):
        return 0
    return int(rnfl_global_um < threshold)


def map_matrix(maps: list[np.ndarray], *, target: tuple[int, int] = CANON_SHAPE) -> np.ndarray:
    """Stack a list of RNFLT maps into an (N, H, W) canonical tensor via :func:`resample`."""
    if not maps:
        return np.empty((0, *target), dtype=float)
    return np.stack([resample(m, target) for m in maps], axis=0)


# --- TSNIT peripapillary profile (the model's read target) ---
# The clinical RNFL read is sampled along a ~3.4 mm calculation circle around the disc; on the 6x6 mm
# canonical map that circle radius ~= 0.55 of the half-width. The profile runs Temporal(0deg) ->
# Superior(90) -> Nasal(180) -> Inferior(270) -> Temporal (TSNIT), OD convention (right = temporal).
PROFILE_N: int = 64
CIRCLE_RADIUS_FRAC: float = 0.55


def _fill_circular(v: np.ndarray) -> np.ndarray:
    """Fill NaN entries of a periodic profile by circular linear interpolation from valid neighbours."""
    v = np.asarray(v, dtype=float).copy()
    bad = ~np.isfinite(v)
    if not bad.any():
        return v
    if bad.all():
        return np.zeros_like(v)  # caller substitutes a sane constant
    idx = np.arange(len(v))
    good = idx[~bad]
    # extend one period either side so wrap-around gaps interpolate correctly
    xp = np.concatenate([good - len(v), good, good + len(v)])
    fp = np.tile(v[~bad], 3)
    v[bad] = np.interp(idx[bad], xp, fp)
    return v


def tsnit_profile(m: np.ndarray, n: int = PROFILE_N, *, radius_frac: float = CIRCLE_RADIUS_FRAC,
                  center: tuple[float, float] | None = None, fill: bool = True) -> np.ndarray:
    """Sample the RNFL thickness (um) along the peripapillary circle at ``n`` angles (TSNIT order).

    NaN-aware (mask-weighted bilinear): a sample whose interpolated validity < 0.5 is NaN. With
    ``fill=True`` (default) those gaps are circular-interpolated from valid neighbours (as the device
    interpolates its calculation circle) so the profile is a finite continuous curve usable as a
    regression target; a wholly-invalid circle falls back to the map's mean thickness. ``fill=False``
    keeps the raw NaN pattern (for coverage diagnostics).
    """
    m = clean(m)
    h, w = m.shape
    cy, cx = (h / 2.0, w / 2.0) if center is None else center
    r = radius_frac * (min(h, w) / 2.0)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)  # 0 = temporal(+x), pi/2 = superior(up)
    rows = cy - r * np.sin(ang)
    cols = cx + r * np.cos(ang)
    valid = np.isfinite(m).astype(float)
    filled = np.where(valid > 0, np.nan_to_num(m, nan=0.0), 0.0)
    num = map_coordinates(filled, [rows, cols], order=1, mode="nearest")
    den = map_coordinates(valid, [rows, cols], order=1, mode="nearest")
    out = np.full(n, np.nan)
    ok = den >= 0.5
    out[ok] = num[ok] / np.maximum(den[ok], 1e-6)
    if fill and not ok.all():
        out = _fill_circular(out) if ok.any() else np.full(n, float(np.nanmean(m)) if np.isfinite(m).any() else 0.0)
    return out


def _profile_angles(n: int) -> np.ndarray:
    """Angles (degrees, 0=temporal, 90=superior) of each profile sample — matches ``tsnit_profile``."""
    return np.linspace(0.0, 360.0, n, endpoint=False)


def profile_sectors(profile: np.ndarray) -> dict[str, float]:
    """Aggregate a TSNIT profile into global + S/N/I/T quadrant means (um). Angle 0=temporal, 90=superior."""
    profile = np.asarray(profile, dtype=float)
    ang = _profile_angles(len(profile))
    masks = {
        "s": (ang >= 45) & (ang < 135),
        "n": (ang >= 135) & (ang < 225),
        "i": (ang >= 225) & (ang < 315),
        "t": (ang >= 315) | (ang < 45),
    }
    out = {"global": float(np.nanmean(profile)) if np.isfinite(profile).any() else float("nan")}
    for s, mk in masks.items():
        v = profile[mk & np.isfinite(profile)]
        out[s] = float(np.mean(v)) if v.size else float("nan")
    return out


def clock_hour_means(profile: np.ndarray) -> np.ndarray:
    """Aggregate a TSNIT profile into 12 clock-hour means (30deg arcs)."""
    profile = np.asarray(profile, dtype=float)
    ang = _profile_angles(len(profile))
    out = np.full(12, np.nan)
    for k in range(12):
        mk = (ang >= k * 30) & (ang < (k + 1) * 30)
        v = profile[mk & np.isfinite(profile)]
        if v.size:
            out[k] = float(np.mean(v))
    return out


@dataclass
class NormativeReference:
    """Empirical normative reference built from HEALTHY-eye TSNIT profiles (the red/yellow/green mechanism).

    Per sector (global + S/N/I/T), stores the 5th/1st percentiles of the healthy distribution. An eye is
    flagged abnormal if >=1 sector falls below its 5th percentile (the sensitivity-favouring clinical
    variant; Yang 2020) — or, in ``mode='global'``, only the global sector. Lit: Silverman 2016 (percentile
    colour code), Yang 2020 (<5th pct of healthy distribution), Budenz 2007 (age dependence).
    """

    sectors: tuple[str, ...]
    pct5: np.ndarray
    pct1: np.ndarray

    @classmethod
    def fit(cls, profiles: np.ndarray, healthy_mask: np.ndarray) -> "NormativeReference":
        profiles = np.asarray(profiles, dtype=float)
        healthy_mask = np.asarray(healthy_mask, dtype=bool)
        healthy = profiles[healthy_mask]
        if healthy.shape[0] < 5:
            raise ValueError(f"need >=5 healthy eyes to fit a normative reference; got {healthy.shape[0]}")
        secs = ("global", "s", "n", "i", "t")
        agg = np.array([[profile_sectors(p)[s] for s in secs] for p in healthy], dtype=float)
        pct5 = np.nanpercentile(agg, 5, axis=0)
        pct1 = np.nanpercentile(agg, 1, axis=0)
        return cls(sectors=secs, pct5=pct5, pct1=pct1)

    def abnormal(self, profile: np.ndarray, *, mode: str = "any_sector") -> int:
        """Abnormal flag. ``any_sector`` (default) = >=1 sector below the 1st pct (~95% healthy specificity,
        Silverman 2016); ``global`` = global sector below 1st pct; ``sensitive`` = >=1 sector below 5th pct."""
        agg = np.array([profile_sectors(profile)[s] for s in self.sectors], dtype=float)
        thr = self.pct5 if mode == "sensitive" else self.pct1
        below = agg < thr
        if mode == "global":
            return int(bool(below[0]))
        return int(bool(np.nansum(below) > 0))  # >=1 sector (incl global) below threshold

    def abnormal_batch(self, profiles: np.ndarray, *, mode: str = "any_sector") -> np.ndarray:
        return np.array([self.abnormal(p, mode=mode) for p in profiles], dtype=int)
