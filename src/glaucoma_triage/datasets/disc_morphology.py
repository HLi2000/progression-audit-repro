"""Optic-disc morphometry from PAPILA expert cup/disc contours (pure numpy).

The fundus-label experiment needs an *imaging-morphology* structural read that goes beyond the
vertical cup-to-disc ratio: neuroretinal-rim area and the full **ISNT rule** on sector rim
widths. This module derives those quantities from the two-expert cup/disc segmentation contours
(``load_contour`` parses one ``RET###{OD,OS}_{cup,disc}_exp{1,2}.txt`` file into an ``(N,2)``
array of pixel ``(x, y)`` points).

ISNT geometry / laterality convention (this is the subtle part, so it is spelled out):
for a boundary point relative to the disc centroid ``(cx, cy)`` we work in a clinical frame

    u = (x - cx) * nasal_sign     # u > 0  ==>  NASAL
    v = (cy - y)                  # v > 0  ==>  SUPERIOR   (image y grows downward)
    theta = atan2(v, u)           # radians, CCW from the nasal axis

with ``nasal_sign = +1`` for the **right eye (OD)** and ``-1`` for the **left eye (OS)** — the
fundus photos are disc-centred crops, so orientation cannot be inferred from image position and
must come from the OD/OS laterality. Superior is *up* and inferior is *down* for both eyes; only
nasal↔temporal flip with laterality. Quadrant sectors (each 90° wide):

    Nasal    : |theta| <= 45°
    Superior : 45° < theta <= 135°
    Temporal : |theta| > 135°
    Inferior : -135° <= theta < -45°

A healthy disc obeys ``I >= S >= N >= T`` (Jonas 1988; Harizman 2006, Arch Ophthalmol
124:1579); ``isnt_violation`` is the negation of that ordering on the expert-averaged widths.

Sector rim widths use **ray–polygon intersection** from the disc centroid: for each angle a ray
is cast and the farthest edge intersection is the boundary radius. Unlike polar vertex
interpolation this stays correct when the centroid falls outside the cup (an eccentric cup) — a
ray that misses the cup returns radius 0, so the rim there is the full disc radius. ``vcdr`` /
``hcdr`` / areas are computed directly (extents, shoelace). Degenerate inputs (fewer than 3
points, zero area) yield ``NaN`` rather than raising.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_N_THETA = 720  # dense angular grid for sector rim-width averaging


def load_contour(path: Path) -> np.ndarray:
    """Parse a segmentation contour file into an ``(N, 2)`` array of ``(x, y)`` points."""
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


# --------------------------------------------------------------------------- primitives
def _extent(poly: np.ndarray, axis: int) -> float:
    if poly.size == 0:
        return float("nan")
    col = poly[:, axis]
    return float(col.max() - col.min())


def vertical_extent(poly: np.ndarray) -> float:
    """Vertical (y) span of a contour."""
    return _extent(poly, 1)


def horizontal_extent(poly: np.ndarray) -> float:
    """Horizontal (x) span of a contour."""
    return _extent(poly, 0)


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace polygon area (always non-negative); ``NaN`` for fewer than 3 points."""
    if poly.ndim != 2 or poly.shape[0] < 3:
        return float("nan")
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area-weighted polygon centroid; falls back to the point mean for degenerate polygons."""
    if poly.ndim != 2 or poly.shape[0] == 0:
        return np.array([np.nan, np.nan])
    if poly.shape[0] < 3:
        return poly.mean(axis=0)
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if not np.isfinite(a) or abs(a) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
    return np.array([cx, cy])


def _clinical_directions(thetas: np.ndarray, nasal_sign: int) -> np.ndarray:
    """Unit ray directions in IMAGE coordinates for each clinical angle ``theta``.

    Clinical frame: ``u = (x-cx)*nasal_sign`` (nasal), ``v = cy-y`` (superior, image y grows down).
    A clinical direction ``(cos θ, sin θ)`` maps to image ``(cos θ * nasal_sign, -sin θ)``."""
    return np.column_stack([np.cos(thetas) * nasal_sign, -np.sin(thetas)])


def radius_profile(
    poly: np.ndarray, center: np.ndarray, thetas: np.ndarray, nasal_sign: int
) -> np.ndarray:
    """Boundary radius ``r(theta)`` by **ray–polygon intersection** from ``center``.

    For each clinical angle, cast a ray from ``center`` and take the farthest intersection with a
    polygon edge as the boundary radius. Robust to a centre that lies outside the polygon (an
    eccentric cup): rays that miss the polygon return ``0`` (the contour is absent in that
    direction). ``NaN`` for a contour with fewer than 3 points or a non-finite centre.
    """
    if poly.ndim != 2 or poly.shape[0] < 3 or not np.all(np.isfinite(center)):
        return np.full(thetas.shape, np.nan)
    d = _clinical_directions(thetas, nasal_sign)          # (T, 2) unit directions, r == t
    a = poly - center                                     # (E, 2) edge starts rel. to centre
    e = np.roll(poly, -1, axis=0) - poly                  # (E, 2) edge vectors
    # Solve center + t*d = A + s*e  →  t = cross(a, e)/cross(d, e), s = cross(a, d)/cross(d, e).
    denom = np.outer(d[:, 0], e[:, 1]) - np.outer(d[:, 1], e[:, 0])   # (T, E) = cross(d, e)
    cross_a_e = a[:, 0] * e[:, 1] - a[:, 1] * e[:, 0]     # (E,) = cross(a, e), independent of theta
    t = np.divide(cross_a_e[None, :], denom, out=np.full_like(denom, -np.inf), where=denom != 0)
    s = np.divide((np.outer(d[:, 1], a[:, 0]) - np.outer(d[:, 0], a[:, 1])),  # cross(a, d) = -cross(d, a)
                  denom, out=np.full_like(denom, np.nan), where=denom != 0)
    valid = (denom != 0) & (t >= -1e-9) & (s >= -1e-9) & (s <= 1 + 1e-9)
    t_valid = np.where(valid, t, -np.inf)
    r = t_valid.max(axis=1)
    return np.where(np.isfinite(r) & (r > 0), r, 0.0)


def rim_width_profile(
    cup: np.ndarray, disc: np.ndarray, center: np.ndarray, thetas: np.ndarray, nasal_sign: int
) -> np.ndarray:
    """Neuroretinal-rim width ``r_disc(theta) - r_cup(theta)`` (clamped at 0) about ``center``."""
    r_disc = radius_profile(disc, center, thetas, nasal_sign)
    r_cup = radius_profile(cup, center, thetas, nasal_sign)
    return np.clip(r_disc - r_cup, 0.0, None)


def min_rim_disc_ratio(
    cup: np.ndarray, disc: np.ndarray, center: np.ndarray, thetas: np.ndarray, nasal_sign: int
) -> float:
    """Minimum local rim fraction ``min_theta rim(theta)/r_disc(theta)`` — a fundus analog of the
    minimum-rim-width (BMO-MRW) that flags **focal** rim narrowing a global CDR averages away.
    Dimensionless in ``[0, 1]`` (small = focal thinning); disc-size-robust. ``NaN`` if degenerate."""
    r_disc = radius_profile(disc, center, thetas, nasal_sign)
    rim = rim_width_profile(cup, disc, center, thetas, nasal_sign)
    ok = np.isfinite(r_disc) & (r_disc > 0)
    if not ok.any():
        return float("nan")
    return float(np.min(rim[ok] / r_disc[ok]))


def sector_means(width: np.ndarray, thetas: np.ndarray) -> dict[str, float]:
    """Mean rim width per ISNT quadrant, keyed ``'S' 'I' 'N' 'T'`` (see module docstring)."""
    deg = np.degrees(thetas)
    masks = {
        "N": np.abs(deg) <= 45,
        "S": (deg > 45) & (deg <= 135),
        "T": np.abs(deg) > 135,
        "I": (deg >= -135) & (deg < -45),
    }

    def _mean(m: np.ndarray) -> float:
        vals = width[m]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    return {k: _mean(m) for k, m in masks.items()}


def isnt_violation(sectors: dict[str, float], rel_tol: float = 0.02) -> bool | None:
    """``True`` if the ``I >= S >= N >= T`` rim ordering is *meaningfully* broken; ``None`` if any
    sector is NaN. A small relative tolerance (``rel_tol`` of the mean rim width) means only a
    real ordering breach counts, not sub-pixel numerical ties (a symmetric disc obeys ISNT)."""
    i, s, n, t = sectors["I"], sectors["S"], sectors["N"], sectors["T"]
    if not all(np.isfinite(v) for v in (i, s, n, t)):
        return None
    tol = rel_tol * max((i + s + n + t) / 4.0, 1e-9)
    return not (i >= s - tol and s >= n - tol and n >= t - tol)


# --------------------------------------------------------------------------- per-eye metrics
_CONTINUOUS_KEYS = (
    "disc_area", "cup_area", "rim_area", "area_cdr", "rim_disc_area_ratio",
    "vcdr", "hcdr", "isnt_s", "isnt_i", "isnt_n", "isnt_t", "min_rim_disc_ratio",
    "isnt_s_ratio", "isnt_i_ratio", "isnt_n_ratio", "isnt_t_ratio",
)

# Scale-invariant disc metrics (safe to compare across cameras/resolutions): dimensionless ratios only,
# excluding the pixel-scaled areas/widths (disc_area, cup_area, rim_area, raw isnt_s/i/n/t). Used by the
# multi-camera GRAPE cohort; single-camera PAPILA (step 0) may use the full pixel-scaled set.
SCALE_INVARIANT_KEYS = (
    "vcdr", "hcdr", "area_cdr", "min_rim_disc_ratio",
    "isnt_s_ratio", "isnt_i_ratio", "isnt_n_ratio", "isnt_t_ratio",
)


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return float("nan")
    return num / den


def expert_metrics(cup: np.ndarray, disc: np.ndarray, laterality: str) -> dict:
    """Disc morphometry for a single expert's cup/disc contours of one eye.

    ``laterality`` is ``"OD"``/``"OS"`` and sets the nasal direction. Returns the continuous
    metrics in :data:`_CONTINUOUS_KEYS` plus ``isnt_violation`` (bool or ``None``).
    """
    nasal_sign = 1 if str(laterality).upper() == "OD" else -1
    disc_area = polygon_area(disc)
    cup_area = polygon_area(cup)
    rim_area = disc_area - cup_area if np.isfinite(disc_area) and np.isfinite(cup_area) else float("nan")

    thetas = np.linspace(-np.pi, np.pi, _N_THETA, endpoint=False)
    center = polygon_centroid(disc)
    rim = rim_width_profile(cup, disc, center, thetas, nasal_sign)
    sectors = sector_means(rim, thetas)
    # scale-invariant per-sector rim/disc-radius ratios (cross-camera-comparable analog of isnt_*)
    r_disc = radius_profile(disc, center, thetas, nasal_sign)
    ratio = np.where(np.isfinite(r_disc) & (r_disc > 0), rim / np.where(r_disc > 0, r_disc, np.nan), np.nan)
    sec_ratio = sector_means(ratio, thetas)

    return {
        "disc_area": disc_area,
        "cup_area": cup_area,
        "rim_area": rim_area,
        "area_cdr": _safe_ratio(cup_area, disc_area),
        "rim_disc_area_ratio": _safe_ratio(rim_area, disc_area),
        "vcdr": _safe_ratio(vertical_extent(cup), vertical_extent(disc)),
        "hcdr": _safe_ratio(horizontal_extent(cup), horizontal_extent(disc)),
        "isnt_s": sectors["S"],
        "isnt_i": sectors["I"],
        "isnt_n": sectors["N"],
        "isnt_t": sectors["T"],
        "min_rim_disc_ratio": min_rim_disc_ratio(cup, disc, center, thetas, nasal_sign),
        "isnt_s_ratio": sec_ratio["S"],
        "isnt_i_ratio": sec_ratio["I"],
        "isnt_n_ratio": sec_ratio["N"],
        "isnt_t_ratio": sec_ratio["T"],
        "isnt_violation": isnt_violation(sectors),
    }


def _expert_contours(seg_dir: Path, patient_tag: str, eye: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """All ``(cup, disc)`` contour pairs for one eye, one per expert with both files present."""
    pairs = []
    for cup_path in sorted(Path(seg_dir).rglob(f"{patient_tag}{eye}*cup*.txt")):
        disc_path = Path(str(cup_path).replace("cup", "disc"))
        if disc_path.exists():
            pairs.append((load_contour(cup_path), load_contour(disc_path)))
    return pairs


def _nanmean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def eye_disc_metrics(seg_dir: Path | None, patient_tag: str, eye: str, laterality: str) -> dict:
    """Expert-averaged disc morphometry for one eye.

    Averages the continuous metrics across the available experts (mean of finite values);
    ``isnt_violation`` is recomputed from the *averaged* sector widths. Returns all-``NaN`` /
    ``None`` metrics when ``seg_dir`` is missing or no complete cup+disc expert pair exists.
    ``per_expert`` holds the individual expert dicts (for the inter-expert reliability check).
    """
    empty = {k: float("nan") for k in _CONTINUOUS_KEYS}
    empty["isnt_violation"] = None
    empty["per_expert"] = []
    if seg_dir is None:
        return empty

    per_expert = [expert_metrics(cup, disc, laterality)
                  for cup, disc in _expert_contours(seg_dir, patient_tag, eye)]
    if not per_expert:
        return empty

    out = {k: _nanmean([m[k] for m in per_expert]) for k in _CONTINUOUS_KEYS}
    out["isnt_violation"] = isnt_violation(
        {s: out[f"isnt_{s.lower()}"] for s in ("I", "S", "N", "T")}
    )
    out["per_expert"] = per_expert
    return out
