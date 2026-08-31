"""Canonical VF field representation: a fixed 24-2 grid, nearest-neighbour resampling onto it,
and the VF-intrinsic interpretation targets (MD = mean TD; abnormal-field flag / 3-stage).

A "field" throughout this module is ``(values, coords)``: ``values`` a 1-D array of total-deviation
(loss = negative; NaN = missing/untested) and ``coords`` an ``(n, 2)`` array of ``(x, y)`` degrees.
The canonical grid :data:`CANON_COORDS` is the standard 54-location Humphrey 24-2 layout, so any
source field (UWHVF, Harvard, or a re-gridded Octopus field) maps to one fixed-length model input.

The interpretation targets reuse the repo's VF thresholds (``features.FIELD_ABNORMAL_MD``):
``normal_limit = -2`` dB (abnormal binary), ``hpa_moderate = -6`` dB (3-stage boundary).
"""

from __future__ import annotations

import numpy as np

from ..datasets.features import FIELD_ABNORMAL_MD

_NORMAL_LIMIT = FIELD_ABNORMAL_MD["normal_limit"]   # -2 dB
_HPA_MODERATE = FIELD_ABNORMAL_MD["hpa_moderate"]   # -6 dB


# The canonical 54-location grid is the **authoritative visualFields ``p24d2`` layout** (Marín-Franch & Swanson,
# ``locmaps$p24d2$coord``; Apache-2.0) — the standard Humphrey 24-2: x = horizontal, y = vertical; 8 rows at
# y = ±3, ±9, ±15, ±21 (counts 4,6,8,9,9,8,6,4) with the two extra nasal points at x = −27 in the y = ±3 rows,
# listed row-major (top→bottom, nasal→temporal within a row). The blind spot is at (15, ±3) (``p24d2$bs``).
# (Phase 1t §0: VERIFIED-REGISTERED. This order is bit-identical to ``locmaps$p24d2$coord`` AND to UWHVF's own
# ``td_seq`` order — confirmed by a coordinate join to the extracted p24d2 map (exact 54-pt bijection, max NN 0.0)
# and by the population blind spot: placing UWHVF ``td_seq`` and SUNY-IU ``l1..l54`` by index both land the
# deepest-sensitivity location at (15, −3) [~3–4 dB], with a clean hill-of-vision (central ≈28 dB > periphery
# ≈23 dB) and corr(UWHVF hill, SUNY-IU hill) = 0.972. So UWHVF/SUNY-IU/Iowa/vfpwg — all in ``l``/``td_seq``
# order — align to this grid by identity, and the SUNY-IU per-location variance σ²ᵢ (:func:`weighted_md`)
# registers directly. The earlier grid was a *scrambled* permutation of this point-set (SUNY-IU blind spot fell
# at (21,±3)); MD/PSD were unaffected (permutation-invariant) but position-sensitive HLO weighting needed this
# fix. Sources shipping real coords (Rotterdam, 10-2, Octopus, Eyecatcher) resample onto it via
# :func:`best_orientation`/:func:`resample`; by-order sources (Harvard ``td1..td54``) are placed by index and
# stay UNWEIGHTED (permutation-safe) — HLO weighting is scoped to the coordinate-aligned rungs.)
_CANON: list[tuple[float, float]] = [
    (-9, 21), (-3, 21), (3, 21), (9, 21),
    (-15, 15), (-9, 15), (-3, 15), (3, 15), (9, 15), (15, 15),
    (-21, 9), (-15, 9), (-9, 9), (-3, 9), (3, 9), (9, 9), (15, 9), (21, 9),
    (-27, 3), (-21, 3), (-15, 3), (-9, 3), (-3, 3), (3, 3), (9, 3), (15, 3), (21, 3),
    (-27, -3), (-21, -3), (-15, -3), (-9, -3), (-3, -3), (3, -3), (9, -3), (15, -3), (21, -3),
    (-21, -9), (-15, -9), (-9, -9), (-3, -9), (3, -9), (9, -9), (15, -9), (21, -9),
    (-15, -15), (-9, -15), (-3, -15), (3, -15), (9, -15), (15, -15),
    (-9, -21), (-3, -21), (3, -21), (9, -21),
]

CANON_COORDS: np.ndarray = np.array(_CANON, dtype=float)
N_CANON: int = len(CANON_COORDS)  # 54
# Index of the two blind-spot locations (15, ±3) on the canonical grid (p24d2$bs), for reference/tests.
BLIND_SPOT_IDX: tuple[int, int] = (25, 34)


def resample(values: np.ndarray, coords: np.ndarray, target: np.ndarray = CANON_COORDS,
             *, max_radius: float = 6.0) -> np.ndarray:
    """Nearest-neighbour resample a field onto ``target`` coordinates.

    For each target location take the value of the nearest source point within ``max_radius``
    degrees (NaN if none, or if that source value is NaN). When ``coords == target`` this is the
    identity; when the source is on a different grid (Octopus, or a 10-2 subset) it re-expresses the
    field on the canonical grid — deliberately lossy, which is the point of the OOD rungs."""
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    out = np.full(len(target), np.nan, dtype=float)
    if len(coords) == 0:
        return out
    for i, (tx, ty) in enumerate(target):
        d = np.hypot(coords[:, 0] - tx, coords[:, 1] - ty)
        j = int(np.argmin(d))
        if d[j] <= max_radius:
            out[i] = values[j]
    return out


def best_orientation(coords: np.ndarray) -> np.ndarray:
    """Return ``coords`` under the axis swap / sign flip that best matches the canonical grid.

    The canonical grid is the anatomical 24-2, but a source that ships **real** degree coordinates in a
    different convention (e.g. Rotterdam is vertically flipped) must be reoriented before resampling or it
    lands scrambled. We pick, among the 8 axis-swap × sign-flip candidates, the one minimizing mean
    nearest-neighbour distance to :data:`CANON_COORDS` — a deterministic, self-verifying alignment (the
    correct orientation drives the cost to ~0)."""
    coords = np.asarray(coords, dtype=float)
    best, best_cost = coords, np.inf
    for swap in (False, True):
        base = coords[:, ::-1] if swap else coords
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                cand = base * np.array([sx, sy])
                d = np.array([np.min(np.hypot(CANON_COORDS[:, 0] - x, CANON_COORDS[:, 1] - y))
                              for x, y in cand])
                cost = float(d.mean())
                if cost < best_cost:
                    best_cost, best = cost, cand
    return best


def md(values: np.ndarray) -> float:
    """Mean total deviation over valid (non-NaN) locations — the MD-analog target."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def psd(values: np.ndarray) -> float:
    """Pattern-spread proxy: population SD of pointwise TD over valid locations. Sensitive to *focal /
    irregular* loss (a scotoma raises it; a uniform depression does not) where MD is insensitive. This is a
    spread proxy, NOT the exact HFA variance-weighted PSD."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.std()) if v.size else float("nan")


def weighted_md(values: np.ndarray, variances: np.ndarray) -> float:
    """HLO inverse-variance-weighted mean deviation (Heijl–Lindgren–Olsson 1987): ``MD_w = Σ(TDᵢ/σ²ᵢ)/Σ(1/σ²ᵢ)``,
    where ``σ²ᵢ`` is the per-location between-subject normal variance (central ≈ 3× tighter than periphery).

    POSITION-SENSITIVE: ``values`` and ``variances`` must be coordinate-aligned (both on :data:`CANON_COORDS`).
    Do NOT apply to by-order-placed (Harvard) or synthesized-coord (Octopus) fields, which are only
    permutation-safe under the *unweighted* :func:`md` — see the Phase-1s soundness note (HLO is scoped to
    coordinate-aligned rungs only)."""
    v = np.asarray(values, dtype=float)
    var = np.asarray(variances, dtype=float)
    m = ~np.isnan(v) & ~np.isnan(var) & (var > 0)
    if not m.any():
        return float("nan")
    w = 1.0 / var[m]
    return float(np.sum(w * v[m]) / np.sum(w))


def weighted_psd(values: np.ndarray, variances: np.ndarray) -> float:
    """HLO variance-weighted pattern SD (Heijl–Lindgren–Olsson 1987 — the classical STATPAC PSD; not bit-identical
    to a modern SITA printout PSD): ``PSD_w = √[ mean(σ²ᵢ)·Σ wᵢ(TDᵢ − MD_w)² / (n−1) ]`` with ``wᵢ = 1/σ²ᵢ``.

    Same alignment requirement as :func:`weighted_md` (position-sensitive; canonical-grid-aligned inputs only)."""
    v = np.asarray(values, dtype=float)
    var = np.asarray(variances, dtype=float)
    m = ~np.isnan(v) & ~np.isnan(var) & (var > 0)
    if int(m.sum()) < 2:
        return float("nan")
    w = 1.0 / var[m]
    mdw = float(np.sum(w * v[m]) / np.sum(w))
    return float(np.sqrt(np.mean(var[m]) * np.sum(w * (v[m] - mdw) ** 2) / (int(m.sum()) - 1)))


# --- HLO weighting as a canonical resource ---------------------------------------------------------
# The per-location normal variance σ²ᵢ (SUNY-IU controls, on THIS grid) is a fixed normative; the driver installs
# it once via :func:`set_normal_variance` so the weighted readouts (:func:`md_hlo`/:func:`psd_hlo`) need no σ²
# threaded through every call. Unset → the weighted readouts fall back to the unweighted mean-TD / SD-of-TD
# (permutation-safe), so callers degrade gracefully when the SUNY-IU normative is absent (e.g. unit tests).
_NORMAL_VAR: np.ndarray | None = None


def set_normal_variance(variances: np.ndarray | None) -> None:
    """Install the canonical per-location normal variance σ²ᵢ (length :data:`N_CANON`, on :data:`CANON_COORDS`),
    or ``None`` to clear it (→ the HLO readouts revert to unweighted)."""
    global _NORMAL_VAR
    _NORMAL_VAR = None if variances is None else np.asarray(variances, dtype=float)


def normal_variance() -> np.ndarray | None:
    """The installed σ²ᵢ (or ``None``)."""
    return _NORMAL_VAR


def md_hlo(values: np.ndarray) -> float:
    """HLO inverse-variance-weighted MD using the installed σ²ᵢ; falls back to the unweighted :func:`md` when no
    normative is installed. POSITION-SENSITIVE (see :func:`weighted_md`) → apply only to coordinate-aligned fields."""
    return weighted_md(values, _NORMAL_VAR) if _NORMAL_VAR is not None else md(values)


def psd_hlo(values: np.ndarray) -> float:
    """HLO variance-weighted PSD using the installed σ²ᵢ; falls back to the unweighted :func:`psd` when no
    normative is installed. POSITION-SENSITIVE → coordinate-aligned fields only."""
    return weighted_psd(values, _NORMAL_VAR) if _NORMAL_VAR is not None else psd(values)


def abnormal_binary(md_value: float, threshold: float = _NORMAL_LIMIT) -> int:
    """1 if the field is abnormal (``MD < threshold``), else 0."""
    return int(md_value < threshold)


def abnormal_stage(md_value: float) -> int:
    """3-stage severity from MD: 0 normal (>= -2), 1 mild-moderate (-6 <= MD < -2), 2 mod-severe (< -6)."""
    if md_value >= _NORMAL_LIMIT:
        return 0
    if md_value >= _HPA_MODERATE:
        return 1
    return 2


def field_matrix(fields: list[dict], *, target: np.ndarray = CANON_COORDS) -> np.ndarray:
    """Stack a list of ``{"values","coords"}`` fields into an ``(n_fields, n_canon)`` model-input
    matrix by resampling each onto ``target``. NaNs are left for the estimator's imputer to handle."""
    return np.vstack([resample(f["values"], f["coords"], target) for f in fields]) \
        if fields else np.empty((0, len(target)), dtype=float)
