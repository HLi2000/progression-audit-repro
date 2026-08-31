"""Real-data grounding for the VF format study, from the visualFields R package (read via pyreadr).

Provides:
- :func:`fit_hill_of_vision` — an empirical HFA normative (centre + age-slope) fitted to the SUNY-IU
  healthy 24-2 controls → replaces the made-up constants in :mod:`vf_format.harmonize`.
- :func:`control_mean_vector` — per-location mean normal sensitivity (54-vector) → converts any
  visualFields sensitivity field to a TD analog (aligned within the shared ``l1..l54`` ordering).
- :func:`retest_series` / :func:`retest_noise_sd` — the ``vfpwgRetest24d2`` 30 × 12-weekly test-retest
  series (real repeatability) and its short-term per-location SD (grounds the noise transform).

All datasets share columns ``id, eye, date, time, age, type, fpr, fnr, fl, duration, l1..l54`` where
``l*`` are pointwise sensitivities (dB). Repeats are grouped by ``(id, eye)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_LOC = [f"l{i}" for i in range(1, 55)]   # l1..l54 pointwise sensitivities (dB) — 24-2
_LOC64 = [f"l{i}" for i in range(1, 65)]  # l1..l64 — Iowa full-field central/peripheral (64 locations)
_ECATCH_RE = re.compile(r"DLS_x(Neg)?(\d+)_y(Neg)?(\d+)_dB")

# Humphrey 10-2 layout (68 locations, central +/-9 deg) — extracted verbatim from the visualFields R
# package `locmaps$p10d2$coord` (Apache-2.0; Marin-Franch & Swanson). Used to place real 10-2 fields on
# the shared grid for the cross-pattern rung.
P10D2 = np.array([
    (-1, 9), (1, 9), (-5, 7), (-3, 7), (-1, 7), (1, 7), (3, 7), (5, 7),
    (-7, 5), (-5, 5), (-3, 5), (-1, 5), (1, 5), (3, 5), (5, 5), (7, 5),
    (-7, 3), (-5, 3), (-3, 3), (-1, 3), (1, 3), (3, 3), (5, 3), (7, 3),
    (-9, 1), (-7, 1), (-5, 1), (-3, 1), (-1, 1), (1, 1), (3, 1), (5, 1),
    (7, 1), (9, 1), (-9, -1), (-7, -1), (-5, -1), (-3, -1), (-1, -1), (1, -1),
    (3, -1), (5, -1), (7, -1), (9, -1), (-7, -3), (-5, -3), (-3, -3), (-1, -3),
    (1, -3), (3, -3), (5, -3), (7, -3), (-7, -5), (-5, -5), (-3, -5), (-1, -5),
    (1, -5), (3, -5), (5, -5), (7, -5), (-5, -7), (-3, -7), (-1, -7), (1, -7),
    (3, -7), (5, -7), (-1, -9), (1, -9),
], dtype=float)


# Iowa full-field CENTRAL grid (≤26°, 64 locations) — extracted verbatim from the visualFields R package
# `locmaps$pPC26$coord` (Apache-2.0; Marín-Franch et al., Data in Brief 21:75-82, 2018). Used to place the
# genuine-healthy Iowa central controls on the shared 24-2 grid for the specificity read. (The complementary
# peripheral map `pPeri` spans 26–81°, OUTSIDE the 24-2 field → not usable on the canonical grid; central only.)
IOWA_PC26 = np.array([
    (-20, 20), (-12, 20), (-4, 20), (4, 20), (12, 20), (20, 20), (-4, 14), (4, 14),
    (-20, 12), (-12, 12), (12, 12), (20, 12), (-8, 8), (-2, 8), (2, 8), (8, 8),
    (24, 8), (-4, 6), (4, 6), (8, 6), (-26, 4), (-20, 4), (-12, 4), (-8, 4),
    (-4, 4), (4, 4), (20, 4), (-6, 2), (-2, 2), (2, 2), (8, 2), (-6, -2),
    (-2, -2), (2, -2), (8, -2), (-26, -4), (-20, -4), (-12, -4), (-8, -4), (-4, -4),
    (4, -4), (20, -4), (-6, -6), (4, -6), (8, -6), (-8, -8), (-2, -8), (2, -8),
    (8, -8), (24, -8), (-18, -12), (-12, -12), (12, -12), (20, -12), (-4, -14), (4, -14),
    (-18, -20), (-12, -20), (-4, -20), (4, -20), (12, -20), (20, -20), (-8, -26), (8, -26),
], dtype=float)


def _read(raw_dir: Path, name: str) -> pd.DataFrame:
    import pyreadr
    res = pyreadr.read_r(str(Path(raw_dir) / f"{name}.rda"))
    return res[name] if name in res else next(iter(res.values()))


def _sens(df: pd.DataFrame, cols: list[str] = _LOC) -> np.ndarray:
    """The ``(n, len(cols))`` sensitivity matrix from the ``l1..lN`` columns (default 24-2 ``l1..l54``)."""
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def fit_hill_of_vision(raw_dir: Path, control: str = "vfctrSunyiu24d2") -> dict:
    """Fit a coord-free hill-of-vision normative to the healthy controls: mean field sensitivity
    regressed on age → ``{'centre': normal sensitivity at age 45, 'age_slope': dB/yr (negative)}``.

    Level + age dependence are the dominant normative terms; the eccentricity slope stays at the
    literature value in :mod:`vf_format.harmonize` (per-location coords aren't reliably available)."""
    df = _read(raw_dir, control)
    sens = _sens(df)
    field_mean = np.nanmean(sens, axis=1)          # mean sensitivity per control field
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(field_mean) & np.isfinite(age)
    slope, intercept = np.polyfit(age[ok], field_mean[ok], 1)
    # A spatially-FLAT normative (mean-location normal at age 45 + age slope). MD = mean(sens − normal)
    # depends only on this mean level, so a flat empirical normal is exact for MD and avoids fabricating
    # an eccentricity curve from an unverifiable coord map (per-location shape → pattern, not MD).
    return {"level_normal_age45": float(intercept + slope * 45.0),
            "age_slope": float(slope), "n": int(ok.sum())}


def control_mean_vector(raw_dir: Path, control: str = "vfctrSunyiu24d2") -> np.ndarray:
    """Per-location mean normal sensitivity (54-vector) over the healthy controls (age-pooled)."""
    return np.nanmean(_sens(_read(raw_dir, control)), axis=0)


def sunyiu_normal_variance(raw_dir: Path, control: str = "vfctrSunyiu24d2", *, floor: float = 0.5) -> np.ndarray:
    """Per-location between-subject normal VARIANCE σ²ᵢ (54-vector, ``l1..l54`` order = the canonical p24d2 grid)
    over the SUNY-IU healthy controls → the HLO inverse-variance weights ``wᵢ = 1/σ²ᵢ`` (Heijl–Lindgren–Olsson
    1987; reference ``visualFields::vfstats``). Central locations are ≈2–3× tighter than the periphery and the
    blind spot is loosest, so weighting pulls the MD/PSD toward the reliable central field — the real STATPAC MD.
    Floored at ``floor`` dB² to bound the weights (the variance sibling of :func:`control_mean_vector`; same 263
    controls that ground the normative mean)."""
    var = np.nanvar(_sens(_read(raw_dir, control)), axis=0)
    return np.maximum(var, floor)


def sunyiu_healthy_fields(raw_dir: Path, control: str = "vfctrSunyiu24d2", *,
                          holdout: float | None = None, seed: int = 0) -> list[dict]:
    """Healthy 24-2 control fields (``label=0``) from the SUNY-IU normative set, as TD analogs
    (sensitivity − control normal) on the canonical grid — the study's ONLY healthy cohort, used for the
    **specificity / over-referral-under-shift** read (the counterpart to the all-glaucoma sensitivity).

    If ``holdout`` ∈ (0, 1) is given, the controls are split: the normative mean is fit on the COMPLEMENTARY
    (1 − holdout) split and only the ``holdout`` fraction is returned — so the returned controls' TD is centred by
    a normative NOT fit on them (reduces the mechanical MD≡0 circularity). With ``holdout=None`` the normative is
    the full-cohort mean (the study-wide usage). NOTE (honest): even held-out, these are genuinely-normal controls
    and the model's own normative is population-derived, so clean-distance specificity stays ~trivially high — the
    informative signal is the **shift-induced false-positive rise**, not the distance-0 specificity. Single-field
    controls (no repeats)."""
    from ..vf_format import canonical
    df = _read(raw_dir, control)
    sens = _sens(df)
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(sens))]
    n = sens.shape[0]
    if holdout is not None and 0.0 < holdout < 1.0:
        perm = np.random.default_rng(seed).permutation(n)
        n_eval = max(1, int(round(holdout * n)))
        eval_idx, fit_idx = perm[:n_eval], perm[n_eval:]
        nv = np.nanmean(sens[fit_idx], axis=0)           # normative fit on the COMPLEMENTARY split
        rows_idx = eval_idx
    else:
        nv = np.nanmean(sens, axis=0)                    # age-pooled control normal (l1..l54 order)
        rows_idx = np.arange(n)
    out = []
    for i in rows_idx:
        td = sens[i] - nv
        if np.isnan(td).all():
            continue
        out.append({"values": td, "coords": canonical.CANON_COORDS, "md": canonical.md(td),
                    "patient_id": ids[int(i)], "label": 0, "aligned": True})
    return out


def iowa_healthy_fields(raw_dir: Path, control: str = "vfctrIowaPC26") -> list[dict]:
    """GENUINE-healthy 24-2 fields (``label=0``) from the Iowa full-field CENTRAL controls (Marín-Franch et al.
    2018; 98 ocularly-healthy eyes, custom ≤26° 64-location pattern → :data:`IOWA_PC26`), as TD analogs
    (sensitivity − Iowa control mean) resampled onto the canonical 24-2 grid. A **separate** genuine-healthy
    cohort from SUNY-IU → the model's normative is NOT fit on it, so the specificity / over-referral read is
    **non-circular without any held-out split** (replaces the SUNY-IU half-split). The dense central grid covers
    the 24-2 field; the complementary peripheral test (``vfctrIowaPeri``, 26–81°) is out-of-field → not used.

    NOTE (honest): Iowa's "two tests" per eye are the spatially-complementary central + peripheral halves of one
    full-field exam, NOT a repeat → Iowa grounds the healthy normative/specificity, it is NOT a test-retest σ
    source (σ stays :func:`retest_noise_sd`, vfpwgRetest)."""
    from ..vf_format import canonical
    df = _read(raw_dir, control)
    sens = _sens(df, _LOC64)                                  # (n, 64) Iowa central sensitivities
    nv = np.nanmean(sens, axis=0)                             # Iowa control mean (its own genuine-healthy normal)
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(sens))]
    oc = canonical.best_orientation(IOWA_PC26)
    out = []
    for i in range(sens.shape[0]):
        vals = canonical.resample(sens[i] - nv, oc)          # TD analog placed on the canonical 24-2 grid
        if np.isnan(vals).all():
            continue
        out.append({"values": vals, "coords": canonical.CANON_COORDS, "md": canonical.md(vals),
                    "patient_id": ids[int(i)], "label": 0, "aligned": True})
    return out


def retest_series(raw_dir: Path, name: str = "vfpwgRetest24d2") -> dict[str, np.ndarray]:
    """``{f"{id}_{eye}": (n_visit, 54) sensitivity matrix}`` — the short-term test-retest series."""
    df = _read(raw_dir, name)
    out: dict[str, np.ndarray] = {}
    for (sid, eye), g in df.groupby(["id", "eye"]):
        out[f"{sid}_{eye}"] = _sens(g)
    return out


def retest_noise_sd(raw_dir: Path, name: str = "vfpwgRetest24d2") -> float:
    """Real short-term test-retest SD (dB) grounding the synthetic noise scale: per eye, the SD of
    **consecutive-visit differences** per location / √2 (the drift-robust short-term repeatability Sw — matches
    the consecutive-pair convention used elsewhere), averaged over locations then eyes. Uses the vfpwgRetest
    weekly repeats — the only repeat-rich HFA 24-2 test-retest. (SUNY-IU controls are single-field → cannot
    supply a retest σ; they ground the normative only. Eyecatcher's HFA repeats are reserved for R1.)"""
    sds = []
    for m in retest_series(raw_dir, name).values():
        if m.shape[0] >= 2:
            d = np.diff(m, axis=0)                                    # consecutive-visit diffs, per location
            sds.append(np.nanmean(np.nanstd(d, axis=0) / np.sqrt(2)))  # Sw = SD(Δ)/√2 (drift-robust)
    return float(np.nanmean(sds)) if sds else float("nan")


def retest_td_pairs(raw_dir: Path, normal_vec: np.ndarray | None = None,
                    name: str = "vfpwgRetest24d2") -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(TD_a, TD_b, patient_id) short-term test-retest pairs — ALL consecutive weekly fields per eye, as TD
    analogs (sensitivity − control normal) in the shared ``l1..l54`` order. patient_id = the series ``id``
    (shared across an eye's repeats) so CIs cluster by patient (effective n = patients, not pairs)."""
    nv = control_mean_vector(raw_dir) if normal_vec is None else np.asarray(normal_vec, dtype=float)
    pairs = []
    for key, m in retest_series(raw_dir, name).items():
        pid = key.rsplit("_", 1)[0]  # "{id}_{eye}" → id (patient)
        for i in range(m.shape[0] - 1):
            pairs.append((m[i] - nv, m[i + 1] - nv, pid))
    return pairs


# --- Eyecatcher: real HFA <-> tablet cross-device pointwise VF -----------------------------------

def _eyecatcher_cols(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """The DLS_x{X}_y{Y}_dB columns and their real (x,y) degree coordinates (parsed from the names)."""
    cols, xy = [], []
    for c in df.columns:
        m = _ECATCH_RE.fullmatch(c)
        if m:
            x = -int(m.group(2)) if m.group(1) else int(m.group(2))
            y = -int(m.group(4)) if m.group(3) else int(m.group(4))
            cols.append(c)
            xy.append((float(x), float(y)))
    return cols, np.array(xy, dtype=float)


def eyecatcher_frame(raw_dir: Path, csv_name: str = "eyecatcher_home_rawdata.csv") -> pd.DataFrame:
    return pd.read_csv(Path(raw_dir) / csv_name)


def _ec_canon_vec(row: pd.Series, cols: list[str], coords: np.ndarray, level: float) -> np.ndarray:
    """One Eyecatcher field → canonical-order TD vector (DLS − flat normative level, resampled to CANON)."""
    from ..vf_format import canonical
    dls = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
    td = dls - level
    return canonical.resample(td, canonical.best_orientation(coords))


def eyecatcher_fields(raw_dir: Path, *, device: str, normative: dict | None = None,
                      attach_native_md: bool = False,
                      csv_name: str = "eyecatcher_home_rawdata.csv") -> list[dict]:
    """Eyecatcher fields for one ``Device`` (``"HFA"`` or ``"Eyecatcher"``) as TD-analog canonical fields.
    ``attach_native_md``: also attach the row's shipped device ``MD_dB`` as ``native_md`` — the device's OWN
    mean-deviation, so the R2 rung is scored against the real device MD (consistency with Rotterdam/10-2/Harvard)
    rather than a resampled central-only mean."""
    from ..vf_format import canonical
    df = eyecatcher_frame(raw_dir, csv_name)
    cols, coords = _eyecatcher_cols(df)
    level = 30.0 if not normative else normative["level_normal_age45"]
    ocoords = canonical.best_orientation(coords)
    out = []
    for _, r in df[df["Device"].astype(str) == device].iterrows():
        dls = pd.to_numeric(r[cols], errors="coerce").to_numpy(dtype=float)
        vals = canonical.resample(dls - level, ocoords)
        if np.isnan(vals).all():
            continue
        fld = {"values": vals, "coords": canonical.CANON_COORDS, "aligned": True,
               "md": canonical.md(vals), "patient_id": f"{r['ParticipantID']}_{r['Eye']}"}
        if attach_native_md:
            nm = pd.to_numeric(r.get("MD_dB"), errors="coerce")
            if pd.notna(nm):
                fld["native_md"] = float(nm)
        out.append(fld)
    return out


def eyecatcher_crossdevice_pairs(raw_dir: Path, normative: dict | None = None,
                                 csv_name: str = "eyecatcher_home_rawdata.csv"
                                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """(HFA, tablet, patient_id) canonical-order TD-vector pairs → REAL cross-device same-signal invariance.
    ALL HFA×tablet combinations per eye (2 HFA × 6 tablet ≈ 12/eye); patient_id = ParticipantID so CIs
    cluster by patient (effective n = patients, not pairs)."""
    df = eyecatcher_frame(raw_dir, csv_name)
    cols, coords = _eyecatcher_cols(df)
    level = 30.0 if not normative else normative["level_normal_age45"]
    pairs = []
    for (ppid, _eye), g in df.groupby(["ParticipantID", "Eye"]):
        hfa = g[g["Device"].astype(str) == "HFA"]
        tab = g[g["Device"].astype(str) == "Eyecatcher"]
        for _, h in hfa.iterrows():
            a = _ec_canon_vec(h, cols, coords, level)
            for _, t in tab.iterrows():
                pairs.append((a, _ec_canon_vec(t, cols, coords, level), str(ppid)))
    return pairs


def eyecatcher_eye_reads(raw_dir: Path, normative: dict | None = None,
                         csv_name: str = "eyecatcher_home_rawdata.csv") -> list[dict]:
    """Per-eye Eyecatcher reads for the within-device **device-switch stability** analysis (R1). **Excludes
    `SessionType == "Practice"`.** One dict per eye: ``{patient_id, eye, hfa_sessions, tablet}`` of canonical
    TD vectors, where ``hfa_sessions`` = a list of **same-session** HFA replicate groups (Baseline pair,
    Followup pair) — the clean short-term repeatability replicates — and ``tablet`` = the 6 HomeMonitoring
    tablet fields (consecutive-pair repeatability). Requires ≥1 HFA session and ≥2 tablet tests per eye."""
    df = eyecatcher_frame(raw_dir, csv_name)
    cols, coords = _eyecatcher_cols(df)
    level = 30.0 if not normative else normative["level_normal_age45"]
    out = []
    for (ppid, eye), g in df.groupby(["ParticipantID", "Eye"]):
        g = g.sort_values("TestN") if "TestN" in g.columns else g
        hfa = g[g["Device"].astype(str) == "HFA"]
        sessions = []
        for st in ("Baseline", "Followup"):
            sub = hfa[hfa["SessionType"].astype(str) == st]
            vecs = [_ec_canon_vec(r, cols, coords, level) for _, r in sub.iterrows()]
            if vecs:
                sessions.append(vecs)
        tab = g[(g["Device"].astype(str) == "Eyecatcher") & (g["SessionType"].astype(str) == "HomeMonitoring")]
        tablet = [_ec_canon_vec(r, cols, coords, level) for _, r in tab.iterrows()]
        if sessions and len(tablet) >= 2:
            out.append({"patient_id": str(ppid), "eye": str(eye),
                        "hfa_sessions": sessions, "tablet": tablet})
    return out


def eyecatcher_device_md(raw_dir: Path, csv_name: str = "eyecatcher_home_rawdata.csv") -> list[dict]:
    """Per-eye **native device MD** (the shipped `MD_dB`, each device's own read) for the R1 native-device
    reference: ``{patient_id, hfa: [MD_dB…], tablet: [MD_dB…]}`` over the same eyes (Practice excluded).
    NOTE: HFA and tablet `MD_dB` are on **different normatives** → this is the *raw-device* agreement (what a
    clinician reads off each printout), distinct from the model-based canonical read that drives R1."""
    df = eyecatcher_frame(raw_dir, csv_name)
    out = []
    for (ppid, _eye), g in df.groupby(["ParticipantID", "Eye"]):
        dev, st = g["Device"].astype(str), g["SessionType"].astype(str)
        hfa = pd.to_numeric(g[dev == "HFA"]["MD_dB"], errors="coerce").dropna().tolist()
        tab = pd.to_numeric(g[(dev == "Eyecatcher") & (st == "HomeMonitoring")]["MD_dB"],
                            errors="coerce").dropna().tolist()
        if hfa and tab:
            out.append({"patient_id": str(ppid), "hfa": hfa, "tablet": tab})
    return out


def retest_eye_visits(raw_dir: Path, normal_vec: np.ndarray | None = None,
                      name: str = "vfpwgRetest24d2") -> list[list[np.ndarray]]:
    """Per-eye list of ALL weekly visit TD vectors (sensitivity − control normal), for the external short-term
    HFA repeatability floor. Each inner list = one eye's ~12 visits in order (consecutive-pair repeatability)."""
    nv = control_mean_vector(raw_dir) if normal_vec is None else np.asarray(normal_vec, dtype=float)
    return [[m[i] - nv for i in range(m.shape[0])]
            for m in retest_series(raw_dir, name).values() if m.shape[0] >= 2]


# --- 10-2 cross-pattern (real Humphrey 10-2 total deviation, placed on P10D2) --------------------

def tenten2_fields(raw_dir: Path, csv_name: str = "vf10_2.csv") -> list[dict]:
    """Real 10-2 glaucoma fields (TD_* columns) placed on the visualFields p10d2 layout, resampled to the
    canonical 24-2 grid. 10-2 samples only the central ±9°, so a resampled field fills just the central
    canonical slots (periphery NaN) — a genuine **cross-pattern** submission. MD here is central-only and
    NOT directly comparable to 24-2 MD; treat it as a different-pattern read (level, not 24-2 MD)."""
    from ..vf_format import canonical
    df = pd.read_csv(Path(raw_dir) / csv_name)
    td_cols = sorted((c for c in df.columns if c.startswith("TD_") and c[3:].isdigit()),
                     key=lambda c: int(c[3:]))
    oc = canonical.best_orientation(P10D2)
    n = min(len(td_cols), len(oc))
    out = []
    for _, r in df.iterrows():
        v = pd.to_numeric(r[td_cols[:n]], errors="coerce").to_numpy(dtype=float)
        vals = canonical.resample(v, oc[:n])
        if np.isnan(vals).all():
            continue
        fld = {"values": vals, "coords": canonical.CANON_COORDS, "aligned": True,
               "md": canonical.md(vals), "patient_id": str(r.get("id", r.name)).removesuffix(".0")}
        nm = pd.to_numeric(r.get("MD"), errors="coerce")     # shipped 10-2 (central) MD → native GT for R2
        if pd.notna(nm):
            fld["native_md"] = float(nm)
        out.append(fld)
    return out
