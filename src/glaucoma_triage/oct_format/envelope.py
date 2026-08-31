"""Interpretation model + metric library for OCT-FormatShift (structural analog of vf_format.envelope).

A model reads a canonical RNFLT map (raw pixels IN — like VF's raw field) and recovers the clinical
**TSNIT peripapillary RNFL profile** (the read OUT); the **abnormal call is then DERIVED** from that
predicted profile via an empirical normative-percentile reference (the clinical red/yellow/green
mechanism), so it is tied to the read and inherits its stability. The benchmark scores how much the
read + call survive a synthetic format shift.

Arms (all on ONE model kind — a multi-output Ridge; ``constant`` kept only as the internal certificate
reference):
  * ``baseline``       — trained on clean maps only.
  * ``dr``             — denoising anchor: trained on synthetic format variants (domain-randomised over
                         the calibrated vendor envelope) -> clean profile target.
  * ``dr_consistency`` — anchor + an up-weighted consistency term ``f(shift_i) == f(shift_j)`` enforced as
                         cross-vendor agreement to the clean read (two DIFFERENT vendor formats of an eye,
                         both -> the same clean profile, weighted by lambda) — the scaler-safe linear form
                         of the Pi-model penalty. Tested IN PARALLEL with ``dr`` so the leaderboard shows
                         whether it earns its place.
Grounding: TSNIT read (Mariottoni 2020; Driessen 2024); denoising-AE (Vincent 2008; MAE He 2022);
consistency (Pi-model Laine&Aila 2017; UDA Xie 2020; DAIR Huang 2022); normative call (Silverman 2016;
Yang 2020). Generic uncertainty helpers are reused from ``vf_format.envelope`` (pure statistics).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The raw-pixel input has ~1/3 of its columns permanently NaN (pixels outside the disc mask are the same
# for every eye). SimpleImputer correctly drops those all-NaN columns; silence its per-fit warning — the
# structural NaN is by design (the disc mask), not a data problem.
warnings.filterwarnings("ignore", message="Skipping features without any observed values",
                        category=UserWarning)

from . import canonical, transforms
from ..vf_format.envelope import cluster_ci, loa_tolerance_ci, mcnemar_kappa, wilson_ci  # noqa: F401

__all__ = [
    "to_features", "profile_true", "sector_matrix", "InterpModel", "build_interp_model",
    "fit_reference", "train_models", "decision_stability", "profile_abs_errors",
    "sector_abs_errors", "pair_agreement", "cluster_ci", "wilson_ci", "loa_tolerance_ci", "mcnemar_kappa",
]


def to_features(canon_maps: np.ndarray) -> np.ndarray:
    """Model INPUT = the flattened raw canonical map (pixels), (N,H,W) -> (N,H*W). NaN imputed downstream.

    The read (sectors/profile) is the OUTPUT, never the input — so input != output (no trivial copy);
    mirrors vf_format (raw field values in, derived MD out).
    """
    canon_maps = np.asarray(canon_maps, dtype=float)
    flat = np.array([canonical.clean(m).ravel() for m in canon_maps], dtype=float)
    return flat


def profile_true(canon_maps: np.ndarray, n: int = canonical.PROFILE_N) -> np.ndarray:
    """Regression TARGET = the clean map's N-point TSNIT profile, (N, n)."""
    return np.array([canonical.tsnit_profile(m, n) for m in canon_maps], dtype=float)


def sector_matrix(profiles: np.ndarray) -> np.ndarray:
    """Aggregate profiles (N,n) -> per-sector [global,S,N,I,T] (N,5) for focal/recovery reads."""
    secs = ("global", "s", "n", "i", "t")
    return np.array([[canonical.profile_sectors(p)[s] for s in secs] for p in profiles], dtype=float)


def fit_reference(train_maps: np.ndarray, glaucoma: np.ndarray, n: int = canonical.PROFILE_N
                  ) -> canonical.NormativeReference:
    """Fit the empirical normative reference from the HEALTHY (glaucoma==0) training eyes' profiles."""
    profs = profile_true(train_maps, n)
    return canonical.NormativeReference.fit(profs, np.asarray(glaucoma) == 0)


def _regressor(kind: str) -> Pipeline:
    if kind == "constant":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", DummyRegressor(strategy="mean"))])
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("reg", Ridge(alpha=10.0))])


class InterpModel:
    """Reads canonical RNFLT maps (pixels) -> TSNIT profile; the abnormal call is DERIVED via the reference."""

    def __init__(self, kind: str = "linear"):
        self.kind = kind
        self.reg = _regressor(kind)
        self.reference: canonical.NormativeReference | None = None

    def fit(self, X: np.ndarray, profiles: np.ndarray, *, sample_weight: np.ndarray | None = None) -> "InterpModel":
        self.reg.fit(X, profiles, reg__sample_weight=sample_weight)
        return self

    def set_reference(self, ref: canonical.NormativeReference) -> "InterpModel":
        self.reference = ref
        return self

    def predict_profile(self, X: np.ndarray) -> np.ndarray:
        return self.reg.predict(X)

    def predict_abn(self, X: np.ndarray, *, mode: str = "any_sector") -> np.ndarray:
        """Derived abnormal call (0/1) = the normative-percentile flag on the PREDICTED profile."""
        if self.reference is None:
            raise RuntimeError("InterpModel.predict_abn needs a NormativeReference (call set_reference)")
        return self.reference.abnormal_batch(self.predict_profile(X), mode=mode)


def build_interp_model(kind: str = "linear") -> InterpModel:
    return InterpModel(kind)


def _dr_samples(clean_maps: np.ndarray, rng: np.random.Generator, n_aug: int, levels: tuple[float, ...],
                pairs: tuple[str, ...], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Anchor samples: features of synthetic variants -> the CLEAN profile target (recover-through-shift).

    Domain randomisation over the calibrated vendor envelope (Tobin 2017) with BALANCED coverage:
    ``n_aug`` variants for EACH pair (random level each), so a wide envelope doesn't starve the hard
    pairs of augmentation — the DR arm learns to recover through the WHOLE envelope, the ``one shared
    model across devices`` contender the taxonomy adjudicates.
    """
    prof0 = profile_true(clean_maps, n)
    Xs, ys = [], []
    for pair in pairs:
        for _ in range(n_aug):
            lv = float(rng.choice(levels))
            variants = np.stack([transforms.apply_format(m, rng, lv, pair=pair) for m in clean_maps])
            Xs.append(to_features(variants))
            ys.append(prof0)
    return np.vstack(Xs), np.vstack(ys)


def _consistency_samples(clean_maps: np.ndarray, rng: np.random.Generator, n_pairs: int,
                         levels: tuple[float, ...], pairs: tuple[str, ...], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Consistency samples: TWO DIFFERENT vendor formats of each eye, BOTH -> the same clean profile.

    The Pi-model consistency term (f(shift_i) == f(shift_j)) realised in a scaler-safe way. The naive
    ``(feat_i - feat_j) -> 0`` difference-pseudo-sample is WRONG here: the InterpModel pipeline centers
    features by their ~80 um mean, so a mean-0 difference fed through it becomes a far-out-of-distribution
    outlier and corrupts the fit. Instead we up-weight the *agreement itself* — heterogeneous vendor
    formats of one eye must collapse to one clean read — as ordinary absolute-feature anchor rows that the
    scaler handles correctly. ``i`` and ``j`` draw DIFFERENT vendor pairs so the emphasis is cross-vendor.
    """
    prof0 = profile_true(clean_maps, n)
    Xs, ys = [], []
    for _ in range(n_pairs):
        pi, pj = rng.choice(len(pairs), size=2, replace=len(pairs) < 2)
        li, lj = float(rng.choice(levels)), float(rng.choice(levels))
        vi = to_features(np.stack([transforms.apply_format(m, rng, li, pair=pairs[pi]) for m in clean_maps]))
        vj = to_features(np.stack([transforms.apply_format(m, rng, lj, pair=pairs[pj]) for m in clean_maps]))
        Xs.extend([vi, vj]); ys.extend([prof0, prof0])
    return np.vstack(Xs), np.vstack(ys)


def train_models(train_maps: np.ndarray, glaucoma: np.ndarray, rng: np.random.Generator, *,
                 kind: str = "linear", n_aug: int = 4, levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                 pairs: tuple[str, ...] = transforms.CALIBRATED_PAIRS, consistency_lambda: float = 1.0,
                 n: int = canonical.PROFILE_N) -> dict[str, InterpModel]:
    """Fit the normative reference (from healthy eyes) + the baseline / dr / dr_consistency arms.

    Returns a dict of arms; every arm carries the shared reference. ``glaucoma`` (0/1) selects the
    healthy eyes for the normative reference (standard normative-DB practice; the tolerance score stays
    label-free). The DR arms domain-randomise over ``pairs`` (the calibrated vendor envelope by default).
    """
    ref = fit_reference(train_maps, glaucoma, n)
    Xc, prof_c = to_features(train_maps), profile_true(train_maps, n)

    baseline = build_interp_model(kind).fit(Xc, prof_c).set_reference(ref)

    Xa, ya = _dr_samples(train_maps, rng, n_aug, levels, pairs, n)
    X_dr, y_dr = np.vstack([Xc, Xa]), np.vstack([prof_c, ya])
    dr = build_interp_model(kind).fit(X_dr, y_dr).set_reference(ref)

    Xk, yk = _consistency_samples(train_maps, rng, n_aug, levels, pairs, n)
    X_c = np.vstack([Xc, Xa, Xk])
    y_c = np.vstack([prof_c, ya, yk])
    w_c = np.concatenate([np.ones(len(Xc) + len(Xa)), np.full(len(Xk), consistency_lambda)])
    dr_consistency = build_interp_model(kind).fit(X_c, y_c, sample_weight=w_c).set_reference(ref)

    return {"baseline": baseline, "dr": dr, "dr_consistency": dr_consistency, "_reference": ref}


def _shift(clean_maps: np.ndarray, level: float, rng: np.random.Generator, pair: str) -> np.ndarray:
    return np.stack([transforms.apply_format(m, rng, level, pair=pair) for m in clean_maps])


def decision_stability(model: InterpModel, clean_maps: np.ndarray, levels: list[float],
                       rng: np.random.Generator, *, pair: str = transforms.DEFAULT_PAIR) -> list[dict]:
    """Per-level decision-flip rate: fraction whose derived abnormal call changes vs the clean read."""
    base_call = model.predict_abn(to_features(clean_maps))
    out = []
    for lv in levels:
        variants = _shift(clean_maps, lv, rng, pair)
        call = model.predict_abn(to_features(variants))
        dists = [transforms.format_distance(m, v) for m, v in zip(clean_maps, variants)]
        out.append({"level": lv, "distance": float(np.mean(dists)),
                    "flip_rate": float(np.mean(call != base_call))})
    return out


def profile_abs_errors(model: InterpModel, clean_maps: np.ndarray, level: float,
                       rng: np.random.Generator, *, pair: str = transforms.DEFAULT_PAIR) -> np.ndarray:
    """Per-eye mean |predicted profile on the shifted map - true clean profile| (um): recovery error."""
    truth = profile_true(clean_maps)
    variants = _shift(clean_maps, level, rng, pair)
    pred = model.predict_profile(to_features(variants))
    return np.nanmean(np.abs(pred - truth), axis=1)


def sector_abs_errors(model: InterpModel, clean_maps: np.ndarray, level: float,
                      rng: np.random.Generator, *, pair: str = transforms.DEFAULT_PAIR) -> np.ndarray:
    """Per-eye WORST-sector |Δ sector RNFL| (um) — the focal-recovery read (S/N/I/T)."""
    truth = sector_matrix(profile_true(clean_maps))[:, 1:]  # drop global -> quadrants
    variants = _shift(clean_maps, level, rng, pair)
    pred = sector_matrix(model.predict_profile(to_features(variants)))[:, 1:]
    return np.nanmax(np.abs(pred - truth), axis=1)


def pair_agreement(model: InterpModel, maps_a: np.ndarray, maps_b: np.ndarray) -> dict:
    """Same-eye invariance across two acquisitions: abnormal-call kappa + |Δ profile| (um)."""
    ca = model.predict_abn(to_features(maps_a))
    cb = model.predict_abn(to_features(maps_b))
    pa = model.predict_profile(to_features(maps_a))
    pb = model.predict_profile(to_features(maps_b))
    k = mcnemar_kappa(ca, cb)
    return {"kappa": k.get("kappa"), "abs_delta_profile": float(np.nanmean(np.abs(pa - pb))), "n": len(ca)}
