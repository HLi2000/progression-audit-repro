"""Device-conditional harmonizers — a SCORED TREATMENT class for a systematic location-scale device bias.

Scope (docs/OCT_FORMATSHIFT_SCOPING.md §8): a systematic per-device location-scale offset ``T(x)=a+b·x``
can be removed by ESTIMATING the per-device parameters and inverting them — the harmonization canon
(Johnson 2007; Fortin 2018). But that removes only the BIAS. Under the STRUCTURAL per-eye LoA model
(``calibrated_offset``: the cross-device disagreement is a per-eye, anatomy-coupled quantity — Warner 2011;
Buchser 2012 bias-vs-imprecision), ComBat CANNOT remove the per-eye LoA it leaves behind, so behind the
referral DECISION it does NOT top the board — the device-blind ``dr`` (domain-randomisation) arm is the
steadier reader. So these harmonizers are SCORED TREATMENTS on the certificate, NOT a headline: the earlier
"ComBat is the efficient/reliable route" and "ComBat scale-hazard" framings are DROPPED (§8 box).

Two CPU, label-free, ``InterpModel``-shaped contenders that wrap a base reader (the clean ``baseline``):

  * ``ComBatCorrector`` — the empirical-Bayes location(γ↔a)-and-scale(δ↔b) batch corrector (Johnson 2007,
    Biostatistics DOI 10.1093/biostatistics/kxj037), applied per profile-point with EB pooling across
    points. **Transductive**: it ESTIMATES the batch's location/scale from the test device's own read
    sample and aligns them to the stored reference-device distribution — it is never handed the injected
    (a,b) (that would be circular). No published ComBat-on-OCT-RNFL precedent → a genuine contribution.
    Valid here because the synthetic axis shifts the SAME eyes across devices (full content overlap), so
    a device's read-distribution differs from the reference ONLY by the injected offset (unlike the real
    aliased cohorts, where device≡cohort and ComBat cannot separate device from case-mix — §0 spine).

    ⚠ ``scale`` (default True = textbook full location+scale ComBat) vs ``scale=False`` (location-only).
    Decision-level finding (docs §8b; memory p0-degradation-panel), stated HONESTLY: on Harvard-GDP full
    ComBat DROPS detection retention (medians 0.39–0.54; 83% of 10 seeds <0.6; GF stays ≥0.86), the known
    ComBat pitfall of compressing disease-driven variance in a case-mix-imbalanced transductive batch
    (Nygaard 2016), on a 64² non-NN reader → NOT yet a load-bearing device finding.
    ``scale=False`` is NOT a clean fix: it holds retention ~1.0 only by inflating abnormal calls (specificity
    0.90→0.65, worst referral-flip + ΔAUC on the GDP panel) — the sens-only retention metric rewards that
    false-positive trade. Score harmonisers behind the decision on MULTIPLE balanced metrics; do not treat
    ``scale=False`` as triage-safe on this evidence (domain-randomisation is the steadier reader).

  * ``AffineRefitCorrector`` — a device-AWARE per-quadrant affine fit on a HELD-OUT paired calibration
    split (clean↔shifted), inverted at inference (Leite 2011 / Buchser 2012 conversion-equation style).
    Recovers the systematic bias but NOT the LoA noise (Buchser's bias-vs-imprecision split) — the honest
    "convert-and-compare still carries a device residual" contrast.

Both expose ``predict_profile`` / ``predict_abn`` / ``reference`` so they drop straight into the
benchmark leaderboard alongside ``baseline`` / ``dr`` / ``constant``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from . import canonical
from .envelope import InterpModel, profile_true, sector_matrix, to_features

_EPS = 1e-6


# --------------------------------------------------------------------------------------------------
# ComBat empirical-Bayes location/scale correction (Johnson 2007), per profile-point, transductive.
# --------------------------------------------------------------------------------------------------
def _aprior(delta_hat: np.ndarray) -> float:
    m, s2 = float(np.mean(delta_hat)), float(np.var(delta_hat))
    return (2.0 * s2 + m * m) / s2 if s2 > 0 else 1.0


def _bprior(delta_hat: np.ndarray) -> float:
    m, s2 = float(np.mean(delta_hat)), float(np.var(delta_hat))
    return (m * s2 + m**3) / s2 if s2 > 0 else 1.0


def _postmean(g_hat: np.ndarray, g_bar: float, n: float, d_star: np.ndarray, t2: float) -> np.ndarray:
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2: np.ndarray, n: float, a: float, b: float) -> np.ndarray:
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _eb_fit(z: np.ndarray, conv: float = 1e-4, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Johnson-2007 parametric EB fixed point for one batch. ``z`` = standardized reads (n_eyes, G).

    Returns EB-shrunk per-feature location ``g_star`` (G,) and scale ``d_star`` (G,).
    """
    n = float(z.shape[0])
    g_hat = z.mean(axis=0)
    d_hat = np.maximum(z.var(axis=0), _EPS)
    g_bar, t2 = float(g_hat.mean()), float(g_hat.var())
    a, b = _aprior(d_hat), _bprior(d_hat)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((z - g_new) ** 2).sum(axis=0)
        d_new = np.maximum(_postvar(sum2, n, a, b), _EPS)
        change = max(float(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + _EPS))),
                     float(np.max(np.abs(d_new - d_old) / (d_old + _EPS))))
        g_old, d_old = g_new, d_new
        if change < conv:
            break
    return g_old, d_old


class ComBatCorrector:
    """Base reader + empirical-Bayes location/scale batch correction to the reference device.

    ``fit`` stores the reference (clean) per-point profile mean/SD from the base reader; ``predict_profile``
    reads the (device-shifted) batch, standardizes by the reference, EB-estimates the batch location γ and
    scale δ, and inverts them — aligning the device's reads back onto the reference distribution.
    """

    def __init__(self, base: InterpModel, *, eb: bool = True, scale: bool = True):
        self.base = base
        self.eb = eb
        self.scale = scale        # False = location-only (triage-safe; skips the δ scale correction)
        self.alpha: np.ndarray | None = None  # reference per-point mean
        self.sigma: np.ndarray | None = None  # reference per-point SD

    def fit(self, X_ref_clean: np.ndarray, _profiles: np.ndarray | None = None) -> "ComBatCorrector":
        P = self.base.predict_profile(X_ref_clean)
        self.alpha = P.mean(axis=0)
        self.sigma = np.maximum(P.std(axis=0), _EPS)
        return self

    def predict_profile(self, X: np.ndarray) -> np.ndarray:
        if self.alpha is None:
            raise RuntimeError("ComBatCorrector.fit must be called first")
        P = self.base.predict_profile(X)
        z = (P - self.alpha) / self.sigma
        if self.eb and z.shape[0] >= 4:
            g_star, d_star = _eb_fit(z)
        else:  # plain location/scale (large batches ≈ EB; guard tiny batches)
            g_star = z.mean(axis=0)
            d_star = np.maximum(z.var(axis=0), _EPS)
        # location-only (scale=False) removes the systematic offset without the δ scale correction that
        # can inflate diseased reads and hide disease (the GDP triage hazard; see class docstring).
        z_adj = (z - g_star) / np.sqrt(d_star) if self.scale else (z - g_star)
        return z_adj * self.sigma + self.alpha

    def predict_abn(self, X: np.ndarray, *, mode: str = "any_sector") -> np.ndarray:
        return self.reference.abnormal_batch(self.predict_profile(X), mode=mode)

    @property
    def reference(self) -> canonical.NormativeReference:
        return self.base.reference


# --------------------------------------------------------------------------------------------------
# Spatial denoiser — the RIGHT method for the NON-location-scale focal segmentation-error axis.
# --------------------------------------------------------------------------------------------------
class DenoiseCorrector:
    """Base reader + a spatial MEDIAN filter of the map before the read — recovers FOCAL segmentation errors.

    The anti-ComBat control: ComBat (per-point location/scale) cannot invert a per-eye-random focal drop-out/
    spike (`transforms.segmentation_error`), but a median filter removes isolated focal errors. On the
    seg-error axis this certifies where ComBat does NOT — showing the certificate MATCHES method to
    corruption rather than rubber-stamping any one harmoniser.
    """

    def __init__(self, base: InterpModel, *, size: int = 3):
        self.base = base
        self.size = size

    def _denoise(self, X: np.ndarray) -> np.ndarray:
        h, w = canonical.CANON_SHAPE
        out = np.empty_like(X)
        for i, row in enumerate(X):
            m = row.reshape(h, w)
            nan = ~np.isfinite(m)
            fill = float(np.nanmedian(m)) if np.isfinite(m).any() else 0.0
            filt = median_filter(np.where(nan, fill, m), size=self.size)
            filt[nan] = np.nan                                   # keep the disc mask
            out[i] = filt.ravel()
        return out

    def predict_profile(self, X: np.ndarray) -> np.ndarray:
        return self.base.predict_profile(self._denoise(X))

    def predict_abn(self, X: np.ndarray, *, mode: str = "any_sector") -> np.ndarray:
        return self.reference.abnormal_batch(self.predict_profile(X), mode=mode)

    @property
    def reference(self) -> canonical.NormativeReference:
        return self.base.reference


# --------------------------------------------------------------------------------------------------
# Device-aware per-quadrant affine refit on a held-out paired calibration split (conversion-equation).
# --------------------------------------------------------------------------------------------------
def _quadrant_of_point(n: int) -> np.ndarray:
    """Map each of the ``n`` TSNIT profile points to a quadrant index (0=s,1=n,2=i,3=t)."""
    ang = canonical._profile_angles(n)
    q = np.full(n, 3, dtype=int)  # temporal default
    q[(ang >= 45) & (ang < 135)] = 0    # superior
    q[(ang >= 135) & (ang < 225)] = 1   # nasal
    q[(ang >= 225) & (ang < 315)] = 2   # inferior
    return q


class AffineRefitCorrector:
    """Base reader + a per-quadrant affine inverse fit on paired (clean, shifted) calibration reads.

    Device-aware (needs a calibration sample of THIS device's paired reads). Recovers the systematic
    fixed+proportional bias; by construction leaves the LoA noise (Buchser 2012 bias-vs-imprecision).
    """

    def __init__(self, base: InterpModel):
        self.base = base
        self.coef_: np.ndarray | None = None  # (n_quadrants, 2): [intercept, slope] mapping shifted->clean

    def fit_calibration(self, clean_maps: np.ndarray, shifted_maps: np.ndarray) -> "AffineRefitCorrector":
        clean = self.base.predict_profile(to_features(clean_maps))
        shifted = self.base.predict_profile(to_features(shifted_maps))
        n = clean.shape[1]
        q = _quadrant_of_point(n)
        self.coef_ = np.zeros((4, 2))
        for k in range(4):
            xs = shifted[:, q == k].ravel()
            ys = clean[:, q == k].ravel()
            A = np.vstack([np.ones_like(xs), xs]).T
            self.coef_[k], *_ = np.linalg.lstsq(A, ys, rcond=None)
        return self

    def predict_profile(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("AffineRefitCorrector.fit_calibration must be called first")
        P = self.base.predict_profile(X)
        q = _quadrant_of_point(P.shape[1])
        out = P.copy()
        for k in range(4):
            a, b = self.coef_[k]
            out[:, q == k] = a + b * P[:, q == k]
        return out

    def predict_abn(self, X: np.ndarray, *, mode: str = "any_sector") -> np.ndarray:
        return self.reference.abnormal_batch(self.predict_profile(X), mode=mode)

    @property
    def reference(self) -> canonical.NormativeReference:
        return self.base.reference
