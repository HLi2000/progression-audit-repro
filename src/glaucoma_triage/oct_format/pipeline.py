"""The decision-pipeline interface — the COMMON GROUND every contender is scored on (OCT-FormatShift v3).

The v3 reframe (plan §PRIMARY; docs/OCT_FORMATSHIFT_SCOPING.md §0): heterogeneous models (a full-res map
CNN, a B-scan foundation model, the Eyened disc specialist, a ComBat-harmonised reader, the Ridge floor)
consume DIFFERENT inputs and emit DIFFERENT native reads (a cpRNFL profile, an embedding, a glaucoma
logit). They cannot be compared on any one native read without privileging that representation. What they
DO share is the clinical act at the end of the pipeline — the **glaucoma referral call**. So the common
ground is a single method:

    predict_referral_proba(eyes) -> ndarray in [0, 1]     # P(refer this eye) per eye

Everything the paired-degradation scorer needs (referral-flip, ΔAUC, Δsens@spec, detection-retention) is
derived from that probability + the eye labels — so a map-reader and a B-scan FM are scored by the SAME
code on the SAME axis, each row LABELLED by representation + adaptation regime (never silently pooled;
the identifiability discipline of §0 — "axis-labelled, never cross-ranked" — is preserved because ranking
lives only on the identified within-Cirrus synthetic axis, see ``degradation.py``).

This module provides the ``OCTModel`` protocol + a ``MapReaderPipeline`` adapter that turns any existing
profile-reader (``envelope.InterpModel`` / ``harmonize.ComBatCorrector`` / ``DenoiseCorrector`` /
``AffineRefitCorrector`` / the ``constant`` exhibit) into an ``OCTModel`` by DERIVING the referral
probability from the reader's predicted TSNIT profile against the shared normative reference — the exact
continuous analog of the clinical red/yellow/green call the readers already make (``predict_abn``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import canonical, envelope

__all__ = ["OCTModel", "referral_scores", "MapReaderPipeline", "ProbaModel", "REFERRAL_TEMP_UM"]

# The logistic temperature (um) mapping the worst-sector normative shortfall -> a referral probability.
# Cosmetic by construction: AUC + sens@spec are rank/threshold-based and INVARIANT to it, and proba>0.5
# reproduces the clinical any-sector call (shortfall>0) for ANY positive temperature. Set to a few um so
# the probability spreads over the clinically-meaningful deficit range.
REFERRAL_TEMP_UM: float = 5.0


@runtime_checkable
class OCTModel(Protocol):
    """A glaucoma-referral decision pipeline — the one interface the common-ground scorer consumes.

    ``name`` / ``representation`` / ``adaptation`` LABEL the row (the identifiability discipline: a map
    CNN and a B-scan FM are never silently pooled — each is tagged and, off the identified synthetic axis,
    never cross-ranked). ``predict_referral_proba`` returns P(refer) in [0, 1] per eye; the 0.5 threshold
    is the clinical call and the ROC over it is the discrimination read.
    """

    name: str
    representation: str          # "map" | "bscan" | "volume" — the native input class
    adaptation: str              # "supervised" | "linear-probe" | "fine-tune" | "harmonizer" | ...

    def predict_referral_proba(self, eyes: Any) -> np.ndarray: ...


def referral_scores(profiles: np.ndarray, reference: canonical.NormativeReference) -> np.ndarray:
    """Continuous referral score per TSNIT profile = the WORST-sector shortfall below the normative 1st pct.

    ``score = max_sector (pct1_sector - read_sector)`` in um: > 0 exactly when >=1 sector falls below its
    1st-percentile normative limit — the continuous form of ``NormativeReference.abnormal`` (``any_sector``
    mode). Monotone in abnormality, so its ROC is the reader's discrimination and thresholding it at 0 is
    the clinical call. Sector order follows ``reference.sectors`` (global, S, N, I, T).
    """
    profiles = np.asarray(profiles, dtype=float)
    pct1 = np.asarray(reference.pct1, dtype=float)
    out = np.empty(len(profiles), dtype=float)
    for i, p in enumerate(profiles):
        agg = np.array([canonical.profile_sectors(p)[s] for s in reference.sectors], dtype=float)
        shortfall = pct1 - agg
        out[i] = float(np.nanmax(shortfall)) if np.isfinite(shortfall).any() else 0.0
    return out


def _sigmoid(x: np.ndarray, temp: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float) / temp))


@dataclass
class MapReaderPipeline:
    """Adapt any profile-reader (``predict_profile(X) -> (N, PROFILE_N)``) into an ``OCTModel``.

    ``eyes`` are canonical RNFLT maps (N, H, W); the pipeline flattens them to features, reads the TSNIT
    profile, scores the worst-sector normative shortfall, and squashes it to a referral probability. Wraps
    ``InterpModel`` (baseline/dr/dr_consistency/constant), ``ComBatCorrector``, ``DenoiseCorrector`` and
    ``AffineRefitCorrector`` unchanged — they all expose ``predict_profile``.
    """

    name: str
    reader: Any
    reference: canonical.NormativeReference
    representation: str = "map"
    adaptation: str = "supervised"
    temp: float = REFERRAL_TEMP_UM

    def predict_referral_proba(self, eyes: np.ndarray) -> np.ndarray:
        X = envelope.to_features(eyes)
        profiles = self.reader.predict_profile(X)
        return _sigmoid(referral_scores(profiles, self.reference), self.temp)


@dataclass
class ProbaModel:
    """Trivial ``OCTModel`` over PRE-COMPUTED per-eye referral probabilities (e.g. an FM linear-probe read
    materialised to disk by a torch script). ``eyes`` is an index array selecting rows of the stored probs,
    so the same held-out split can be scored by the shared degradation code as any in-process reader."""

    name: str
    probs: np.ndarray
    representation: str = "bscan"
    adaptation: str = "linear-probe"
    _by_id: dict[int, float] = field(default_factory=dict)

    def predict_referral_proba(self, eyes: np.ndarray) -> np.ndarray:
        idx = np.asarray(eyes, dtype=int)
        return np.asarray(self.probs, dtype=float)[idx]
