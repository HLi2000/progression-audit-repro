"""Physics-DERIVED cross-device OCT B-scan appearance parameters (OCT-FormatShift Pillar 1).

The audit (memory `p0-degradation-panel`; docs §12) found the B-scan shift's magnitudes were hand-set with
no citation (`bscan_transforms.CONTRAST_MAX`, `BLUR_MAX_SIGMA`). This module replaces them with values
DERIVED from published device specifications, so the B-scan corruption family has FEW, PHYSICAL, CITED free
parameters — the precondition for the endpoint-calibration in Pillar 1 (a small physical family, then tune
its overall dose so the segmenter-induced RNFL-thickness change reproduces the published Bland-Altman LoA).

Textbook OCT relations (Agent-2 research, Path 3):
  * axial resolution (FWHM, in air):  Δz = (2 ln2 / π) · λ0² / Δλ     → /n (n≈1.38 tissue)
  * lateral resolution (Gaussian beam): Δx = 0.37 · λ0 / NA
  * speckle amplitude contrast:  C = 0.52 / √N_ART   (fully-developed Rayleigh amplitude contrast 0.52;
    Schmitt 1999 DOI 10.1117/1.429925; ART frame-averaging of N frames divides speckle SD by √N)
  * wavelength backscatter: μ_s falls with λ (Almasian 2017 PMC5665955) → longer-λ swept-source devices
    (Topcon Triton 1050 nm) read weaker backscatter / lower display contrast than the 840-870 nm SD devices.

The cross-DEVICE shift is the DELTA between two devices' derived parameters (e.g. Cirrus full speckle
C=0.52 → Spectralis ART-averaged C≈0.13; Cirrus 5 µm axial → Topcon 8 µm axial = added blur). The absolute
dose is then set by the endpoint calibration, NOT here — this module only fixes the physically-correct
RATIOS/directions between devices so the family is not hand-tuned.

Device specs (with sources) — Agent-2 Path 2 / datasheets:
  Cirrus 5000/6000 (Zeiss): λ0 840 nm, axial 5 µm, lateral 15 µm, single-frame (no B-scan averaging).
  Spectralis (Heidelberg):  λ0 870 nm, axial 7 µm (3.9 µm marketed), lateral 14 µm, ART frame-averaging.
  RTVue/Avanti (Optovue):   λ0 840 nm, axial 5 µm, lateral 15 µm, single-frame.
  Topcon DRI Triton (SS):   λ0 1050 nm, axial 8 µm, lateral 20 µm, single-frame, swept-source.
  Measured cross-device anchors: Spectralis OCT1→OCT2 +2.4 dB SNR (PMC6710930); 11-device phantom layer
  over-measure 2.7-4.3 µm (PMC7075602).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548: FWHM = 2.355·σ for a Gaussian
_RAYLEIGH_AMPLITUDE_CONTRAST = 0.52                  # fully-developed amplitude speckle contrast (Schmitt 1999)


@dataclass(frozen=True)
class DeviceSpec:
    """Published OCT device parameters (with the source noted in the module docstring)."""

    name: str
    lambda0_nm: float          # center wavelength
    axial_res_um: float        # axial resolution (FWHM, tissue) — datasheet
    lateral_res_um: float      # lateral resolution
    n_art: int                 # B-scan frame-averaging (1 = single frame → full speckle)
    relative_backscatter: float  # ∝ μ_s(λ), normalised to Cirrus=1.0 (falls with wavelength; Almasian 2017)


# relative_backscatter: heuristic from μ_s decreasing with wavelength (Almasian 2017), Cirrus(840nm)=1.0;
# Spectralis 870 nm ~0.97; Topcon 1050 nm materially weaker ~0.75 (documented approximate, used only for
# the cross-device contrast RATIO, whose absolute dose the endpoint calibration sets).
DEVICES: dict[str, DeviceSpec] = {
    "cirrus":     DeviceSpec("cirrus", 840.0, 5.0, 15.0, 1, 1.00),
    "spectralis": DeviceSpec("spectralis", 870.0, 7.0, 14.0, 16, 0.97),
    "rtvue":      DeviceSpec("rtvue", 840.0, 5.0, 15.0, 1, 1.00),
    "topcon":     DeviceSpec("topcon", 1050.0, 8.0, 20.0, 1, 0.75),
}

REFERENCE_DEVICE = "cirrus"   # our source device (Harvard-GF/GDP)

# EMPIRICAL per-vendor DISPLAY speckle contrast (median SD/mean, 9 px window over the tissue region), MEASURED
# on RETOUCH real B-scans (Cirrus/Spectralis/Topcon, SAME preprocessing, n=30 each). REPRODUCIBLE from the raw
# data by validity.retouch_display_speckle (asserted in tests/test_oct_retouch_speckle — these constants are
# the median it returns, not hand-set).
# These are the LOG-COMPRESSED 8-bit DISPLAY contrasts, NOT the raw amplitude 0.52/√N — so they set a REALISTIC
# injection magnitude (the raw amplitude overshoots the display). They VALIDATE the frame-averaging ORDERING
# (Cirrus single-frame > Topcon single-frame > Spectralis 16-frame ART) that speckle_contrast(n_art) predicts.
# ⚠ RETOUCH is MACULAR: the device-characteristic ORDERING is region-INDEPENDENT (frame-averaging) and transfers
# to the disc, but the ABSOLUTE level is a PROXY for the disc region (tissue-dependent) — disclosed downstream.
DISPLAY_SPECKLE_CONTRAST: dict[str, float] = {
    "cirrus": 0.365, "spectralis": 0.148, "topcon": 0.184, "rtvue": 0.365,  # rtvue single-frame ≈ cirrus
}


def axial_resolution_um(lambda0_nm: float, bandwidth_nm: float, n_tissue: float = 1.38) -> float:
    """Axial resolution FWHM (µm) from center wavelength + spectral bandwidth: Δz = (2ln2/π)·λ0²/Δλ / n."""
    lam_um, bw_um = lambda0_nm / 1000.0, bandwidth_nm / 1000.0
    return (2.0 * np.log(2.0) / np.pi) * (lam_um ** 2) / bw_um / n_tissue


def bandwidth_for_axial_um(lambda0_nm: float, axial_res_um: float, n_tissue: float = 1.38) -> float:
    """Invert the axial-resolution relation → the spectral bandwidth (nm) implied by a datasheet axial res."""
    lam_um = lambda0_nm / 1000.0
    bw_um = (2.0 * np.log(2.0) / np.pi) * (lam_um ** 2) / axial_res_um / n_tissue
    return bw_um * 1000.0


def lateral_resolution_um(lambda0_nm: float, na: float) -> float:
    """Lateral (Gaussian-beam) resolution: Δx = 0.37·λ0/NA."""
    return 0.37 * (lambda0_nm / 1000.0) / na


def speckle_contrast(n_art: int) -> float:
    """Amplitude speckle contrast after ART frame-averaging: C = 0.52/√N (Rayleigh 0.52; ÷√N averaging)."""
    return _RAYLEIGH_AMPLITUDE_CONTRAST / np.sqrt(max(1, int(n_art)))


def added_blur_sigma_um(ref_axial_um: float, target_axial_um: float) -> float:
    """Extra Gaussian σ (µm) that degrades a ``ref``-resolution B-scan to a coarser ``target`` resolution.

    Convolving two Gaussians adds their variances in FWHM: added-FWHM = √(max(0, target² − ref²)); σ =
    FWHM/2.355. Zero when the target is equal/finer (we do not sharpen — a coarser device only adds blur).
    """
    added_fwhm = np.sqrt(max(0.0, target_axial_um ** 2 - ref_axial_um ** 2))
    return added_fwhm / _FWHM_PER_SIGMA


@dataclass(frozen=True)
class CrossDeviceAppearance:
    """Physically-derived cross-device B-scan appearance shift (ref → target), pre-endpoint-calibration."""

    ref: str
    target: str
    speckle_contrast_ref: float
    speckle_contrast_target: float
    speckle_ratio: float          # target/ref amplitude contrast (<1 = target smoother, e.g. ART averaging)
    added_blur_sigma_um: float    # extra axial blur to reach the coarser device (0 if target is finer)
    contrast_ratio: float         # target/ref display contrast from relative backscatter (μ_s(λ))


def cross_device_appearance(ref: str = "spectralis", target: str = "topcon") -> CrossDeviceAppearance:
    """Derive the ref→target B-scan appearance shift purely from the published device specs.

    Returns physically-correct RATIOS/magnitudes (speckle change, added blur, contrast change); the absolute
    dose is fixed downstream by the endpoint calibration (so the family stays physical, not hand-tuned).
    """
    r, t = DEVICES[ref], DEVICES[target]
    # DISPLAY speckle contrasts, RETOUCH-calibrated (realistic dose) — NOT the raw amplitude speckle_contrast(n_art)
    # (which overshoots the log-compressed 8-bit display). speckle_contrast(n_art) still documents the physics
    # ORDERING that RETOUCH validates.
    c_ref, c_tar = DISPLAY_SPECKLE_CONTRAST[ref], DISPLAY_SPECKLE_CONTRAST[target]
    return CrossDeviceAppearance(
        ref=ref, target=target,
        speckle_contrast_ref=c_ref, speckle_contrast_target=c_tar,
        speckle_ratio=c_tar / c_ref,
        added_blur_sigma_um=added_blur_sigma_um(r.axial_res_um, t.axial_res_um),
        contrast_ratio=t.relative_backscatter / r.relative_backscatter,
    )
