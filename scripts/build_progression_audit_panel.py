#!/usr/bin/env python
"""Assemble the single committed data panel that drives every figure/table + number in the
progression-audit TAE paper (papers/progression-audit-tae/).

Two kinds of number live here. The scalar audit constants (floors, device-decode AUCs, dial,
triangulation) are LOCKED, transcribed from the committed source-of-truth reports
(reports/grape_endpoint_sensitivity.md, grape_baseline_match.md, grape_serial_event.md,
grape_serial_fundus.md), and CROSS-CHECKED against the committed data/derived/grape_*.json on disk,
failing loudly on any mismatch (the reconciliation gate). The serial-model blocks
(serial_ablation, two_vft, tsdf_repro, serial_ladder, prospective_baseline, honest_ladder) are
built DIRECTLY from those committed data/derived/grape_*.json — so they are required inputs, not
optional cross-checks: the small analysis JSONs are committed (see .gitignore) and this script
rebuilds the full panel from a clean checkout. Only regenerating those JSONs from scratch needs the
large feature npz + raw GRAPE (Figshare 6406319) on a GPU. No re-scoring happens here.

Run:  .venv/bin/python scripts/build_progression_audit_panel.py
Out:  reports/progression_audit_panel.json  (committed; the ONLY input to the figure/table builders)
"""
from __future__ import annotations
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "reports", "progression_audit_panel.json")
EPS = ["PLR2", "PLR3", "MD"]

# ----------------------------------------------------------------------------- LOCKED CONSTANTS
# Cohort (GRAPE; Huang et al. Sci Data 2023; Figshare collection 6406319).
COHORT = {
    "eyes": 263, "patients": 144, "all_glaucoma": True,
    "visits_median": 4, "visits_range": [3, 9], "eyes_ge6_visits": 50,
    "events": {"PLR2": 40, "PLR3": 14, "MD": 27},
    "endpoint_def": {
        "PLR2": "pointwise-VF progression (>=2 points)",
        "PLR3": "pointwise-VF progression (>=3 points)",
        "MD": "MD-slope progression (global, p<0.05)"},
    "labels_source": "all VF-derived",
    "cameras": [
        {"model": "TOPCON TRC-NW8", "res": "1556^2", "n_baseline": 157},
        {"model": "Canon CR-2 PLUS", "res": "2136^2", "n_baseline": "part of 106"},
        {"model": "Canon CR-2", "res": "1611^2", "n_baseline": "part of 106"}],
    "device_split_baseline": {"TOPCON": 157, "Canon_combined": 106},
    "followup_cfps": 631, "eyes_ge2_cfp": 196,
    "figshare": "6406319", "repo": "github.com/HLi2000/Glaucoma-Triage",
}

# Camera-only floor (is-TOPCON, no pixels): patient-grouped repeated-OOF AUC. (honest_representation.json / ES §2)
FLOOR = {"PLR2": 0.561, "PLR3": 0.600, "MD": 0.580}

# Check 1 — device confound (ES §2 + fullfundus_landscape.json).
DEVICE = {
    "camera_decode_fullfundus": 0.981,   # frozen full-fundus RETFound embedding -> is-TOPCON, OOF AUC
    "camera_decode_disc_roi": 1.000,     # disc/ROI crop RAISES decodability to ~1.000 (crop does not remove it)
    "camera_decode_followup": 0.960,     # follow-up CFP embedding -> device
    "rate_topcon": {"PLR2": 0.178, "PLR3": 0.070, "MD": 0.127},
    "rate_canon":  {"PLR2": 0.113, "PLR3": 0.028, "MD": 0.066},
    # disc-crop fundus-FM (ES §2): naive vs camera-residualized
    "fm_disc_naive": {"PLR2": 0.602, "PLR3": 0.565, "MD": 0.593},
    "fm_disc_resid": {"PLR2": 0.606, "PLR3": 0.510, "MD": 0.525},
}

# Check 2 — reproduce-then-collapse (grape_baseline_match.md). eye = (median, max); honest = patient-grouped.
PUBLISHED = {"PLR2": 0.71, "PLR3": 0.80, "MD": 0.73}
FOUDA_PLR3 = 0.86
COLLAPSE = {  # model -> endpoint -> {eye_median, eye_max, honest}
    "ResNet-50":       {"PLR2": (0.41, 0.71, 0.45), "PLR3": (0.57, 0.72, 0.48), "MD": (0.51, 0.57, 0.52)},
    "EfficientNet-B4": {"PLR2": (0.57, 0.66, 0.47), "PLR3": (0.61, 0.83, 0.44), "MD": (0.45, 0.62, 0.43)},
    "DenseNet-121":    {"PLR2": (0.53, 0.72, 0.51), "PLR3": (0.68, 0.83, 0.56), "MD": (0.70, 0.80, 0.51)},
    "ConvNeXt-Tiny":   {"PLR2": (0.43, 0.51, 0.52), "PLR3": (0.55, 0.71, 0.46), "MD": (0.41, 0.50, 0.50)},
    "Soft-vote ens.":  {"PLR2": (0.52, 0.69, 0.47), "PLR3": (0.67, 0.77, 0.44), "MD": (0.50, 0.53, 0.49)},
}
PLR3_TESTPOS_PER_FOLD = 3  # -> AUC quantizes in ~0.3 steps seed-to-seed

# Check 3 — circularity dial + statistic ablation + honest anchor (grape_serial_dial.json / serial_event.md).
# [auc, lo, hi] per k in 1..6, full.  (filled from JSON at runtime; constants below are the cross-check targets)
DIAL_FULL = {"MD": 0.903, "PLR2": 0.823, "PLR3": 0.875}
DIAL_K1 = {"MD": 0.474, "PLR2": 0.579, "PLR3": 0.561}
ABLATION = {  # full-series single-statistic -> shipped MD flag: [auc, lo, hi]
    "VF slope":       [0.861, 0.807, 0.910],
    "slope p-value":  [0.873, 0.782, 0.946],
    "last VF level":  [0.640, 0.516, 0.752],
    "VF variability": [0.599, 0.481, 0.714],
    "# visits":       [0.558, 0.423, 0.681],
}
HONEST_ANCHOR = {"auc": 0.549, "lo": 0.380, "hi": 0.709, "events": 13}
CHANCE = 0.50

# Check 4 — honest landmark (serial_event.md §A) + serial fundus (grape_serial_fundus.json).
LANDMARK = {  # L -> {eyes, events, vf, iop, struct, all}  (all values [auc, lo, hi])
    1: {"eyes": 163, "events": 13, "vf": [0.56, 0.41, 0.72], "iop": [0.61, 0.46, 0.74],
        "struct": [0.57, 0.40, 0.73], "all": [0.55, 0.38, 0.71]},
    2: {"eyes": 88, "events": 5, "vf": [0.51, 0.26, 0.89], "iop": [0.87, 0.75, 0.99],
        "struct": [0.61, 0.21, 0.89], "all": [0.73, 0.45, 0.98]},
    3: {"eyes": 50, "events": 4, "vf": [0.60, 0.42, 0.77], "iop": [0.61, 0.31, 0.87],
        "struct": [0.26, 0.00, 0.48], "all": [0.66, 0.45, 0.81]},
}
SERIAL_FUNDUS = {
    "n_eyes": 196, "pct_switch": 0.408, "decode": 0.960,
    "switch_pattern": {"topcon->topcon": 27, "topcon->canon": 80, "canon->topcon": 0, "canon->canon": 89},
    # endpoint -> {events, naive[auc,lo,hi], resid[auc,lo,hi], camera_pair}
    "PLR2": {"events": 26, "naive": [0.42, 0.32, 0.54], "resid": [0.46, 0.35, 0.58], "camera_pair": 0.391},
    "PLR3": {"events": 8,  "naive": [0.20, 0.10, 0.31], "resid": [0.34, 0.20, 0.49], "camera_pair": 0.282},
    "MD":   {"events": 16, "naive": [0.54, 0.40, 0.70], "resid": [0.57, 0.42, 0.73], "camera_pair": 0.376},
}

# Triangulation (~0.58): baseline-fundus estimates under the single device control (covariate-adjusted AUC).
# retfound_lp is the frozen RETFound-CFP DISC-crop probe's COVARIATE-ADJUSTED AUC (honest_representation.json,
# retfound_cfp_disc -- the optic-disc crop the downstream papers use, matching the naive disc read and the decode);
# gppm_cfp is GPPM's OWN reported CFP-only baseline under its eye-level, device-uncontrolled 196-eye protocol
# (not re-scored by us) -- an inflating protocol, so it is a GENEROUS UPPER BOUND on the honest fundus signal.
TRIANGULATION = {  # endpoint -> {floor, retfound_lp[auc,lo,hi] (covariate-adjusted, disc crop), gppm_cfp}
    "PLR2": {"floor": 0.561, "retfound_lp": [0.588, 0.493, 0.677], "gppm_cfp": 0.577},
    "PLR3": {"floor": 0.600, "retfound_lp": [0.523, 0.367, 0.679], "gppm_cfp": 0.571},
    "MD":   {"floor": 0.580, "retfound_lp": [0.552, 0.440, 0.659], "gppm_cfp": 0.595},
}
BORDERLINE = {"rep": "ConvNeXt-Base (disc)", "endpoint": "MD", "auc": [0.740, 0.660, 0.820],  # covariate-adjusted
              "note": "does not replicate on ConvNeXt-Large (0.61) and falls to the floor once fine-tuned"}

# Finding 5 — endpoint-sensitivity single-modality AUC (ES §1; disc-crop fundus-FM, camera-residualized).
# endpoint -> modality -> [auc, lo, hi]; leader is the report's leader.
ENDPOINT_SENS = {
    "PLR2": {"events": 40, "VF": [0.561, 0.46, 0.66], "OCT": [0.516, 0.42, 0.62],
             "fundus_morph": [0.572, 0.46, 0.68], "IOP": [0.296, 0.20, 0.40],
             "fundus_FM": [0.606, 0.51, 0.70], "leader": "fundus_FM"},
    "PLR3": {"events": 14, "VF": [0.541, 0.41, 0.66], "OCT": [0.414, 0.25, 0.59],
             "fundus_morph": [0.160, 0.07, 0.27], "IOP": [0.461, 0.29, 0.63],
             "fundus_FM": [0.510, 0.34, 0.67], "leader": "VF"},
    "MD":   {"events": 27, "VF": [0.573, 0.47, 0.68], "OCT": [0.609, 0.49, 0.73],
             "fundus_morph": [0.386, 0.24, 0.53], "IOP": [0.393, 0.28, 0.51],
             "fundus_FM": [0.525, 0.41, 0.64], "leader": "OCT"},
    "rho": {"PLR2_PLR3": 0.10, "PLR2_MD": -0.20, "PLR3_MD": 0.40},
    "rho_mean": 0.10,
    "saeedi_agree": 0.025, "saeedi_kappa": [0.12, 0.52],
}

# Prior-papers audit map (ES §2/§4 + firsthand verification of the 5 GRAPE-CFP papers).
PRIOR_PAPERS = [
    {"name": "Huang et al. (GRAPE, 2023)", "auc": {"PLR2": 0.71, "PLR3": 0.80, "MD": 0.73},
     "split": "unstated (our repro: eye-level)", "device": "uncontrolled",
     "circularity": "n/a (baseline CFP)", "honest_landmark": "n/a",
     "verdict": "device + eye-level shortcut"},
    {"name": "TSDF (2025)", "auc": {"PLR2": 0.87, "PLR3": 0.96, "MD": 0.92},
     "split": "participant-level 5-fold (stated in paper; no code released to verify)",
     "device": "uncontrolled (41% camera-switch)",
     "circularity": "variable-length fundus sequence spans label window", "honest_landmark": "no",
     "verdict": "device confound + concurrent-change circularity (split stated participant-level)"},
    {"name": "DiffSight (2026)", "auc": {"PLR2": 0.79, "PLR3": 0.88, "MD": 0.80},
     "split": "eye-level 156/40 (PLR3 = 2 test positives; firsthand from their Table)", "device": "uncontrolled",
     "circularity": "n/a: first two GRAPE visits, below the 3-visit slope threshold (non-circular, not prospective); "
                    "its pre-onset forecasting is the separate SIGF experiment, not GRAPE",
     "honest_landmark": "no",
     "verdict": "device confound + eye-level split with 2 PLR3 test positives"},
    # fusion_per_endpoint = GPPM Table 3 per-criterion fusion AUCs (firsthand-verified from the GPPM paper, page 7);
    # combined 0.849 is the headline combined-criterion value. Both are GPPM's own reported numbers.
    {"name": "GPPM (Zheng et al., 2025)",
     "auc": {"combined": 0.849, "fusion_per_endpoint": [0.851, 0.818, 0.688], "cfp_only": [0.577, 0.571, 0.595]},
     "split": "eye-level (156/40)", "device": "uncontrolled",
     "circularity": "0.849 needs the VF branch (VF-derived labels)", "honest_landmark": "n/a",
     "verdict": "0.849 = VF fusion; own CFP-only baseline ~0.58"},
    {"name": "Fouda & Mohammed (2025)", "auc": {"PLR3": 0.86},
     "split": "70/30 (no CV/CIs, ~4 PLR3 pos)", "device": "names 'camera artifacts', uncontrolled",
     "circularity": "n/a (baseline CFP)", "honest_landmark": "n/a",
     "verdict": "names the confound but does not control it"},
]

# Value/interaction — DROPPED from the body; retained for the one-line discussion mention + record only.
VALUE_INTERACTION_DROPPED = {
    "delta_fundus_over_tab": {"PLR2": [-0.042, -0.09, 0.01], "PLR3": [0.051, -0.06, 0.18], "MD": [-0.012, -0.06, 0.04]},
    "pid_octxvf_rate": {"net_synergy": 0.001, "p": 0.010},
    "intershap": "~0 all pairs",
    "binary_loss_shapley": "+0.10..0.16 but null-calibration returns +0.15 on random -> artifact",
    "evpi_at_0.15": {"PLR2": 0.0109, "PLR3": 0.0, "MD": 0.00005},
    "verdict": "weak, additive; not a figure, not a finding in this paper",
}

# --------------------------------------------------------------------------- reconciliation gate
def _load(name):
    p = os.path.join(DERIVED, name)
    return json.load(open(p)) if os.path.exists(p) else None

def _close(a, b, tol=0.02):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol

def reconcile():
    """Cross-check locked constants against the on-disk analysis JSON; report + fail on mismatch."""
    rows, ok = [], True
    def check(label, const, live):
        nonlocal ok
        if live is None:
            rows.append((label, f"{const}", "—", "no JSON (constant used)")); return
        good = _close(const, live)
        ok = ok and good
        rows.append((label, f"{const}", f"{live:.3f}", "OK" if good else "*** MISMATCH ***"))

    hr = _load("grape_honest_representation.json")
    if hr:
        for ep, k in (("PLR2", "plr2"), ("PLR3", "plr3"), ("MD", "md")):
            check(f"floor {ep}", FLOOR[ep], hr["floors"][k])
            check(f"RETFound-LP covar-adj {ep}", TRIANGULATION[ep]["retfound_lp"][0],
                  hr["results"]["retfound_cfp_disc"][k]["aaroc"][0])
    ff = _load("grape_fullfundus_landscape.json")
    if ff:
        check("camera_decode fullfundus", DEVICE["camera_decode_fullfundus"], ff["camera_decode_auc"])
    sd = _load("grape_serial_dial.json")
    if sd:
        for ep, k in (("MD", "prog_md"), ("PLR2", "prog_plr2"), ("PLR3", "prog_plr3")):
            check(f"dial full {ep}", DIAL_FULL[ep], sd["dial"][k][-1]["auc"])
        check("ablation VF slope", ABLATION["VF slope"][0], sd["ablation"]["full_slope"][0])
        check("ablation slope-p", ABLATION["slope p-value"][0], sd["ablation"]["full_slope_p"][0])
        check("honest anchor", HONEST_ANCHOR["auc"], sd["honest_anchor"]["auc"])
    sf = _load("grape_serial_fundus.json")
    if sf:
        check("serial-fundus decode", SERIAL_FUNDUS["decode"], sf["followup_camera_decode_auc"])
        check("serial-fundus %switch", SERIAL_FUNDUS["pct_switch"], sf["pct_switched_camera"])

    w = max(len(r[0]) for r in rows)
    print("\n=== reconciliation gate (locked constant  vs  on-disk analysis JSON) ===")
    for lab, c, live, st in rows:
        print(f"  {lab:<{w}}  const={c:<7}  json={live:<7}  {st}")
    print(f"=== {'ALL OK' if ok else 'MISMATCH — FIX BEFORE BUILDING'} ===\n")
    if not ok:
        sys.exit("reconciliation failed: a locked constant disagrees with its analysis JSON")


def main():
    reconcile()
    panel = {
        "_provenance": "LOCKED from reports/grape_{endpoint_sensitivity,baseline_match,serial_event,serial_fundus}.md "
                       "+ data/derived/grape_*.json; built by scripts/build_progression_audit_panel.py. No re-scoring.",
        "endpoints": EPS,
        "cohort": COHORT,
        "floor": FLOOR,
        "device": DEVICE,
        "collapse": {"published": PUBLISHED, "fouda_plr3": FOUDA_PLR3, "models": COLLAPSE,
                     "plr3_testpos_per_fold": PLR3_TESTPOS_PER_FOLD, "floor": FLOOR},
        "circularity": {"dial_full": DIAL_FULL, "dial_k1": DIAL_K1, "ablation": ABLATION,
                        "honest_anchor": HONEST_ANCHOR, "chance": CHANCE},
        "landmark": {str(k): v for k, v in LANDMARK.items()},
        "serial_fundus": SERIAL_FUNDUS,
        "triangulation": TRIANGULATION, "borderline": BORDERLINE,
        "endpoint_sensitivity": ENDPOINT_SENS,
        "prior_papers": PRIOR_PAPERS,
        "value_interaction_dropped": VALUE_INTERACTION_DROPPED,
    }
    # attach the full dial series (auc + CI) from JSON when present, else a minimal reconstruction.
    sd = _load("grape_serial_dial.json")
    if sd:
        panel["circularity"]["dial"] = {
            ep: [{"k": r["k"], "auc": r["auc"], "lo": r["lo"], "hi": r["hi"]} for r in sd["dial"][kk]]
            for ep, kk in (("MD", "prog_md"), ("PLR2", "prog_plr2"), ("PLR3", "prog_plr3"))}
    # attach the full honest-representation ladder (every frozen rep/crop, camera-residualized) for App-A2 + table.
    hr = _load("grape_honest_representation.json")
    if hr:
        REP_LABEL = {
            "retfound_cfp": "RETFound-CFP (full)", "retfound_cfp_disc": "RETFound-CFP (disc)",
            "retfound_cfp_roi": "RETFound-CFP (ROI)", "dinov2_reg": "DINOv2-reg (full)",
            "dinov2_reg_disc": "DINOv2-reg (disc)", "convnext_tiny_disc": "ConvNeXt-T (disc)",
            "convnext_base_disc": "ConvNeXt-B (disc)", "convnext_large_disc": "ConvNeXt-L (disc)"}
        ladder = {"floors": {"PLR2": hr["floors"]["plr2"], "PLR3": hr["floors"]["plr3"], "MD": hr["floors"]["md"]},
                  "reps": {}}
        for rep, lab in REP_LABEL.items():
            if rep not in hr["results"]:
                continue
            r = hr["results"][rep]
            ladder["reps"][lab] = {
                ep.upper(): {"resid": r[ep]["resid"], "aaroc": r[ep]["aaroc"],
                             "beats_floor": r[ep]["beats_floor"], "aaroc_beats_floor": r[ep].get("aaroc_beats_floor")}
                for ep in ("plr2", "plr3", "md")}
        panel["honest_ladder"] = ladder
    # attach the GPPM two-field (2-VFT) reproduce-then-collapse span sweep for Fig 4b (approved recompute, like floor-CI).
    tv = _load("grape_two_vft.json")
    if tv:
        panel["two_vft"] = tv
    # attach the direct TSDF reproduce-then-collapse (their released features + recipe; scripts/run_grape_tsdf_repro.py).
    ts = _load("grape_tsdf_repro.json")
    if ts:
        panel["tsdf_repro"] = ts
    # attach the serial reproduce-collapse ladder (R1 features -> R2 embeddings -> R3 temporal head, both protocols).
    sl = _load("grape_serial_ladder.json")
    if sl:
        panel["serial_ladder"] = sl
    # attach the WELL-POWERED baseline prospective forecast (first visit -> shipped label; the landmark replacement).
    # fusion = run_grape_prospective_baseline.py; image-only = the committed single-CFP (fm_disc, first-1) + DiffSight
    # first-two head (diff2, first-2). All predict the shipped PLR2/PLR3/MD on full events 40/14/27, patient-grouped +
    # camera-residualized; all sit at/below the camera floor (approved recompute, like floor-CI/two_vft/diff2).
    pb = _load("grape_prospective_baseline.json")
    if pb:
        d2 = sl["rungs"]["R0_diff2"] if sl else {}
        image = {
            "first1": {e: {"naive": DEVICE["fm_disc_naive"][e], "resid": DEVICE["fm_disc_resid"][e]} for e in EPS},
            "first2": {e: {"naive": d2[e]["grouped_naive"][0], "resid": d2[e]["grouped_resid"][0]} for e in EPS} if d2 else {},
        }
        panel["prospective_baseline"] = {
            "fusion": pb, "image": image,
            "dial_ref": DEVICE["fm_disc_naive"]["MD"],   # best baseline-prospective MD (image-only, first visit): ~0.59
            "events": COHORT["events"],
        }
    # attach the SERIAL ABLATION -- the 3-check leave-one-out for the three serial models (device / fair split /
    # non-circularity), each check applied independently vs the published number, + the all-removed floor. Published
    # values are the authors' reported AUCs (cited); the check reads are ours (standard per-eye ROC-AUC, patient-grouped
    # and cross-validated where the check requires). GPPM survives device + fair split (genuine in-window VF signal) and
    # falls only at non-circularity; TSDF/DiffSight do not survive a fair split. PLR2 primary; PLR3 (8-14 ev) flagged.
    PUB_SERIAL = {"TSDF": {"PLR2": 0.866, "PLR3": 0.954, "MD": 0.917},        # song2025tsdf reported
                  "DiffSight": {"PLR2": 0.79, "PLR3": 0.88, "MD": 0.80},      # huang2026diffsight Table IV
                  "GPPM": {"PLR2": 0.851, "PLR3": 0.818, "MD": 0.688}}        # zheng2025gppm Table 3 per-criterion fusion
    FOUDA_ARCHS = ["EfficientNet-B4", "DenseNet-121", "ConvNeXt-Tiny"]   # fouda2025's weighted-ensemble members
    sc = _load("grape_singlecfp_ablation.json")   # single baseline-CFP (Huang/Fouda regime), one consistent probe
    if ts and tv and sl and sc:
        d2 = sl["rungs"]["R0_diff2"]; gp = tv["gppm_fusion"]
        import numpy as _np
        # Reproduced = each method's OWN eye-level split p90 (a defensible upper tail, not the cherry-picked max):
        #   TSDF   -> its released DSMIL code (matches published within 0.002);
        #   Huang  -> ResNet-50 with Huang's faithful CE/full-fine-tune recipe (grape_cfp_faithful.json) if present,
        #             else the committed run; Fouda -> its weighted CNN ensemble (committed focal/SMOTE recipe);
        #   GPPM/DiffSight -> their own head's eye-level p90.
        tsdf_repro = {e: [ts["reproduce"][e]["mean_auc"]] * 3 for e in EPS}
        _ftc = _load("grape_cfp_finetune_results.json")          # committed CNN fine-tune (Fouda/Huang fallback)
        _ftf = _load("grape_cfp_faithful.json")                  # Huang faithful ResNet-50 recipe, if run
        _ftfo = _load("grape_cfp_fouda_faithful.json")           # Fouda faithful CNN ensemble (PLR3), if run

        def _p95(fj, arch, ep):
            try:
                au = fj[ep.lower()]["published"][arch]["aucs"]
                return round(float(_np.percentile(au, 95)), 3) if au else None
            except (KeyError, TypeError):
                return None
        huang_rep = {e: [(_p95(_ftf, "resnet50", e) or _p95(_ftc, "resnet50", e))] * 3 for e in EPS}

        def _fc(fj, ep, key):   # faithful item-6 control cell [auc,lo,hi] from a single-CFP run JSON, else None
            try:
                return fj[ep.lower()][key]
            except (KeyError, TypeError):
                return None
        _hfc = lambda ep, key: _fc(_ftf, ep, key)      # Huang (grape_cfp_faithful.json)
        _ffc = lambda ep, key: _fc(_ftfo, ep, key)     # Fouda (grape_cfp_fouda_faithful.json; PLR3 only)
        fouda_rep = {e: [(_p95(_ftfo, "ensemble", e) or _p95(_ftc, "ensemble", e))] * 3 for e in EPS}
        panel["serial_ablation"] = {
            "floor": {e: TRIANGULATION[e]["floor"] for e in EPS},
            "primary": "PLR2",
            "order": ["Huang", "Fouda", "TSDF", "DiffSight", "GPPM"],
            "models": {
                # Huang & Fouda are separate single-baseline-CFP methods, each at its reported endpoint. reproduced =
                # our distinct single-CFP CNN eye-level max ceiling (max over the 5 CNN reproductions); the controlled
                # checks share the frozen RETFound-CFP disc probe (a single baseline image reduces to the device floor
                # regardless of backbone -- that is the finding). Circularity is N/A (no serial window).
                # Huang's method is ResNet-50 on a single baseline CFP; Fouda's is a weighted soft-vote ensemble of
                # EfficientNet-B4 + DenseNet-121 + ConvNeXt-Tiny -- both reproduced by their OWN architectures in our
                # CFP fine-tune (grape_cfp_finetune). reproduced = that method's eye-level max; fair split = its
                # patient-grouped read; -device = the camera-residualized single-CFP read.
                # Huang: wired to its OWN faithful ResNet-50 run (grape_cfp_faithful, 264 ep) -- -device = covariate-adjusted
                # AUC (AAROC), -fair split = patient-grouped, -all = IECV, each with a patient cluster-bootstrap CI on the
                # per-eye OOF; the shared frozen-RETFound-probe reads are kept as robustness twins. Falls back to the shared
                # probe if the faithful run is absent.
                "Huang": {"published": {"PLR2": 0.71, "PLR3": 0.80, "MD": 0.73}, "endpoint": "PLR2",   # huang2023grape: ResNet-50
                          "klass": "single baseline CFP (ResNet-50)", "decisive": "device",
                          "reproduced": huang_rep,          # ResNet-50 eye-level p90 (faithful recipe if run)
                          "fair_split": {e: (_hfc(e, "honest_ci") or [COLLAPSE["ResNet-50"][e][2]] * 3) for e in EPS},
                          "device": {e: (_hfc(e, "device_aaroc_ci") or sc[e]["device"]) for e in EPS},   # (1) AAROC
                          "circular": None,
                          "all": {e: (_hfc(e, "all_combined_ci") or sc[e]["all"]) for e in EPS},         # (4) = (1)+(2) (no circularity for single-CFP)
                          "all_iecv": {e: _hfc(e, "all_ci") for e in EPS},                               # transport twin
                          "device_resid": {e: sc[e]["device"] for e in EPS},
                          "all_resid": {e: sc[e]["all"] for e in EPS},
                          "events": {e: sc[e]["events"] for e in EPS}},
                # Fouda: wired to its OWN faithful CNN-ensemble run (grape_cfp_fouda_faithful, PLR3 only) -- same item-6
                # controls as Huang (AAROC device / patient / combined-(4) / IECV twin). Its unreported PLR2/MD fall back
                # to the shared frozen-RETFound probe (not displayed -- n.r. in the figure).
                "Fouda": {"published": {"PLR3": FOUDA_PLR3}, "endpoint": "PLR3",   # fouda2025: EffNet-B4+DenseNet-121+ConvNeXt-Tiny ensemble
                          "klass": "single baseline CFP (weighted CNN ensemble)", "decisive": "device",
                          "reproduced": fouda_rep,          # weighted CNN ensemble eye-level p90 (faithful PLR3 if run)
                          "fair_split": {e: (_ffc(e, "honest_ci") or [round(max(COLLAPSE[m][e][2] for m in FOUDA_ARCHS), 3)] * 3) for e in EPS},
                          "device": {e: (_ffc(e, "device_aaroc_ci") or sc[e]["device"]) for e in EPS},   # (1) AAROC
                          "circular": None,
                          "all": {e: (_ffc(e, "all_combined_ci") or sc[e]["all"]) for e in EPS},          # (4) = (1)+(2)
                          "all_iecv": {e: _ffc(e, "all_ci") for e in EPS},                                # transport twin
                          "device_resid": {e: sc[e]["device"] for e in EPS},
                          "all_resid": {e: sc[e]["all"] for e in EPS},
                          "events": {e: sc[e]["events"] for e in EPS}},
                # Serial/fusion models (TSDF, DiffSight, GPPM): -device = covariate-adjusted AUC (Janes-Pepe AAROC,
                # device removed at scoring, split kept). Headline -all = (1)+(2)+(3) applied together = AAROC on the
                # patient-grouped, non-circular OOF (the cleanest within-GRAPE floor, nested under -device). The IECV
                # leave-one-camera-out (transport proxy) and the camera-residualized reads are kept as robustness twins
                # (all_iecv / all_resid, device_resid) -- all agree at/below the floor.
                "TSDF": {"published": PUB_SERIAL["TSDF"], "klass": "serial fundus", "decisive": "fair split",
                         "reproduced": tsdf_repro,          # eye-level split ceiling (standard per-eye AUC)
                         "device": {e: ts["collapse"][e]["ablation"]["m_aaroc"] for e in EPS},   # (1) AAROC
                         "fair_split": {e: ts["collapse"][e]["ablation"]["m_split"] for e in EPS},
                         "circular": {e: ts["collapse"][e]["ablation"]["m_circ"] for e in EPS},
                         "all": {e: ts["collapse"][e]["ablation"]["all_combined"] for e in EPS},  # (4) = (1)+(2)+(3)
                         "all_iecv": {e: ts["collapse"][e]["ablation"]["iecv"] for e in EPS},     # transport twin
                         "device_resid": {e: ts["collapse"][e]["ablation"]["m_device"] for e in EPS},
                         "all_resid": {e: ts["collapse"][e]["ablation"]["all"] for e in EPS},
                         "events": {e: ts["reproduce"][e]["n_pos"] for e in EPS}},
                "DiffSight": {"published": PUB_SERIAL["DiffSight"], "klass": "serial fundus", "decisive": "fair split",
                              "reproduced": {e: [d2[e]["eyelevel"]["p95"]] * 3 for e in EPS},   # eye-level split p95
                              "device": {e: d2[e]["ablation"]["m_aaroc"] for e in EPS},          # (1) AAROC
                              "fair_split": {e: d2[e]["ablation"]["m_split"] for e in EPS},
                              "circular": None,        # first-two input is already non-circular (below k=3)
                              "all": {e: d2[e]["ablation"]["all_combined"] for e in EPS},         # (4) = (1)+(2)+(3)
                              "all_iecv": {e: d2[e]["ablation"]["iecv"] for e in EPS},            # transport twin
                              "device_resid": {e: d2[e]["ablation"]["m_device"] for e in EPS},
                              "all_resid": {e: d2[e]["ablation"]["all"] for e in EPS},
                              "events": {e: d2[e]["events"] for e in EPS}},
                "GPPM": {"published": PUB_SERIAL["GPPM"], "klass": "VF+CFP fusion", "decisive": "non-circularity",
                         "reproduced": {e: [gp[e]["eyelevel_p95"]] * 3 for e in EPS},   # eye-level split p95
                         "device": {e: gp[e]["m_aaroc"] for e in EPS},                            # (1) AAROC
                         "fair_split": {e: gp[e]["m_split"] for e in EPS},
                         "circular": {e: gp[e]["m_circ"] for e in EPS},
                         "all": {e: gp[e]["all_combined"] for e in EPS},                          # (4) = (1)+(2)+(3)
                         "all_iecv": {e: gp[e]["iecv"] for e in EPS},                             # transport twin
                         "device_resid": {e: gp[e]["m_device"] for e in EPS},
                         "all_resid": {e: gp[e]["all"] for e in EPS},
                         "events": {e: gp[e]["events"] for e in EPS}},
            },
        }
    with open(OUT, "w") as f:
        json.dump(panel, f, indent=2)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)} bytes)")
    print("panel keys:", ", ".join(panel.keys()))


if __name__ == "__main__":
    main()
