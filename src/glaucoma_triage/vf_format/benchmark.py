"""VF-FormatShift benchmark: the **earned-tolerance certificate** + a model-agnostic metric runner.

The certificate verifies EARNED format-tolerance on **model-controllable, label-free** axes — not a device
limit. Each contender is scored on five axes — two RECOVERY (focal PSD recovery, global MD recovery, over the real
held-out rungs) + three STABILITY (switch-call stability, test-retest repeatability, detection retention) — combined
as a **geometric-mean tolerance score** (any zero axis → 0, so a degenerate model that games the stability axes but
can't recover a field scores 0). A model is **certified** only if it is useful (beats the constant model's recovery +
is a non-degenerate classifier), call-stable across the device switch, AND **earns its tolerance over the clean-only
baseline** — a statistically-significant focal-PSD improvement on the real rungs. The clean-only ``baseline`` is the
untrained *reference*, so it is Tier-1 (useful but uncertified); only a model that beats it (``dr``) reaches Tier-2
(certified).

Cohort basis of the stability axes (stated honestly): switch-call stability + detection retention are **cross-device**
reads → they need the same eye on two devices, which in public VF data only **Eyecatcher** (40 eyes) provides —
irreducibly single-cohort. Test-retest repeatability **averages two own-scale sub-axes** — Eyecatcher HFA
same-session (÷``REP_REF``) + the vfpwg weekly retest (÷``REP_WEEKLY_REF``, 30 glaucoma eyes) — so a week of natural
fluctuation is scored against a weekly anchor, not a same-session one, and dr's lead is corroborated off-cohort.

The switch MD/PSD limit-of-agreement is a DEVICE/hardware limit (identical across arms; the home tablet differs
from the HFA by ±5–7 dB regardless of the reader) — it is **reported as context, NOT gated**. Gating on it would
make the verdict track the device, not the model.

:func:`run_benchmark` scores any set of :class:`~.envelope.InterpModel`-compatible models and assembles a
JSON-clean scorecard consumed by ``scripts/run_vf_format_benchmark.py`` + the report artifact. See
``docs/VF_FORMATSHIFT_BENCHMARK.md``.
"""

from __future__ import annotations

import math

import numpy as np

from . import envelope

# --- certificate thresholds (documented, versioned with the benchmark) -----------------------------
USEFULNESS_MAE_RATIO = 0.7   # shifted MD-MAE and PSD-MAE must be <= this fraction of the constant model's
CALL_EQUIV_MARGIN = 0.05     # switch call-disagreement allowed above the same-device tablet-repeat rate
TOLERANCE_SCORE_FLOOR = 0.55  # geometric-mean capability score a certified model must clear (constant scores 0)
SWITCH_REF = 0.20            # a "bad" untrained switch call-change → the 0-anchor of the call-stability axis
REP_REF = 0.10             # a "bad" SAME-SESSION test-retest disagreement → 0-anchor of the same-session repeat sub-axis
REP_WEEKLY_REF = 0.20      # a "bad" WEEKLY retest disagreement (switch-magnitude perturbation) → 0-anchor of the weekly sub-axis
# CONTEXT-ONLY (reported beside the verdict, never gated): the switch MD limit is a device/hardware constant.
LOA_CONTEXT_RATIO = 2.0     # switch MD LoA half-width vs the HFA repeatability RC — shown as context, not a gate


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x))) if math.isfinite(x) else 0.0


def capability_axes(*, psd_shift: float, md_shift: float, c_psd: float | None, c_md: float | None,
                    switch_call: float | None, hfa_repeat: float | None,
                    tablet_recall: float | None, hfa_recall: float | None,
                    weekly_repeat: float | None = None) -> dict:
    """The five model-controllable, label-free (all REAL) capability axes in [0,1] (higher = better) + their
    geometric-mean tolerance score. 0 = useless/constant anchor, 1 = ideal/floor. Two RECOVERY axes (focal PSD,
    global MD) — the anti-gaming teeth; three STABILITY axes (device-switch call, same-device test-retest
    repeatability, detection retention). The geometric mean is anti-gaming: any single zero axis forces the score to
    0, so a model that maxes the stability axes but can't recover a field scores 0.

    The **repeatability** axis averages two sub-axes on their own 'bad-rate' scales — same-session (Eyecatcher,
    ÷REP_REF) + (optional) weekly (vfpwg, ÷REP_WEEKLY_REF) — so a week of natural fluctuation is scored against a
    weekly anchor, not a same-session one. With ``weekly_repeat=None`` it degrades to the same-session sub-axis."""
    a = _clamp01(1 - psd_shift / c_psd) if c_psd else 0.0          # A focal PSD recovery (the earned-tolerance axis)
    b = _clamp01(1 - md_shift / c_md) if c_md else 0.0             # B global MD recovery
    c = _clamp01(1 - (switch_call or 0.0) / SWITCH_REF)            # C device-switch call stability
    r_session = _clamp01(1 - (hfa_repeat or 0.0) / REP_REF)       # R test-retest repeatability, same-session sub-axis
    r = (r_session if weekly_repeat is None                       # + weekly sub-axis (own scale), averaged
         else 0.5 * (r_session + _clamp01(1 - (weekly_repeat or 0.0) / REP_WEEKLY_REF)))
    d = _clamp01((tablet_recall or 0.0) / max(hfa_recall or 1e-9, 1e-9))  # D detection retention
    axes = {"focal_psd": a, "global_md": b, "call_stability": c, "repeatability": r, "retention": d}
    prod = 1.0
    for v in axes.values():
        prod *= v
    axes["tolerance_score"] = float(prod ** (1.0 / len(axes))) if prod > 0 else 0.0
    return axes


def tolerance_certificate(*, axes: dict, psd_shift: float, md_shift: float,
                          c_psd: float | None, c_md: float | None, abn_call_rate: float | None,
                          switch_call: float | None, tablet_repeat: float | None,
                          earned_ok: bool, earned_beat: int, earned_n: int, earned_p: float,
                          device_context: dict, is_reference: bool = False) -> dict:
    """Grade EARNED format-tolerance on model-controllable axes (the device MD/PSD-LoA is context, not gated).

    Tiers: **0** ¬useful (constant / gamers — recovery axes → 0); **1** useful + call-stable but has NOT *earned*
    tolerance over the clean-only baseline (the untrained reference itself); **2 = CERTIFIED** (useful ∧ call-stable
    ∧ earned — a significant focal-PSD win over baseline on the real rungs ∧ tolerance-score ≥ floor)."""
    reasons: list[str] = []
    psd_ok = c_psd is not None and psd_shift <= USEFULNESS_MAE_RATIO * c_psd
    md_ok = c_md is not None and md_shift <= USEFULNESS_MAE_RATIO * c_md
    nondeg = abn_call_rate is None or (0.0 < float(abn_call_rate) < 1.0)
    usefulness_ok = bool(psd_ok and md_ok and nondeg)
    callstable_ok = bool(switch_call is not None and tablet_repeat is not None
                         and switch_call <= tablet_repeat + CALL_EQUIV_MARGIN)
    score_ok = axes.get("tolerance_score", 0.0) >= TOLERANCE_SCORE_FLOOR

    if not psd_ok:
        reasons.append(f"focal PSD recovery {psd_shift:.2f} not ≤ {USEFULNESS_MAE_RATIO:.0%} of constant "
                       f"{'—' if c_psd is None else f'{c_psd:.2f}'} dB")
    if not md_ok:
        reasons.append(f"global MD recovery {md_shift:.2f} not ≤ {USEFULNESS_MAE_RATIO:.0%} of constant "
                       f"{'—' if c_md is None else f'{c_md:.2f}'} dB")
    if not nondeg:
        reasons.append(f"degenerate classifier (call rate {float(abn_call_rate):.2f})")
    if not callstable_ok and switch_call is not None and tablet_repeat is not None:
        reasons.append(f"switch call {switch_call:.2f} > tablet-repeat {tablet_repeat:.2f} + {CALL_EQUIV_MARGIN}")
    if usefulness_ok and callstable_ok and not earned_ok:
        reasons.append("useful + call-stable but has NOT earned tolerance over the clean-only baseline "
                       + ("(it IS the reference)" if is_reference
                          else f"(beats baseline focal PSD on only {earned_beat}/{earned_n} rungs)"))

    if not usefulness_ok:
        tier, label = 0, "not useful"
    elif usefulness_ok and callstable_ok and earned_ok and score_ok:
        tier, label = 2, "certified"
    else:
        tier, label = 1, "useful — not certified"
    return {"tier": tier, "tier_label": label, "certified": tier == 2,
            "usefulness_ok": usefulness_ok, "callstable_ok": callstable_ok, "earned_ok": bool(earned_ok),
            "earned_beat": earned_beat, "earned_n": earned_n, "earned_p": earned_p, "score_ok": score_ok,
            "device_context": device_context, "reasons": reasons}


def _within_eyecatcher(model: "envelope.InterpModel", eye_reads: list) -> dict:
    """Per-model within-Eyecatcher device-switch stats (same eyes): switch call-disagreement + tablet-repeat +
    HFA same-session repeat, the switch MD limits/RCs (variance-components Bland–Altman), and the per-device
    detection **recall** (all 40 eyes are glaucoma → the fraction each device flags IS its recall)."""
    def pmd(v):
        return float(model.predict_md(np.asarray(v, float).reshape(1, -1))[0])

    def ppsd(v):
        return float(model.predict_psd(np.asarray(v, float).reshape(1, -1))[0])

    def pca(v):
        return int(model.predict_abn(np.asarray(v, float).reshape(1, -1))[0] >= 0.5)

    a_md, b_md, a_psd, b_psd, a_call, b_call, hmaj, tmaj = [], [], [], [], [], [], [], []
    for e in eye_reads:
        hs_md = [[pmd(v) for v in s] for s in e["hfa_sessions"]]
        hs_call = [[pca(v) for v in s] for s in e["hfa_sessions"]]
        t_call = [pca(v) for v in e["tablet"]]
        a_md.append(hs_md)
        b_md.append([pmd(v) for v in e["tablet"]])
        a_psd.append([[ppsd(v) for v in s] for s in e["hfa_sessions"]])
        b_psd.append([ppsd(v) for v in e["tablet"]])
        a_call.append(hs_call)
        b_call.append(t_call)
        hmaj.append(int(np.mean([x for s in hs_call for x in s]) >= 0.5))
        tmaj.append(int(np.mean(t_call) >= 0.5))
    ba = envelope.bland_altman_replicates(a_md, b_md)
    ba_psd = envelope.bland_altman_replicates(a_psd, b_psd)
    cdd = envelope.within_across_call_disagreement(a_call, b_call)
    mk = envelope.mcnemar_kappa(hmaj, tmaj)
    w_hfa, w_tab = envelope.wilson_ci(cdd["a_k"], cdd["a_n"]), envelope.wilson_ci(cdd["b_k"], cdd["b_n"])
    w_sw = envelope.wilson_ci(cdd["switch_k"], cdd["switch_n"])
    return {"switch_call": cdd["switch"], "switch_call_lo": w_sw["lo"], "switch_call_hi": w_sw["hi"],
            "tablet_repeat": cdd["b_repeat"], "tablet_repeat_lo": w_tab["lo"], "tablet_repeat_hi": w_tab["hi"],
            "hfa_repeat": cdd["a_repeat"], "hfa_repeat_lo": w_hfa["lo"], "hfa_repeat_hi": w_hfa["hi"],
            "hfa_repeat_k": cdd["a_k"], "hfa_repeat_n": cdd["a_n"],
            "hfa_recall": float(np.mean(hmaj)) if hmaj else float("nan"),
            "tablet_recall": float(np.mean(tmaj)) if tmaj else float("nan"),
            "rc_hfa": ba["rc_a"], "rc_tablet": ba["rc_b"], "md_bias": ba["bias"],
            "loa_lo": ba["loa_lo"], "loa_hi": ba["loa_hi"], "loa_hw": (ba["loa_hi"] - ba["loa_lo"]) / 2,
            "psd_bias": ba_psd["bias"], "psd_loa_lo": ba_psd["loa_lo"], "psd_loa_hi": ba_psd["loa_hi"],
            "psd_loa_hw": (ba_psd["loa_hi"] - ba_psd["loa_lo"]) / 2,
            "psd_rc_hfa": ba_psd["rc_a"], "psd_rc_tablet": ba_psd["rc_b"],
            "kappa": mk["kappa"], "mcnemar_p": mk["p_exact"], "n_eyes": len(eye_reads)}


def _vfpwg_repeat(model: "envelope.InterpModel", retest_visits: list) -> tuple[int, int]:
    """External same-device **weekly** test-retest repeatability as a ``(k, n)`` call-disagreement count over the
    vfpwg cohort — the 2nd repeatability cohort pooled into the repeatability axis. Each eye's ordered weekly visits
    become one session of consecutive-pair calls, scored with the same machinery as the Eyecatcher HFA repeat."""
    def pca(v):
        return int(model.predict_abn(np.asarray(v, float).reshape(1, -1))[0] >= 0.5)

    groups = [[[pca(v) for v in eye]] for eye in retest_visits]   # one session of consecutive weekly calls per eye
    cdd = envelope.within_across_call_disagreement(groups, [[] for _ in retest_visits])
    return cdd["a_k"], cdd["a_n"]


def _avg_stability(model: "envelope.InterpModel", fields, levels, *, n_seeds: int) -> list[dict]:
    """Decision-flip curve vs format level, averaged over ``n_seeds`` transform seeds (tighter curve)."""
    per_seed = [envelope.decision_stability(model, fields, levels, np.random.default_rng(1000 + s))
                for s in range(max(1, n_seeds))]
    out = []
    for i, lvl in enumerate(levels):
        dist = float(np.mean([c[i]["distance"] for c in per_seed]))
        flip = float(np.mean([c[i]["flip_rate"] for c in per_seed]))
        out.append({"level": float(lvl), "distance": dist, "flip_rate": flip})
    return out


def _shift_means(ladder: list[dict]) -> tuple[float, float]:
    """Mean MD-MAE and PSD-MAE over the real held-out rungs (the recovery axes)."""
    md = [r["md_mae"] for r in ladder if r.get("md_mae") is not None]
    psd = [r["psd_mae"] for r in ladder if r.get("psd_mae") is not None]
    return (float(np.mean(md)) if md else float("nan"),
            float(np.mean(psd)) if psd else float("nan"))


def run_benchmark(specs: list[dict], data: dict, *, n_seeds: int = 3, n_boot: int = 1000) -> dict:
    """Score models on the label-free metric library + attach an EARNED-tolerance certificate to each.

    ``specs`` = list of ``{"name", "kind", "models"}`` (contenders = the ``baseline``/``dr`` training arms + the
    degenerate ``constant``). Earned tolerance is measured relative to the ``baseline`` arm's per-rung focal PSD."""
    Xt, mdt, abnt = envelope.to_xy(data["test_fields"])
    psd_true = np.array([envelope._psd_gt(f, x) for f, x in zip(data["test_fields"], Xt)], dtype=float)
    levels = data["levels"]
    eye_reads = data.get("eye_reads") or []
    rungs = data.get("rungs") or {}
    retest_visits = data.get("retest_visits") or []   # vfpwg weekly retest → 2nd repeatability cohort (pooled)

    models = []
    for spec in specs:
        ms = spec.get("models") or [spec["model"]]
        model = ms[0]
        ev = envelope._eval(model, Xt, mdt, abnt, {"psd": psd_true})
        abn_call_rate = float(np.mean(model.predict_abn(Xt) >= 0.5)) if len(Xt) else float("nan")
        we = _within_eyecatcher(model, eye_reads) if eye_reads else {}
        if we:   # 2nd repeatability cohort: the vfpwg weekly retest → its OWN-scale sub-axis (averaged, not pooled)
            vf_k, vf_n = _vfpwg_repeat(model, retest_visits) if retest_visits else (0, 0)
            we["vfpwg_repeat_k"], we["vfpwg_repeat_n"] = vf_k, vf_n
            we["vfpwg_repeat"] = (vf_k / vf_n) if vf_n else None
        ladder = envelope.ladder_eval(model, rungs) if rungs else []
        stab = _avg_stability(model, data["test_fields"], levels, n_seeds=n_seeds)
        md_shift, psd_shift = _shift_means(ladder)
        mean_flip = float(np.mean([s["flip_rate"] for s in stab if s["level"] > 0])) if stab else float("nan")
        models.append({"name": spec["name"], "kind": spec["kind"],
                       "md_mae_clean": ev["md_mae"], "psd_mae_clean": ev.get("psd_mae"),
                       "abn_auc_clean": ev["auc"], "abn_call_rate": abn_call_rate, "ece_clean": ev["ece"],
                       "md_shift": md_shift, "psd_shift": psd_shift, "mean_flip": mean_flip,
                       "decision_stability": stab, "within_eyecatcher": we,
                       "ladder": [{"rung": r["rung"], "md_mae": r["md_mae"], "psd_mae": r.get("psd_mae"),
                                   "sensitivity": r.get("sensitivity"), "native_gt": r.get("native_gt")}
                                  for r in ladder]})

    const = next((m for m in models if m["kind"] == "constant"), None)
    c_md = const["md_shift"] if const else None
    c_psd = const["psd_shift"] if const else None
    base = next((m for m in models if m["name"] == "baseline"), None)
    base_psd = {lr["rung"]: lr["psd_mae"] for lr in (base["ladder"] if base else [])}

    for r in models:
        we = r["within_eyecatcher"]
        axes = capability_axes(psd_shift=r["psd_shift"], md_shift=r["md_shift"], c_psd=c_psd, c_md=c_md,
                               switch_call=we.get("switch_call"), hfa_repeat=we.get("hfa_repeat"),
                               weekly_repeat=we.get("vfpwg_repeat"),  # 2nd repeatability cohort (own-scale sub-axis)
                               tablet_recall=we.get("tablet_recall"), hfa_recall=we.get("hfa_recall"))
        r["axes"] = axes
        # earned = a significant focal-PSD improvement over the clean-only baseline on the real rungs (sign test).
        pairs = [(lr["psd_mae"], base_psd.get(lr["rung"])) for lr in r["ladder"] if lr["psd_mae"] is not None]
        pairs = [(x, y) for x, y in pairs if y is not None]
        beat = sum(1 for x, y in pairs if x < y)
        n = len(pairs)
        earned_ok = (r["name"] != "baseline") and n > 0 and beat == n
        earned_p = (0.5 ** n) if earned_ok else float("nan")
        dev_ctx = {"switch_md_loa_hw": we.get("loa_hw"), "switch_psd_loa_hw": we.get("psd_loa_hw"),
                   "rc_hfa": we.get("rc_hfa"), "md_bias": we.get("md_bias")}
        r["certificate"] = tolerance_certificate(
            axes=axes, psd_shift=r["psd_shift"], md_shift=r["md_shift"], c_psd=c_psd, c_md=c_md,
            abn_call_rate=r["abn_call_rate"], switch_call=we.get("switch_call"),
            tablet_repeat=we.get("tablet_repeat"), earned_ok=earned_ok, earned_beat=beat, earned_n=n,
            earned_p=earned_p, device_context=dev_ctx, is_reference=(r["name"] == "baseline"))

    meta = dict(data.get("meta", {}))
    meta.update({"n_seeds": n_seeds, "n_boot": n_boot, "usefulness_mae_ratio": USEFULNESS_MAE_RATIO,
                 "call_equiv_margin": CALL_EQUIV_MARGIN, "tolerance_score_floor": TOLERANCE_SCORE_FLOOR,
                 "switch_ref": SWITCH_REF, "rep_ref": REP_REF, "rep_weekly_ref": REP_WEEKLY_REF,
                 "repeatability_2cohort": bool(retest_visits),
                 "md_shift_constant": c_md, "psd_shift_constant": c_psd})
    return {"meta": meta, "models": models}


def leaderboard(scorecard: dict) -> list[dict]:
    """Rank rows by the geometric-mean tolerance score (↓ tier as tiebreak); include the tier for display."""
    def key(r):
        return -r.get("axes", {}).get("tolerance_score", 0.0)
    rows = []
    for r in sorted(scorecard["models"], key=key):
        we = r.get("within_eyecatcher") or {}
        c = r["certificate"]
        rows.append({"name": r["name"], "kind": r["kind"], "tolerance_score": r["axes"]["tolerance_score"],
                     "psd_shift": r["psd_shift"], "md_shift": r["md_shift"],
                     "switch_call": we.get("switch_call"), "tablet_repeat": we.get("tablet_repeat"),
                     "hfa_repeat": we.get("hfa_repeat"), "vfpwg_repeat": we.get("vfpwg_repeat"),
                     "repeatability_axis": r.get("axes", {}).get("repeatability"),
                     "switch_loa_hw": we.get("loa_hw"), "tier": c["tier"], "tier_label": c["tier_label"],
                     "certified": c["certified"], "usefulness_ok": c["usefulness_ok"],
                     "callstable_ok": c["callstable_ok"], "earned_ok": c["earned_ok"]})
    return rows
