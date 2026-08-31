"""The go/no-go incremental-value probe over a small modality spec, with bootstrap CIs.

Reuses the SAGE-proper coalition value function (:mod:`glaucoma_triage.models.fusion`). Reports, for a
progression label:
* per-subset descriptive **AUC** (each single modality and the full set),
* SAGE loss-reduction **coalition values** `v(S)` and per-modality **Shapley** values,
* the **incremental value** of each modality given the other(s) `v(N) − v(N∖{i})`,
* the pairwise **interaction** `I[a,b]` (redundancy < 0 / synergy > 0) for a 2-modality spec,
each with a patient/eye **bootstrap** CI.

This tests the crux the progression paper depends on: is there non-redundant incremental value between
OCT and VF for future progression, or is it null? (Cf. PMC9560647 finding OCT-given-VF ≈ null.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..interaction.shapley import interaction_matrix, shapley_values
from ..models.fusion import ModalitySpec, coalition_values, subset_auc


@dataclass
class ProbeResult:
    label: str
    n: int
    n_pos: int
    prevalence: float
    aucs: dict[str, float]                 # "vf","oct","oct+vf"  (point)
    coalition: dict[str, float]            # v(S) by "+"-joined subset
    shapley: dict[str, float]
    incremental: dict[str, float]          # loss-based: "vf|oct","oct|vf" = v(N)-v(N\{i})
    interaction: float | None              # I[a,b] for 2-modality spec
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)  # metric -> (lo,hi)


def _subset_key(s) -> str:
    return "+".join(sorted(s)) if s else "∅"


def _point(df: pd.DataFrame, spec: ModalitySpec, y: pd.Series, k: int, seed: int, n_repeats: int,
           est=None):
    players = list(spec)
    table = coalition_values(df, spec, y, k=k, seed=seed, n_repeats=n_repeats, estimator_factory=est)
    full = frozenset(players)
    incremental = {
        f"{i}|{'+'.join(p for p in players if p != i)}": table[full] - table[full - {i}]
        for i in players
    }
    inter = None
    if len(players) == 2:
        inter = interaction_matrix(table, players)[tuple(sorted(players))]
    aucs = {_subset_key([p]): subset_auc(df, spec, [p], y, k=k, seed=seed, n_repeats=n_repeats,
                                         estimator_factory=est) for p in players}
    aucs[_subset_key(players)] = subset_auc(df, spec, players, y, k=k, seed=seed,
                                            n_repeats=n_repeats, estimator_factory=est)
    return table, aucs, shapley_values(table, players), incremental, inter


def modality_probe(
    df: pd.DataFrame, spec: ModalitySpec, y: pd.Series, *, label: str = "progression",
    k: int = 5, seed: int = 0, n_repeats: int = 3, n_boot: int = 200, estimator_factory=None,
) -> ProbeResult:
    """Point estimates + eye-level bootstrap CIs (on the full AUC, incremental values, interaction).

    ``estimator_factory`` overrides the model (e.g. ``fusion.regularized_estimator`` for low-event runs)."""
    y = pd.to_numeric(y, errors="coerce")
    keep = y.notna().to_numpy()
    sub = df.loc[keep].reset_index(drop=True)
    yv = y[keep].reset_index(drop=True)

    table, aucs, shap, incr, inter = _point(sub, spec, yv, k, seed, n_repeats, estimator_factory)
    players = list(spec)
    full_key = _subset_key(players)

    boot: dict[str, list[float]] = {f"auc[{full_key}]": []}
    boot.update({f"incr[{key}]": [] for key in incr})
    if inter is not None:
        boot["interaction"] = []
    rng = np.random.default_rng(seed)
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = sub.iloc[idx].reset_index(drop=True).copy()
        bs["patient_id"] = [f"b{j}" for j in range(n)]  # fresh ids so each draw is its own group
        yb = yv.iloc[idx].reset_index(drop=True)
        try:
            tb, ab, _, ib, itb = _point(bs, spec, yb, k, seed, 1, estimator_factory)
        except Exception:
            continue
        if ab[full_key] == ab[full_key]:
            boot[f"auc[{full_key}]"].append(ab[full_key])
        for key, val in ib.items():
            if val == val:
                boot[f"incr[{key}]"].append(val)
        if inter is not None and itb == itb:
            boot["interaction"].append(itb)

    def _ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) >= 2 \
            else (float("nan"), float("nan"))

    ci = {m: _ci(v) for m, v in boot.items()}
    return ProbeResult(
        label=label, n=int(n), n_pos=int((yv == 1).sum()), prevalence=float((yv == 1).mean()),
        aucs=aucs, coalition={_subset_key(s): v for s, v in table.items()},
        shapley=shap, incremental=incr, interaction=inter, ci=ci,
    )


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:+.3f}"


def render_markdown(results: list[ProbeResult], *, oct_present: bool) -> str:
    out = ["# Progression go/no-go probe — Harvard-GDP1000", ""]
    out.append(f"Stage: **{'1 (OCT + VF)' if oct_present else '0 (VF-only, no OCT downloaded)'}**. "
               "Value function = SAGE loss reduction `v(S)=L(∅)−L(S)` (log-loss); AUC descriptive. "
               "Interaction <0 redundancy / >0 synergy. CIs = eye-level bootstrap. **Label is VF-derived "
               "(VF-anchored): weight OCT|VF and the interaction, not VF|OCT alone.**")
    out.append("")
    for r in results:
        out.append(f"## label: `{r.label}`  —  n={r.n}, progressors={r.n_pos} "
                   f"(prevalence {r.prevalence:.3f})")
        out.append("")
        out.append("| subset | AUC |")
        out.append("|---|---|")
        for key, v in r.aucs.items():
            out.append(f"| {key} | {v:.3f} |")
        aci = r.ci.get(f"auc[{'+'.join(sorted(k for k in ['oct','vf'] if k in r.shapley))}]")
        out.append("")
        if oct_present and r.interaction is not None:
            out.append(f"- coalition values: {', '.join(f'{k}={_fmt(v)}' for k,v in r.coalition.items())}")
            for key, v in r.incremental.items():
                lo, hi = r.ci.get(f"incr[{key}]", (float('nan'), float('nan')))
                out.append(f"- **incremental value {key}** (adding {key.split('|')[0]} given "
                           f"{key.split('|')[1]}): {_fmt(v)} (95% CI {_fmt(lo)}, {_fmt(hi)})")
            lo, hi = r.ci.get("interaction", (float('nan'), float('nan')))
            read = "redundancy" if r.interaction < 0 else "synergy"
            out.append(f"- **interaction I[oct,vf]**: {_fmt(r.interaction)} ({read}; "
                       f"95% CI {_fmt(lo)}, {_fmt(hi)})")
        else:
            fk = next(iter(r.aucs))
            lo, hi = r.ci.get(f"auc[{fk}]", (float('nan'), float('nan')))
            out.append(f"- VF-only AUC {r.aucs[fk]:.3f} (95% CI {_fmt(lo).lstrip('+')}, "
                       f"{_fmt(hi).lstrip('+')}); coalition v(vf)={_fmt(r.coalition.get('vf', float('nan')))}")
        out.append("")
    return "\n".join(out)
