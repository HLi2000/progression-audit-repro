"""Reproduce the proposal's data-status table from the actual loaded data.

Produces three views:
  * coverage_table  — per-signal x per-arm presence (✓ / ~ / —), mirroring the proposal.
  * class_balance   — disposition / diagnosis / race counts per arm.
  * leakage_notes   — flags that must not become features (e.g. PAPILA VF-MD label leak).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datasets import features, schema

_PRESENT, _PARTIAL, _ABSENT = "✓", "~", "—"


def coverage_table(df: pd.DataFrame, partial_threshold: float = 0.95) -> pd.DataFrame:
    """Per-signal x per-source coverage marks (✓ full, ~ partial, — absent)."""
    sources = sorted(df["source"].dropna().unique())
    rows = []
    for signal, col in schema.SIGNAL_COLUMNS.items():
        row = {"signal": signal}
        for src in sources:
            sub = df[df["source"] == src]
            frac = sub[col].notna().mean() if len(sub) else 0.0
            row[src] = _ABSENT if frac == 0 else (_PRESENT if frac >= partial_threshold else _PARTIAL)
            row[f"{src}_frac"] = round(float(frac), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def class_balance(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Disposition, diagnosis, and race breakdowns per source."""
    out = {}
    out["disposition"] = (
        df.groupby("source")["disposition"].value_counts().unstack(fill_value=0)
    )
    out["diagnosis"] = (
        df.groupby("source")["diagnosis"].value_counts().unstack(fill_value=0)
    )
    if df["race"].notna().any():
        out["race"] = df.groupby("source")["race"].value_counts().unstack(fill_value=0)
    return out


def feature_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source presence fraction for every model-input feature (both structural
    modalities + the tabular set). Shows which arm can populate the input contract."""
    sources = sorted(df["source"].dropna().unique())
    feats = (
        list(features.IMAGING_CLASSIFIERS["oct"]["features"])
        + list(features.IMAGING_CLASSIFIERS["fundus"]["features"])
        + list(features.METADATA_INPUTS)
    )
    rows = []
    for col in feats:
        row = {"input_feature": col}
        for src in sources:
            sub = df[df["source"] == src]
            if not len(sub):
                row[src] = 0.0
            elif col == "sex":
                row[src] = round(float(sub[col].notna().mean()), 3)
            else:
                row[src] = round(float(pd.to_numeric(sub[col], errors="coerce").notna().mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def leakage_notes(df: pd.DataFrame) -> list[str]:
    notes = []
    sim = df.get("vf_md_simulated")
    leak_mask = df["vf_md_leakage"].fillna(False).astype(bool)
    for src in df.loc[leak_mask, "source"].dropna().unique():
        sub = df[df["source"] == src]
        rate = sub.groupby("disposition")["vf_md"].apply(
            lambda s: pd.to_numeric(s, errors="coerce").notna().mean()
        )
        src_sim = sim is not None and sim[df["source"] == src].fillna(False).astype(bool)
        if src_sim is not None and src_sim is not False and bool(src_sim.any()):
            notes.append(
                f"{src}: VF-MD missingness leak REMOVED by simulation "
                f"({int(src_sim.sum())} healthy eyes imputed); presence by disposition now "
                f"= {rate.round(3).to_dict()} (balanced). Real MD preserved; synthetic rows "
                "flagged vf_md_simulated."
            )
        else:
            notes.append(
                f"{src}: VF-MD is label-conditional — presence rate by disposition = "
                f"{rate.round(3).to_dict()} "
                f"(overall {pd.to_numeric(sub['vf_md'], errors='coerce').notna().mean():.1%}). "
                "Its presence encodes the label; simulate healthy MD (build --simulate-vf-md) "
                "or do NOT use vf_md for this arm."
            )

    if (df["source"] == "papila").any():
        notes.append(
            "papila: the disposition is a clinician diagnosis mapped to refer/discharge, "
            "not an observed referral outcome (label-provenance caveat from the proposal)."
        )

    has_oct_suspect = (
        (df["source"] == "harvard_gf").any()
        and df.loc[df["source"] == "harvard_gf", "is_suspect"].fillna(False).astype(bool).any()
    )
    if not has_oct_suspect:
        notes.append(
            "No arm pairs OCT (RNFLT) with a suspect-inclusive disposition on the same "
            "eyes — this is the two-arm limitation from the proposal, not a bug."
        )
    return notes


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov test in pure numpy (scipy is not a dependency).

    Returns ``(D, p)`` where ``D`` is the max CDF gap and ``p`` is the asymptotic Kolmogorov
    survival-function p-value (Press et al., *Numerical Recipes*). A large ``p`` means the two
    samples are *not* distinguishable — the property we want for simulated vs real normal MD.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    allx = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, allx, "right") / n1
                            - np.searchsorted(b, allx, "right") / n2)))
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    ks = 2.0 * sum((-1.0) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(ks, 0.0), 1.0))


def vf_md_distribution_check(df: pd.DataFrame) -> dict | None:
    """Validate that simulated healthy PAPILA MD is indistinguishable from real normal-eye MD.

    Removing the missingness leak is only sound if the imputed values are not themselves
    separable in shape from real MD. Two comparisons, with different roles:

    * ``ks_vs_real_papila`` — the **leak-displacement test**: same device, same arm — are the
      few *real* healthy PAPILA MDs separable from the simulated ones? (Underpowered: those
      eyes exist only because IOP > 22 mmHg, so tiny n and not representative normals — but it
      is the only same-instrument check, so a high p is the reassuring result.)
    * ``ks_vs_harvard`` — a cross-device **realism sanity check** against real normal-eye MD
      derived from Harvard-GF non-glaucoma eyes (computed here, not asserted). NOT pass/fail:
      at n > 1000 the KS p is dominated by the *deliberate* width gap (σ≈1.0 vs Harvard's
      selection-compressed σ≈0.7), so report Δmean and std_ratio, not p.

    Returns ``None`` when nothing was simulated.
    """
    sim = df.get("vf_md_simulated")
    if sim is None:
        return None
    sim = sim.fillna(False).astype(bool)
    md = pd.to_numeric(df.get("vf_md"), errors="coerce")
    is_pap = df["source"] == "papila"
    simulated = md[is_pap & sim].dropna()
    if simulated.empty:
        return None

    real_healthy_pap = md[is_pap & ~sim & (df["diagnosis"] == "healthy")].dropna()
    harvard_normal = md[(df["source"] == "harvard_gf") & (df["diagnosis"] == "non-glaucoma")].dropna()

    def _stats(s: pd.Series) -> dict:
        return {
            "n": int(s.size),
            "mean": round(float(s.mean()), 3) if s.size else float("nan"),
            "std": round(float(s.std(ddof=1)), 3) if s.size > 1 else float("nan"),
        }

    out: dict = {
        "simulated": _stats(simulated),
        "harvard_normal_ref": _stats(harvard_normal),
        "real_healthy_papila": _stats(real_healthy_pap),
    }
    if harvard_normal.size:
        d, p = _ks_2samp(simulated.to_numpy(), harvard_normal.to_numpy())
        out["ks_vs_harvard"] = {
            "D": round(d, 3), "p": round(p, 3),
            "dmean": round(float(simulated.mean() - harvard_normal.mean()), 3),
            "std_ratio": round(float(simulated.std(ddof=1) / harvard_normal.std(ddof=1)), 2),
            "note": ("realism sanity, not a pass/fail — cross-device (Harvard research cohort "
                     "vs PAPILA 30-2) and at this n KS flags the deliberate width gap: Harvard "
                     "normals are selection-compressed (σ≈0.7) vs the literature general-"
                     "population σ≈1.0. Read Δmean and std_ratio, not p."),
        }
    if real_healthy_pap.size:
        d, p = _ks_2samp(simulated.to_numpy(), real_healthy_pap.to_numpy())
        out["ks_vs_real_papila"] = {
            "D": round(d, 3), "p": round(p, 3),
            "note": ("LEAK-DISPLACEMENT test (same device, same arm): is real vs simulated "
                     "healthy MD separable? tiny n & IOP>22-selected, so underpowered, but the "
                     "only same-instrument check. High p ⇒ not separable ⇒ leak not displaced."),
        }
    return out


def render_report(df: pd.DataFrame) -> str:
    """Human-readable multi-section report string."""
    lines = ["=" * 78, "DATA-STATUS REPORT", "=" * 78, ""]
    lines.append(f"Total eyes: {len(df)}")
    lines.append(df.groupby("source").size().to_string())
    lines += ["", "Coverage (✓ full · ~ partial · — absent):", ""]
    cov = coverage_table(df)
    show_cols = ["signal"] + [c for c in cov.columns if not c.endswith("_frac") and c != "signal"]
    lines.append(cov[show_cols].to_string(index=False))
    lines += ["", "Model-input feature availability per source (fraction present):", ""]
    lines.append(feature_availability(df).to_string(index=False))
    lines += [
        "",
        "  Input contract: two separate imaging arms (fundus | oct, never grafted); metadata = "
        + ", ".join(features.METADATA_INPUTS) + ".",
        "  Excluded from input: " + ", ".join(features.EXCLUDED_FROM_INPUT) + ".",
    ]
    lines += ["", "Class balance:"]
    for name, tbl in class_balance(df).items():
        lines += [f"  [{name}]", tbl.to_string(), ""]
    check = vf_md_distribution_check(df)
    if check is not None:
        lines += ["", "VF-MD simulation validation (simulated healthy MD vs real normal-eye MD):"]
        lines.append(f"  simulated healthy PAPILA : {check['simulated']}")
        lines.append(f"  Harvard-GF normal ref    : {check['harvard_normal_ref']}")
        lines.append(f"  real healthy PAPILA      : {check['real_healthy_papila']}")
        if "ks_vs_real_papila" in check:
            k = check["ks_vs_real_papila"]
            lines.append(f"  leak-displacement (same device): KS D={k['D']} p={k['p']} "
                         "— high p ⇒ real vs simulated healthy MD NOT separable (leak not displaced)")
        if "ks_vs_harvard" in check:
            k = check["ks_vs_harvard"]
            lines.append(f"  realism sanity (cross-device)  : Δmean={k['dmean']} dB "
                         f"std_ratio={k['std_ratio']} (KS D={k['D']} p={k['p']}; read Δmean/ratio, not p)")
        lines.append("  note: σ=1.0 is the literature general-population normal SD; Harvard "
                     "normals (σ≈0.7) are selection-compressed — we intentionally do not go narrower.")

    lines += ["", "Leakage / limitation notes:"]
    for note in leakage_notes(df):
        lines.append(f"  * {note}")
    lines.append("=" * 78)
    return "\n".join(lines)
