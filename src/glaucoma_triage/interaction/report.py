"""Phase-1 diagnosis modality-interaction decomposition — engine validation for the P1 plan.

Runs the coalition-Shapley decomposition (:mod:`.shapley`) over the SAGE-proper loss-reduction value
function (:mod:`glaucoma_triage.models.fusion`) on the **diagnosis** task, using data already on disk.
This is the method-validation warm-up before the Harvard-GDP1000 progression headline — it must produce
sign-sane interaction terms and a leak-free structure read.

Reads produced (see the plan's committed direction):
* **Headline (leak-free screening):** PAPILA `fundus × iop × cct × refraction × age` on the full cohort.
  `fundus` is the pre-specified **3-feature clinical panel** (`features.FUNDUS_CLINICAL_PANEL` =
  vCDR·area-CDR·min-rim), identical to the progression study's panel for cross-study consistency; `vf_md`
  is EXCLUDED because in PAPILA it is MNAR-on-label (value + presence leak).
* **Structure↔function reference:** Harvard-GF `oct × vf` — flagged: its glaucoma label is VF-derived,
  so the redundancy is real in direction but leak-inflated.
* **Sensitivity (labeled):** `fundus × iop × cct × vf_md` on the real-VF subcohort only (complete-case);
  glaucoma-vs-OHT/suspect among worked-up eyes, NOT screening.
* **Secondary A:** within-`fundus` feature-level Shapley (interpretability, low-novelty).
* **Secondary B:** fundus-representation ablation — does the structure↔function interaction survive
  representing `fundus` as full morphometry vs the clinical panel vs the gold/sensitive derived reads.

Value function is loss-based (Shapley efficiency holds); AUC is descriptive only. CIs are a patient-level
cluster bootstrap.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..datasets import features
from ..models.fusion import ModalitySpec, coalition_values, per_class_auc, subset_auc
from .shapley import interaction_matrix, shapley_values


def _log(msg: str) -> None:
    """Progress line to stderr (flushed) so long/background runs are observable."""
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Decomposition:
    """Point-estimate + optional bootstrap-CI results for one decomposition read."""

    name: str
    n_eyes: int
    n_patients: int
    prevalence: float
    class_counts: dict[int, int]
    full_auc: float
    shapley: dict[str, float]
    interaction: dict[tuple[str, str], float]
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_class_auc: dict[int, float] = field(default_factory=dict)
    single_auc: dict[str, float] = field(default_factory=dict)
    note: str = ""


# --- specs -----------------------------------------------------------------------------------
def _papila_headline_spec() -> ModalitySpec:
    """Leak-free screening, enriched with metadata modalities: fundus (the 3-feature clinical panel
    vCDR·area-CDR·min-rim, identical to the progression study) × IOP × CCT × refraction × age
    (NO vf_md; NO sex — a negligible glaucoma risk factor)."""
    return {
        "fundus": list(features.FUNDUS_CLINICAL_PANEL),
        "iop": ["iop"], "cct": ["cct"], "refraction": ["refraction"], "age": ["age"],
    }


def _papila_sensitivity_spec() -> ModalitySpec:
    """Real-VF subcohort: the SAME modality set as the leak-free headline (fundus 3-feat panel × iop ×
    cct × refraction × age) PLUS vf_md (the genuinely-measured function modality) — so the incremental
    value / redundancy of VF is measured over the exact headline baseline (apples-to-apples)."""
    return {"fundus": list(features.FUNDUS_CLINICAL_PANEL),
            "iop": ["iop"], "cct": ["cct"], "refraction": ["refraction"], "age": ["age"],
            "vf_md": ["vf_md"]}


_CLINICAL_CLASS_CODE = {"healthy": 0, "OHT": 1, "glaucoma": 2}


def clinical_class_coded(df: pd.DataFrame) -> pd.Series:
    """Integer-coded 3-class benchmark label (healthy=0, OHT=1, glaucoma=2; <NA> otherwise).

    OHT ⟺ non-glaucoma & IOP≥24 (GIRFT) — so the healthy↔OHT axis is IOP-definitional (circular for
    the `iop` modality); the glaucoma axis is clean. See `features.clinical_class`."""
    cc = features.clinical_class(df)
    return cc.map(_CLINICAL_CLASS_CODE).astype("Int64")


# --- core decomposition ----------------------------------------------------------------------
def _decompose(
    sub: pd.DataFrame, spec: ModalitySpec, y: pd.Series, k: int, seed: int, n_repeats: int, est=None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    table = coalition_values(sub, spec, y, k=k, seed=seed, n_repeats=n_repeats, estimator_factory=est)
    players = list(spec)
    return shapley_values(table, players), interaction_matrix(table, players)


def _patient_index(sub: pd.DataFrame, patient_col: str = "patient_id"):
    """(patients, flat, bounds): unique patient ids in first-appearance order + a FLAT positional-row
    index (`flat`) split by `bounds` (CSR-style: patient p's rows = flat[bounds[p]:bounds[p+1]]). The flat
    form pickles as 2 numpy arrays (cheap to ship to workers) rather than a list of n_patients arrays, and
    makes a bootstrap draw O(n_rows) not O(n_patients²)."""
    gi = sub.groupby(patient_col, sort=False).indices          # {pid: ascending positional np.array}
    patients = pd.unique(sub[patient_col].to_numpy())
    idx_by = [gi[p] for p in patients]
    flat = np.concatenate(idx_by) if idx_by else np.empty(0, dtype=int)
    bounds = np.concatenate([[0], np.cumsum([len(a) for a in idx_by])]).astype(int)
    return patients, flat, bounds


def _resample_rows(rng, patients, flat, bounds):
    """Same ``rng.choice`` patient picks as the naive per-patient bootstrap → (row positions, per-row
    relabeled patient-id list). Single source of truth for the resample so every caller is bit-identical."""
    draws = rng.choice(len(patients), size=len(patients), replace=True)
    rows = np.concatenate([flat[bounds[p]:bounds[p + 1]] for p in draws]) if len(draws) else np.empty(0, int)
    labels = np.repeat([f"{patients[p]}__b{j}" for j, p in enumerate(draws)],
                       [bounds[p + 1] - bounds[p] for p in draws])
    return rows, labels.tolist()


def _cluster_bootstrap(sub: pd.DataFrame, rng: np.random.Generator,
                       patients=None, flat=None, bounds=None) -> pd.DataFrame:
    """Resample patients with replacement; fresh patient ids per draw (cluster bootstrap).

    Vectorized + bit-identical to the naive per-patient version (same ``rng.choice`` picks + ``f"{pid}__b{j}"``
    labels): one ``np.concatenate`` + a single ``.iloc`` instead of an O(n_patients²) per-patient boolean scan."""
    if patients is None:
        patients, flat, bounds = _patient_index(sub)
    rows, labels = _resample_rows(rng, patients, flat, bounds)
    out = sub.iloc[rows].reset_index(drop=True)
    out["patient_id"] = labels                              # object dtype, matching the naive version
    return out


def _boot_draw(slim, spec, y_full, k, cv_seed, draw_seed, est, patients, flat, bounds):
    """One patient-cluster bootstrap draw → ``(full_auc, interaction_dict)`` or ``None`` on failure.

    Operates on a **slim** frame (only the spec feature columns + patient_id) and a **precomputed** label
    Series ``y_full`` (row-wise labels, so resampling by row is identical to recomputing ``label`` on the
    resampled frame). This keeps the per-task pickle small. Its RNG is seeded from ``draw_seed`` (a spawned
    ``SeedSequence``) so the draw set is reproducible for any ``n_jobs``."""
    rng = np.random.default_rng(draw_seed)
    rows, labels = _resample_rows(rng, patients, flat, bounds)
    bs = slim.iloc[rows].reset_index(drop=True)
    bs["patient_id"] = labels
    y_boot = y_full.iloc[rows].reset_index(drop=True)
    try:
        _, i_b = _decompose(bs, spec, y_boot, k, cv_seed, 1, est)   # n_repeats=1 for speed
        f_b = subset_auc(bs, spec, list(spec), y_boot, k=k, seed=cv_seed, n_repeats=1, estimator_factory=est)
    except Exception:
        return None
    return f_b, i_b


def decompose(
    sub: pd.DataFrame, spec: ModalitySpec, label: Callable[[pd.DataFrame], pd.Series], name: str,
    *, note: str = "", k: int = 5, seed: int = 0, n_repeats: int = 3, n_boot: int = 100,
    progress: str = "", estimator_factory=None, n_jobs: int = -1,
) -> Decomposition:
    """Point-estimate decomposition + patient cluster-bootstrap CIs on the full-model AUC and each
    pairwise interaction index. ``progress`` is a prefix (e.g. ``"[2/6]"``) for stderr progress.
    ``estimator_factory`` overrides the model (e.g. a regularized logistic for low-event atlases).
    The bootstrap is parallelized over draws (``n_jobs``; ``-1`` = all cores) with per-draw seeding, so
    results are identical for any ``n_jobs``."""
    y = label(sub)
    scorable = pd.to_numeric(y, errors="coerce").notna()
    sub_s = sub[scorable].reset_index(drop=True)
    yv = pd.to_numeric(label(sub_s), errors="coerce")

    if progress:
        _log(f"{progress} {name} — point estimate ({len(sub_s)} eyes, {2**len(spec)} coalitions)…")
    shap, inter = _decompose(sub_s, spec, label(sub_s), k, seed, n_repeats, estimator_factory)
    full_auc = subset_auc(sub_s, spec, list(spec), label(sub_s), k=k, seed=seed, n_repeats=n_repeats,
                          estimator_factory=estimator_factory)
    pc_auc = per_class_auc(sub_s, spec, list(spec), label(sub_s), k=k, seed=seed, n_repeats=n_repeats,
                           estimator_factory=estimator_factory)
    single_auc = {                                     # descriptive per-modality discrimination (point est.)
        m: subset_auc(sub_s, spec, [m], label(sub_s), k=k, seed=seed, n_repeats=n_repeats,
                      estimator_factory=estimator_factory)
        for m in spec
    }

    boot_full: list[float] = []
    boot_inter: dict[tuple[str, str], list[float]] = {p: [] for p in inter}
    if n_boot:
        if progress:
            _log(f"{progress}   bootstrap ×{n_boot} (n_jobs={n_jobs}) …")
        patients, flat, bounds = _patient_index(sub_s)     # precompute once (per read, not per draw)
        feat_cols = list(dict.fromkeys(c for cols in spec.values() for c in cols))
        slim = sub_s[feat_cols + ["patient_id"]].copy()    # only what a draw needs → small per-task pickle
        y_full = label(sub_s)                              # row-wise label, resampled by row in the worker
        child_seeds = np.random.SeedSequence(seed).spawn(n_boot)
        # loky (process) backend: the many small logistic fits are GIL-bound (LBFGS holds the GIL), so a
        # threading backend serializes them on one core — processes give true multi-core parallelism.
        draws = Parallel(n_jobs=n_jobs)(
            delayed(_boot_draw)(slim, spec, y_full, k, seed, cs, estimator_factory, patients, flat, bounds)
            for cs in child_seeds)
        for res in draws:
            if res is None:
                continue
            f_b, i_b = res
            if f_b == f_b:
                boot_full.append(f_b)
            for pair, val in i_b.items():
                if val == val:
                    boot_inter[pair].append(val)

    def _ci(vals: list[float]) -> tuple[float, float]:
        if len(vals) < 2:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    ci = {"full_auc": _ci(boot_full)}
    for pair, vals in boot_inter.items():
        ci[f"I[{pair[0]},{pair[1]}]"] = _ci(vals)

    yv_int = pd.to_numeric(yv, errors="coerce").dropna().astype(int)
    class_counts = {int(k_): int(v_) for k_, v_ in yv_int.value_counts().sort_index().items()}
    return Decomposition(
        name=name, n_eyes=int(len(sub_s)), n_patients=int(pd.unique(sub_s["patient_id"]).size),
        prevalence=float(yv.mean()), class_counts=class_counts, full_auc=full_auc,
        shapley=shap, interaction=inter, ci=ci, per_class_auc=pc_auc, single_auc=single_auc, note=note,
    )


# --- the real-VF sensitivity subcohort -------------------------------------------------------
def real_vf_subcohort(df: pd.DataFrame) -> pd.DataFrame:
    """PAPILA eyes with a GENUINELY-MEASURED vf_md (not simulated), complete-case on fundus/iop/cct/vf_md.

    ~164 eyes (9 real-healthy + 68 suspect + 87 glaucoma). Quarantines the simulated healthy values."""
    pap = df[df["source"] == "papila"].copy()
    simulated = (pap["vf_md_simulated"] == True).fillna(False)  # noqa: E712 — NA-safe
    vf = pd.to_numeric(pap["vf_md"], errors="coerce")
    real = pap[(~simulated) & vf.notna()]
    need = ["vcdr", "iop", "cct", "vf_md"]  # complete-case on the modality keys
    for c in need:
        real = real[pd.to_numeric(real[c], errors="coerce").notna()]
    return real.reset_index(drop=True)


# --- Secondary A: within-fundus feature-level Shapley ----------------------------------------
def within_fundus_shapley(df: pd.DataFrame, *, k: int = 5, seed: int = 0, n_repeats: int = 3,
                          ) -> dict[str, float]:
    """Feature-level Shapley over the **full available disc morphometry** (10 metrics,
    `features.FUNDUS_MORPHOMETRY_FULL`) — a superset of the 3-feature headline panel — so it shows which
    disc biomarker carries the structural signal, not just the three headline features. Each biomarker is
    its own coalition player (2^10=1024 coalitions); interpretability only, explicitly NOT the modality headline."""
    pap = df[df["source"] == "papila"].reset_index(drop=True)
    spec = {f: [f] for f in features.FUNDUS_MORPHOMETRY_FULL}
    y = features.fundus_structural_label(pap)
    scorable = pd.to_numeric(y, errors="coerce").notna()
    pap_s = pap[scorable].reset_index(drop=True)
    table = coalition_values(pap_s, spec, features.fundus_structural_label(pap_s),
                             k=k, seed=seed, n_repeats=n_repeats)
    return shapley_values(table, list(spec))


# --- Secondary B: fundus-representation robustness ablation -----------------------------------
def _add_derived_reads(sub: pd.DataFrame) -> pd.DataFrame:
    """Attach the gold / sensitive derived binary reads as numeric columns for the ablation."""
    out = sub.copy()
    out["fundus_gold"] = pd.to_numeric(
        features.fundus_structural_label_derived(sub, combine="mean"), errors="coerce")
    out["fundus_sensitive"] = pd.to_numeric(
        features.fundus_suspicious_disc_label(sub), errors="coerce")
    return out


def representation_ablation(df: pd.DataFrame, *, k: int = 5, seed: int = 0, n_repeats: int = 3,
                            ) -> dict[str, float]:
    """The structure↔function interaction index I[fundus, vf_md] on the real-VF subcohort under four
    fundus representations. A stable sign/magnitude means the redundancy conclusion is not an artifact
    of the feature choice — the empirical justification for the a-priori rule."""
    sub = _add_derived_reads(real_vf_subcohort(df))
    reps: dict[str, list[str]] = {
        "full-morphometry": list(features.FUNDUS_MORPHOMETRY_FULL),
        "clinical-panel": list(features.FUNDUS_CLINICAL_PANEL),
        "gold-derived": ["fundus_gold"],
        "sensitive-derived": ["fundus_sensitive"],
    }
    out: dict[str, float] = {}
    for name, cols in reps.items():
        spec = {"fundus": cols, "iop": ["iop"], "cct": ["cct"], "vf_md": ["vf_md"]}
        y = features.fundus_structural_label(sub)
        table = coalition_values(sub, spec, y, k=k, seed=seed, n_repeats=n_repeats)
        inter = interaction_matrix(table, list(spec))
        pair = tuple(sorted(("fundus", "vf_md")))
        out[name] = inter[pair]
    return out


# --- reporting -------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.3f}"


def run(df: pd.DataFrame, *, k: int = 5, seed: int = 0, n_repeats: int = 3, n_boot: int = 100,
        n_jobs: int = -1) -> dict:
    """Compute all Phase-1 reads present in ``df``, with stderr progress logging."""
    out: dict = {"headline": [], "secondary_a": None, "secondary_b": None}
    # assemble the read list first so we can number progress [i/N]
    reads: list[tuple] = []
    if (df["source"] == "papila").any():
        pap = df[df["source"] == "papila"].reset_index(drop=True)
        sub = real_vf_subcohort(df)
        reads += [
            (pap, _papila_headline_spec(), features.fundus_structural_label,
             "PAPILA — leak-free screening, binary glaucoma-vs-rest (fundus × iop × cct × refraction × age)",
             "PRIMARY read. vf_md excluded (MNAR-on-label). iop is NOT circular here — glaucoma is "
             "~86% normal-tension (IOP-independent). Missing iop/cct/refraction (≤3%) median-imputed "
             "within-fold (no leakage); age complete."),
            (pap, _papila_headline_spec(), clinical_class_coded,
             "PAPILA — 3-class healthy/OHT/glaucoma (fundus × iop × cct × refraction × age) [FLAGGED]",
             "SECONDARY read. OHT ⟺ non-glaucoma & IOP≥24, so the healthy↔OHT axis is IOP-definitional "
             "→ iop's value here is CIRCULAR (expected artifact, not a finding); the glaucoma axis is "
             "clean. OHT n≈20 → wide CIs. AUC is OVR-macro."),
            (sub, _papila_sensitivity_spec(), features.fundus_structural_label,
             "PAPILA — real-VF sensitivity, binary glaucoma-vs-rest (fundus × iop × cct × vf_md)",
             "SELECTED subcohort (real VF only, complete-case): glaucoma vs OHT/suspect among worked-up "
             "eyes, NOT screening. vf_md genuinely measured here."),
            (sub, _papila_sensitivity_spec(), clinical_class_coded,
             "PAPILA — real-VF 3-class healthy/OHT/glaucoma (fundus × iop × cct × vf_md) [FLAGGED]",
             "SECONDARY read on the real-VF subcohort. VF (field damage) is what separates glaucoma from "
             "OHT/healthy → expect vf_md to lead the glaucoma axis and fundus×vf_md redundancy to persist. "
             "OHT axis IOP-definitional (iop CIRCULAR); OHT small → wide CIs; AUC OVR-macro."),
        ]
    if (df["source"] == "harvard_gf").any():
        hgf = df[df["source"] == "harvard_gf"].reset_index(drop=True)
        reads.append(
            (hgf, {"oct": features.imaging_columns("oct"), "vf_md": ["vf_md"]}, features.triage_label,
             "Harvard-GF — structure↔function reference (oct × vf_md)",
             "LEAK-FLAGGED: the glaucoma label is VF-derived, so the oct×vf_md redundancy is real in "
             "direction but leak-inflated (AUC saturates). Kept as engine validation only. OCT is the "
             "3-feature arcuate panel (mean·inferior·superior), consistent with the progression OCT panel."))

    total = len(reads)
    for i, (sub_i, spec_i, label_i, name_i, note_i) in enumerate(reads, 1):
        out["headline"].append(decompose(
            sub_i, spec_i, label_i, name_i, note=note_i,
            k=k, seed=seed, n_repeats=n_repeats, n_boot=n_boot, progress=f"[{i}/{total}]", n_jobs=n_jobs))

    if (df["source"] == "papila").any():
        _log("[secondary] within-fundus feature Shapley …")
        out["secondary_a"] = within_fundus_shapley(df, k=k, seed=seed, n_repeats=n_repeats)
        _log("[secondary] fundus-representation ablation …")
        out["secondary_b"] = representation_ablation(df, k=k, seed=seed, n_repeats=n_repeats)
    return out


def render_markdown(results: dict) -> str:
    out: list[str] = ["# Phase 1 — diagnosis modality-interaction decomposition", ""]
    out.append("Engine validation for the P1 progression plan. Value function = SAGE-proper loss "
               "reduction `v(S)=L(∅)−L(S)` (log-loss); `v(∅)=0`; efficiency `Σφ=v(N)` holds. AUC is "
               "descriptive only. Interaction index: **>0 synergy**, **<0 redundancy**. CV is "
               "patient-grouped `StratifiedGroupKFold`, repeated; CIs are a patient cluster bootstrap.")
    out.append("")
    out.append("*Scope: this read validates the coalition-Shapley / interaction engine and demonstrates the "
               "label-leakage discipline. The progression-side methods (Gaussian PID, decision-curve / "
               "EVPI-EVSI, survival landmarking) are separate and are validated by their own synthetic "
               "unit tests, not by this diagnosis read.*")
    out.append("")
    for r in results.get("headline", []):
        ci_full = r.ci.get("full_auc", (float("nan"), float("nan")))
        out.append(f"## {r.name}")
        out.append("")
        if len(r.class_counts) <= 2:
            out.append(f"- eyes: {r.n_eyes}, patients: {r.n_patients}, positive prevalence: "
                       f"{_fmt(r.prevalence)}")
        else:
            dist = ", ".join(f"{k_}:{v_}" for k_, v_ in r.class_counts.items())
            out.append(f"- eyes: {r.n_eyes}, patients: {r.n_patients}, class counts (0=healthy, "
                       f"1=OHT, 2=glaucoma): {dist}")
        out.append(f"- full-model AUC (descriptive): **{_fmt(r.full_auc)}** "
                   f"(95% CI {_fmt(ci_full[0])}–{_fmt(ci_full[1])})")
        if r.per_class_auc:
            names = {0: "healthy", 1: "OHT", 2: "glaucoma"}
            pc = " · ".join(f"{names.get(c, c)} {_fmt(a)}" for c, a in sorted(r.per_class_auc.items()))
            out.append(f"- per-class one-vs-rest AUC: {pc}")
        if r.note:
            out.append(f"- note: {r.note}")
        out.append("")
        out.append("| modality | Shapley value (loss explained) | single-modality AUC (descriptive) |")
        out.append("|---|---|---|")
        for m, v in sorted(r.shapley.items(), key=lambda kv: -kv[1]):
            out.append(f"| {m} | {_fmt(v)} | {_fmt(r.single_auc.get(m, float('nan')))} |")
        out.append("")
        out.append("| modality pair | interaction index | read | 95% CI |")
        out.append("|---|---|---|---|")
        for pair, v in sorted(r.interaction.items(), key=lambda kv: kv[1]):
            lo, hi = r.ci.get(f"I[{pair[0]},{pair[1]}]", (float("nan"), float("nan")))
            read = "synergy" if v > 0 else "redundancy"
            out.append(f"| {pair[0]} × {pair[1]} | {_fmt(v)} | {read} | {_fmt(lo)}–{_fmt(hi)} |")
        out.append("")
    if results.get("secondary_a"):
        out.append("## Secondary A — within-`fundus` feature-level Shapley (interpretability, low-novelty)")
        out.append("")
        out.append("Feature-level, NOT the modality-level headline. Which disc biomarker carries the "
                   "structure signal, over the full disc morphometry (the 10 metrics of the fundus modality).")
        out.append("")
        out.append("| fundus biomarker | Shapley value (loss explained) |")
        out.append("|---|---|")
        for m, v in sorted(results["secondary_a"].items(), key=lambda kv: -kv[1]):
            out.append(f"| {m} | {_fmt(v)} |")
        out.append("")
    if results.get("secondary_b"):
        out.append("## Secondary B — fundus-representation robustness ablation")
        out.append("")
        out.append("Structure↔function interaction `I[fundus, vf_md]` on the real-VF subcohort under "
                   "four fundus representations. A stable sign/magnitude means the redundancy "
                   "conclusion is not an artifact of the feature choice.")
        out.append("")
        out.append("| fundus representation | I[fundus, vf_md] | read |")
        out.append("|---|---|---|")
        for name, v in results["secondary_b"].items():
            read = "synergy" if v > 0 else "redundancy"
            out.append(f"| {name} | {_fmt(v)} | {read} |")
        out.append("")
    return "\n".join(out)
