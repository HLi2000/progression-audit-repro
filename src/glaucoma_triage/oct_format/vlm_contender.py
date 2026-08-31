"""Pure (no-API) helpers for bridging a multimodal LLM (Gemini) onto the OCT appearance-axis benchmark.

Kept import-light on PURPOSE — NO ``google.genai``, NO torch, NO network — so it unit-tests without the SDK or an
API key (mirrors ``fm_contender``). The producer ``scripts/run_gemini_bscan_contender.py`` assembles the image
``contents``, calls the Gemini API, and hands each response here to :func:`parse_referral_prob`; the extracted
per-eye probabilities are then scored by the SAME ``fm_contender.score_proba_degradation`` as every other
appearance-axis arm — head-to-head on the seed-0 360-eye GLEAM split, under the same Spectralis->Topcon shift.

Elicitation (validated against an independent GLEAM run): ask the model for a **binary glaucoma VERDICT + a
confidence + a severity**, NOT a bare probability. A bare "give me P(glaucoma)" prompt makes the model hedge to a
degenerate ~0.05/0.5 (near-chance); forcing a boolean *decision* graded by confidence recovers real discrimination
(≈79% acc / 0.79 AUC few-shot vs ≈0.58 for the probability prompt). We then map the verdict to a referral
probability = ``confidence if glaucoma else 1-confidence`` (a graded score for AUC / referral-flip / ECE).
Few-shot = in-context learning (Brown 2020) with **severity-graded** exemplars (healthy + early/moderate/advanced)
drawn ONLY from the TRAIN split — the leakage guard.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class GlaucomaVerdict(BaseModel):
    """The structured output every VLM arm returns (the Gemini ``response_schema``). A boolean decision + a
    confidence + a severity — the framing that recovers real discrimination (vs a hedged bare probability)."""

    glaucoma: bool = Field(..., description="True if this eye shows glaucoma and should be REFERRED, else False.")
    severity: str = Field(default="", description="none | early | moderate | advanced")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Your confidence in the glaucoma call, 0-1.")
    rationale: str = Field(default="", description="One short sentence (RNFL thinning, rim change, …).")


def referral_prob(glaucoma: bool, confidence: float) -> float:
    """Map a boolean glaucoma call + its confidence to a referral probability in [0,1]: ``confidence`` if the call
    is glaucoma, else ``1-confidence``. Forcing a boolean decision (then grading it by confidence) avoids the
    degenerate hedge a bare 'give a probability' prompt elicits, while still yielding a graded score for
    AUC / referral-flip / ECE."""
    c = float(np.clip(confidence, 0.0, 1.0))
    return c if bool(glaucoma) else 1.0 - c


# ── prompt pieces (the producer stitches these around the PIL/bytes images) ───────────────────────────
_INTRO_BASE = (
    "You are grading circumpapillary OCT B-scans for glaucoma. Each image is a cross-sectional retinal scan taken "
    "in a ring around the optic nerve head. Glaucoma shows as thinning/loss of the retinal nerve fibre layer (RNFL) "
    "and neuroretinal rim."
)
FEWSHOT_INTRO = _INTRO_BASE + (" Below are LABELLED reference B-scans in this exact format so you can calibrate "
                               "what healthy vs glaucomatous scans look like here.")
ZEROSHOT_INTRO = _INTRO_BASE
QUERY = ("Now classify THIS new B-scan. Return JSON: glaucoma (bool), severity (none|early|moderate|advanced), "
         "confidence (0-1, your confidence in the glaucoma call), rationale (one short sentence).")


class ReferralProb(BaseModel):
    """Direct verbalized-probability schema (Tian et al., EMNLP 2023): the model states P(refer for glaucoma) in
    [0,1]. A GRADED score (better ROC/ECE granularity than the verdict's bimodal 0.05/0.95), AUC-equal to the
    verdict per the n=100 ablation. This is the DEFAULT schema."""

    glaucoma_referral_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Calibrated probability in [0,1] that this eye should be REFERRED for glaucoma; use the full range.")
    rationale: str = Field(default="", description="One short clause (RNFL thinning, rim change, …).")


PROB_QUERY = ("Now assess THIS new B-scan. Return JSON: glaucoma_referral_probability (0-1, your calibrated "
              "probability that this eye should be referred for glaucoma), rationale (one short sentence).")

# schema name -> (pydantic response_schema, query text). 'probability' = Tian-2023 verbalized confidence (DEFAULT);
# 'verdict' = the boolean-call variant (AUC-equal per the ablation). parse_referral_prob handles both outputs.
SCHEMAS = {"probability": (ReferralProb, PROB_QUERY), "verdict": (GlaucomaVerdict, QUERY)}

_SEVERITY = {0: "HEALTHY (no glaucoma)", 1: "GLAUCOMA — early", 2: "GLAUCOMA — moderate", 3: "GLAUCOMA — advanced"}


def exemplar_caption(label4: int) -> str:
    """Text tag paired with a few-shot exemplar image (4-class GLEAM label: 0=healthy, 1/2/3=glaucoma severity)."""
    return f"Reference label: {_SEVERITY.get(int(label4), 'GLAUCOMA')}."


def select_fewshot_severity(labels4: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """6 exemplars from the TRAIN ``pool``: 3 healthy (sev 0) + 1 each early/moderate/advanced (sev 1/2/3), spanning
    the glaucoma spectrum. If a severity is absent, top up with any remaining glaucoma. Exemplars are guaranteed
    ``⊆ pool`` (disjoint from a held-out eval set disjoint from ``pool``) — the leakage guard (Brown 2020)."""
    pool = np.asarray(pool, dtype=int)
    lab = np.asarray(labels4)[pool]
    pick: list[int] = []
    heal = pool[lab == 0]
    if len(heal):
        pick += [int(x) for x in rng.choice(heal, size=min(3, len(heal)), replace=False)]
    for sev in (1, 2, 3):
        s = pool[lab == sev]
        if len(s):
            pick.append(int(rng.choice(s)))
    need = 6 - len(pick)
    if need > 0:
        rest = pool[(lab > 0) & ~np.isin(pool, pick)]
        if len(rest):
            pick += [int(x) for x in rng.choice(rest, size=min(need, len(rest)), replace=False)]
    return np.array(pick, dtype=int)


def select_fewshot_indices(labels: np.ndarray, pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Binary-balanced exemplar selection (``k//2`` glaucoma + rest healthy) from the TRAIN ``pool`` — kept for the
    simple/binary case; the producer uses :func:`select_fewshot_severity`. Exemplars ``⊆ pool`` (leakage guard)."""
    pool = np.asarray(pool, dtype=int)
    lab = (np.asarray(labels)[pool] > 0).astype(int)
    pos, neg = pool[lab == 1], pool[lab == 0]
    n_pos = k // 2
    take_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False) if len(pos) else np.array([], int)
    take_neg = rng.choice(neg, size=min(k - n_pos, len(neg)), replace=False) if len(neg) else np.array([], int)
    return rng.permutation(np.concatenate([take_pos, take_neg]).astype(int))


# ── response parsing (verdict -> referral probability) ────────────────────────────────────────────────
def _coerce_prob(x: Any) -> float:
    p = float(x)
    if not np.isfinite(p):
        raise ValueError(f"non-finite probability {x!r}")
    return float(np.clip(p, 0.0, 1.0))


def _loads_lenient(text: str) -> Any:
    """``json.loads`` tolerating ```json fences / surrounding prose."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.DOTALL).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _prob_from_dict(d: dict) -> float | None:
    if "glaucoma" in d:                                          # the verdict schema
        conf = d.get("confidence", 0.5)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        return referral_prob(bool(d["glaucoma"]), conf)
    if "glaucoma_referral_probability" in d:                     # legacy probability field
        return _coerce_prob(d["glaucoma_referral_probability"])
    return None


def parse_referral_prob(response: Any) -> float:
    """Extract the referral probability [0,1] from a Gemini response (or a test stand-in).

    Priority: ``.parsed`` (a :class:`GlaucomaVerdict` / ``BaseModel`` / dict) → bare model/dict → text JSON. A
    verdict maps via :func:`referral_prob` (``confidence`` signed by the boolean ``glaucoma`` call); a legacy
    ``glaucoma_referral_probability`` is used directly. Raises ``ValueError`` if nothing parseable is found — the
    producer catches it, retries, then imputes 0.5 and logs the failure.
    """
    obj = getattr(response, "parsed", None)
    if obj is None and isinstance(response, (BaseModel, dict)):
        obj = response
    if isinstance(obj, BaseModel):
        if hasattr(obj, "glaucoma"):
            return referral_prob(bool(obj.glaucoma), float(getattr(obj, "confidence", 0.5)))
        if hasattr(obj, "glaucoma_referral_probability"):
            return _coerce_prob(obj.glaucoma_referral_probability)
    if isinstance(obj, dict):
        p = _prob_from_dict(obj)
        if p is not None:
            return p

    text = getattr(response, "text", None)
    if text is None and isinstance(response, str):
        text = response
    if isinstance(text, str) and text.strip():
        d = _loads_lenient(text)
        if isinstance(d, dict):
            p = _prob_from_dict(d)
            if p is not None:
                return p
        m = re.search(r"glaucoma_referral_probability\"?\s*[:=]\s*(-?\d*\.?\d+)", text)
        if m:
            return _coerce_prob(m.group(1))
    raise ValueError(f"could not parse a referral verdict/probability from response: {str(response)[:200]!r}")
