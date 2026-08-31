# progression-audit-repro

Reproducibility bundle for the paper

> **The Camera, the Split, and the Label: A Three-Check Audit of Glaucoma-Progression Prediction**
> NeurIPS 2026 Workshop — TAE, *"Can We Trust AI Evaluation?"* (under review)

The paper audits the entire published deep-learning record on the public **GRAPE** glaucoma-progression benchmark and
finds that, at this sample size, none of the reported numbers can be distinguished from a trivial baseline once three
checks are applied: a **device floor** (the acquisition camera), a **fair patient-grouped split**, and a **circularity**
test (input disjoint from the window that defines the label). This repo carries everything needed to rebuild the paper
and to inspect (and, with the licensed dataset, regenerate) every number in it.

---

## What's here

```
papers/progression-audit-tae/   LaTeX source, figures (PNG+PDF), compiled main.pdf
scripts/                        build_progression_audit_{panel,tables,figures}.py  (paper from committed data)
                                run_grape_*.py                                     (raw -> derived pipeline)
src/glaucoma_triage/            the Python package the run_grape_*.py scripts import
data/derived/                   committed derived artifacts (aggregate stats + the released split); the paper builds from these
data/raw/README.md              how to obtain the licensed GRAPE data (NOT redistributed here)
reports/progression_audit_panel.json   the single reconciled data panel the tables/figures read
```

Not included, by design: the six **source-paper PDFs** we audit (copyrighted — see the reference list in the paper), and
the **raw GRAPE dataset** (licensed medical data — obtain it from Figshare, see `data/raw/README.md`).

---

## Quick reproduction — rebuild the paper from committed data (no raw data needed)

The build path uses only `numpy` / `matplotlib` and the committed `data/derived/` artifacts.

```bash
# 1. Python env (uv recommended; or: python -m venv .venv && pip install -e .)
uv sync

# 2. Rebuild the reconciled panel, then the tables and figures
uv run python scripts/build_progression_audit_panel.py      # cross-checks every constant vs committed JSON -> "ALL OK"
uv run python scripts/build_progression_audit_tables.py     # -> papers/progression-audit-tae/appendix_tables.tex
uv run python scripts/build_progression_audit_figures.py    # -> papers/progression-audit-tae/figures/*.pdf

# 3. Compile the PDF (install tectonic: `brew install tectonic` or `cargo install tectonic`)
tectonic -X compile papers/progression-audit-tae/main.tex   # -> main.pdf
# (or use the paper's Makefile with a system TeX: `cd papers/progression-audit-tae && make`)
```

The panel builder is the source of truth: it holds every headline constant and reconciles it against the committed
`data/derived/grape_*.json`, printing `ALL OK` when they agree.

### Core scripts behind each finding
- **F1 device floor / representation ladder** — `run_grape_honest_representation.py`, `run_grape_fullfundus_landscape.py`, `run_grape_camera_casemix.py`
- **F2 reproduce-then-revert** — `run_grape_cfp_finetune.py`, `run_grape_tsdf_repro.py`, `run_grape_serial_ladder.py`
- **F3 circularity dial** — `run_grape_serial_event.py`, `run_grape_dial_robustness.py`, `run_grape_two_vft.py`
- **Prospective baseline & split** — `run_grape_prospective_baseline.py`, `run_grape_standard_splits.py`

## Full regeneration from raw (optional; needs the licensed dataset + compute)

The `run_grape_*.py` scripts regenerate `data/derived/` from scratch. They require:
1. the **GRAPE** dataset in `data/raw/grape/` (see `data/raw/README.md`, Figshare collection 6406319);
2. **RETFound-CFP** foundation-model weights (Hugging Face) for the fundus embeddings;
3. `torch` (+ `torchvision`, `timm`) and a GPU for the fine-tuning runs — install alongside the base deps.

These are not needed to rebuild the paper: the committed `data/derived/` artifacts already carry every number the PDF
uses.

## The released split

`data/derived/grape_standard_splits.csv` is the fixed patient-grouped 5-fold split the paper releases (263 eyes / 144
patients; both eyes of a patient in one fold). Score on these fixed folds to remove the single-favorable-split degree of
freedom.

---

## Data, code, and licensing
- **Code** (`src/`, `scripts/`) — MIT, see `LICENSE`.
- **GRAPE data** — © its authors, distributed on Figshare under its own terms; obtain it there, do not assume it is
  redistributable. The committed `data/derived/` files are aggregate statistics plus GRAPE's own pseudonymous IDs and
  derived labels (no images, no PHI).
- **Audited source papers** — their PDFs are copyrighted and are *not* included; see the paper's reference list.

## Citation
See `CITATION.cff`. The paper is under anonymous review; formal authorship will be set on acceptance.
