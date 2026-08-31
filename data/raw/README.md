# Raw data (not included)

The raw **GRAPE** dataset is a licensed medical dataset and is **not** redistributed in this repository. You do not need
it to rebuild the paper — the committed files in `../derived/` already carry every number the PDF uses. You only need the
raw data to regenerate `../derived/` from scratch via the `run_grape_*.py` scripts.

## Obtain GRAPE

GRAPE is published on Figshare (collection **6406319**):

- Huang et al., *GRAPE: A multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma
  progression prediction*, Scientific Data (2023).
- Download the collection from Figshare and accept its terms of use.

## Layout expected by the pipeline

Place the extracted dataset here so the loaders in `src/glaucoma_triage/datasets/grape.py` find it:

```
data/raw/grape/
  <the GRAPE files/folders as distributed on Figshare>
```

The fundus-embedding and fine-tuning scripts additionally need RETFound-CFP weights (Hugging Face) and `torch`; see the
top-level `README.md`.
