# References verification — progression-audit / TAE

Firsthand PDFs archived in `refs/` (read directly via pypdf): `gppm_zheng_eswa2025.pdf`,
`fouda_mohammed_imsa2025.pdf` (confirmed to be Fouda & Mohammed, IMSA 2025), `huang_grape_scidata2023.pdf`,
`tsdf_song_lmid2025.pdf` (Springer LMID-2025 chapter, version of record), and `diffsight_huang_2026.pdf` (arXiv). All
five audited GRAPE-CFP papers are now firsthand-archived; their reported AUCs and splits were read from the source tables. GPPM Table 3
(CFP-only baseline 0.577/0.571/0.595; own CFP branch 0.668/0.727/0.555 at zero PLR3/MD sensitivity; 0.849 = VF+CFP
fusion, combined criterion; 156/40 eyes) and Fouda ("camera artifacts" verbatim; 70/30; 4 PLR3 positives; ensemble
0.863) were re-verified against these PDFs.

Discipline: no BibTeX from memory. Every entry is checked against a primary source. This session ran a
**programmatic CrossRef DOI verification** (title + year matched from `api.crossref.org/works/<doi>`) for all
DOI-bearing entries, firsthand **arXiv abstract-page** checks for the preprints, and firsthand **web** checks for
the dataset/venue facts. Entries that cannot be fully confirmed are flagged, never invented.

Verification status:
- **DOI-verified (CrossRef, this session)** — title/year returned by CrossRef match the entry.
- **arXiv-verified** — title + full author list confirmed from the arXiv abstract page.
- **venue-standard** — stable conference proceedings metadata (page range to re-confirm at camera-ready).
- **author-PDF** — confirmed from the authors' PDF earlier in this project; not web-indexed; confirm at camera-ready.

| key | status | source checked / note |
|---|---|---|
| `huang2023grape` | DOI-verified | 10.1038/s41597-023-02424-4 → "GRAPE: A multi-modal dataset of longitudinal follow-up visual field…", Sci Data 10:520, 2023. |
| `saeedi2019` | DOI-verified | 10.1016/j.ophtha.2019.01.029 → "Agreement and Predictors of Discordance of 6 Visual Field Progression Algorithms", Ophthalmology 126(6):822–828, 2019. Author list re-check at camera-ready. |
| `kass2002ohts` | DOI-verified | 10.1001/archopht.120.6.701 → "The Ocular Hypertension Treatment Study", Arch Ophthalmol 120(6):701–713, 2002. |
| `tham2014global` | DOI-verified | 10.1016/j.ophtha.2014.05.013 → Ophthalmology 121(11):2081–2090, 2014. |
| `gedde2021ppp` | DOI-verified (CrossRef, reused from sibling) | 10.1016/j.ophtha.2020.10.022 → Primary Open-Angle Glaucoma PPP, Ophthalmology 128(1):P71–P150, 2021. |
| `geirhos2020` | DOI-verified | 10.1038/s42256-020-00257-z → "Shortcut learning in deep neural networks", Nat Mach Intell 2:665–673, 2020. |
| `zech2018` | DOI-verified | 10.1371/journal.pmed.1002683 → "Variable generalization performance of a deep learning model…", PLoS Med 15(11):e1002683, 2018. |
| `degrave2021` | DOI-verified | 10.1038/s42256-021-00338-7 → "AI for radiographic COVID-19 detection selects shortcuts over signal", Nat Mach Intell 3:610–619, 2021. |
| `gichoya2022` | DOI-verified | 10.1016/S2589-7500(22)00063-2 → "AI recognition of patient race in medical imaging: a modelling study", Lancet Digit Health 4(6):e406–e414, 2022. |
| `poplin2018` | DOI-verified | 10.1038/s41551-018-0195-0 → "Prediction of cardiovascular risk factors from retinal fundus photographs…", Nat Biomed Eng 2:158–164, 2018. |
| `kapoor2023leakage` | DOI-verified | 10.1016/j.patter.2023.100804 → "Leakage and the reproducibility crisis in machine-learning-based science", Patterns 4(9):100804, 2023. |
| `zhou2023retfound` | DOI-verified | 10.1038/s41586-023-06555-x → "A foundation model for generalizable disease detection from retinal images", Nature 622:156–163, 2023. |
| `diciccio1996bootstrap` | DOI-verified | 10.1214/ss/1032280214 → "Bootstrap confidence intervals", Stat Sci 11(3):189–228, 1996. |
| `zheng2025gppm` | **DOI-verified + firsthand Table 3** | CrossRef record: 10.1016/j.eswa.2025.128225, "Glaucoma progression prediction using multi-modal data and trusted multi-view learning", authors **Xiujuan Zheng, Wo Wang, Zhiqing Lv, Ni Li**, Expert Systems with Applications 288:128225, 2025. **Table 3 (page 7, read firsthand from `refs/gppm_zheng_eswa2025.pdf`):** GPPM full model (VF+CFP fusion) per-criterion **AUC = PLR2 0.851 / PLR3 0.818 / MD 0.688 / Combined 0.849**; sensitivities PLR2 0.667, **PLR3 0** (Kappa 0), MD 0.200 — PLR3/MD are near-degenerate operating points on ~40 test eyes. CFP-only "Baseline" (disc-ROI evidential CFP) = **0.577/0.571/0.595** (our floor corroboration); CFP\_M branch 0.668/0.727/0.555; VFT\_M 0.726/0.565/0.768. `0.849` is the separate Combined-criterion headline the abstract/intro cite. Table 1 now shows the verified per-criterion triple 0.851/0.818/0.688 (fusion), note "0.849 comb." |
| `mohammadzadeh2026` | **firsthand-PDF-verified** | AJO 2026;284:43–55, doi:10.1016/j.ajo.2025.12.026, archived `refs/mohammadzadeh_ajo2026.pdf` (13 pp, read directly). "An AI-Based Prognostic Model for Prediction of Functional Glaucoma Progression from Clinical and Structural Data", 13 authors (Mohammadzadeh V … Caprioli J, Nouri-Mahdavi K). **Passes all four of our checks**, so it is a safe positive citation: baseline structural input (ODP + RNFL/macular OCT) → future MD-rate progression (Check 3 non-circular); **patient-level 80/20 split** ("Patients were split into 80%/20% bins", Check 2); **external-device validation** on a separate cohort (AUC 0.754 [0.671–0.837], Check 1); **399/1599 eyes progressed**, well-powered (Check 4). Best-model AUC 0.839 (0.771–0.906); individual baseline modalities ODP 0.664 / RNFL 0.732 / macular 0.781; outperforms glaucoma specialists (0.629/0.680). Cited in Discussion as the on-endpoint, honest-design proof that trajectory progression is predictable in a proper cohort. |
| `song2025tsdf` | **firsthand-PDF + CODE verified** | Springer chapter (LMID 2025, doi:10.1007/978-3-032-16128-4_9), archived `refs/tsdf_song_lmid2025.pdf`. Table 2 GRAPE **AUC 0.866/0.956/0.917** (PLR2/PLR3/MD) → our 0.87/0.96/0.92 ✓; ResNet-50 baseline 0.71/0.80/0.73 ✓. The paper states a participant-level 70/10/20 5-fold split, **but the released code — found this session via the PDF-embedded abstract hyperlink and downloaded (github.com/zongzi13545329/MICCAI: `train_grape.py`, `dure.py`, `GRAPE.zip`) — runs an eye-level `KFold(shuffle, random_state=42)`** over 263 per-eye MAE-feature bags with a DSMIL head, unweighted CE, and best-epoch-by-test-accuracy mean-per-fold AUC (no patient grouping). We reproduce 0.867/0.954/0.917 on its own features/recipe, then collapse to 0.46/0.56/0.59 under GroupKFold-by-patient + honest selection + pooled OOF (`scripts/run_grape_tsdf_repro.py`). TSDF therefore fails Check 1 (device), Check 2 (eye-level split, from code), and Check 3 (sequence spans the label window). **Supersedes the earlier "no public TSDF code" note** — the code exists and is firsthand-verified. |
| `huang2026diffsight` | **firsthand-PDF-verified** | arXiv:2606.09140, archived `refs/diffsight_huang_2026.pdf`, read via pypdf. GRAPE per-criterion **AUC 0.79/0.88/0.80** (confirmed AUCs, not accuracy). Split table (firsthand): eye-level 196 eyes → 156 train / 40 test; **PLR3 = 6 train / 2 test progressors** ✓. Frames the task as forecasting **future onset**; circularity marked "not assessed" rather than asserted. Authors Yi Huang, Lei Bi, Jinman Kim. |
| `luo2023harvardgdp` | venue-standard | ICCV 2023, pp. 20471–20482 (reused from the sibling paper's verified bib). |
| `koh2021wilds` | venue-standard | ICML 2021, PMLR 139:5637–5664 (reused from sibling). |
| `raji2021benchmark` | venue-standard | NeurIPS 2021 Datasets & Benchmarks Track; confirm final page range at camera-ready. |
| `fouda2025` | author-PDF | IEEE IMSA 2025; verified from the authors' PDF earlier (names "camera artifacts"; 70/30 split; ~4 PLR3 positives). **Confirm exact title/authors/pages/DOI from IEEE Xplore at camera-ready** (not web-indexed at check time). |

## Camera-ready TODO
1. `fouda2025`: confirm title/authors/pages/DOI from the IEEE Xplore IMSA-2025 record.
2. `saeedi2019`, `raji2021benchmark`, `luo2023harvardgdp`, `koh2021wilds`: confirm full author lists / final page ranges via CrossRef/DBLP.
3. Every `\cite` key resolves in the compiled bibliography (checked: 0 undefined citations in the tectonic run).
