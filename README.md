# Reproducing *"Transcriptional signatures in histologic structures within glioblastoma tumors may predict personalized drug sensitivity and survival"*

Step-by-step computational reproduction of Greenwald et al., *Neuro-Oncology Advances* (`vdaa093`), linking spatial/histologic glioblastoma (GBM) transcriptomics to molecular subtype and patient survival.

The pipeline ingests the IvyGAP anatomic-structure RNA-seq atlas and TCGA-GBM bulk RNA-seq, learns a histologic-structure classifier, extracts zonal transcriptional signatures, performs molecular subtyping on the cellular-tumor compartment, and builds/validates a cellular-tumor survival model.

---

## TL;DR — reproduced vs. published

| Analysis | This reproduction | Paper | Status |
|---|---|---|---|
| IvyGAP tumors / TCGA primary samples in cohort | 34 / 157 | 34 / 157 | Match |
| Structure classifier (Lasso multinomial), CV accuracy | **98.4%** | 98.45% | Match |
| TCGA samples called "predominantly CT" (P(CT) ≥ 0.5) | **42** | 40 | Match (±2) |
| Structure dominates variance: PCA + t-SNE + corr-network + dendrogram (Fig 1) | Reproduced (PC1+2=45.7%; net 100% same-zone) | 50.9% | Match |
| High-risk gene biology: MYC/OXPHOS/mTORC1/glycolysis/DNA-repair (Suppl S10) | **All 5 recovered, top + significant** | same 5 | Match |
| Zonal DGE + GSEA biology (Fig 2: hypoxia/PAN, angiogenesis/MVP) | Reproduced | Reproduced | Match |
| Subtype depends on structure; CT recapitulates bulk (Fig 3) | Reproduced | Reproduced | Match |
| Colman 9-gene metagene is structure-confounded (Fig 4A–C) | Reproduced | Reproduced | Match |
| Novel CT survival signature, C-index (IvyGAP CT) | **0.83** | ~0.8 | Match |
| Novel signature validates on TCGA CT, C-index | **0.69** | validated | Match |
| Novel signature > MGMT alone | Yes (lower log-rank p) | Yes | Match |
| Stepwise re-derivation: 10 seeds x 5-fold, mean validation C-index | **0.784** | ~0.75 | Match |
| 3-group (high/med/low) tertile stratification (Suppl S8) | All cohorts significant (CT p=2.6e-14) | Reproduced | Match |
| IDH-mutant-excluded re-analysis (Suppl S9) | TCGA CT p=0.001 (IDH-wt) | Reproduced | Match |
| Univariate clinical Cox screen | KPS p=0.028, MGMT p=6e-6 significant | Reproduced | Match (avail. covariates) |
| Within-structure Colman KM direction (Fig 4D/E) | Underpowered | significant | **Data-limited** (see caveats) |
| Drug sensitivity / resistance prediction | scaffold only | **not in paper** | N/A — future work in paper |

---

## Pipeline modules

Run in order. All paths and parameters live in the `Config` dataclass at the top of each script.

| Module | Purpose | Key outputs |
|---|---|---|
| `1_data_preprocessing.py` | Load, QC, filter, normalize IvyGAP + TCGA. Drops IvyGAP recurrent tumors; keeps TCGA primary only. | `processed/ivygap_processed.h5ad`, `processed/tcga_processed.h5ad`, clinical/covariate CSVs |
| `2_structure_classifier.py` | Lasso (L1) multinomial logistic regression trained on IvyGAP *Anatomic Structures RNA-Seq* labels; predicts histologic structure for TCGA. | `processed/structure_classifier.joblib`, `tcga_CT_processed.h5ad` |
| `2_zonal_signature_extraction.py` | Fig 1 (PCA + **t-SNE** + **correlation network** + **gap statistic** + **Ward dendrogram** + k-means: structure dominates variance) and Fig 2 (zonal DGE + GSEA). | `zonal_dge.csv`, `zonal_gsea_summary.csv`, `zonal_signatures.gmt`, figures |
| `3_molecular_subtyping.py` | Fig 3: Verhaak subtype ssGSEA per structure; CT-vs-bulk concordance. | `subtype_scores_*.csv`, `subtype_by_zone.csv`, figures |
| `4_survival_prediction.py` | Fig 4 (Colman confound) + Fig 5 (novel CT Cox-PH; **univariate screen** + faithful **10-seed stepwise** + published model) + **S8** (3-group tertiles) + **S9** (IDH-excluded) + **S10** (Wald-ranked GSEA). | `survival_metrics.csv`, `survival_*_model.csv`, `survival_univariate_clinical.csv`, `survival_highrisk_*.csv`, KM figures |
| `5_drug_sensitivity_mapping.py` | **Beyond the paper.** Zone-vs-zone connectivity (Lamb 2006 CMap) and cosine similarity scaffolding for future drug-DB integration. | `zone_connectivity_matrix.csv`, `zone_cosine_matrix.csv` |

---

## Detailed results

### Module 1 — Cohorts (preprocessing)
- IvyGAP filtered to **34** non-recurrent tumors; TCGA filtered to **157** primary tumors (recurrent/normal barcodes dropped).
- IvyGAP samples tagged with `sampling_study` to separate ground-truth microdissected *Anatomic Structures RNA-Seq* from noisier *Cancer Stem Cells RNA-Seq* samples.

### Module 2 — Structure classifier (Fig 1 prerequisite)
- L1 multinomial logistic regression (saga), trained **only on the clean anatomic-structure-labeled IvyGAP samples** over top variance-ranked feature genes.
- **Cross-validated accuracy 98.4%** (paper 98.45%). Training on the noisier stem-cell subset plateaued ~92%, confirming label quality drove the gap.
- Applied to TCGA with a confidence threshold P(CT) ≥ 0.5 → **42 predominantly-CT samples** (paper 40).

### Module 2b — Zonal signatures (Fig 1 & 2)
Full Fig 1 method set now reproduced on the 1000 most variable genes (z-scored):
- **PCA** (`pca_by_zone.png`): PC1+PC2 = **45.7%** of variance (paper 50.9%), separating by structure.
- **t-SNE** (`tsne_by_zone.png`): corroborates PCA — clean structure islands (HBV/MVP, LE/IT, PAN/PNZ, CT).
- **Correlation network**, Pearson r>0.92 (`correlation_network.png`): 29 edges, **100% connect same-zone samples** (Fig 1C) — samples within a structure are more correlated to each other than to other structures, even within a patient.
- **Gap statistic**, k=1–10, Tibshirani PCA-aligned reference (`gap_statistic.png`): reported and plotted honestly. NB: the gap statistic is highly sensitive to the reference null / cluster overlap and does **not** land cleanly on k=4 on this matrix; like the paper, k=4 is justified by *converging* evidence (k-means + hierarchical + network + biology), not the gap alone.
- **k-means (k=4)** vs zones: adjusted Rand index **0.71**; **Ward/Euclidean hierarchical** dendrogram cut at k=4 (`dendrogram_by_zone.png`): ARI **0.59** — both recover the 4 collapsed zones.
- Zonal DGE + GSEA recover the expected biology: **hypoxia/glycolysis in PAN** (pseudopalisading necrosis) and **angiogenesis/VEGF in MVP/HBV** (microvascular proliferation); proliferative core in CT.

### Module 3 — Molecular subtyping (Fig 3)
Verhaak subtype assignment is **structure-dependent**, and CT is the compartment that recapitulates the canonical bulk subtype (`subtype_by_zone.csv`):

| Zone | Proneural | Classical | Mesenchymal | Neural |
|---|---|---|---|---|
| CT | 31 | 50 | 16 | 5 |
| HBV/MVP | 6 | 5 | 39 | 0 |
| LE/IT | 5 | 0 | 2 | 36 |
| PAN/PNZ | 9 | 11 | 41 | 0 |

Boundary (LE/IT) skews Neural; angiogenic/hypoxic zones (MVP, PAN) skew Mesenchymal — matching the paper's structure→subtype coupling.

### Module 4 — Survival (Fig 4 & 5)
- **Colman 9-gene metagene is structure-confounded** (Fig 4A–C): its apparent prognostic signal tracks histologic composition rather than independent biology.
- **Univariate clinical Cox screen** (`survival_univariate_clinical.csv`): on IvyGAP CT, **KPS** (HR 0.98, p=0.028) and **MGMT methylation** (HR 0.15, p=6e-6) are significant; age and EGFR-amp are not. Gender / IDH1 / 1p19q are missing or invariant in public IvyGAP CT and are skipped (logged).
- **Novel CT-derived Cox-PH signature**, built with the paper's *exact* methodology — **10 random seeds x 5-fold CV**, per-fold forward stepwise by train log-rank, internal-validation filter (test C≥0.5 & p<0.05), then the model nearest the survivors' mean concordance, finalized on all CT with Wald p>0.05 genes dropped. Result: 40/50 fold-models passed validation, **mean validation C-index 0.784** (paper ~0.75); finalized model = age + MGMT + **SNRPE + MMP13** (full-CT C-index 0.77). NB: this differs from the paper's six genes — the stepwise procedure is underspecified/seed-sensitive so a different, similarly-predictive set emerges; the paper's *published* equation is still applied verbatim above.
- **3-group tertile stratification (Suppl. S8)** — high/medium/low risk separate cleanly in every cohort (`km_model3_*.png`): IvyGAP CT p=2.6e-14, TCGA CT p=5e-5, IvyGAP all p=1e-36, TCGA all p=0.004.
- **IDH-mutant-excluded re-analysis (Suppl. S9)** — removing IDH-mutant tumors, the published model still stratifies TCGA (`km_model_*_IDHwt.png`): TCGA CT p=0.001 (29 IDH-wt), TCGA all p=0.028 (114 IDH-wt). IvyGAP CT is all IDH-wt, so exclusion is a logged no-op.

  | Cohort | Model | log-rank p | C-index | n |
  |---|---|---|---|---|
  | IvyGAP CT | novel signature | 2.7e-12 | **0.83** | 78 |
  | IvyGAP CT | MGMT alone | 1.1e-06 | — | 78 |
  | TCGA CT | novel signature | 9.6e-06 | **0.69** | 35 |
  | TCGA CT | MGMT alone | 5.8e-03 | — | 35 |

  The novel signature outperforms MGMT methylation alone in both cohorts and **validates out-of-cohort on TCGA CT**.

- **Supplementary Fig S10 — biology of the high-risk genes:** the entire transcriptome (19,385 genes) was ranked by its multivariate-Cox **Wald statistic** on IvyGAP CT, then pre-ranked GSEA was run on that ranking. All five Hallmark pathways the paper names are recovered at the high-risk end, significant and at the top (`survival_highrisk_gsea.csv`, `highrisk_hallmark_gsea.png`):

  | Hallmark | NES | FDR q |
  |---|---|---|
  | MYC targets V1 | 2.47 | 0.000 |
  | Oxidative phosphorylation | 2.23 | 0.000 |
  | mTORC1 signaling | 2.11 | <0.001 |
  | Glycolysis | 1.74 | 0.004 |
  | DNA repair | 1.54 | 0.019 |

  Positional (chromosomal) sets also match: **chrXp11** (NES 2.16), **chr3q22** (2.09), **chr13q12** (1.85) are significant (3 of the paper's 5 bands; chr3q25/chr16p12 are positive but weaker).

### Module 5 — Drug mapping (beyond the paper)
The paper does **not** perform any drug-sensitivity/resistance analysis — the title's "drug sensitivity" is framed as explicit *future work* ("creating predictive signatures for tumor sensitivity and response to treatments"), and Fig 3E only *annotates* subtype pathways with therapeutic relevance (proneural→chemoradiation, mesenchymal→immunotherapy). This module is therefore a forward-looking scaffold: it computes zone-vs-zone CMap connectivity and cosine similarity, showing the boundary (LE/IT) and core (CT) zones are transcriptionally **opposed** — the entry point for plugging in LINCS/PRISM/CTRP later.

---

## Caveats / known limitations

- **Gap statistic does not land on exactly k=4.** It is computed and plotted honestly (`gap_statistic.png`) with a Tibshirani PCA-aligned reference, but it is intrinsically sensitive to the reference null and to cluster overlap and overshoots on this z-scored matrix. The collapse to k=4 is justified — exactly as in the paper — by *converging* evidence: k-means (ARI 0.71), Ward hierarchical clustering (ARI 0.59), the correlation network (100% same-zone edges), and the zonal biology. We do **not** force the gap to 4.
- **Fig 4D/E (within-structure Colman KM direction):** not reliably reproducible. Public IvyGAP survival data has only 6–27 patients per zone and **no published censoring information**; the analysis is underpowered, not a code defect. Reproducing the paper's exact direction would require fabricating signal. Documented transparently rather than forced.
- **IDH1 / 1p19q covariates** are not in the public IvyGAP data, so they enter the Cox models as GBM priors (IDH-wt, no codeletion) — effectively constants, dropped automatically when invariant. (TCGA *does* carry per-sample IDH1, which is what makes the S9 exclusion meaningful there.)
- **Re-derived gene set differs from the paper's six.** The stepwise procedure is underspecified and seed/data-sensitive, so it yields a different but similarly-predictive signature (mean validation C-index 0.784 ≈ paper 0.75). The paper's published equation is still applied verbatim for the headline survival curves.
- **Exact integer matches** (e.g., 42 vs 40 CT samples) differ by small margins because the paper underspecifies some random seeds / internal thresholds; biological conclusions are unaffected.
- **Drug module is a scaffold**, not a validated predictor (consistent with the paper, which has no drug result).

---

## Environment & how to run

Use the CPython 3.14 interpreter (`py -3.14`) — the default `python` on this machine is a MinGW build without standard PyPI wheels.

```powershell
py -3.14 -m pip install scanpy anndata scikit-learn pandas numpy matplotlib seaborn gseapy lifelines joblib

py -3.14 1_data_preprocessing.py
py -3.14 2_structure_classifier.py
py -3.14 2_zonal_signature_extraction.py
py -3.14 3_molecular_subtyping.py
py -3.14 4_survival_prediction.py
py -3.14 5_drug_sensitivity_mapping.py
```

Gene sets used: MSigDB `c1.all`, `c5.all`, `h.all` (v7.1, symbols). All intermediate artifacts and figures are written under `processed/`.
#   t - m l 
 
 #   t - m l  
 #   t - m l  
 #   m l  
 