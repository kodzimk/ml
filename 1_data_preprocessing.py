"""
1_data_preprocessing.py
========================
PHASE 1 of the replication pipeline for:

    Kersch, Claunch et al. (2020) "Transcriptional signatures in histologic
    structures within glioblastoma tumors may predict personalized drug
    sensitivity and survival." Neuro-Oncology Advances, doi:10.1093/noajnl/vdaa093

This script performs the *foundational* data engineering that everything else in
the project depends on:

    (1) Robust loading of the two independent cohorts:
            - IvyGAP  (Allen Brain Institute, anatomic-structure RNA-seq, FPKM)   -> n=34 tumors
            - TCGA-GBM (Genomic Data Commons, HTSeq/STAR-FPKM)                     -> n=156 cases
    (2) Mapping gene identifiers to a common HGNC gene-symbol space.
    (3) Attaching histologic-structure labels (LE / IT / CT / PNZ / PAN / HBV / MVP)
        AND the 4 collapsed transcriptionally-distinct zones used in the paper
        (CT | LE/IT | HBV/MVP | PAN/PNZ).
    (4) Quality control + the paper's EXACT low-expression gene filter
        ("genes whose mean across all samples falls below the lower quartile").
    (5) The 3 normalization representations the authors used interchangeably:
        FPKM, log2-transformed, and z-score-normalized.
    (6) Survival/clinical formatting into a clean (time, event) table ready for
        downstream Cox-PH / Kaplan-Meier modelling.

WHY a dedicated, paper-faithful preprocessing module?
The central thesis of the paper is that *histologic structure*, not algorithmic
batch artifact, is the dominant axis of variation in GBM transcriptomes. The
authors therefore deliberately AVOID cross-cohort batch correction (e.g. ComBat)
and instead (a) process each cohort independently and (b) lean on per-gene
z-scoring -- sometimes computed *within a single structure* -- to make samples
comparable. This script reproduces that philosophy faithfully so that the zonal
signatures extracted in module 2 are biological, not technical.

Outputs (written to ./processed/):
    - ivygap_processed.h5ad      (AnnData: samples x genes, FPKM/log2/zscore layers)
    - tcga_processed.h5ad        (AnnData: samples x genes, FPKM/log2/zscore layers)
    - ivygap_clinical.csv        (clean survival table keyed by sample)
    - tcga_clinical.csv          (clean survival table keyed by sample)

Primary stack: pandas, numpy, scanpy/anndata.
Author: GBM spatial-transcriptomics replication pipeline.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import scanpy as sc
import anndata as ad



@dataclass
class Config:
    ivy_fpkm: str = "gene_expression_matrix_2014-11-25/fpkm_table.csv"
    ivy_genes: str = "gene_expression_matrix_2014-11-25/rows-genes.csv"
    ivy_samples: str = "gene_expression_matrix_2014-11-25/columns-samples.csv"
    ivy_sample_details: str = "IvyGAP.rna_seq_samples_details.csv"
    ivy_tumor_details: str = "IvyGAP.tumor_details.csv"

    tcga_fpkm: str = "TCGA-GBM.star_fpkm.tsv"
    tcga_clinical: str = "TCGA-GBM.clinical.tsv"
    tcga_probemap: str = "gencode.v36.annotation.gtf.gene.probemap"
    tcga_mol_cov: str = "processed/tcga_molecular_covariates.csv"
    ivy_mol_cov: str = "processed/ivygap_molecular_covariates.csv"


    tcga_values_are_log2: bool | None = None 

    out_dir: str = "processed"

    low_expression_quantile: float = 0.25

    n_top_variable_genes: int = 1000

    compute_within_structure_zscore: bool = True

    ivy_drop_recurrent: bool = True

    tcga_primary_only: bool = True
    tcga_keep_sample_type_codes: tuple = ("01",)


CFG = Config()


def _exists(path: str) -> bool:
    ok = os.path.isfile(path)
    if not ok:
        warnings.warn(f"[MISSING FILE] {path} -- the relevant cohort will be skipped.")
    return ok


def _parse_age(value) -> float:
    if pd.isna(value):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) if m else np.nan


def _binarize_yes_no(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"yes", "y", "1", "true", "methylated", "positive", "pos", "mutant",
             "mutated", "amplified", "present"}:
        return 1.0
    if s in {"no", "n", "0", "false", "unmethylated", "not methylated", "negative",
             "neg", "wild type", "wildtype", "wt", "not amplified", "absent"}:
        return 0.0
    return np.nan


def _looks_like_log2(matrix: pd.DataFrame) -> bool:
    sample = matrix.values
    if sample.size > 200_000:  # subsample for speed
        flat = sample.ravel()
        flat = flat[np.isfinite(flat)]
        sample = np.random.default_rng(0).choice(flat, size=200_000, replace=False)
    finite = sample[np.isfinite(sample)]
    return float(np.nanmax(finite)) < 30.0


CANONICAL_STRUCTURES = ["LE", "IT", "CT", "PNZ", "PAN", "HBV", "MVP"]

ZONE_COLLAPSE = {
    "LE": "LE/IT",      # tumor invasion / leading edge  -> boundary
    "IT": "LE/IT",
    "CT": "CT",         # high cellularity neoplastic core (the paper's focus)
    "HBV": "HBV/MVP",   # vasculature / angiogenic zone
    "MVP": "HBV/MVP",
    "PAN": "PAN/PNZ",   # hypoxic / pseudopalisading necrosis zone
    "PNZ": "PAN/PNZ",
}


def classify_structure(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    t = text.lower()
    if "microvascular" in t or re.search(r"\bmvp\b", t) or "ctmvp" in t:
        return "MVP"
    if "hyperplastic blood vessel" in t or re.search(r"\bhbv\b", t) or "cthbv" in t:
        return "HBV"
    if "pseudopalisad" in t or re.search(r"\bpan\b", t) or "ctpan" in t:
        return "PAN"
    if "perinecro" in t or re.search(r"\bpnz\b", t) or "ctpnz" in t:
        return "PNZ"
    if "leading edge" in t or re.search(r"\ble\b", t) or t.startswith("le"):
        return "LE"
    if "infiltrat" in t or re.search(r"\bit\b", t) or t.startswith("it"):
        return "IT"
    if "cellular tumor" in t or re.search(r"\bct\b", t) or t.startswith("ct"):
        return "CT"
    return None


def collapse_to_gene_symbols(expr: pd.DataFrame) -> pd.DataFrame:
    if expr.index.duplicated().any():
        n_dup = int(expr.index.duplicated().sum())
        print(f"   - collapsing {n_dup} duplicate gene-symbol rows by mean")
        expr = expr.groupby(level=0).mean()
    return expr


def filter_low_expression(expr: pd.DataFrame, quantile: float) -> pd.DataFrame:
    gene_means = expr.mean(axis=1)
    threshold = gene_means.quantile(quantile)
    keep = gene_means >= threshold
    print(f"   - low-expression filter: threshold(mean FPKM @ q{quantile:.2f}) "
          f"= {threshold:.4f}; kept {int(keep.sum())}/{len(keep)} genes")
    return expr.loc[keep]


def build_anndata(expr: pd.DataFrame, obs: pd.DataFrame, cohort: str) -> ad.AnnData:
    X_fpkm = expr.T 
    common = [s for s in X_fpkm.index if s in obs.index]
    if len(common) < len(X_fpkm.index):
        warnings.warn(
            f"[{cohort}] {len(X_fpkm.index) - len(common)} expression samples had "
            f"no metadata and were dropped.")
    X_fpkm = X_fpkm.loc[common]
    obs = obs.loc[common]

    adata = ad.AnnData(
        X=X_fpkm.values.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=X_fpkm.columns),
    )
    adata.uns["cohort"] = cohort

    adata.layers["fpkm"] = adata.X.copy()

    adata.layers["log2"] = np.log2(adata.layers["fpkm"] + 1.0)

    log2 = adata.layers["log2"]
    mu = log2.mean(axis=0, keepdims=True)
    sd = log2.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0 
    adata.layers["zscore"] = (log2 - mu) / sd

    adata.X = adata.layers["log2"].copy()

    gene_var = pd.Series(adata.layers["log2"].var(axis=0), index=adata.var_names)
    top = gene_var.sort_values(ascending=False).head(CFG.n_top_variable_genes).index
    adata.var["highly_variable"] = adata.var_names.isin(top)
    adata.var["variance_log2"] = gene_var.values
    print(f"   - flagged top {CFG.n_top_variable_genes} most variable genes "
          f"(highly_variable=True)")

    return adata


def add_within_structure_zscore(adata: ad.AnnData, group_key: str = "structure") -> None:
    if group_key not in adata.obs:
        warnings.warn(f"[within-structure z-score] '{group_key}' not in obs; skipping.")
        return
    log2 = adata.layers["log2"]
    out = np.zeros_like(log2)
    for grp in adata.obs[group_key].dropna().unique():
        idx = np.where(adata.obs[group_key].values == grp)[0]
        if len(idx) < 2:  
            out[idx] = 0.0
            continue
        block = log2[idx]
        mu = block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, ddof=0, keepdims=True)
        sd[sd == 0] = 1.0
        out[idx] = (block - mu) / sd
    adata.layers["zscore_within_structure"] = out
    print(f"   - added 'zscore_within_structure' layer (grouped by {group_key})")


def load_ivygap(cfg: Config) -> ad.AnnData | None:
    print("\n=== IvyGAP cohort ===")
    if not (_exists(cfg.ivy_fpkm) and _exists(cfg.ivy_genes)):
        return None

    expr = pd.read_csv(cfg.ivy_fpkm, index_col=0)
    expr.index.name = "gene_id"
    expr.columns = expr.columns.astype(str)  
    print(f"   - raw expression matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")

    genes = pd.read_csv(cfg.ivy_genes)
    gid_col = "gene_id"
    sym_col = "gene_symbol" if "gene_symbol" in genes.columns else _guess_col(
        genes, ["symbol", "gene_name"])
    gid2sym = dict(zip(genes[gid_col].astype(str), genes[sym_col].astype(str)))
    expr.index = expr.index.astype(str).map(gid2sym)
    expr = expr.loc[expr.index.notna() & (expr.index != "nan")]
    expr = collapse_to_gene_symbols(expr)

    if cfg.ivy_drop_recurrent:
        drop = _ivygap_recurrent_sample_ids(cfg)
        if drop:
            before = expr.shape[1]
            expr = expr.loc[:, [c for c in expr.columns if c not in drop]]
            print(f"   - dropped {before - expr.shape[1]} recurrent-tumour samples "
                  f"(newly-diagnosed only); {expr.shape[1]} samples remain")

    expr = filter_low_expression(expr, cfg.low_expression_quantile)

    obs = _build_ivygap_obs(cfg, sample_ids=expr.columns)

    adata = build_anndata(expr, obs, cohort="IvyGAP")
    if cfg.compute_within_structure_zscore:
        add_within_structure_zscore(adata, group_key="structure")

    print(f"   - FINAL IvyGAP AnnData: {adata.n_obs} samples x {adata.n_vars} genes")
    _structure_summary(adata)
    return adata


def _build_ivygap_obs(cfg: Config, sample_ids) -> pd.DataFrame:
    details = None
    if _exists(cfg.ivy_sample_details):
        details = pd.read_csv(cfg.ivy_sample_details)
        key = "sample_id" if "sample_id" in details.columns else _guess_col(
            details, ["rna_well_id", "well_id"])
        details[key] = details[key].astype(str)
        details = details.drop_duplicates(subset=key).set_index(key)

    obs = pd.DataFrame(index=pd.Index([str(s) for s in sample_ids], name="sample_id"))

    if details is not None:
        details = details.reindex(obs.index)
        struct_src = details.get("structure_name")
        if struct_src is None or struct_src.isna().all():
            struct_src = details.get("structure_acronym")
        acr = details.get("structure_acronym")
        obs["structure"] = [
            classify_structure(n) or classify_structure(a)
            for n, a in zip(
                struct_src if struct_src is not None else [None] * len(obs),
                acr if acr is not None else [None] * len(obs),
            )
        ]
        obs["sampling_study"] = details.get("study_name")
        obs["tumor_name"] = details.get("tumor_name")
        obs["donor_id"] = details.get("donor_id")
        obs["molecular_subtype"] = details.get("molecular_subtype")
        obs["age"] = details.get("age_in_years").map(_parse_age) \
            if "age_in_years" in details.columns else np.nan
        obs["survival_days"] = pd.to_numeric(
            details.get("survival_days"), errors="coerce") \
            if "survival_days" in details.columns else np.nan
        obs["mgmt_methylated"] = details.get("mgmt_methylation").map(_binarize_yes_no) \
            if "mgmt_methylation" in details.columns else np.nan
        obs["kps"] = pd.to_numeric(details.get("initial_kps"), errors="coerce") \
            if "initial_kps" in details.columns else np.nan
        obs["egfr_amplified"] = details.get("egfr_amplification").map(_binarize_yes_no) \
            if "egfr_amplification" in details.columns else np.nan
        obs["idh1_mutant"] = details.get("idh1_mutation").map(_binarize_yes_no) \
            if "idh1_mutation" in details.columns else np.nan
    else:
        obs["structure"] = None

    obs["zone"] = obs["structure"].map(ZONE_COLLAPSE)

    if "tumor_name" in obs.columns:
        obs = _merge_derived_covariates(
            obs, cfg.ivy_mol_cov, obs_key="tumor_name", csv_key="tumor_name",
            fields={"idh1_mutant": "idh1_mutant", "codel_1p19q": "codel_1p19q"})

    obs["time_days"] = obs.get("survival_days")
    obs["event"] = obs["time_days"].notna().astype(int)
    if obs["event"].sum() == len(obs):
        warnings.warn(
            "[IvyGAP] No vital-status column found: assuming every recorded "
            "survival_days is an observed death (event=1). Override in module 4 "
            "if you have censoring information.")

    return obs


def _ivygap_recurrent_sample_ids(cfg: Config) -> set[str]:
    drop: set[str] = set()
    if _exists(cfg.ivy_sample_details):
        det = pd.read_csv(cfg.ivy_sample_details)
        key = "sample_id" if "sample_id" in det.columns else _guess_col(
            det, ["rna_well_id", "well_id"])
        if key is not None and "surgery" in det.columns:
            rec = det[det["surgery"].astype(str).str.lower().str.strip() == "recurrent"]
            drop.update(rec[key].astype(str).tolist())
        if "tumor_name" in det.columns and key is not None:
            mask = det["tumor_name"].astype(str).str.match(r"^W\d+-(?!1-)\d+-")
            drop.update(det.loc[mask, key].astype(str).tolist())
    if drop:
        print(f"   - identified {len(drop)} recurrent-tumour RNA samples to exclude")
    return drop


def load_tcga(cfg: Config) -> ad.AnnData | None:
    print("\n=== TCGA-GBM cohort ===")
    if not (_exists(cfg.tcga_fpkm) and _exists(cfg.tcga_probemap)):
        return None

    expr = pd.read_csv(cfg.tcga_fpkm, sep="\t", index_col=0)
    expr.index = expr.index.astype(str)
    expr.columns = expr.columns.astype(str)
    print(f"   - raw expression matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")

    if cfg.tcga_primary_only:
        keep = [c for c in expr.columns
                if _tcga_sample_type_code(c) in cfg.tcga_keep_sample_type_codes]
        if keep:
            dropped = expr.shape[1] - len(keep)
            expr = expr.loc[:, keep]
            print(f"   - kept {len(keep)} primary-tumour samples "
                  f"(type code {cfg.tcga_keep_sample_type_codes}); dropped {dropped} "
                  f"recurrent/normal specimens")

    is_log2 = cfg.tcga_values_are_log2
    if is_log2 is None:
        is_log2 = _looks_like_log2(expr)
    if is_log2:
        print("   - detected log2(FPKM+1) input -> converting back to linear FPKM "
              "so QC/normalization match the IvyGAP (FPKM) pipeline")
        expr = np.power(2.0, expr) - 1.0
        expr[expr < 0] = 0.0

    probemap = pd.read_csv(cfg.tcga_probemap, sep="\t")
    id_col = "id" if "id" in probemap.columns else probemap.columns[0]
    gene_col = "gene" if "gene" in probemap.columns else _guess_col(
        probemap, ["gene", "symbol", "name"])
    ens2sym = dict(zip(probemap[id_col].astype(str), probemap[gene_col].astype(str)))
    mapped = expr.index.map(ens2sym)
    if mapped.isna().mean() > 0.5:
        stripped = {k.split(".")[0]: v for k, v in ens2sym.items()}
        mapped = expr.index.to_series().str.split(".").str[0].map(stripped)
    expr.index = mapped
    expr = expr.loc[expr.index.notna() & (expr.index != "nan")]
    expr = collapse_to_gene_symbols(expr)

    expr = filter_low_expression(expr, cfg.low_expression_quantile)

    obs = _build_tcga_obs(cfg, sample_ids=expr.columns)

    adata = build_anndata(expr, obs, cohort="TCGA-GBM")
    print(f"   - FINAL TCGA AnnData: {adata.n_obs} samples x {adata.n_vars} genes")
    return adata


def _build_tcga_obs(cfg: Config, sample_ids) -> pd.DataFrame:
    obs = pd.DataFrame(index=pd.Index([str(s) for s in sample_ids], name="sample"))
    obs["patient_barcode"] = obs.index.str.slice(0, 12)  

    if not _exists(cfg.tcga_clinical):
        warnings.warn("[TCGA] clinical file missing; survival fields left as NaN.")
        for c in ["age", "gender", "vital_status", "time_days", "event",
                  "mgmt_methylated", "idh1_mutant"]:
            obs[c] = np.nan
        return obs

    clin = pd.read_csv(cfg.tcga_clinical, sep="\t", low_memory=False)

    samp_col = _guess_col(clin, ["sample", "submitter_id.samples"])
    case_col = _guess_col(clin, ["submitter_id", "case_submitter_id", "_PATIENT"])

    clin = clin.copy()
    if samp_col is not None:
        clin["_sample"] = clin[samp_col].astype(str)
    clin["_patient"] = clin[case_col].astype(str) if case_col else \
        clin.get("_sample", pd.Series(index=clin.index)).str.slice(0, 12)

    vital_col = _guess_col(clin, ["vital_status.demographic", "vital_status"])
    death_col = _guess_col(clin, ["days_to_death.demographic", "days_to_death"])
    follow_col = _guess_col(clin, ["days_to_last_follow_up.diagnoses",
                                   "days_to_last_follow_up"])
    age_col = _guess_col(clin, ["age_at_index.demographic",
                                "age_at_earliest_diagnosis_in_years.diagnoses.xena_derived",
                                "age_at_index"])
    gender_col = _guess_col(clin, ["gender.demographic", "gender"])

    join_on = "_sample" if "_sample" in clin.columns and \
        clin["_sample"].isin(obs.index).any() else "_patient"
    clin_keyed = clin.drop_duplicates(subset=join_on).set_index(join_on)

    obs_key = obs.index if join_on == "_sample" else obs["patient_barcode"]
    look = clin_keyed.reindex(obs_key.values)
    look.index = obs.index  

    obs["vital_status"] = look[vital_col].values if vital_col else np.nan
    obs["age"] = pd.to_numeric(look[age_col], errors="coerce").values if age_col else np.nan
    obs["gender"] = look[gender_col].values if gender_col else np.nan

    dead = obs["vital_status"].astype(str).str.lower().eq("dead")
    days_death = pd.to_numeric(look[death_col], errors="coerce").values if death_col else np.nan
    days_follow = pd.to_numeric(look[follow_col], errors="coerce").values if follow_col else np.nan
    obs["event"] = dead.astype(int).values
    obs["time_days"] = np.where(dead.values, days_death, days_follow)

    obs["mgmt_methylated"] = np.nan
    obs["idh1_mutant"] = np.nan
    obs = _merge_derived_covariates(
        obs, cfg.tcga_mol_cov, obs_key="patient_barcode", csv_key="patient_barcode",
        fields={"mgmt_methylated": "mgmt_methylated", "idh1_mutant": "idh1_mutant"})
    obs["structure"] = np.nan
    obs["zone"] = np.nan
    return obs


def _merge_derived_covariates(obs: pd.DataFrame, csv_path: str, obs_key: str,
                              csv_key: str, fields: dict[str, str]) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        warnings.warn(f"[molecular covariates] {csv_path} not found -- run "
                      f"download_molecular_covariates.py first to fetch these. "
                      f"Leaving {list(fields.values())} as-is.")
        return obs
    cov = pd.read_csv(csv_path, dtype=str)
    if csv_key not in cov.columns:
        warnings.warn(f"[derived covariates] key '{csv_key}' missing in {csv_path}.")
        return obs
    cov = cov.drop_duplicates(subset=csv_key).set_index(csv_key)

    keys = obs[obs_key] if obs_key in obs.columns else obs.index.to_series()
    look = cov.reindex(keys.values)
    look.index = obs.index
    for csv_col, obs_col in fields.items():
        if csv_col not in look.columns:
            continue
        vals = pd.to_numeric(look[csv_col], errors="coerce")
        if obs_col in obs.columns:
            obs[obs_col] = obs[obs_col].where(obs[obs_col].notna(), vals)
        else:
            obs[obs_col] = vals
    n_filled = int(pd.to_numeric(look[list(fields)[0]], errors="coerce").notna().sum())
    print(f"   - merged derived covariates from {os.path.basename(csv_path)} "
          f"({n_filled}/{len(obs)} matched on {obs_key})")
    return obs


def _tcga_sample_type_code(barcode: str) -> str:
    parts = str(barcode).split("-")
    if len(parts) >= 4 and len(parts[3]) >= 2 and parts[3][:2].isdigit():
        return parts[3][:2]
    return ""


def _guess_col(df: pd.DataFrame, candidates) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:  
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def _structure_summary(adata: ad.AnnData) -> None:
    if "structure" in adata.obs:
        vc = adata.obs["structure"].value_counts(dropna=False)
        print("   - structure counts:", dict(vc))
    if "zone" in adata.obs:
        vc = adata.obs["zone"].value_counts(dropna=False)
        print("   - collapsed-zone counts:", dict(vc))


def export(adata: ad.AnnData, name: str, cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    h5 = os.path.join(cfg.out_dir, f"{name}_processed.h5ad")
    adata.write_h5ad(h5)
    clin_cols = [c for c in
                 ["tumor_name", "donor_id", "patient_barcode", "structure", "zone",
                  "molecular_subtype", "age", "gender", "kps", "mgmt_methylated",
                  "idh1_mutant", "codel_1p19q", "egfr_amplified", "vital_status",
                  "time_days", "event"]
                 if c in adata.obs.columns]
    adata.obs[clin_cols].to_csv(os.path.join(cfg.out_dir, f"{name}_clinical.csv"))
    print(f"   - wrote {h5} and {name}_clinical.csv")


def main(cfg: Config = CFG) -> None:
    sc.settings.verbosity = 1
    print("Phase 1: data preprocessing (loading, QC, normalization)")
    print("=" * 70)

    ivy = load_ivygap(cfg)
    if ivy is not None:
        export(ivy, "ivygap", cfg)

    tcga = load_tcga(cfg)
    if tcga is not None:
        export(tcga, "tcga", cfg)

    print("\nDone. Processed AnnData + clinical tables are in ./%s/" % cfg.out_dir)


if __name__ == "__main__":
    main()
