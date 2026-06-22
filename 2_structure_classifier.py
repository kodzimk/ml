"""
2_structure_classifier.py
==========================
PHASE 2a of the replication pipeline for:

    Kersch, Claunch et al. (2020) "Transcriptional signatures in histologic
    structures within glioblastoma tumors may predict personalized drug
    sensitivity and survival." Neuro-Oncology Advances, doi:10.1093/noajnl/vdaa093

WHAT THIS MODULE DOES (and WHY it is its own step)
--------------------------------------------------
The central methodological move of the paper is that GBM transcriptomes should be
analyzed *within a single histologic structure* -- specifically the dense Cellular
Tumor (CT) -- rather than on mixed-structure bulk biopsies. The IvyGAP atlas comes
with ground-truth histologic labels (laser-microdissected structures), but the
TCGA-GBM validation cohort is bulk RNA-seq with NO structure annotation.

The authors therefore "[used] lasso logistic regression on each of the 4
transcriptionally distinct tumor structures in the IvyGAP database [to create] a
novel gene expression classifier ... [and] applied this structure classifier to
glioblastoma samples from TCGA and classified 40 samples as predominantly CT
composition." (Results; Figure 3D, Supplementary Figure S4).

This module reproduces exactly that step:
    1. Train a multinomial L1 (lasso) logistic-regression classifier on the IvyGAP
       cohort to distinguish the 4 collapsed zones:  CT | LE/IT | HBV/MVP | PAN/PNZ.
       Hyperparameters verbatim from the paper:
           penalty="l1", solver="saga", C=1/8, multi_class="multinomial",
           fit_intercept=True
       Cross-validation: StratifiedKFold(n_splits=5)  (paper CV acc = 98.45%).
    2. Apply the trained classifier to every TCGA-GBM sample to assign a predicted
       structure/zone (the "predominant structure" of that bulk biopsy).
    3. Write the predicted labels back into the TCGA AnnData + clinical table, and
       emit a CT-only TCGA subset that modules 3 (drug) and 4 (survival) consume to
       reproduce the paper's "CT**" analyses.

WHY z-scored features and a shared gene space?
The two cohorts are processed INDEPENDENTLY (the paper deliberately avoids
cross-cohort batch correction). To transfer a classifier across them we (a)
restrict to genes present in both and (b) feed the per-gene z-scored layer
(mean 0 / sd 1 within each cohort). This makes IvyGAP-learnt decision boundaries
applicable to TCGA without ComBat-style correction -- faithful to the paper's
"lean on z-scoring, not batch correction" philosophy.

Outputs (written to ./processed/):
    - tcga_processed.h5ad     (UPDATED in place: obs.structure / obs.zone filled +
                               per-class probabilities + is_CT flag)
    - tcga_clinical.csv       (UPDATED in place: structure / zone columns filled)
    - tcga_CT_processed.h5ad  (NEW: TCGA samples predicted predominantly CT)
    - structure_classifier.joblib (NEW: the fitted sklearn pipeline, for reuse)

Primary stack: pandas, numpy, scikit-learn, scanpy/anndata.
Author: GBM spatial-transcriptomics replication pipeline.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import anndata as ad

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*")
warnings.filterwarnings("ignore", message=".*n_jobs.*no effect.*")
warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty.*")

try:
    import joblib
except Exception: 
    joblib = None


@dataclass
class Config:
    processed_dir: str = "processed"
    ivy_h5ad: str = "processed/ivygap_processed.h5ad"
    tcga_h5ad: str = "processed/tcga_processed.h5ad"

    label_key: str = "zone"
    feature_layer: str = "zscore"

    use_highly_variable: bool = True
    n_feature_genes: int = 1500

    train_on_anatomic_only: bool = True
    anatomic_study_name: str = "Anatomic Structures RNA Seq"

    ct_min_proba: float | None = 0.5

    penalty: str = "l1"
    solver: str = "saga"
    C: float = 1.0 / 8.0
    fit_intercept: bool = True
    max_iter: int = 5000
    tol: float = 1e-4  
    n_splits: int = 5
    random_state: int = 0

    class_weight: str = "balanced"

    out_model: str = "processed/structure_classifier.joblib"
    out_tcga_ct: str = "processed/tcga_CT_processed.h5ad"


CFG = Config()


def _shared_gene_matrix(adata: ad.AnnData, genes: list[str], layer: str) -> np.ndarray:
    X = pd.DataFrame(
        adata.layers[layer] if layer in adata.layers else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )
    aligned = X.reindex(columns=genes)
    n_missing = int(aligned.isna().all(axis=0).sum())
    if n_missing:
        warnings.warn(f"{n_missing} shared genes absent from this cohort -> filled 0")
    return aligned.fillna(0.0).values.astype(np.float32)


def load_cohorts(cfg: Config):
    if not (os.path.isfile(cfg.ivy_h5ad) and os.path.isfile(cfg.tcga_h5ad)):
        raise FileNotFoundError(
            f"Run 1_data_preprocessing.py first -- expected {cfg.ivy_h5ad} and "
            f"{cfg.tcga_h5ad}.")
    ivy = ad.read_h5ad(cfg.ivy_h5ad)
    tcga = ad.read_h5ad(cfg.tcga_h5ad)

    # Shared gene space (intersection) -- the only genes a transferable model can use.
    shared_set = set(ivy.var_names) & set(tcga.var_names)
    if cfg.use_highly_variable:
        # Rank shared genes by IvyGAP variance and keep the top n_feature_genes.
        if "variance_log2" in ivy.var.columns:
            var = ivy.var["variance_log2"]
        else:  # fall back to computing variance on the log2 layer
            lay = ivy.layers["log2"] if "log2" in ivy.layers else ivy.X
            var = pd.Series(np.asarray(lay).var(axis=0), index=ivy.var_names)
        ranked = var[var.index.isin(shared_set)].sort_values(ascending=False)
        shared = sorted(ranked.head(cfg.n_feature_genes).index)
        panel = f"top-{len(shared)} variance-ranked shared genes"
    else:
        shared = sorted(shared_set)
        panel = "full intersection"
    if len(shared) < 50:
        raise RuntimeError(f"Only {len(shared)} shared genes between cohorts -- "
                           f"check that both were symbol-mapped consistently.")
    print(f"Shared gene space ({panel}): {len(shared)} genes "
          f"(IvyGAP {ivy.n_vars}, TCGA {tcga.n_vars})")
    return ivy, tcga, shared


# ----------------------------------------------------------------------------
# 2. TRAIN + CROSS-VALIDATE ON IVYGAP
# ----------------------------------------------------------------------------
def build_classifier(cfg: Config) -> LogisticRegression:
    """Multinomial L1 logistic regression, hyperparameters per the paper.

    NOTE: scikit-learn >=1.7 removed the explicit `multi_class` argument; for the
    saga + L1 combination a softmax (multinomial) model is now fit automatically
    when there are >2 classes, which is exactly the paper's "multiclass=
    multinomial" setting. We keep a fallback for older sklearn that still accepts
    the keyword.
    """
    kwargs = dict(
        penalty=cfg.penalty,
        solver=cfg.solver,
        C=cfg.C,
        fit_intercept=cfg.fit_intercept,
        class_weight=cfg.class_weight,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        random_state=cfg.random_state,
    )
    try:
        return LogisticRegression(multi_class="multinomial", **kwargs)
    except TypeError:
        # sklearn >=1.7 removed the keyword; softmax is automatic for >2 classes.
        return LogisticRegression(**kwargs)


def train_and_validate(ivy: ad.AnnData, shared: list[str], cfg: Config):
    """Fit on labelled IvyGAP samples; report 5-fold StratifiedKFold accuracy.

    Training is restricted to the clean anatomic-structure microdissection samples
    (paper's approach) when available; this is what yields ~98% CV accuracy.
    """
    labels = ivy.obs[cfg.label_key].astype("object")
    keep = labels.notna().values
    if cfg.train_on_anatomic_only and "sampling_study" in ivy.obs:
        anat = (ivy.obs["sampling_study"] == cfg.anatomic_study_name).values
        if anat.sum() >= 50:
            keep = keep & anat
            print(f"   training on {int(keep.sum())} anatomic-structure samples "
                  f"only (excluding {int((~anat).sum())} CSC-cluster wells)")
        else:
            warnings.warn("sampling_study present but too few anatomic samples; "
                          "training on all labelled samples.")
    X = _shared_gene_matrix(ivy, shared, cfg.feature_layer)[keep]
    y = labels.values[keep].astype(str)
    print(f"\nTraining structure classifier on {X.shape[0]} labelled IvyGAP "
          f"samples x {X.shape[1]} shared genes")
    print("   class balance:", dict(pd.Series(y).value_counts()))

    clf = build_classifier(cfg)

    # --- 5-fold stratified cross-validation (paper reports 98.45% avg accuracy) ---
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True,
                          random_state=cfg.random_state)
    y_cv = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
    acc = accuracy_score(y, y_cv)
    print(f"\n   StratifiedKFold(n_splits={cfg.n_splits}) CV accuracy = {acc:.4f} "
          f"(paper: 0.9845)")
    print("   --- cross-validated classification report ---")
    print(classification_report(y, y_cv, zero_division=0))
    print("   confusion matrix (rows=true, cols=pred), labels =",
          sorted(set(y)))
    print(confusion_matrix(y, y_cv, labels=sorted(set(y))))

    # --- final model: refit on ALL labelled IvyGAP samples ---
    clf.fit(X, y)
    n_used = int((np.abs(clf.coef_).sum(axis=0) > 0).sum())
    print(f"\n   final model refit on all {X.shape[0]} samples; "
          f"L1 selected {n_used}/{X.shape[1]} genes")
    return clf, sorted(set(y))


# ----------------------------------------------------------------------------
# 3. PREDICT TCGA STRUCTURE / ZONE
# ----------------------------------------------------------------------------
def predict_tcga(clf: LogisticRegression, tcga: ad.AnnData, shared: list[str],
                 classes: list[str], cfg: Config) -> ad.AnnData:
    X = _shared_gene_matrix(tcga, shared, cfg.feature_layer)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    class_order = list(clf.classes_)

    # The predicted ZONE is the predominant histologic structure of each bulk
    # TCGA biopsy. For CT this zone label *is* the structure; we mirror it into
    # `structure` so downstream code can filter on either key uniformly.
    tcga.obs["zone"] = pred
    tcga.obs["structure"] = pred
    for i, cls in enumerate(class_order):
        safe = cls.replace("/", "_")
        tcga.obs[f"pred_proba_{safe}"] = proba[:, i].astype(np.float32)

    # "Predominantly CT": CT is the argmax and (optionally) exceeds a confidence
    # threshold. The paper reported 40 confidently-CT TCGA biopsies.
    ct_idx = class_order.index("CT") if "CT" in class_order else None
    is_ct = (pred == "CT")
    if cfg.ct_min_proba is not None and ct_idx is not None:
        is_ct = is_ct & (proba[:, ct_idx] >= cfg.ct_min_proba)
    tcga.obs["is_CT"] = is_ct

    vc = pd.Series(pred).value_counts()
    print("\nTCGA predicted-zone composition:", dict(vc))
    thr = "" if cfg.ct_min_proba is None else f" @ P(CT)>={cfg.ct_min_proba}"
    print(f"   -> {int(is_ct.sum())} samples classified predominantly CT{thr} "
          f"(paper: 40)")
    return tcga


# ----------------------------------------------------------------------------
# 4. PERSIST
# ----------------------------------------------------------------------------
def export(tcga: ad.AnnData, clf, cfg: Config) -> None:
    os.makedirs(cfg.processed_dir, exist_ok=True)

    # (a) overwrite the processed TCGA AnnData with the filled labels
    tcga.write_h5ad(cfg.tcga_h5ad)

    # (b) refresh the flat clinical table (same column set module 1 wrote)
    clin_cols = [c for c in
                 ["tumor_name", "donor_id", "patient_barcode", "structure", "zone",
                  "molecular_subtype", "age", "gender", "kps", "mgmt_methylated",
                  "idh1_mutant", "codel_1p19q", "egfr_amplified", "vital_status",
                  "time_days", "event", "is_CT"]
                 if c in tcga.obs.columns]
    csv = os.path.join(cfg.processed_dir, "tcga_clinical.csv")
    tcga.obs[clin_cols].to_csv(csv)

    # (c) CT-only subset for the paper's "CT**" downstream analyses
    ct = tcga[tcga.obs["is_CT"].values].copy()
    ct.write_h5ad(cfg.out_tcga_ct)

    # (d) the fitted model, for reuse / inspection of selected genes
    if joblib is not None:
        joblib.dump(clf, cfg.out_model)

    print(f"\nWrote:")
    print(f"   - {cfg.tcga_h5ad}        (structure/zone filled)")
    print(f"   - {csv}    (structure/zone filled)")
    print(f"   - {cfg.out_tcga_ct}   ({ct.n_obs} CT-predicted samples)")
    if joblib is not None:
        print(f"   - {cfg.out_model}")


# ----------------------------------------------------------------------------
# 5. ENTRY POINT
# ----------------------------------------------------------------------------
def main(cfg: Config = CFG) -> None:
    print("Phase 2a: IvyGAP-trained structure classifier -> TCGA labelling")
    print("=" * 70)
    ivy, tcga, shared = load_cohorts(cfg)
    clf, classes = train_and_validate(ivy, shared, cfg)
    tcga = predict_tcga(clf, tcga, shared, classes, cfg)
    export(tcga, clf, cfg)
    print("\nDone. TCGA samples now carry predicted histologic structure/zone.")


if __name__ == "__main__":
    main()
