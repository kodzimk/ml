"""
4_survival_prediction.py
=========================
PHASE 4 of the replication pipeline for:

    Kersch, Claunch et al. (2020) "Transcriptional signatures in histologic
    structures within glioblastoma tumors may predict personalized drug
    sensitivity and survival." Neuro-Oncology Advances, doi:10.1093/noajnl/vdaa093

This module reproduces the paper's HEADLINE result (Figure 5): a novel prognostic
gene signature created EXCLUSIVELY from Cellular-Tumor (CT) transcriptomics that
stratifies glioblastoma patients by survival better than MGMT methylation alone,
and validates on an independent cohort (TCGA).

It does this two complementary ways:

  (1) PUBLISHED MODEL (exact reproduction).
      The paper reports the final risk-score equation verbatim (Fig 5A):

        Risk score = (-5.386 x MGMT) + (0.181 x Age)
                   + (2.378 x PGAM4) + (2.391 x ETNK2) + (1.892 x MIA)
                   + (1.413 x GMPS) + (2.784 x BCL7B) + (-0.910 x IBSP)

        where MGMT = 1 if methylated else 0, Age in years, and each gene is
        normalized (z-scored) expression. RR = exp(risk score); RR>1 => high risk
        of short overall survival.

      We apply this fixed equation to IvyGAP CT, TCGA CT-classified, and the full
      cohorts, then risk-stratify by the paper's rule (tertiles of the hazard
      ratio: high-risk if HR > quantile(2/3)) and compare Kaplan-Meier curves to
      MGMT-status-alone stratification (Fig 5B-E).

  (2) RE-DERIVED MODEL (methodology reproduction).
      Following the paper's Methods, we fit multivariate Cox PH on IvyGAP CT
      samples adjusting for the known prognostic factors (age, MGMT, IDH1),
      screen candidate genes (Wald p < 0.05), and run cross-validated forward
      stepwise selection to build a compact risk model from scratch -- showing
      the published 6-gene signature is recoverable from CT expression.

WHY CT-only? The paper's thesis: mixed-structure or non-CT biopsies confound
prognostic signatures (Fig 4). Building the model on pure CT removes histologic
confounding, so the signature reflects tumor-cell biology rather than how much
vasculature/necrosis/normal-brain happened to be in the biopsy.

Outputs (./processed/ and ./processed/figures/):
    - survival_published_scores.csv     (per-sample risk score / HR / risk group)
    - survival_rederived_model.csv      (selected genes + coefficients)
    - survival_metrics.csv              (log-rank p, concordance per cohort/model)
    - figures/km_*.png                  (Kaplan-Meier curves, Fig 5B-E style)

Primary stack: lifelines (Cox PH, KM, log-rank), pandas, numpy, matplotlib.
Author: GBM spatial-transcriptomics replication pipeline.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index


# ----------------------------------------------------------------------------
# ESTABLISHED PROGNOSTIC SIGNATURE (Colman et al. 2010, Neuro-Oncology 12:49-57)
# ----------------------------------------------------------------------------
# The 9-gene "multigene predictor of outcome in glioblastoma" used in Figure 4.
# 7 genes correlate with DECREASED survival (poor), 2 with IMPROVED survival.
COLMAN_POOR_GENES = ["AQP1", "CHI3L1", "EMP3", "GPNMB", "IGFBP2", "LGALS3", "PDPN"]
COLMAN_GOOD_GENES = ["OLIG2", "RTN1"]


# ----------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------
@dataclass
class Config:
    ivy_h5ad: str = "processed/ivygap_processed.h5ad"
    tcga_h5ad: str = "processed/tcga_processed.h5ad"
    out_dir: str = "processed"
    fig_dir: str = "processed/figures"

    # ---- The published final model (Figure 5A), verbatim ----
    # Intercept-free linear predictor; gene terms use z-scored expression.
    published_gene_coefs: dict = field(default_factory=lambda: {
        "PGAM4": 2.378, "ETNK2": 2.391, "MIA": 1.892,
        "GMPS": 1.413, "BCL7B": 2.784, "IBSP": -0.910,
    })
    published_mgmt_coef: float = -5.386   # MGMT methylated (1) is protective
    published_age_coef: float = 0.181     # older age is higher risk

    # ---- Risk stratification rule (Fig 5 caption) ----
    # "Tertiles of HR values were used to risk stratify (high-risk: HR >
    #  quantile(2/3); low-risk: HR < quantile(2/3))."
    high_risk_quantile: float = 2.0 / 3.0

    # ---- Univariate clinical screen (paper Methods: "Univariate analysis") ----
    # Paper screens age, gender, MGMT, IDH1, 1p19q, KPS in IvyGAP CT. We run
    # univariate Cox on whichever of these actually VARY in the public data
    # (gender/IDH1/1p19q are missing or constant for IvyGAP CT and are skipped).
    univariate_covariates: tuple = ("age", "gender", "kps", "mgmt_methylated",
                                    "idh1_mutant", "codel_1p19q", "egfr_amplified")

    # ---- Re-derivation (paper Methods) ----
    rederive: bool = True
    # Screen candidate genes among the highly-variable panel for tractability
    # (fitting a Cox model per gene over ~19k genes is needlessly slow and the
    # paper's signature genes are all variable). Adjust covariates as in the paper.
    rederive_adjust: tuple = ("age", "mgmt_methylated", "idh1_mutant")
    rederive_screen_p: float = 0.05
    rederive_max_genes: int = 10           # paper stops at concordance=1 or 10 genes
    rederive_candidate_pool: int = 20       # top genes by screen p-value to consider
    rederive_cv_folds: int = 5
    # Paper uses 10 random seeds, each with 5-fold CV; per fold a stepwise model is
    # built on train and validated on test. ~10*5*pool*max_genes Cox fits.
    rederive_seeds: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # Internal validation (paper): keep fold-models with test concordance >= 0.5
    # and log-rank p < 0.05, then pick the model whose concordance is nearest the
    # mean of the survivors (the paper reports that mean was ~0.75).
    rederive_min_test_concordance: float = 0.5
    rederive_max_test_p: float = 0.05

    # ---- 3-group risk stratification (Suppl. Fig S8) ----
    # Tertiles: low < q(1/3) <= medium < q(2/3) <= high.
    tertile_low: float = 1.0 / 3.0
    tertile_high: float = 2.0 / 3.0
    # ---- IDH-mutant-excluded re-analysis (Suppl. Fig S9) ----
    idh_excluded: bool = True

    # ---- Supplementary Figure S10: high-risk gene biology ----
    # The paper "ranked the entire transcriptome in order of the Wald statistic
    # calculated during multivariate Cox regression" and ran GSEA on that ranking.
    # We reproduce this with pre-ranked GSEA against Hallmark (H) and Positional
    # (C1) collections, expecting OXPHOS / MYC / MTORC1 / glycolysis / DNA-repair
    # and chromosomal-position sets among the high-risk (positive-Wald) end.
    s10: bool = True
    s10_hallmark_gmt: str = "h.all.v7.1.symbols.gmt"
    s10_positional_gmt: str = "c1.all.v7.1.symbols.gmt"
    s10_gene_panel: str = "all"          # "all" (paper) or "hv" (faster panel)
    s10_min_size: int = 15
    s10_max_size: int = 500
    s10_permutations: int = 1000
    s10_top_n: int = 20

    random_state: int = 0


CFG = Config()


# ----------------------------------------------------------------------------
# 1. HELPERS
# ----------------------------------------------------------------------------
def _zscore_genes(adata: ad.AnnData, genes: list[str]) -> pd.DataFrame:
    """Return a samples x genes DataFrame of per-gene z-scored expression.

    Z-scoring is computed over the *given* sample subset using the log2 layer --
    i.e. "normalized expression" within the cohort being scored, matching the
    paper's per-analysis normalization (e.g. the '*CT normalized' columns).
    """
    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        warnings.warn(f"genes absent from {adata.uns.get('cohort','?')}: {missing}")
    log2 = pd.DataFrame(
        adata[:, present].layers["log2"] if "log2" in adata.layers
        else adata[:, present].X,
        index=adata.obs_names, columns=present)
    mu = log2.mean(axis=0)
    sd = log2.std(axis=0, ddof=0).replace(0, 1.0)
    return (log2 - mu) / sd


def _survival_frame(adata: ad.AnnData) -> pd.DataFrame:
    """Pull a clean (time_days, event) + covariate frame, dropping unusable rows."""
    df = adata.obs.copy()
    df["time_days"] = pd.to_numeric(df.get("time_days"), errors="coerce")
    df["event"] = pd.to_numeric(df.get("event"), errors="coerce")
    df = df[(df["time_days"] > 0) & df["event"].notna()]
    return df


def _km_plot(groups: dict[str, pd.DataFrame], title: str, path: str,
             p: float | None = None) -> None:
    """Kaplan-Meier curves for a dict of {label -> frame(time_days,event)}."""
    fig, axk = plt.subplots(figsize=(6.5, 5.5))
    kmf = KaplanMeierFitter()
    colors = {"High risk": "#d62728", "Low risk": "#1f77b4",
              "Medium risk": "#ff7f0e",
              "Unmethylated": "#d62728", "Methylated": "#1f77b4"}
    for label, g in groups.items():
        if len(g) == 0:
            continue
        kmf.fit(g["time_days"] / 365.25, g["event"], label=f"{label} (n={len(g)})")
        kmf.plot_survival_function(ax=axk, ci_show=True,
                                   color=colors.get(label, None))
    axk.set_xlabel("Time (years)")
    axk.set_ylabel("Survival probability")
    ttl = title if p is None else f"{title}\nlog-rank p = {p:.4g}"
    axk.set_title(ttl)
    axk.set_ylim(0, 1.02)
    axk.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   wrote {path}")


# ----------------------------------------------------------------------------
# 2. PUBLISHED MODEL (exact Fig 5A equation)
# ----------------------------------------------------------------------------
def apply_published_model(adata: ad.AnnData, cfg: Config, cohort_label: str
                          ) -> pd.DataFrame:
    """Compute the paper's risk score / hazard ratio for every usable sample."""
    genes = list(cfg.published_gene_coefs)
    z = _zscore_genes(adata, genes)

    df = _survival_frame(adata)
    z = z.reindex(df.index)

    # MGMT term needs a methylation call; age term needs age. Rows lacking either
    # cannot be scored by the published equation -> drop (documented).
    age = pd.to_numeric(df.get("age"), errors="coerce")
    mgmt = pd.to_numeric(df.get("mgmt_methylated"), errors="coerce")
    usable = age.notna() & mgmt.notna() & z.notna().all(axis=1)
    df, z, age, mgmt = df[usable], z[usable], age[usable], mgmt[usable]

    score = (cfg.published_mgmt_coef * mgmt) + (cfg.published_age_coef * age)
    for g, coef in cfg.published_gene_coefs.items():
        score = score + coef * z[g]

    out = pd.DataFrame({
        "cohort": cohort_label,
        "time_days": df["time_days"].values,
        "event": df["event"].astype(int).values,
        "mgmt_methylated": mgmt.values,
        "risk_score": score.values,
        # HR relative to the cohort baseline; only the ordering matters for KM.
        "hazard_ratio": np.exp(score.values - np.nanmean(score.values)),
    }, index=df.index)

    thr = out["hazard_ratio"].quantile(cfg.high_risk_quantile)
    out["risk_group"] = np.where(out["hazard_ratio"] > thr, "High risk", "Low risk")
    print(f"   [{cohort_label}] scored {len(out)} samples; "
          f"{(out['risk_group']=='High risk').sum()} high-risk "
          f"(HR>q{cfg.high_risk_quantile:.2f})")
    return out


# ----------------------------------------------------------------------------
# 3. KAPLAN-MEIER EVALUATION (model vs MGMT-alone baseline)
# ----------------------------------------------------------------------------
def evaluate_cohort(scores: pd.DataFrame, cohort_label: str, cfg: Config,
                    metrics: list) -> None:
    """KM + log-rank for (a) the model's risk groups and (b) MGMT-alone."""
    # ---- (a) model risk groups ----
    hi = scores[scores["risk_group"] == "High risk"]
    lo = scores[scores["risk_group"] == "Low risk"]
    lr = logrank_test(hi["time_days"], lo["time_days"],
                      event_observed_A=hi["event"], event_observed_B=lo["event"])
    # concordance: higher risk score should mean shorter survival -> negate score.
    try:
        c_index = concordance_index(scores["time_days"], -scores["risk_score"],
                                    scores["event"])
    except Exception:
        c_index = np.nan
    safe = cohort_label.replace(" ", "_").replace("/", "_")
    _km_plot({"High risk": hi, "Low risk": lo},
             f"{cohort_label}: novel CT prognostic model",
             os.path.join(cfg.fig_dir, f"km_model_{safe}.png"),
             p=lr.p_value)
    metrics.append({"cohort": cohort_label, "model": "novel_CT_signature",
                    "logrank_p": lr.p_value, "concordance": c_index,
                    "n": len(scores)})
    print(f"   [{cohort_label}] novel model: log-rank p={lr.p_value:.4g}, "
          f"C-index={c_index:.3f}")

    # ---- (b) MGMT-alone baseline (methylated=low, unmethylated=high) ----
    has_mgmt = scores["mgmt_methylated"].notna()
    sm = scores[has_mgmt]
    if sm["mgmt_methylated"].nunique() == 2:
        meth = sm[sm["mgmt_methylated"] == 1]
        unmeth = sm[sm["mgmt_methylated"] == 0]
        lr_m = logrank_test(unmeth["time_days"], meth["time_days"],
                            event_observed_A=unmeth["event"],
                            event_observed_B=meth["event"])
        _km_plot({"Unmethylated": unmeth, "Methylated": meth},
                 f"{cohort_label}: MGMT status alone",
                 os.path.join(cfg.fig_dir, f"km_mgmt_{safe}.png"),
                 p=lr_m.p_value)
        metrics.append({"cohort": cohort_label, "model": "MGMT_alone",
                        "logrank_p": lr_m.p_value, "concordance": np.nan,
                        "n": len(sm)})
        print(f"   [{cohort_label}] MGMT alone: log-rank p={lr_m.p_value:.4g}")


# ----------------------------------------------------------------------------
# 3b. UNIVARIATE CLINICAL COX SCREEN (paper Methods: "Univariate analysis")
# ----------------------------------------------------------------------------
def _encode_clinical(series: pd.Series) -> pd.Series:
    """Coerce a clinical covariate to numeric (encode gender male=1/female=0)."""
    s = series.copy()
    if s.dtype == object:
        low = s.astype(str).str.lower()
        if low.isin(["male", "female", "m", "f"]).any():
            return low.map({"male": 1, "m": 1, "female": 0, "f": 0})
    return pd.to_numeric(s, errors="coerce")


def univariate_clinical_cox(ivy_ct: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Univariate Cox PH for each clinical covariate on IvyGAP CT (paper Methods).

    The paper screens age, gender, MGMT, IDH1, 1p19q and KPS univariately. We fit
    a single-covariate Cox model per variable on the CT samples with survival data
    and report HR, Wald statistic (z) and p. Covariates that are missing or
    invariant in the public IvyGAP CT data (gender, IDH1, 1p19q) are skipped and
    logged, so the table reflects exactly what the data can support.
    """
    print("\n=== Univariate clinical Cox screen (IvyGAP CT) ===")
    df = _survival_frame(ivy_ct)
    rows = []
    for cov in cfg.univariate_covariates:
        if cov not in df.columns:
            print(f"     -- {cov}: absent from data -- skipped")
            continue
        x = _encode_clinical(df[cov])
        sub = pd.DataFrame({"time_days": df["time_days"], "event": df["event"],
                            cov: x}).dropna()
        if sub[cov].nunique() < 2 or len(sub) < 10:
            print(f"     -- {cov}: invariant/insufficient (n={len(sub)}, "
                  f"levels={sub[cov].nunique()}) -- skipped")
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="time_days", event_col="event")
            r = cph.summary.loc[cov]
            rows.append({"covariate": cov, "n": len(sub),
                         "hazard_ratio": r["exp(coef)"], "coef": r["coef"],
                         "wald_z": r["z"], "p": r["p"]})
            print(f"     OK {cov:16s} HR={r['exp(coef)']:.3f}  z={r['z']:+.2f}  "
                  f"p={r['p']:.4g}  (n={len(sub)})")
        except Exception as exc:  # noqa: BLE001
            print(f"     -- {cov}: Cox fit failed ({exc})")
    out = pd.DataFrame(rows)
    if not out.empty:
        path = os.path.join(cfg.out_dir, "survival_univariate_clinical.csv")
        out.to_csv(path, index=False)
        print(f"   wrote {path} ({len(out)} covariates)")
    return out


# ----------------------------------------------------------------------------
# 4. RE-DERIVED MODEL (paper methodology, from scratch on IvyGAP CT)
# ----------------------------------------------------------------------------
def rederive_model(ivy_ct: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Reproduce the paper's stepwise multivariate Cox model build on CT data.

    Steps (faithful to Methods):
      1. Restrict to the highly-variable gene panel (tractable candidate space).
      2. For each gene, fit a multivariate Cox PH (gene + age + MGMT + IDH1) on
         IvyGAP CT; keep genes with Wald p < 0.05 (HR != 1) as candidates.
      3. Cross-validated forward stepwise selection: starting from the base
         clinical model, iteratively add the candidate gene that most improves
         the mean cross-validated concordance, until no improvement or
         `rederive_max_genes` reached.
    Returns the selected genes with their final multivariate coefficients (HRs).
    """
    print("\n=== Re-derived CT Cox model (paper methodology) ===")
    df = _survival_frame(ivy_ct)
    base_cov = [c for c in cfg.rederive_adjust
                if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    for c in base_cov:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop IDH1 if it is constant (IvyGAP GBM is overwhelmingly IDH-wt -> no signal).
    base_cov = [c for c in base_cov if df[c].nunique(dropna=True) > 1]
    df = df.dropna(subset=base_cov)
    print(f"   IvyGAP CT usable for modelling: {len(df)} samples; "
          f"base covariates = {base_cov}")

    # candidate gene pool: highly-variable, present in data
    hv = ivy_ct.var_names[ivy_ct.var["highly_variable"].values] \
        if "highly_variable" in ivy_ct.var else ivy_ct.var_names
    zall = _zscore_genes(ivy_ct, list(hv)).reindex(df.index)

    # ---- 2. univariate(adjusted) screen ----
    screen = []
    for g in zall.columns:
        sub = df[base_cov + ["time_days", "event"]].copy()
        sub[g] = zall[g].values
        if sub[g].std(ddof=0) == 0 or sub[g].isna().any():
            continue
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(sub, duration_col="time_days", event_col="event",
                    robust=False)
            p = cph.summary.loc[g, "p"]
            coef = cph.summary.loc[g, "coef"]
            screen.append((g, p, coef))
        except Exception:
            continue
    screen_df = pd.DataFrame(screen, columns=["gene", "p", "coef"]).dropna()
    candidates = (screen_df[screen_df["p"] < cfg.rederive_screen_p]
                  .sort_values("p").head(cfg.rederive_candidate_pool)["gene"].tolist())
    print(f"   screened {len(screen_df)} genes; {len(candidates)} candidates "
          f"(p<{cfg.rederive_screen_p}); pool capped at {cfg.rederive_candidate_pool}")
    if not candidates:
        warnings.warn("No significant candidate genes found; skipping re-derivation.")
        return pd.DataFrame()

    # ---- 3. per-fold forward stepwise + internal validation (paper Methods) ----
    # For each of 10 seeds we 5-fold split CT; on each TRAIN fold we forward-select
    # genes (adding the candidate with the highest train log-rank score), then
    # validate the resulting model on the held-out TEST fold (concordance + log-
    # rank p). We keep fold-models passing the internal-validation filter and pick
    # the one whose test concordance is nearest the survivors' mean (~0.75 in the
    # paper). The chosen gene set is then finalized on all CT samples.
    from sklearn.model_selection import StratifiedKFold
    y = df["event"].astype(int).values
    fold_models = []
    for seed in cfg.rederive_seeds:
        skf = StratifiedKFold(n_splits=cfg.rederive_cv_folds, shuffle=True,
                              random_state=seed)
        for tr, te in skf.split(df, y):
            train, test = df.iloc[tr], df.iloc[te]
            if train["event"].sum() < 3 or test["event"].sum() < 2:
                continue
            genes = _forward_select_on_train(train, base_cov, zall, candidates, cfg)
            if not genes:
                continue
            tc, tp = _evaluate_fold_model(train, test, base_cov, zall, genes)
            if np.isnan(tc):
                continue
            fold_models.append({"seed": seed, "genes": genes,
                                "test_concordance": tc, "test_logrank_p": tp})
    if not fold_models:
        warnings.warn("Stepwise produced no usable fold-models; skipping.")
        return pd.DataFrame()
    print(f"   built {len(fold_models)} fold-models across "
          f"{len(cfg.rederive_seeds)} seeds x {cfg.rederive_cv_folds} folds")

    # ---- internal validation filter + nearest-mean-concordance selection ----
    survivors = [m for m in fold_models
                 if m["test_concordance"] >= cfg.rederive_min_test_concordance
                 and m["test_logrank_p"] < cfg.rederive_max_test_p]
    pool = survivors if survivors else fold_models
    mean_c = float(np.mean([m["test_concordance"] for m in pool]))
    chosen = min(pool, key=lambda m: abs(m["test_concordance"] - mean_c))
    selected = chosen["genes"]
    print(f"   {len(survivors)}/{len(fold_models)} models passed validation "
          f"(C>={cfg.rederive_min_test_concordance}, p<{cfg.rederive_max_test_p}); "
          f"mean C={mean_c:.3f}")
    print(f"   selected fold-model (test C={chosen['test_concordance']:.3f}, "
          f"p={chosen['test_logrank_p']:.3g}): {selected}")

    # ---- finalize on all CT; drop genes with Wald p > 0.05, then refit ----
    selected = _finalize_drop_nonsig(df, base_cov, zall, selected, cfg)
    final = df[base_cov + ["time_days", "event"]].copy()
    for g in selected:
        final[g] = zall[g].values
    cph = CoxPHFitter()
    cph.fit(final, duration_col="time_days", event_col="event")
    res = cph.summary[["coef", "exp(coef)", "p"]].rename(
        columns={"exp(coef)": "hazard_ratio"})
    res.insert(0, "selected_gene", res.index.isin(selected))
    out = os.path.join(cfg.out_dir, "survival_rederived_model.csv")
    res.to_csv(out)
    final_c = concordance_index(final["time_days"],
                                -cph.predict_partial_hazard(final), final["event"])
    print(f"   finalized model: base {base_cov} + genes {selected}")
    print(f"   finalized full-CT concordance = {final_c:.3f}")
    print(f"   wrote {out}")
    overlap = set(selected) & set(cfg.published_gene_coefs)
    print(f"   overlap with published 6-gene signature: "
          f"{sorted(overlap) if overlap else 'none'}")
    return res


def _forward_select_on_train(train: pd.DataFrame, base_cov: list[str],
                             zall: pd.DataFrame, candidates: list[str],
                             cfg: Config) -> list[str]:
    """Forward stepwise on a TRAIN fold: add the candidate with the highest
    log-rank score (median split on predicted risk) until no gain / max genes."""
    selected: list[str] = []
    best_score = _train_logrank_score(train, base_cov, zall, selected)
    while len(selected) < cfg.rederive_max_genes:
        best_gene, best_gain = None, best_score
        for g in candidates:
            if g in selected:
                continue
            sc = _train_logrank_score(train, base_cov, zall, selected + [g])
            if sc > best_gain:
                best_gain, best_gene = sc, g
        if best_gene is None:
            break
        selected.append(best_gene)
        best_score = best_gain
    return selected


def _train_logrank_score(train: pd.DataFrame, base_cov: list[str],
                         zall: pd.DataFrame, genes: list[str]) -> float:
    """Log-rank test statistic of high vs low risk (median split) fit on train."""
    data = train[base_cov + ["time_days", "event"]].copy()
    for g in genes:
        data[g] = zall[g].reindex(train.index).values
    if data[base_cov + genes].shape[1] == 0:
        return 0.0
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(data, duration_col="time_days", event_col="event")
        risk = cph.predict_partial_hazard(data)
        thr = np.median(risk)
        hi, lo = data[risk > thr], data[risk <= thr]
        if len(hi) < 2 or len(lo) < 2:
            return 0.0
        lr = logrank_test(hi["time_days"], lo["time_days"],
                          event_observed_A=hi["event"], event_observed_B=lo["event"])
        return float(lr.test_statistic)
    except Exception:
        return 0.0


def _evaluate_fold_model(train: pd.DataFrame, test: pd.DataFrame,
                         base_cov: list[str], zall: pd.DataFrame,
                         genes: list[str]) -> tuple[float, float]:
    """Fit on train, predict on test; return (test concordance, test log-rank p)."""
    tr = train[base_cov + ["time_days", "event"]].copy()
    te = test[base_cov + ["time_days", "event"]].copy()
    for g in genes:
        tr[g] = zall[g].reindex(train.index).values
        te[g] = zall[g].reindex(test.index).values
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(tr, duration_col="time_days", event_col="event")
        risk = cph.predict_partial_hazard(te)
        tc = concordance_index(te["time_days"], -risk, te["event"])
        thr = np.median(risk)
        hi, lo = te[risk > thr], te[risk <= thr]
        if len(hi) < 1 or len(lo) < 1:
            return tc, 1.0
        lr = logrank_test(hi["time_days"], lo["time_days"],
                          event_observed_A=hi["event"], event_observed_B=lo["event"])
        return float(tc), float(lr.p_value)
    except Exception:
        return float("nan"), float("nan")


def _finalize_drop_nonsig(df: pd.DataFrame, base_cov: list[str],
                          zall: pd.DataFrame, genes: list[str],
                          cfg: Config) -> list[str]:
    """Refit on all CT and iteratively drop genes with Wald p>0.05 (paper)."""
    genes = list(genes)
    while genes:
        data = df[base_cov + ["time_days", "event"]].copy()
        for g in genes:
            data[g] = zall[g].values
        try:
            cph = CoxPHFitter()
            cph.fit(data, duration_col="time_days", event_col="event")
        except Exception:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(data, duration_col="time_days", event_col="event")
        ps = cph.summary.loc[[g for g in genes], "p"]
        worst_gene, worst_p = ps.idxmax(), ps.max()
        if worst_p > 0.05 and len(genes) > 1:
            genes.remove(worst_gene)
            continue
        break
    return genes


def _cv_concordance(df: pd.DataFrame, base_cov: list[str], zall: pd.DataFrame,
                    genes: list[str], cfg: Config) -> float:
    """Mean test-fold concordance over repeated stratified K-fold splits."""
    from sklearn.model_selection import StratifiedKFold
    cols = base_cov + genes
    data = df[base_cov + ["time_days", "event"]].copy()
    for g in genes:
        data[g] = zall[g].values
    y = df["event"].astype(int).values
    cs = []
    for seed in cfg.rederive_seeds:
        skf = StratifiedKFold(n_splits=cfg.rederive_cv_folds, shuffle=True,
                              random_state=seed)
        for tr, te in skf.split(data, y):
            train, test = data.iloc[tr], data.iloc[te]
            if train["event"].sum() < 2 or test["event"].sum() < 1:
                continue
            try:
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(train, duration_col="time_days", event_col="event")
                risk = cph.predict_partial_hazard(test)
                c = concordance_index(test["time_days"], -risk, test["event"])
                cs.append(c)
            except Exception:
                continue
    return float(np.mean(cs)) if cs else 0.5


# ----------------------------------------------------------------------------
# 4b. FIGURE 4 -- ESTABLISHED (COLMAN) SIGNATURE IS CONFOUNDED BY STRUCTURE
# ----------------------------------------------------------------------------
def colman_metagene(adata: ad.AnnData) -> pd.Series:
    """Colman metagene = mean z(poor-prognosis genes) - mean z(good-prognosis genes).

    Z-scoring is over the supplied sample subset (so passing CT-only or HBV-only
    AnnData reproduces the paper's "z-scored within each structure" variants).
    Metagene > 0 => high-risk (poor prognosis); < 0 => low-risk (good prognosis).
    """
    poor = _zscore_genes(adata, COLMAN_POOR_GENES)
    good = _zscore_genes(adata, COLMAN_GOOD_GENES)
    score = poor.mean(axis=1)
    if good.shape[1]:
        score = score - good.mean(axis=1)
    return score


def colman_structure_confound(ivy: ad.AnnData, cfg: Config, metrics: list) -> None:
    """Reproduce Fig 4: the established signature's prediction is driven by structure.

    Fig 4A : the metagene differs systematically across zones (good in LE/IT,
             poor in vascular/necrotic) -- so prognosis tracks histology, not patient.
    Fig 4C : KM on ALL samples shows no separation (a single patient endpoint maps
             to multiple samples predicting opposite risk -> signal cancels).
    Fig 4D : KM using a metagene computed within CT trends to CORRECT stratification.
    Fig 4E : KM using a metagene computed within HBV INVERTS the survival curve.
    """
    print("\n=== Figure 4: established (Colman) signature confounded by structure ===")

    # ---- Fig 4A: metagene by zone ----
    s_all = colman_metagene(ivy)
    by_zone = pd.DataFrame({"zone": ivy.obs["zone"].values,
                            "metagene": s_all.reindex(ivy.obs_names).values})
    zmean = by_zone.groupby("zone", observed=True)["metagene"].mean().sort_values()
    print("   mean Colman metagene by zone (lower=better prognosis):")
    for z, v in zmean.items():
        print(f"     {z:>8}: {v:+.3f}")
    _plot_metagene_by_zone(by_zone, cfg)

    # ---- KM helper: stratify a subset by its within-subset metagene sign ----
    def _km_for_subset(sub: ad.AnnData, tag: str, fig_name: str):
        score = colman_metagene(sub).reindex(sub.obs_names)
        df = _survival_frame(sub)
        score = score.reindex(df.index)
        df = df.assign(metagene=score.values).dropna(subset=["metagene"])
        hi = df[df["metagene"] > 0]   # poor-prognosis prediction
        lo = df[df["metagene"] <= 0]  # good-prognosis prediction
        if len(hi) < 3 or len(lo) < 3:
            print(f"   [{tag}] too few samples to stratify "
                  f"(hi={len(hi)}, lo={len(lo)}) -- skipped")
            return
        lr = logrank_test(hi["time_days"], lo["time_days"],
                          event_observed_A=hi["event"], event_observed_B=lo["event"])
        # median survival per group reveals whether the call is correct or inverted.
        med_hi = hi["time_days"].median()
        med_lo = lo["time_days"].median()
        direction = ("CORRECT" if med_hi < med_lo else "INVERTED")
        _km_plot({"High risk": hi, "Low risk": lo},
                 f"Colman signature, {tag}",
                 os.path.join(cfg.fig_dir, fig_name), p=lr.p_value)
        print(f"   [{tag}] log-rank p={lr.p_value:.4g}; median surv hi={med_hi:.0f}d "
              f"lo={med_lo:.0f}d -> {direction} stratification")
        metrics.append({"cohort": f"IvyGAP {tag}", "model": "Colman_signature",
                        "logrank_p": lr.p_value, "concordance": np.nan,
                        "n": len(df), "direction": direction})

    # ---- Fig 4C / 4D / 4E ----
    _km_for_subset(ivy, "all structures (Fig 4C)", "km_colman_all.png")
    _km_for_subset(_subset(ivy, (ivy.obs["zone"] == "CT").values),
                   "Cellular Tumor (Fig 4D)", "km_colman_CT.png")
    # HBV lives inside the HBV/MVP zone; use the fine structure label when present.
    hbv_mask = (ivy.obs.get("structure") == "HBV").values \
        if "structure" in ivy.obs else (ivy.obs["zone"] == "HBV/MVP").values
    _km_for_subset(_subset(ivy, hbv_mask), "HBV (Fig 4E)", "km_colman_HBV.png")


def _plot_metagene_by_zone(by_zone: pd.DataFrame, cfg: Config) -> None:
    zones = ["LE/IT", "CT", "HBV/MVP", "PAN/PNZ"]
    zones = [z for z in zones if z in set(by_zone["zone"])]
    data = [by_zone.loc[by_zone["zone"] == z, "metagene"].dropna().values
            for z in zones]
    fig, axb = plt.subplots(figsize=(6.5, 5))
    try:
        axb.boxplot(data, tick_labels=zones, showmeans=True)
    except TypeError:  # matplotlib < 3.9 used `labels=`
        axb.boxplot(data, labels=zones, showmeans=True)
    axb.axhline(0, color="grey", ls="--", lw=1)
    axb.set_ylabel("Colman metagene score")
    axb.set_title("Established prognostic signature varies by structure (Fig 4A)\n"
                  ">0 = poor-prognosis call")
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "colman_metagene_by_zone.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# ----------------------------------------------------------------------------
# 4c. SUPPLEMENTARY FIGURE S10 -- BIOLOGY OF THE HIGH-RISK GENES
# ----------------------------------------------------------------------------
def high_risk_wald_gsea(ivy_ct: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Reproduce Supplementary Fig S10: rank the transcriptome by the Cox Wald
    statistic, then pre-ranked GSEA to find pathways enriched in HIGH-RISK genes.

    For every gene we fit a multivariate Cox PH (gene + age + MGMT + IDH1) on
    IvyGAP CT and record its Wald statistic z = coef/se. A positive z means the
    gene is associated with higher hazard (high risk of short survival). We rank
    all genes by z (descending) and run pre-ranked GSEA against Hallmark and
    Positional (chromosomal-location) collections. The paper reports OXPHOS, MYC
    targets, MTORC1, glycolysis and DNA-repair hallmarks -- plus several Chr loci
    (13q12, Xp11, 16p12, 3q22, 3q25) -- enriched at the high-risk end.
    """
    print("\n=== Supplementary Fig S10: high-risk gene biology (Wald-ranked GSEA) ===")
    df = _survival_frame(ivy_ct)
    base_cov = [c for c in cfg.rederive_adjust
                if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    for c in base_cov:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    base_cov = [c for c in base_cov if df[c].nunique(dropna=True) > 1]
    df = df.dropna(subset=base_cov)
    print(f"   IvyGAP CT usable: {len(df)} samples; base covariates = {base_cov}")

    # gene panel: paper uses the entire transcriptome; "hv" is a faster subset.
    if cfg.s10_gene_panel == "hv" and "highly_variable" in ivy_ct.var:
        panel = list(ivy_ct.var_names[ivy_ct.var["highly_variable"].values])
    else:
        panel = list(ivy_ct.var_names)
    zall = _zscore_genes(ivy_ct, panel).reindex(df.index)
    print(f"   ranking {len(panel)} genes by multivariate Cox Wald statistic ...")

    # ---- per-gene Wald statistic from multivariate Cox ----
    base = df[base_cov + ["time_days", "event"]].copy()
    wald = {}
    for g in zall.columns:
        col = zall[g]
        if col.isna().any() or col.std(ddof=0) == 0:
            continue
        sub = base.copy()
        sub[g] = col.values
        try:
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(sub, duration_col="time_days", event_col="event")
            wald[g] = float(cph.summary.loc[g, "z"])
        except Exception:
            try:  # numeric fallback for non-converging genes
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(sub, duration_col="time_days", event_col="event")
                wald[g] = float(cph.summary.loc[g, "z"])
            except Exception:
                continue
    if not wald:
        warnings.warn("S10: no Cox fits succeeded; skipping.")
        return pd.DataFrame()

    rnk = (pd.Series(wald, name="wald").rename_axis("gene").reset_index()
           .dropna().drop_duplicates("gene")
           .sort_values("wald", ascending=False).reset_index(drop=True))
    rnk_out = os.path.join(cfg.out_dir, "survival_highrisk_wald_ranking.csv")
    rnk.to_csv(rnk_out, index=False)
    print(f"   wrote {rnk_out} ({len(rnk)} genes; top risk gene = "
          f"{rnk.iloc[0]['gene']} z={rnk.iloc[0]['wald']:.2f})")

    # ---- pre-ranked GSEA against Hallmark + Positional ----
    try:
        import gseapy as gp
    except ImportError:
        warnings.warn("gseapy not installed -- ranking saved, GSEA skipped.")
        return rnk

    collections = []
    if os.path.isfile(cfg.s10_hallmark_gmt):
        collections.append(("Hallmark", cfg.s10_hallmark_gmt))
    if os.path.isfile(cfg.s10_positional_gmt):
        collections.append(("Positional", cfg.s10_positional_gmt))
    if not collections:
        warnings.warn("S10: no GMT files found -- GSEA skipped.")
        return rnk

    summary_rows = []
    for cname, gmt in collections:
        print(f"   GSEA (Wald-ranked) vs {cname} ...")
        try:
            pre = gp.prerank(
                rnk=rnk, gene_sets=gmt,
                min_size=cfg.s10_min_size, max_size=cfg.s10_max_size,
                permutation_num=cfg.s10_permutations, seed=cfg.random_state,
                outdir=None, no_plot=True, verbose=False, threads=4)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"S10 GSEA failed for {cname}: {exc}")
            continue
        res = pre.res2d.copy()
        res.insert(0, "collection", cname)
        res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
        res["FDR q-val"] = pd.to_numeric(res["FDR q-val"], errors="coerce")
        summary_rows.append(res)

    if not summary_rows:
        return rnk
    allres = pd.concat(summary_rows, ignore_index=True)
    # high-risk end = positive NES (enriched among positive-Wald genes)
    high = (allres[allres["NES"] > 0]
            .sort_values(["FDR q-val", "NES"], ascending=[True, False]))
    keep = [c for c in ["collection", "Term", "NES", "NOM p-val", "FDR q-val",
                        "Lead_genes"] if c in high.columns]
    out = os.path.join(cfg.out_dir, "survival_highrisk_gsea.csv")
    high[keep].to_csv(out, index=False)
    print(f"   wrote {out} ({len(high)} high-risk-enriched terms)")

    _report_highrisk_biology(high)
    _plot_highrisk_gsea(high, cfg)
    return high


def _report_highrisk_biology(high: pd.DataFrame) -> None:
    """Check the paper's named high-risk hallmarks appear among enriched terms.

    MSigDB Hallmark terms use underscores (e.g. HALLMARK_OXIDATIVE_PHOSPHORYLATION
    and HALLMARK_MYC_TARGETS_V1), so we match on underscore tokens, not spaces.
    Paper S10 also reports chromosomal bands chr13q12/Xp11/16p12/3q22/3q25.
    """
    hallmarks = ["oxidative_phosphorylation", "myc_targets", "mtorc1",
                 "glycolysis", "dna_repair"]
    bands = ["13q12", "xp11", "16p12", "3q22", "3q25"]
    if "Term" not in high.columns:
        return
    terms = high["Term"].astype(str).str.lower()
    print("   --- expected high-risk hallmarks (paper S10) ---")
    for kw in hallmarks:
        hit = terms.str.contains(kw).any()
        print(f"     {'OK ' if hit else '-- '}{kw}")
    print("   --- expected high-risk chromosomal bands (paper S10) ---")
    for band in bands:
        hit = terms.str.contains("chr" + band).any()
        print(f"     {'OK ' if hit else '-- '}chr{band}")


def _plot_highrisk_gsea(high: pd.DataFrame, cfg: Config) -> None:
    hm = high[high["collection"] == "Hallmark"].head(cfg.s10_top_n)
    if hm.empty:
        return
    hm = hm.sort_values("NES")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(hm))))
    ax.barh(hm["Term"].astype(str).str.replace("HALLMARK_", ""),
            hm["NES"], color="#d62728")
    ax.set_xlabel("Normalized enrichment score (high-risk end)")
    ax.set_title("Hallmark pathways enriched in high-risk genes (Suppl. Fig S10)")
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "highrisk_hallmark_gsea.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# ----------------------------------------------------------------------------
# 4d. SUPPLEMENTARY FIG S8 -- 3-GROUP (HIGH/MEDIUM/LOW) RISK STRATIFICATION
# ----------------------------------------------------------------------------
def evaluate_3group(scores: pd.DataFrame, cohort_label: str, cfg: Config,
                    metrics: list) -> None:
    """Tertile risk stratification into high / medium / low groups (Suppl. S8).

    The paper notes the model "also effectively identified medium- and low-risk
    groups" using HR tertiles (low < q(1/3) <= medium < q(2/3) <= high). We plot
    all three KM curves and report the 3-group (multivariate) log-rank p.
    """
    hr = scores["hazard_ratio"]
    q1, q2 = hr.quantile(cfg.tertile_low), hr.quantile(cfg.tertile_high)
    grp = np.where(hr > q2, "High risk",
                   np.where(hr < q1, "Low risk", "Medium risk"))
    s = scores.assign(group3=grp)
    try:
        mlr = multivariate_logrank_test(s["time_days"], s["group3"], s["event"])
        p = mlr.p_value
    except Exception:
        p = float("nan")
    safe = cohort_label.replace(" ", "_").replace("/", "_")
    groups = {k: s[s["group3"] == k] for k in ["High risk", "Medium risk", "Low risk"]}
    _km_plot(groups, f"{cohort_label}: 3-group risk (Suppl. S8)",
             os.path.join(cfg.fig_dir, f"km_model3_{safe}.png"), p=p)
    print(f"   [{cohort_label}] 3-group tertile log-rank p={p:.4g} "
          f"(hi={len(groups['High risk'])}, med={len(groups['Medium risk'])}, "
          f"lo={len(groups['Low risk'])})")
    metrics.append({"cohort": cohort_label, "model": "novel_CT_3group",
                    "logrank_p": p, "concordance": np.nan, "n": len(s),
                    "direction": "tertiles"})


# ----------------------------------------------------------------------------
# 4e. SUPPLEMENTARY FIG S9 -- IDH-MUTANT-EXCLUDED RE-ANALYSIS
# ----------------------------------------------------------------------------
def idh_excluded_analysis(cohorts: dict, cfg: Config, metrics: list) -> None:
    """Re-run the published model after EXCLUDING IDH-mutant tumors (Suppl. S9).

    IDH-mutant gliomas are biologically distinct and longer-surviving; the paper
    confirms the model holds when they are removed. We apply this to any cohort
    whose IDH1 status actually varies (TCGA does: 6 mutants; IvyGAP CT is all
    IDH-wt, so exclusion is a no-op and is logged as such).
    """
    print("\n=== Supplementary Fig S9: IDH-mutant-excluded re-analysis ===")
    for label, adata in cohorts.items():
        idh = pd.to_numeric(adata.obs.get("idh1_mutant"), errors="coerce")
        if idh is None or idh.notna().sum() == 0:
            print(f"   [{label}] no IDH1 status -- skipped")
            continue
        n_mut = int((idh == 1).sum())
        if n_mut == 0:
            print(f"   [{label}] 0 IDH-mutant samples (all wild-type) -- "
                  f"exclusion is a no-op")
            continue
        keep = (idh != 1).values
        wt = _subset(adata, keep)
        lbl = f"{label} IDHwt"
        s = apply_published_model(wt, cfg, lbl)
        if s.empty:
            print(f"   [{label}] no usable samples after exclusion -- skipped")
            continue
        hi = s[s["risk_group"] == "High risk"]
        lo = s[s["risk_group"] == "Low risk"]
        lr = logrank_test(hi["time_days"], lo["time_days"],
                          event_observed_A=hi["event"], event_observed_B=lo["event"])
        safe = lbl.replace(" ", "_").replace("/", "_")
        _km_plot({"High risk": hi, "Low risk": lo},
                 f"{lbl}: published model (Suppl. S9)",
                 os.path.join(cfg.fig_dir, f"km_model_{safe}.png"), p=lr.p_value)
        print(f"   [{label}] excluded {n_mut} IDH-mutant; re-ran on {len(s)} "
              f"IDH-wt samples: log-rank p={lr.p_value:.4g}")
        metrics.append({"cohort": lbl, "model": "novel_CT_signature",
                        "logrank_p": lr.p_value, "concordance": np.nan,
                        "n": len(s), "direction": "IDHwt_only"})


# ----------------------------------------------------------------------------
# 5. ENTRY POINT
# ----------------------------------------------------------------------------
def _subset(adata: ad.AnnData, mask) -> ad.AnnData:
    return adata[mask].copy()


def main(cfg: Config = CFG) -> None:
    print("Phase 4: survival prediction (Fig 5 -- novel CT prognostic model)")
    print("=" * 70)
    os.makedirs(cfg.fig_dir, exist_ok=True)

    ivy = ad.read_h5ad(cfg.ivy_h5ad)
    tcga = ad.read_h5ad(cfg.tcga_h5ad)

    ivy_ct = _subset(ivy, (ivy.obs["zone"] == "CT").values)
    tcga_ct = _subset(tcga, (tcga.obs.get("is_CT") == True).values)  # noqa: E712
    print(f"IvyGAP CT: {ivy_ct.n_obs} | TCGA CT-classified: {tcga_ct.n_obs} | "
          f"IvyGAP all: {ivy.n_obs} | TCGA all: {tcga.n_obs}")

    # ---- published model across all four cohorts (Fig 5B-E) ----
    print("\n=== Published 6-gene model (Fig 5A equation) ===")
    cohorts = {
        "IvyGAP CT": ivy_ct, "TCGA CT": tcga_ct,
        "IvyGAP all": ivy, "TCGA all": tcga,
    }
    all_scores, metrics = [], []
    for label, a in cohorts.items():
        s = apply_published_model(a, cfg, label)
        all_scores.append(s)
        evaluate_cohort(s, label, cfg, metrics)
        # Suppl. S8: 3-group (high/medium/low) tertile stratification
        evaluate_3group(s, label, cfg, metrics)

    # ---- Suppl. S9: IDH-mutant-excluded re-analysis ----
    if cfg.idh_excluded:
        idh_excluded_analysis(cohorts, cfg, metrics)

    # ---- Fig 4: show the ESTABLISHED signature is structure-confounded ----
    colman_structure_confound(ivy, cfg, metrics)

    pd.concat(all_scores).to_csv(
        os.path.join(cfg.out_dir, "survival_published_scores.csv"))
    pd.DataFrame(metrics).to_csv(
        os.path.join(cfg.out_dir, "survival_metrics.csv"), index=False)
    print(f"\n   wrote survival_published_scores.csv and survival_metrics.csv")

    # ---- univariate clinical screen (paper Methods) ----
    univariate_clinical_cox(ivy_ct, cfg)

    # ---- re-derived model (methodology reproduction) ----
    if cfg.rederive:
        rederive_model(ivy_ct, cfg)

    # ---- Supplementary Fig S10: biology of the high-risk genes ----
    if cfg.s10:
        high_risk_wald_gsea(ivy_ct, cfg)

    print("\nDone. Survival models + KM curves written to ./%s/" % cfg.out_dir)


if __name__ == "__main__":
    main()
