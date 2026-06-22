"""
3_molecular_subtyping.py
=========================
PHASE 3 of the replication pipeline for:

    Kersch, Claunch et al. (2020) "Transcriptional signatures in histologic
    structures within glioblastoma tumors may predict personalized drug
    sensitivity and survival." Neuro-Oncology Advances, doi:10.1093/noajnl/vdaa093

This module reproduces the paper's Figure 3: molecular subtype classification of
glioblastoma (Verhaak proneural / classical / neural / mesenchymal) and the key
finding that *the subtype call depends on which histologic structure was
sampled*, while the Cellular Tumor (CT) most cleanly distinguishes biologically
distinct subtypes.

What it does
------------
  1. Single-sample GSEA (ssGSEA) of the 4 Verhaak subtype signatures across all
     IvyGAP samples -> per-sample subtype enrichment scores; subtype = argmax.
  2. Figure 3A/B: cross-tabulate subtype call vs histologic zone to show that
     LE/IT tends proneural/neural while HBV/MVP tends mesenchymal -- i.e. the
     same tumor can be called different subtypes depending on the structure.
  3. CT-focused analysis: within CT, recover the 3 main cohorts (proneural,
     classical, mesenchymal) and validate calls against IvyGAP's own reported
     molecular_subtype.
  4. Propagate subtype calls to the CT-classified TCGA samples (Fig 3D).

WHY this matters for the thesis
The paper argues that subtyping a mixed/edge biopsy is misleading because subtype
signatures are themselves structure-dependent (e.g. mesenchymal genes light up in
vascular regions). Restricting to CT removes this confounder, enabling valid
inter-patient subtype comparison -- the prerequisite for the prognostic and
drug-sensitivity work in modules 4-5.

Subtype signatures
------------------
We use curated canonical marker panels for the four Verhaak (2010) subtypes
(see VERHAAK_SUBTYPE_MARKERS). These are the widely-used, well-validated marker
genes for each subtype; a custom GMT can be supplied via Config.subtype_gmt to
swap in the full 840-gene classifier if available.

Outputs (./processed/ and ./processed/figures/):
    - subtype_scores_ivygap.csv     (per-sample ssGSEA score per subtype + call)
    - subtype_scores_tcga.csv       (per-sample calls for CT-classified TCGA)
    - subtype_by_zone.csv           (zone x subtype contingency table, Fig 3A/B)
    - figures/subtype_by_zone.png   (heatmap of subtype enrichment by zone)

Primary stack: gseapy (ssGSEA), pandas, numpy, matplotlib.
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


# ----------------------------------------------------------------------------
# 0. VERHAAK SUBTYPE MARKER PANELS (canonical genes, Verhaak et al. 2010)
# ----------------------------------------------------------------------------
# Curated, widely-used marker genes per subtype. ssGSEA on these recovers the
# subtype structure; substitute the full classifier via Config.subtype_gmt if
# you have it.
VERHAAK_SUBTYPE_MARKERS: dict[str, list[str]] = {
    "Proneural": [
        "PDGFRA", "OLIG2", "SOX2", "DLL3", "NKX2-2", "ERBB3", "DCX", "TCF4",
        "ASCL1", "SOX10", "NCAM1", "CDKN1B", "EPHB1", "CHD7", "BCL11A",
        "GABRB3", "SNAP91", "NMU", "DCC", "MYT1",
    ],
    "Classical": [
        "EGFR", "NES", "AKT2", "NOTCH3", "JAG1", "HES1", "SMO", "GLI2", "GAS1",
        "SOX9", "PDGFA", "FGFR3", "MEOX2", "ACSBG1", "PAX6", "MLC1", "MEST",
    ],
    "Mesenchymal": [
        "CHI3L1", "MET", "CD44", "MERTK", "RELB", "TRADD", "TLR2", "TLR4",
        "CASP1", "CASP4", "CASP8", "TGFBI", "SERPINE1", "TNFRSF1A", "NFKB1",
        "BCL3", "PLAUR", "VEGFA", "TIMP1", "TNC", "LOX", "IL6",
    ],
    "Neural": [
        "NEFL", "GABRA1", "SYT1", "SLC12A5", "GRIN1", "NRGN", "SNCB", "SYN1",
        "GRIA2", "KCNJ10", "SLC17A7", "GAD1",
    ],
}

# Two-tier subtype palette for the zone-vs-subtype heatmap.
SUBTYPE_ORDER = ["Proneural", "Classical", "Mesenchymal", "Neural"]


# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
@dataclass
class Config:
    ivy_h5ad: str = "processed/ivygap_processed.h5ad"
    tcga_h5ad: str = "processed/tcga_processed.h5ad"
    out_dir: str = "processed"
    fig_dir: str = "processed/figures"
    zone_key: str = "zone"

    # Optional external subtype GMT (overrides the curated marker panels).
    subtype_gmt: str | None = None

    # ssGSEA parameters.
    ssgsea_min_size: int = 5
    ssgsea_permutations: int = 0  # ssGSEA enrichment is deterministic; no perms needed
    random_state: int = 0

    markers: dict = field(default_factory=lambda: VERHAAK_SUBTYPE_MARKERS)


CFG = Config()


# ----------------------------------------------------------------------------
# 2. ssGSEA SCORING
# ----------------------------------------------------------------------------
def _expression_frame(adata: ad.AnnData) -> pd.DataFrame:
    """genes x samples log2 expression DataFrame (ssGSEA expects this orientation)."""
    X = adata.layers["log2"] if "log2" in adata.layers else adata.X
    return pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs_names)


def _load_gene_sets(cfg: Config) -> dict[str, list[str]]:
    if cfg.subtype_gmt and os.path.isfile(cfg.subtype_gmt):
        sets: dict[str, list[str]] = {}
        with open(cfg.subtype_gmt) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    sets[parts[0]] = parts[2:]
        print(f"   using external subtype GMT: {cfg.subtype_gmt} ({len(sets)} sets)")
        return sets
    return dict(cfg.markers)


def score_subtypes(adata: ad.AnnData, cfg: Config, label: str) -> pd.DataFrame:
    """ssGSEA-score every sample for each subtype signature; return samples x subtypes."""
    import gseapy as gp
    expr = _expression_frame(adata)
    gene_sets = _load_gene_sets(cfg)
    # Restrict each set to genes actually measured (ssGSEA ignores missing anyway).
    present = {k: [g for g in v if g in expr.index] for k, v in gene_sets.items()}
    for k, v in present.items():
        if len(v) < cfg.ssgsea_min_size:
            warnings.warn(f"[{label}] subtype '{k}' has only {len(v)} measured "
                          f"markers -- scores may be noisy.")

    res = gp.ssgsea(data=expr, gene_sets=present, outdir=None,
                    sample_norm_method="rank", no_plot=True,
                    min_size=cfg.ssgsea_min_size, threads=4, seed=cfg.random_state)
    nes = res.res2d.copy()
    nes["NES"] = pd.to_numeric(nes["NES"], errors="coerce")
    # pivot: sample (Name) x subtype (Term)
    mat = nes.pivot_table(index="Name", columns="Term", values="NES")
    mat = mat.reindex(index=adata.obs_names)
    # z-score each subtype score across samples so the argmax is scale-comparable.
    matz = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0).replace(0, 1.0)
    matz["subtype_call"] = matz[[c for c in SUBTYPE_ORDER if c in matz.columns]].idxmax(axis=1)
    print(f"   [{label}] ssGSEA subtype calls:",
          dict(matz["subtype_call"].value_counts()))
    return matz


# ----------------------------------------------------------------------------
# 3. FIGURE 3A/B -- SUBTYPE DEPENDS ON STRUCTURE
# ----------------------------------------------------------------------------
def subtype_by_zone(scores: pd.DataFrame, adata: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Cross-tabulate subtype call vs histologic zone (the core Fig 3A/B point)."""
    zones = adata.obs[cfg.zone_key].reindex(scores.index)
    df = scores.copy()
    df["zone"] = zones.values
    ct = pd.crosstab(df["zone"], df["subtype_call"])
    ct = ct.reindex(columns=[c for c in SUBTYPE_ORDER if c in ct.columns],
                    fill_value=0)
    out = os.path.join(cfg.out_dir, "subtype_by_zone.csv")
    ct.to_csv(out)
    print(f"   wrote {out}")
    print("   zone x subtype contingency:\n", ct.to_string())

    # mean subtype enrichment per zone (heatmap, Fig 3A flavor)
    subcols = [c for c in SUBTYPE_ORDER if c in scores.columns]
    mean_by_zone = df.groupby("zone", observed=True)[subcols].mean()
    _plot_subtype_heatmap(mean_by_zone, cfg)
    return ct


def _plot_subtype_heatmap(mean_by_zone: pd.DataFrame, cfg: Config) -> None:
    fig, axh = plt.subplots(figsize=(6.5, 4.5))
    im = axh.imshow(mean_by_zone.values, cmap="RdBu_r", aspect="auto",
                    vmin=-1.2, vmax=1.2)
    axh.set_xticks(range(mean_by_zone.shape[1]))
    axh.set_xticklabels(mean_by_zone.columns, rotation=30, ha="right")
    axh.set_yticks(range(mean_by_zone.shape[0]))
    axh.set_yticklabels(mean_by_zone.index.tolist())
    axh.set_title("Mean subtype enrichment by zone (Fig 3A)\n"
                  "structure drives subtype signal")
    for i in range(mean_by_zone.shape[0]):
        for j in range(mean_by_zone.shape[1]):
            axh.text(j, i, f"{mean_by_zone.values[i, j]:.2f}",
                     ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axh, fraction=0.046, label="mean z(ssGSEA NES)")
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "subtype_by_zone.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# ----------------------------------------------------------------------------
# 4. CT VALIDATION AGAINST REPORTED SUBTYPE
# ----------------------------------------------------------------------------
def validate_ct_calls(scores: pd.DataFrame, adata: ad.AnnData, cfg: Config) -> None:
    """Within CT, compare ssGSEA subtype calls to IvyGAP's reported subtype.

    IvyGAP reports a (sometimes multi-label) molecular_subtype per tumour. We
    count a call as concordant if our argmax subtype appears among the reported
    labels. The paper's point is that CT yields cleaner, biologically-valid
    3-subtype structure (proneural/classical/mesenchymal), so we also report the
    CT subtype distribution.
    """
    ctmask = (adata.obs[cfg.zone_key] == "CT").reindex(scores.index).fillna(False)
    ct = scores[ctmask.values].copy()
    reported = adata.obs["molecular_subtype"].reindex(ct.index) \
        if "molecular_subtype" in adata.obs else None
    print(f"\n   CT subtype distribution (n={len(ct)}):",
          dict(ct["subtype_call"].value_counts()))
    print("   3 main subtypes present (proneural/classical/mesenchymal):",
          {s: int((ct["subtype_call"] == s).sum())
           for s in ["Proneural", "Classical", "Mesenchymal"]})

    if reported is not None and reported.notna().any():
        ok = tot = 0
        for idx, call in ct["subtype_call"].items():
            rep = reported.get(idx)
            if pd.isna(rep):
                continue
            tot += 1
            if str(call).lower() in str(rep).lower():
                ok += 1
        if tot:
            print(f"   concordance with IvyGAP reported subtype (CT): "
                  f"{ok}/{tot} = {ok/tot:.2f}")


# ----------------------------------------------------------------------------
# 5. ENTRY POINT
# ----------------------------------------------------------------------------
def main(cfg: Config = CFG) -> None:
    print("Phase 3: molecular subtyping (Fig 3 -- subtype depends on structure)")
    print("=" * 70)
    os.makedirs(cfg.fig_dir, exist_ok=True)

    ivy = ad.read_h5ad(cfg.ivy_h5ad)
    print(f"IvyGAP: {ivy.n_obs} samples")

    print("\n=== ssGSEA subtype scoring (IvyGAP, all structures) ===")
    ivy_scores = score_subtypes(ivy, cfg, "IvyGAP")
    ivy_scores_out = ivy_scores.copy()
    ivy_scores_out["zone"] = ivy.obs[cfg.zone_key].reindex(ivy_scores.index).values
    ivy_scores_out.to_csv(os.path.join(cfg.out_dir, "subtype_scores_ivygap.csv"))

    print("\n=== Figure 3A/B: subtype call vs histologic zone ===")
    subtype_by_zone(ivy_scores, ivy, cfg)
    validate_ct_calls(ivy_scores, ivy, cfg)

    # ---- propagate to CT-classified TCGA (Fig 3D) ----
    if os.path.isfile(cfg.tcga_h5ad):
        tcga = ad.read_h5ad(cfg.tcga_h5ad)
        if "is_CT" in tcga.obs:
            tcga_ct = tcga[(tcga.obs["is_CT"] == True).values].copy()  # noqa: E712
            if tcga_ct.n_obs:
                print(f"\n=== Subtyping CT-classified TCGA (Fig 3D; "
                      f"n={tcga_ct.n_obs}) ===")
                tcga_scores = score_subtypes(tcga_ct, cfg, "TCGA CT")
                tcga_scores.to_csv(
                    os.path.join(cfg.out_dir, "subtype_scores_tcga.csv"))

    print("\nDone. Subtype calls + zone-dependence written to ./%s/" % cfg.out_dir)


if __name__ == "__main__":
    main()
