"""
2_zonal_signature_extraction.py
===============================
PHASE 2 of the replication pipeline for:

    Kersch, Claunch et al. (2020) "Transcriptional signatures in histologic
    structures within glioblastoma tumors may predict personalized drug
    sensitivity and survival." Neuro-Oncology Advances, doi:10.1093/noajnl/vdaa093

This module reproduces the paper's first two results sections (Figures 1-2),
which establish the BIOLOGICAL FOUNDATION for everything downstream:

    Figure 1  "Variation in glioblastoma sample gene expression is primarily
               explained by histologic structure."
        -> PCA on the 1000 most variable genes; PC1+PC2 should explain ~50.9%
           of variance and separate samples BY STRUCTURE.
        -> Unsupervised k-means / hierarchical clustering (k=4) should recover
           the 4 collapsed zones (CT | LE/IT | HBV/MVP | PAN/PNZ), confirming
           the structures are molecularly distinct.

    Figure 2  "Biological processes enriched in glioblastoma structures."
        -> Differential gene expression (DGE) for each zone vs the rest of the
           tumor -> the per-zone "transcriptional signatures" of the title.
        -> GSEA of those signatures against Gene Ontology (C5) and Hallmark (H)
           gene sets, validating the expected zone biology:
               LE/IT    -> normal-CNS / neuronal / synaptic processes
               HBV/MVP  -> angiogenesis / vascular / inflammatory processes
               PAN/PNZ  -> hypoxia / necrosis / cellular-starvation processes
               CT       -> cell-cycle / DNA replication & repair / stem-cell

WHY this module matters for the thesis
The paper's central claim is that histologic structure -- not patient identity or
batch -- is the dominant axis of transcriptional variation, and that each zone
carries a distinct, biologically-interpretable signature. We therefore (a) verify
structure dominates variance (Fig 1) and (b) extract + biologically validate the
zonal signatures (Fig 2). These signatures are the inputs to subtyping (Fig 3),
survival (Fig 5), and drug-sensitivity mapping.

Outputs (written to ./processed/ and ./processed/figures/):
    - zonal_dge.csv                  (per-zone DGE: gene, logFC, pval, padj, score, dir)
    - zonal_signatures.gmt           (up/down gene lists per zone, GMT format)
    - gsea/<ZONE>__<COLLECTION>.csv  (GSEA enrichment tables per zone & gene-set DB)
    - zonal_gsea_summary.csv         (top enriched terms per zone, all collections)
    - figures/pca_by_zone.png        (Fig 1B reproduction)
    - figures/kmeans_vs_zone.png     (Fig 1D-style clustering concordance)

Primary stack: scanpy (DGE), scikit-learn (PCA/k-means), gseapy (GSEA), matplotlib.
Author: GBM spatial-transcriptomics replication pipeline.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display server
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist


# ----------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------
@dataclass
class Config:
    ivy_h5ad: str = "processed/ivygap_processed.h5ad"
    out_dir: str = "processed"
    fig_dir: str = "processed/figures"
    gsea_dir: str = "processed/gsea"

    # Grouping for all zonal contrasts. The paper collapses 7 structures -> 4 zones.
    zone_key: str = "zone"

    # ---- Gene-set collections (MSigDB v7.1, already in the workspace) ----
    # The paper used Gene Ontology (C5), Hallmark (H), and Positional (C1).
    hallmark_gmt: str = "h.all.v7.1.symbols.gmt"
    go_gmt: str = "c5.all.v7.1.symbols.gmt"
    positional_gmt: str = "c1.all.v7.1.symbols.gmt"

    # C5 (GO) holds ~10k sets and is SLOW under permutation GSEA. Default to
    # Hallmark (fast, interpretable) + GO; flip run_go off for a quick pass.
    run_hallmark: bool = True
    run_go: bool = True
    run_positional: bool = False  # positional bins are used later (high-risk genes)

    # ---- DGE parameters ----
    # Wilcoxon rank-sum (scanpy default, robust for small n) on log2(FPKM+1).
    dge_method: str = "wilcoxon"
    # A gene enters a zone's signature if FDR < padj_cutoff AND |log2FC| >= lfc_cutoff.
    padj_cutoff: float = 0.05
    lfc_cutoff: float = 0.5
    # Cap each zone's up/down signature for GMT export + downstream query signatures.
    max_signature_genes: int = 150

    # ---- GSEA parameters (paper Methods: sets 15-500 genes, 1000 permutations) ----
    gsea_min_size: int = 15
    gsea_max_size: int = 500
    gsea_permutations: int = 1000
    gsea_seed: int = 0
    gsea_top_n: int = 15  # rows kept per zone in the human-readable summary

    n_top_variable_genes: int = 1000
    random_state: int = 0

    # ---- Fig 1 companion analyses (paper Methods, "Variation in Gene
    #      Expression Is Primarily Explained by Histologic Structure") ----
    tsne_perplexity: float = 15.0      # small-n (<300 samples) -> modest perplexity
    corr_network_threshold: float = 0.92   # paper: Pearson r > 0.92 edges (Fig 1C)
    gap_k_max: int = 10                # paper: gap statistic over k = 1..10
    gap_n_refs: int = 20               # Monte-Carlo reference datasets (Tibshirani)
    hclust_k: int = 4                  # paper: k=4 groups, Ward/Euclidean (Fig 1D)


CFG = Config()


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_ivygap(cfg: Config) -> ad.AnnData:
    if not os.path.isfile(cfg.ivy_h5ad):
        raise FileNotFoundError(
            f"{cfg.ivy_h5ad} missing -- run 1_data_preprocessing.py first.")
    adata = ad.read_h5ad(cfg.ivy_h5ad)
    if cfg.zone_key not in adata.obs:
        raise KeyError(f"obs['{cfg.zone_key}'] not found in {cfg.ivy_h5ad}.")
    # Keep only zone-labelled samples (all IvyGAP samples should be labelled).
    labelled = adata.obs[cfg.zone_key].notna().values
    adata = adata[labelled].copy()
    print(f"Loaded IvyGAP: {adata.n_obs} zone-labelled samples x {adata.n_vars} genes")
    print("   zone counts:", dict(adata.obs[cfg.zone_key].value_counts()))
    return adata


# ----------------------------------------------------------------------------
# 2. FIGURE 1 -- STRUCTURE DOMINATES TRANSCRIPTIONAL VARIANCE
# ----------------------------------------------------------------------------
def structure_variance_analysis(adata: ad.AnnData, cfg: Config) -> dict:
    """Reproduce Fig 1B/1D: PCA + unsupervised clustering on top-1000 variable genes.

    We operate on the z-scored layer restricted to the highly-variable panel --
    exactly the "1000 most variable genes ... log2-transformed and z-score-
    normalized" matrix the paper analyzes. If PC1+PC2 explains ~half the variance
    AND k-means (k=4) recovers the zone labels (high adjusted Rand index), we have
    confirmed that histologic structure is the dominant axis of variation.
    """
    print("\n=== Figure 1: structure-driven variance ===")
    os.makedirs(cfg.fig_dir, exist_ok=True)

    # ---- feature matrix: HV genes, z-scored (interpatient-comparable) ----
    hv = adata.var["highly_variable"].values if "highly_variable" in adata.var \
        else np.ones(adata.n_vars, dtype=bool)
    layer = "zscore" if "zscore" in adata.layers else None
    X = (adata.layers["zscore"] if layer else adata.X)[:, hv]
    zones = adata.obs[cfg.zone_key].astype(str).values
    print(f"   PCA/clustering on {X.shape[1]} highly-variable genes "
          f"(z-scored), {X.shape[0]} samples")

    # ---- PCA ----
    pca = PCA(n_components=10, random_state=cfg.random_state).fit(X)
    pcs = pca.transform(X)
    var_ratio = pca.explained_variance_ratio_
    pc12 = float(var_ratio[0] + var_ratio[1]) * 100
    print(f"   PC1={var_ratio[0]*100:.1f}%  PC2={var_ratio[1]*100:.1f}%  "
          f"PC1+PC2={pc12:.1f}%  (paper: 50.9%)")

    # ---- gap statistic: estimate optimal k on the 1000-gene matrix ----
    # NOTE: the gap statistic is notoriously sensitive to the reference null and
    # to cluster overlap; on this z-scored matrix it does not land cleanly on 4
    # (the paper's factoextra run did). We therefore report/plot it honestly but
    # COLLAPSE to k=4 downstream exactly as the paper did -- a choice the paper
    # supports by *converging* evidence (k-means + hierarchical + correlation
    # network + biology), not by the gap statistic alone.
    best_k = _gap_statistic(X, cfg)

    # ---- k-means concordance with the 4 zones (k=4, the paper's collapse) ----
    k_use = cfg.hclust_k
    km = KMeans(n_clusters=k_use, n_init=10,
                random_state=cfg.random_state).fit(X)
    ari = adjusted_rand_score(zones, km.labels_)
    print(f"   k-means(k={k_use}) vs zone labels: adjusted Rand index = {ari:.3f} "
          f"(1.0 = perfect structure recovery)  [gap-derived k={best_k}]")

    # ---- hierarchical (Ward/Euclidean) concordance, k=4 (paper Fig 1D) ----
    hclust_ari = _hierarchical_clustering(X, zones, cfg)

    # ---- plots ----
    _plot_pca(pcs, zones, var_ratio, cfg)
    _plot_kmeans_concordance(zones, km.labels_, cfg)
    _plot_tsne(X, zones, cfg)
    _plot_correlation_network(X, zones, cfg)

    return {"pc1": var_ratio[0], "pc2": var_ratio[1], "pc1p2": pc12 / 100,
            "kmeans_ari": ari, "gap_best_k": best_k, "hclust_ari": hclust_ari}


# --- Fig 1 companion: t-SNE -------------------------------------------------
def _plot_tsne(X: np.ndarray, zones: np.ndarray, cfg: Config) -> None:
    """t-SNE of the top-1000-variable z-scored matrix (paper ran t-SNE with PCA).

    t-SNE corroborates PCA: samples should embed into structure-coherent islands
    rather than patient-coherent ones. Purely a visualization, like the paper's.
    """
    n = X.shape[0]
    perp = float(min(cfg.tsne_perplexity, max(5.0, (n - 1) / 3.0)))
    ts = TSNE(n_components=2, perplexity=perp, init="pca",
              learning_rate="auto", random_state=cfg.random_state)
    emb = ts.fit_transform(X)
    palette = {"CT": "#2ca02c", "LE/IT": "#9467bd",
               "HBV/MVP": "#ff7f0e", "PAN/PNZ": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for z in sorted(set(zones)):
        m = zones == z
        ax.scatter(emb[m, 0], emb[m, 1], s=28, alpha=0.8,
                   label=z, color=palette.get(z, "#888888"))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"IvyGAP t-SNE on 1000 most variable genes (perplexity={perp:.0f})")
    ax.legend(title="Zone", frameon=False)
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "tsne_by_zone.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# --- Fig 1C companion: sample-sample correlation network --------------------
def _plot_correlation_network(X: np.ndarray, zones: np.ndarray,
                              cfg: Config) -> None:
    """Sample-to-sample Pearson correlation graph; edges where r > threshold.

    Reproduces Fig 1C (originally BioLayout Express3D). Nodes = samples coloured
    by zone; an edge connects two samples whose expression correlation exceeds
    `corr_network_threshold` (paper: r > 0.92). If structure dominates variance,
    nodes of the same zone form densely-connected, separated communities.
    """
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("networkx not installed -- skipping correlation network.")
        return

    r = np.corrcoef(X)              # sample x sample Pearson correlation
    thr = cfg.corr_network_threshold
    G = nx.Graph()
    G.add_nodes_from(range(len(zones)))
    iu = np.triu_indices_from(r, k=1)
    for i, j in zip(*iu):
        if r[i, j] > thr:
            G.add_edge(int(i), int(j), weight=float(r[i, j]))

    # edge homophily: fraction of edges that connect same-zone samples
    same = sum(1 for u, v in G.edges() if zones[u] == zones[v])
    homophily = same / G.number_of_edges() if G.number_of_edges() else float("nan")
    print(f"   correlation network (r>{thr}): {G.number_of_edges()} edges, "
          f"same-zone fraction = {homophily:.3f} (1.0 = perfect separation)")

    palette = {"CT": "#2ca02c", "LE/IT": "#9467bd",
               "HBV/MVP": "#ff7f0e", "PAN/PNZ": "#1f77b4"}
    node_colors = [palette.get(z, "#888888") for z in zones]
    pos = nx.spring_layout(G, seed=cfg.random_state, k=0.3)
    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=70,
                           edgecolors="black", linewidths=0.3, ax=ax)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor=c, markeredgecolor="black", label=z)
               for z, c in palette.items() if z in set(zones)]
    ax.legend(handles=handles, title="Zone", frameon=False, loc="best")
    ax.set_title(f"Sample correlation network (Pearson r > {thr}, Fig 1C)")
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "correlation_network.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# --- Fig 1 companion: gap statistic to DERIVE optimal k ---------------------
def _gap_statistic(X: np.ndarray, cfg: Config) -> int | None:
    """Tibshirani gap statistic over k = 1..gap_k_max (paper: factoextra).

    For each k we compute within-cluster dispersion Wk, then compare log(Wk) to
    its expectation under B uniform reference datasets drawn from the data's
    bounding box. The optimal k is the smallest k whose gap is within one
    standard error of the next k's gap (Tibshirani's 1-SE rule) -- the paper
    uses this to justify collapsing 7 structures into k=4 clusters.
    """
    rng = np.random.default_rng(cfg.random_state)
    ks = list(range(1, cfg.gap_k_max + 1))

    def _Wk(data: np.ndarray, k: int) -> float:
        if k == 1:
            c = data.mean(axis=0, keepdims=True)
            return float(((data - c) ** 2).sum())
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=cfg.random_state).fit(data)
        tot = 0.0
        for lbl in np.unique(km.labels_):
            pts = data[km.labels_ == lbl]
            tot += ((pts - pts.mean(axis=0, keepdims=True)) ** 2).sum()
        return float(tot)

    # Tibshirani et al. (2001) reference "method (b)": instead of sampling a
    # uniform box over the raw features (a poor null in high dimensions that makes
    # the gap grow monotonically), align the box with the data's principal
    # components, sample uniformly there, then rotate back. This is what
    # factoextra/clusGap effectively does and is required to recover a finite k.
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt.T                      # data projected onto its PCs
    mins, maxs = Xp.min(axis=0), Xp.max(axis=0)

    gaps, sks = [], []
    for k in ks:
        logWk = np.log(_Wk(X, k) + 1e-12)
        ref_logs = []
        for _ in range(cfg.gap_n_refs):
            ref_p = rng.uniform(mins, maxs, size=Xp.shape)
            ref = ref_p @ Vt + X.mean(axis=0, keepdims=True)  # rotate back
            ref_logs.append(np.log(_Wk(ref, k) + 1e-12))
        ref_logs = np.asarray(ref_logs)
        gaps.append(ref_logs.mean() - logWk)
        sks.append(ref_logs.std() * np.sqrt(1.0 + 1.0 / cfg.gap_n_refs))
    gaps, sks = np.asarray(gaps), np.asarray(sks)

    best_k = ks[-1]
    for idx in range(len(ks) - 1):
        if gaps[idx] >= gaps[idx + 1] - sks[idx + 1]:
            best_k = ks[idx]
            break
    print(f"   gap statistic (k=1..{cfg.gap_k_max}): optimal k = {best_k} "
          f"(paper derived k=4)")
    _plot_gap(ks, gaps, sks, best_k, cfg)
    return best_k


def _plot_gap(ks, gaps, sks, best_k, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(ks, gaps, yerr=sks, marker="o", capsize=3, color="#1f77b4")
    ax.axvline(best_k, color="#d62728", linestyle="--",
               label=f"optimal k = {best_k}")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Gap statistic")
    ax.set_title("Gap statistic on 1000 most variable genes")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "gap_statistic.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# --- Fig 1D: Ward/Euclidean hierarchical clustering dendrogram --------------
def _hierarchical_clustering(X: np.ndarray, zones: np.ndarray,
                             cfg: Config) -> float:
    """Ward-linkage / Euclidean dendrogram, cut at k=4 (paper Fig 1D).

    Computes the exact clustering the paper describes ("Euclidean distance ...
    Ward's method ... k = 4 groups") and reports the adjusted Rand index of the
    4 dendrogram groups against the true zone labels, alongside a labelled,
    zone-coloured dendrogram.
    """
    d = pdist(X, metric="euclidean")
    Z = linkage(d, method="ward")
    labels = fcluster(Z, t=cfg.hclust_k, criterion="maxclust")
    ari = adjusted_rand_score(zones, labels)
    print(f"   hierarchical (Ward/Euclidean, k={cfg.hclust_k}) vs zones: "
          f"adjusted Rand index = {ari:.3f}")

    palette = {"CT": "#2ca02c", "LE/IT": "#9467bd",
               "HBV/MVP": "#ff7f0e", "PAN/PNZ": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, ax=ax, labels=zones.tolist(), color_threshold=0,
               leaf_rotation=90, leaf_font_size=6,
               above_threshold_color="#555555")
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(palette.get(lbl.get_text(), "#000000"))
    ax.set_ylabel("Ward distance")
    ax.set_title(f"Hierarchical clustering of IvyGAP samples "
                 f"(Ward/Euclidean, k={cfg.hclust_k}, Fig 1D)")
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "dendrogram_by_zone.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")
    return ari


def _plot_pca(pcs: np.ndarray, zones: np.ndarray, var_ratio: np.ndarray,
              cfg: Config) -> None:
    fig, axratio = plt.subplots(figsize=(7, 6))
    palette = {"CT": "#2ca02c", "LE/IT": "#9467bd",
               "HBV/MVP": "#ff7f0e", "PAN/PNZ": "#1f77b4"}
    for z in sorted(set(zones)):
        m = zones == z
        axratio.scatter(pcs[m, 0], pcs[m, 1], s=28, alpha=0.8,
                        label=z, color=palette.get(z, "#888888"))
    axratio.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    axratio.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    axratio.set_title("IvyGAP PCA on 1000 most variable genes (Fig 1B)")
    axratio.legend(title="Zone", frameon=False)
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "pca_by_zone.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


def _plot_kmeans_concordance(zones: np.ndarray, km_labels: np.ndarray,
                             cfg: Config) -> None:
    # crosstab (not sklearn confusion_matrix) because the two labelings live in
    # different spaces: string zones vs integer k-means cluster ids.
    ct = pd.crosstab(pd.Series(zones, name="zone"),
                     pd.Series(km_labels, name="cluster"))
    fig, axc = plt.subplots(figsize=(6, 5))
    im = axc.imshow(ct.values, cmap="Blues")
    axc.set_xticks(range(ct.shape[1]))
    axc.set_xticklabels([f"k{c}" for c in ct.columns])
    axc.set_yticks(range(ct.shape[0]))
    axc.set_yticklabels(ct.index.tolist())
    axc.set_xlabel("k-means cluster")
    axc.set_ylabel("Histologic zone")
    axc.set_title("Zone vs unsupervised k-means (Fig 1D)")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            axc.text(j, i, int(ct.values[i, j]), ha="center", va="center",
                     color="black", fontsize=9)
    fig.colorbar(im, ax=axc, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "kmeans_vs_zone.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# ----------------------------------------------------------------------------
# 3. FIGURE 2a -- ZONAL DIFFERENTIAL GENE EXPRESSION (the "signatures")
# ----------------------------------------------------------------------------
def zonal_dge(adata: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Zone-vs-rest DGE producing the per-zone transcriptional signatures.

    Implementation note (faithful to the GBM workflow in .cursorrules): we run
    scanpy.rank_genes_groups with the Wilcoxon test on log2(FPKM+1) values, which
    is the standard, distribution-robust choice for RNA-seq signature extraction
    with modest per-group n. Each zone is contrasted against ALL other zones --
    i.e. "enriched/depleted relative to the rest of the tumor" (Fig 2 caption).
    """
    print("\n=== Figure 2a: zonal differential gene expression ===")
    # Work on a copy whose X is the log2 layer (rank_genes_groups reads .X).
    work = adata.copy()
    if "log2" in work.layers:
        work.X = work.layers["log2"].copy()
    work.obs[cfg.zone_key] = work.obs[cfg.zone_key].astype("category")

    sc.tl.rank_genes_groups(
        work, groupby=cfg.zone_key, method=cfg.dge_method,
        pts=True, tie_correct=True)

    # Flatten scanpy's per-group record arrays into a tidy long table.
    rec = work.uns["rank_genes_groups"]
    groups = rec["names"].dtype.names
    frames = []
    for g in groups:
        df = pd.DataFrame({
            "zone": g,
            "gene": rec["names"][g],
            "score": rec["scores"][g],
            "log2fc": rec["logfoldchanges"][g],
            "pval": rec["pvals"][g],
            "padj": rec["pvals_adj"][g],
        })
        frames.append(df)
    dge = pd.concat(frames, ignore_index=True)
    dge["direction"] = np.where(dge["log2fc"] >= 0, "up", "down")

    # Significance flag = the paper's signature membership criterion.
    dge["significant"] = (dge["padj"] < cfg.padj_cutoff) & \
                         (dge["log2fc"].abs() >= cfg.lfc_cutoff)

    out = os.path.join(cfg.out_dir, "zonal_dge.csv")
    dge.to_csv(out, index=False)
    sig = dge[dge["significant"]]
    print(f"   wrote {out} ({len(dge)} rows; {len(sig)} significant)")
    for g in groups:
        gz = sig[sig["zone"] == g]
        print(f"     {g:>8}: {(gz['direction']=='up').sum():4d} up / "
              f"{(gz['direction']=='down').sum():4d} down")
    return dge


def export_signatures_gmt(dge: pd.DataFrame, cfg: Config) -> dict:
    """Write per-zone up/down signatures as a GMT file (for GSEA/drug queries).

    For each zone we take the top `max_signature_genes` significant genes by
    score in each direction. The resulting up/down lists are the "query
    signatures" consumed by module 3 (drug sensitivity) and the subtyping module.
    """
    sig_sets: dict[str, list[str]] = {}
    lines = []
    for zone in sorted(dge["zone"].unique()):
        for direction in ("up", "down"):
            sub = dge[(dge["zone"] == zone) & (dge["significant"]) &
                      (dge["direction"] == direction)]
            sub = sub.reindex(sub["score"].abs().sort_values(ascending=False).index)
            genes = sub["gene"].head(cfg.max_signature_genes).tolist()
            if not genes:
                continue
            name = f"{zone.replace('/', '_')}_{direction.upper()}"
            sig_sets[name] = genes
            lines.append("\t".join([name, "ivygap_zonal_signature", *genes]))
    out = os.path.join(cfg.out_dir, "zonal_signatures.gmt")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"   wrote {out} ({len(sig_sets)} signature sets)")
    return sig_sets


# ----------------------------------------------------------------------------
# 4. FIGURE 2b -- GSEA OF ZONAL SIGNATURES
# ----------------------------------------------------------------------------
def zonal_gsea(dge: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pre-ranked GSEA of each zone's full ranked gene list vs MSigDB collections.

    We rank the WHOLE transcriptome per zone by the signed Wilcoxon score (the
    paper uses Signal2Noise; the rank-sum z-score is a robust, equivalent ranking
    metric for pre-ranked GSEA) and test enrichment against Gene Ontology (C5)
    and Hallmark (H). Positive NES => enriched in that zone relative to the rest.
    """
    print("\n=== Figure 2b: GSEA of zonal signatures ===")
    try:
        import gseapy as gp
    except ImportError:
        warnings.warn("gseapy not installed -- skipping GSEA. "
                      "`pip install gseapy` to enable.")
        return pd.DataFrame()

    os.makedirs(cfg.gsea_dir, exist_ok=True)
    collections = []
    if cfg.run_hallmark and os.path.isfile(cfg.hallmark_gmt):
        collections.append(("Hallmark", cfg.hallmark_gmt))
    if cfg.run_go and os.path.isfile(cfg.go_gmt):
        collections.append(("GO", cfg.go_gmt))
    if cfg.run_positional and os.path.isfile(cfg.positional_gmt):
        collections.append(("Positional", cfg.positional_gmt))
    if not collections:
        warnings.warn("No gene-set GMT files found -- skipping GSEA.")
        return pd.DataFrame()

    summary_rows = []
    for zone in sorted(dge["zone"].unique()):
        zdf = dge[dge["zone"] == zone].copy()
        # pre-ranked list: gene -> signed score, deduplicated, sorted descending.
        rnk = (zdf[["gene", "score"]]
               .dropna()
               .drop_duplicates(subset="gene")
               .sort_values("score", ascending=False)
               .reset_index(drop=True))
        for cname, gmt in collections:
            print(f"   GSEA: zone={zone:>8}  collection={cname} "
                  f"({len(rnk)} ranked genes) ...")
            try:
                pre = gp.prerank(
                    rnk=rnk, gene_sets=gmt,
                    min_size=cfg.gsea_min_size, max_size=cfg.gsea_max_size,
                    permutation_num=cfg.gsea_permutations, seed=cfg.gsea_seed,
                    outdir=None, no_plot=True, verbose=False, threads=4)
            except Exception as exc:  # noqa: BLE001 -- GSEA can fail on edge cases
                warnings.warn(f"GSEA failed for {zone}/{cname}: {exc}")
                continue
            res = pre.res2d.copy()
            res.insert(0, "zone", zone)
            res.insert(1, "collection", cname)
            safe = zone.replace("/", "_")
            res.to_csv(os.path.join(cfg.gsea_dir, f"{safe}__{cname}.csv"),
                       index=False)
            # keep top positively-enriched terms for the summary
            res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
            res["FDR q-val"] = pd.to_numeric(res["FDR q-val"], errors="coerce")
            top = (res[res["NES"] > 0]
                   .sort_values(["FDR q-val", "NES"], ascending=[True, False])
                   .head(cfg.gsea_top_n))
            summary_rows.append(top)

    if not summary_rows:
        return pd.DataFrame()
    summary = pd.concat(summary_rows, ignore_index=True)
    keep = [c for c in ["zone", "collection", "Term", "NES", "NOM p-val",
                        "FDR q-val", "Lead_genes"] if c in summary.columns]
    summary = summary[keep]
    out = os.path.join(cfg.out_dir, "zonal_gsea_summary.csv")
    summary.to_csv(out, index=False)
    print(f"   wrote {out} ({len(summary)} top enriched terms)")
    _report_expected_biology(summary)
    return summary


def _report_expected_biology(summary: pd.DataFrame) -> None:
    """Sanity-check the hypothesized zone biology appears among enriched terms."""
    expect = {
        "LE/IT": ["neuron", "synap", "axon", "nervous"],
        "HBV/MVP": ["angiogen", "vascul", "blood", "immune", "inflamm"],
        "PAN/PNZ": ["hypoxia", "necro", "starvation", "glycolysis"],
        "CT": ["cell cycle", "dna repair", "replication", "e2f", "g2m", "myc"],
    }
    print("\n   --- expected zone biology check (Fig 2) ---")
    if "Term" not in summary.columns:
        return
    for zone, kws in expect.items():
        terms = summary.loc[summary["zone"] == zone, "Term"].astype(str).str.lower()
        hits = sorted({kw for kw in kws if terms.str.contains(kw).any()})
        flag = "OK " if hits else "-- "
        print(f"     {flag}{zone:>8}: expected~{kws}  found~{hits}")


# ----------------------------------------------------------------------------
# 5. ENTRY POINT
# ----------------------------------------------------------------------------
def main(cfg: Config = CFG) -> None:
    print("Phase 2: zonal signature extraction (Fig 1 variance + Fig 2 DGE/GSEA)")
    print("=" * 70)
    os.makedirs(cfg.out_dir, exist_ok=True)

    adata = load_ivygap(cfg)
    structure_variance_analysis(adata, cfg)
    dge = zonal_dge(adata, cfg)
    export_signatures_gmt(dge, cfg)
    zonal_gsea(dge, cfg)

    print("\nDone. Zonal signatures + GSEA written to ./%s/" % cfg.out_dir)


if __name__ == "__main__":
    main()
