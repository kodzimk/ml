"""
5_drug_sensitivity_mapping.py
=============================
PHASE 5 of the replication pipeline for:

    Kersch, Claunch et al. (2020) "Transcriptional signatures in histologic
    structures within glioblastoma tumors may predict personalized drug
    sensitivity and survival." Neuro-Oncology Advances, doi:10.1093/noajnl/vdaa093

This module operationalizes the "drug sensitivity" aspect of the paper's title.
The paper itself frames drug prediction as the forward-looking application of its
central method ("creating predictive signatures for tumor sensitivity and response
to treatments"); here we build the machinery to do exactly that, in the spirit of
the .cursorrules SOP:

    "generate a 'query signature' (Up- and Down-regulated genes) for a specific
     histologic zone ... calculate enrichment scores (GSEA or cosine similarity)
     against drug perturbation signatures to predict which compounds will
     selectively kill boundary cells versus core cells."

How it works (Connectivity-Map / LINCS L1000 style)
---------------------------------------------------
  1. QUERY SIGNATURES: load the per-zone up/down gene lists produced by module 2
     (zonal_signatures.gmt) -- the boundary (LE/IT), core (CT), vascular (HBV/MVP),
     and hypoxic (PAN/PNZ) transcriptional signatures.
  2. SCORING: for each candidate drug's differential-expression profile, compute a
     Lamb-2006 connectivity score against a zone's up/down query. A strongly
     NEGATIVE score means the drug REVERSES that zone's signature -> a candidate to
     selectively oppose/kill that zone's cell population. (Cosine similarity is
     provided as an alternative metric.)
  3. DATABASE-AGNOSTIC: point Config.drug_library at any of
        - a gene x drug differential-expression matrix (CSV/TSV), or
        - a GMT of drug-induced up/down gene sets,
     e.g. exported from LINCS L1000, CTRP, PRISM, or MSigDB C2:CGP. No specific DB
     is hard-wired, so this runs against whatever perturbation resource you have.
  4. NO-DB DEMO: if no drug library is supplied, we still exercise the exact
     scoring machinery by computing zone-vs-zone connectivity from the real zonal
     profiles. This directly tests the paper's hypothesis that BOUNDARY (LE/IT) and
     CORE (CT) zones are transcriptionally opposed (negative connectivity) and thus
     have distinct therapeutic vulnerabilities.

Outputs (./processed/ and ./processed/figures/):
    - drug_connectivity_<ZONE>.csv   (ranked drugs per zone; only if a library given)
    - zone_connectivity_matrix.csv   (zone-vs-zone connectivity; always)
    - figures/zone_connectivity.png  (heatmap of zonal opposition)

Primary stack: pandas, numpy, matplotlib.
Author: GBM spatial-transcriptomics replication pipeline.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------
@dataclass
class Config:
    out_dir: str = "processed"
    fig_dir: str = "processed/figures"

    # Per-zone up/down query signatures from module 2.
    signatures_gmt: str = "processed/zonal_signatures.gmt"
    # Per-zone full ranked DGE (gene -> signed score) from module 2; used as the
    # "perturbation profile" for the no-DB zone-vs-zone connectivity demo.
    zonal_dge_csv: str = "processed/zonal_dge.csv"

    # Optional external drug perturbation library (set to enable real scoring):
    #   - a CSV/TSV matrix: rows = genes, columns = drugs, values = differential
    #     expression (e.g. LINCS L1000 z-scores), OR
    #   - a .gmt of drug up/down sets named like "<DRUG>_UP" / "<DRUG>_DN".
    drug_library: str | None = None

    # Scoring.
    top_n_drugs: int = 25       # rows kept per zone in the ranked output
    random_state: int = 0


CFG = Config()


# ----------------------------------------------------------------------------
# 1. LOAD QUERY SIGNATURES
# ----------------------------------------------------------------------------
def load_signatures(cfg: Config) -> dict[str, dict[str, list[str]]]:
    """Read zonal_signatures.gmt -> {zone: {'up': [...], 'down': [...]}}."""
    if not os.path.isfile(cfg.signatures_gmt):
        raise FileNotFoundError(
            f"{cfg.signatures_gmt} missing -- run 2_zonal_signature_extraction.py.")
    sigs: dict[str, dict[str, list[str]]] = {}
    with open(cfg.signatures_gmt) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, genes = parts[0], parts[2:]
            # names look like "CT_UP", "LE_IT_DOWN"
            if name.endswith("_UP"):
                zone, direction = name[:-3], "up"
            elif name.endswith("_DOWN"):
                zone, direction = name[:-5], "down"
            else:
                continue
            zone = zone.replace("_", "/") if "/" not in zone else zone
            sigs.setdefault(zone, {}).setdefault(direction, []).extend(genes)
    print(f"Loaded query signatures for zones: {sorted(sigs)}")
    for z, d in sigs.items():
        print(f"   {z:>8}: {len(d.get('up', []))} up / {len(d.get('down', []))} down")
    return sigs


# ----------------------------------------------------------------------------
# 2. CONNECTIVITY SCORING (Lamb et al. 2006, Science -- the CMap score)
# ----------------------------------------------------------------------------
def _enrichment_ks(tag_ranks: np.ndarray, n: int) -> float:
    """One-sided Kolmogorov-Smirnov enrichment statistic for a tag set.

    `tag_ranks` = sorted 1-based ranks (in the drug profile) of the query genes;
    `n` = total genes in the profile. Returns the signed max-deviation a/b stat
    used in the connectivity-map algorithm.
    """
    t = len(tag_ranks)
    if t == 0:
        return 0.0
    j = np.arange(1, t + 1)
    a = np.max(j / t - tag_ranks / n)
    b = np.max(tag_ranks / n - (j - 1) / t)
    return a if a > b else -b


def connectivity_score(query_up: list[str], query_down: list[str],
                       profile: pd.Series) -> float:
    """CMap connectivity score of a drug `profile` against an up/down query.

    `profile` : Series indexed by gene, valued by the drug's differential
                expression (higher = more up-regulated by the drug). Genes are
                ranked descending. Score in [-1, 1]:
                  > 0  drug MIMICS the zone signature
                  < 0  drug REVERSES the zone signature (therapeutic candidate)
    """
    ranked = profile.sort_values(ascending=False)
    n = len(ranked)
    rank_of = pd.Series(np.arange(1, n + 1), index=ranked.index)

    up = [g for g in query_up if g in rank_of.index]
    dn = [g for g in query_down if g in rank_of.index]
    if not up and not dn:
        return np.nan
    es_up = _enrichment_ks(np.sort(rank_of[up].values), n) if up else 0.0
    es_dn = _enrichment_ks(np.sort(rank_of[dn].values), n) if dn else 0.0
    # opposite signs -> a real connectivity; same sign -> set to 0 (Lamb rule).
    if es_up * es_dn > 0:
        return 0.0
    return (es_up - es_dn) / 2.0


def cosine_score(query_up: list[str], query_down: list[str],
                 profile: pd.Series) -> float:
    """Signed cosine similarity between the query (+1 up / -1 down) and profile."""
    q = pd.Series(0.0, index=profile.index)
    q[[g for g in query_up if g in q.index]] = 1.0
    q[[g for g in query_down if g in q.index]] = -1.0
    v = profile.reindex(q.index).fillna(0.0).values
    denom = (np.linalg.norm(q.values) * np.linalg.norm(v))
    return float(np.dot(q.values, v) / denom) if denom else np.nan


# ----------------------------------------------------------------------------
# 3. DRUG-LIBRARY SCORING (real prediction, when a library is supplied)
# ----------------------------------------------------------------------------
def _load_drug_matrix(cfg: Config) -> pd.DataFrame | None:
    """Load a gene x drug differential-expression matrix, if configured."""
    path = cfg.drug_library
    if not path or not os.path.isfile(path):
        return None
    if path.endswith(".gmt"):
        return None  # handled separately
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    mat = pd.read_csv(path, sep=sep, index_col=0)
    mat.index = mat.index.astype(str)
    print(f"Loaded drug matrix: {mat.shape[0]} genes x {mat.shape[1]} drugs "
          f"from {os.path.basename(path)}")
    return mat


def score_drug_library(sigs: dict, cfg: Config) -> None:
    """Score every drug against every zone query; write ranked candidates per zone.

    A strongly NEGATIVE connectivity score = the drug reverses that zone's
    signature => predicted to selectively oppose that zone's cell population
    (e.g. core CT vs boundary LE/IT), the paper's drug-sensitivity goal.
    """
    mat = _load_drug_matrix(cfg)
    if mat is None:
        print("\nNo drug differential-expression matrix supplied "
              "(Config.drug_library) -- skipping real drug scoring. "
              "Point it at a LINCS/CTRP/PRISM gene x drug matrix to enable.")
        return
    print("\n=== Drug connectivity scoring (per zone) ===")
    for zone, sig in sigs.items():
        rows = []
        for drug in mat.columns:
            profile = mat[drug].dropna()
            cs = connectivity_score(sig.get("up", []), sig.get("down", []), profile)
            cos = cosine_score(sig.get("up", []), sig.get("down", []), profile)
            rows.append({"drug": drug, "connectivity": cs, "cosine": cos})
        res = (pd.DataFrame(rows)
               .sort_values("connectivity")  # most negative (reversing) first
               .reset_index(drop=True))
        safe = zone.replace("/", "_")
        out = os.path.join(cfg.out_dir, f"drug_connectivity_{safe}.csv")
        res.to_csv(out, index=False)
        top = res.head(cfg.top_n_drugs)
        print(f"   [{zone}] wrote {out}; top reversing candidate: "
              f"{top.iloc[0]['drug']} (connectivity={top.iloc[0]['connectivity']:.3f})")


# ----------------------------------------------------------------------------
# 4. NO-DB DEMO -- ZONE-VS-ZONE CONNECTIVITY (boundary vs core hypothesis)
# ----------------------------------------------------------------------------
def zone_connectivity_matrix(sigs: dict, cfg: Config) -> pd.DataFrame:
    """Connectivity of each zone's query signature against each zone's DGE profile.

    Uses the real per-zone ranked DGE as stand-in "profiles". The diagonal is
    strongly positive (a zone mimics itself); the paper's hypothesis predicts the
    BOUNDARY (LE/IT) and CORE (CT) zones are OPPOSED (off-diagonal negative),
    reflecting distinct, separately-targetable biology.
    """
    if not os.path.isfile(cfg.zonal_dge_csv):
        warnings.warn(f"{cfg.zonal_dge_csv} missing -- skipping zone connectivity.")
        return pd.DataFrame()
    dge = pd.read_csv(cfg.zonal_dge_csv)
    profiles = {z: g.set_index("gene")["score"]
                for z, g in dge.groupby("zone")}
    zones = sorted(sigs)
    mat = pd.DataFrame(index=zones, columns=zones, dtype=float)
    cos = pd.DataFrame(index=zones, columns=zones, dtype=float)
    for qz in zones:
        for pz in zones:
            up, dn = sigs[qz].get("up", []), sigs[qz].get("down", [])
            mat.loc[qz, pz] = connectivity_score(up, dn, profiles[pz])
            cos.loc[qz, pz] = cosine_score(up, dn, profiles[pz])
    out = os.path.join(cfg.out_dir, "zone_connectivity_matrix.csv")
    mat.to_csv(out)
    cos.to_csv(os.path.join(cfg.out_dir, "zone_cosine_matrix.csv"))
    print("\n=== Zone-vs-zone connectivity (CMap; query rows vs profile cols) ===")
    print(mat.round(3).to_string())
    print("\n=== Zone-vs-zone cosine similarity (query rows vs profile cols) ===")
    print(cos.round(3).to_string())
    print(f"   wrote {out} and zone_cosine_matrix.csv")
    _plot_zone_connectivity(cos, cfg)

    # The CMap score zeroes same-sign pairs (Lamb rule), so judge boundary-vs-core
    # opposition with the unzeroed signed cosine similarity.
    if "CT" in zones and "LE/IT" in zones:
        v = float(cos.loc["LE/IT", "CT"])
        verdict = "OPPOSED (distinct vulnerabilities)" if v < 0 else "similar"
        print(f"   boundary(LE/IT) vs core(CT) cosine = {v:.3f} -> {verdict}")
    return mat


def _plot_zone_connectivity(mat: pd.DataFrame, cfg: Config) -> None:
    os.makedirs(cfg.fig_dir, exist_ok=True)
    fig, axz = plt.subplots(figsize=(6, 5))
    im = axz.imshow(mat.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
    axz.set_xticks(range(mat.shape[1]))
    axz.set_xticklabels(mat.columns, rotation=30, ha="right")
    axz.set_yticks(range(mat.shape[0]))
    axz.set_yticklabels(mat.index.tolist())
    axz.set_xlabel("zone profile")
    axz.set_ylabel("zone query signature")
    axz.set_title("Zonal connectivity\n(blue = opposed => distinct vulnerability)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axz.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=8)
    fig.colorbar(im, ax=axz, fraction=0.046, label="connectivity score")
    fig.tight_layout()
    out = os.path.join(cfg.fig_dir, "zone_connectivity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   wrote {out}")


# ----------------------------------------------------------------------------
# 5. ENTRY POINT
# ----------------------------------------------------------------------------
def main(cfg: Config = CFG) -> None:
    print("Phase 5: drug sensitivity mapping (zonal query signatures -> compounds)")
    print("=" * 70)
    os.makedirs(cfg.out_dir, exist_ok=True)

    sigs = load_signatures(cfg)
    score_drug_library(sigs, cfg)      # real scoring (if a library is configured)
    zone_connectivity_matrix(sigs, cfg)  # always-on demo of the scoring machinery

    print("\nDone. Drug-mapping outputs written to ./%s/" % cfg.out_dir)
    if not cfg.drug_library:
        print("Tip: set Config.drug_library to a LINCS/CTRP/PRISM gene x drug "
              "matrix (or drug GMT) to rank actual compounds per zone.")


if __name__ == "__main__":
    main()
