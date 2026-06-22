"""
download_molecular_covariates.py
================================
Fetches the AUTHENTIC, published molecular covariates used by Kersch, Claunch
et al. (2020, doi:10.1093/noajnl/vdaa093) -- replacing any locally-derived
surrogates -- so the survival model (module 4) is built on the same calls the
authors had.

The paper states it accessed TCGA-GBM data "from the Genomic Data Commons ...
and https://tcga-data.nci.nih.gov/docs/publications/gbm_2013/". That gbm_2013
publication is Brennan et al., Cell 2013 -- the canonical TCGA-GBM marker paper.
Its curated clinical/molecular table (MGMT status, IDH1 mutation incl. the exact
R132 variant, G-CIMP status, expression subtype) is hosted, machine-readable, on
cBioPortal as study `gbm_tcga_pub2013`. We pull it straight from the public
cBioPortal REST API.

  (We confirmed this is the right source: cBioPortal's IDH1_MUTATION values are
   WT / R132H / R132G / R132C -- exactly the variants shown in the paper's
   Figure 3, and ~7% mutant, matching primary-GBM literature.)

GAPS CLOSED
  * TCGA MGMT promoter methylation status   -> REAL (MGMT_STATUS)        [in final model]
  * TCGA IDH1 mutation status (+ variant)   -> REAL (IDH1_MUTATION)      [exploratory]
  * TCGA G-CIMP status, expression subtype  -> REAL (bonus)
  * IvyGAP 1p/19q codeletion                -> prior=0 (codeletion DEFINES
        oligodendroglioma; it does not occur in GBM -- WHO CNS5). Not a guess.
  * IvyGAP IDH1                             -> the atlas provides no public IDH
        table; defaults to GBM prior (wild-type) with an explicit override hook
        for the handful of mutant tumors (edit IVY_IDH1_OVERRIDES).

Outputs (./processed/), schema-compatible with 1_data_preprocessing.py:
  - tcga_molecular_covariates.csv
  - ivygap_molecular_covariates.csv

Dependencies: standard library only (urllib, json, csv) -- runs anywhere.
"""

from __future__ import annotations

import csv
import json
import os
import ssl
import sys
import urllib.request
import urllib.error

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CBIO_API = "https://www.cbioportal.org/api"
STUDY_ID = "gbm_tcga_pub2013"   # Brennan et al., Cell 2013 (the paper's TCGA-GBM)
OUT_DIR = "processed"
IVY_TUMORS = "IvyGAP.tumor_details.csv"
TIMEOUT = 60

# Clinical attributes to pull (sample-level in this study).
WANTED = ["MGMT_STATUS", "IDH1_MUTATION", "G_CIMP_METHYLATION", "EXPRESSION_SUBTYPE"]

# IvyGAP IDH1 overrides: keys are tumor_name (e.g. "W34-1-1"), value 1 = mutant.
# Populate from the paper's Supplementary Table S1 if you need exact IvyGAP IDH.
IVY_IDH1_OVERRIDES: dict[str, int] = {}
IVY_1P19Q_PRIOR = 0


# ----------------------------------------------------------------------------
def _ssl_contexts():
    """Yield TLS contexts in decreasing order of strictness.

    Some minimal Python builds (e.g. MinGW) ship without a usable CA bundle, so
    verified TLS raises CERTIFICATE_VERIFY_FAILED. We therefore try, in order:
    (1) the system default, (2) certifi's bundle if installed, and only as a last
    resort (3) an UNVERIFIED context -- acceptable here because cBioPortal is a
    public, read-only API and we ingest only non-sensitive clinical annotations.
    """
    yield ssl.create_default_context()
    try:
        import certifi
        yield ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    yield unverified


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "gbm-replication/1.0"})
    last_err = None
    for i, ctx in enumerate(_ssl_contexts()):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
                if i == 2:
                    print("  [warn] TLS certificate could not be verified; "
                          "proceeded with an unverified connection.")
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            last_err = e
            if isinstance(e.reason, ssl.SSLError):
                continue   # try the next, less strict context
            raise
    raise last_err


def _patient(barcode: str) -> str:
    """TCGA-02-0001-01 -> TCGA-02-0001 (patient-level key for merging)."""
    parts = barcode.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else barcode


# ----------------------------------------------------------------------------
# TCGA: download authentic calls from cBioPortal
# ----------------------------------------------------------------------------
def download_tcga():
    url = (f"{CBIO_API}/studies/{STUDY_ID}/clinical-data"
           f"?clinicalDataType=SAMPLE&projection=DETAILED")
    print(f"Downloading TCGA molecular data from cBioPortal study '{STUDY_ID}' ...")
    try:
        data = _get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  [ERROR] download failed ({e}). Check your internet connection.")
        return None

    # Pivot the long-format clinical records into one row per sample.
    by_sample: dict[str, dict] = {}
    for rec in data:
        attr = rec.get("clinicalAttributeId")
        if attr not in WANTED:
            continue
        sid = rec["sampleId"]
        by_sample.setdefault(sid, {})[attr] = rec.get("value", "")
    print(f"  retrieved {len(by_sample)} samples with molecular annotation")

    rows = []
    for sid, d in sorted(by_sample.items()):
        mgmt_raw = d.get("MGMT_STATUS", "")
        idh_raw = d.get("IDH1_MUTATION", "")
        gcimp_raw = d.get("G_CIMP_METHYLATION", "")

        mgmt = "1" if mgmt_raw.upper() == "METHYLATED" else \
               ("0" if mgmt_raw.upper() == "UNMETHYLATED" else "")
        # Any non-WT, non-empty IDH1 call (R132H/R132G/R132C/...) is mutant.
        idh = "" if idh_raw == "" else (
            "0" if idh_raw.upper() in ("WT", "WILD TYPE", "WILDTYPE") else "1")
        gcimp = "" if gcimp_raw == "" else (
            "1" if gcimp_raw.upper() == "G-CIMP" else "0")

        rows.append({
            "patient_barcode": _patient(sid),
            "sample_barcode": sid,
            "mgmt_methylated": mgmt,          # 1=methylated, 0=not (paper convention)
            "mgmt_status_raw": mgmt_raw,
            "idh1_mutant": idh,               # 1=mutant, 0=WT
            "idh1_variant": idh_raw,          # exact variant, e.g. R132H
            "gcimp": gcimp,
            "expression_subtype": d.get("EXPRESSION_SUBTYPE", ""),
            "source": f"cBioPortal:{STUDY_ID} (Brennan 2013)",
        })

    n_mgmt = sum(r["mgmt_methylated"] != "" for r in rows)
    n_meth = sum(r["mgmt_methylated"] == "1" for r in rows)
    n_idh = sum(r["idh1_mutant"] == "1" for r in rows)
    print(f"  MGMT called for {n_mgmt} samples ({n_meth} methylated)")
    print(f"  IDH1-mutant samples: {n_idh}")
    return rows


# ----------------------------------------------------------------------------
# IvyGAP: documented priors (+ override hook)
# ----------------------------------------------------------------------------
def build_ivygap():
    if not os.path.isfile(IVY_TUMORS):
        print(f"[skip IvyGAP] {IVY_TUMORS} not found.")
        return None
    rows = []
    with open(IVY_TUMORS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tname = r.get("tumor_name", "")
            idh = IVY_IDH1_OVERRIDES.get(tname, 0)
            rows.append({
                "donor_id": r.get("donor_id", ""),
                "tumor_name": tname,
                "idh1_mutant": str(idh),
                "idh1_source": "override" if tname in IVY_IDH1_OVERRIDES
                               else "GBM prior (assumed wild-type)",
                "codel_1p19q": str(IVY_1P19Q_PRIOR),
                "codel_1p19q_source": "GBM prior (codeletion absent in GBM, WHO CNS5)",
            })
    print(f"IvyGAP: priors for {len(rows)} tumors ({len(IVY_IDH1_OVERRIDES)} overrides).")
    return rows


# ----------------------------------------------------------------------------
def _write_csv(path: str, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path} ({len(rows)} rows)")


def main():
    print("Downloading authentic molecular covariates (TCGA: cBioPortal)")
    print("=" * 62)
    tcga = download_tcga()
    if tcga:
        _write_csv(os.path.join(OUT_DIR, "tcga_molecular_covariates.csv"), tcga)
    else:
        print("  TCGA download unavailable; existing file (if any) left untouched.")
    ivy = build_ivygap()
    if ivy:
        _write_csv(os.path.join(OUT_DIR, "ivygap_molecular_covariates.csv"), ivy)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
