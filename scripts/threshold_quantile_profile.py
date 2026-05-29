#!/usr/bin/env python
"""Table S4: threshold-resolved directional support for the HHL utility contrast.

Splits HHL neurons into equal-sized quintiles of across-context median adapted
threshold, repeats the optimization-prior bootstrap within each quintile, and
reports Pr(NE>SH) on normalized utility per HPR. Writes
data/derived/tables/threshold_quantile_profile.csv.

    PYTHONPATH=scripts python scripts/threshold_quantile_profile.py [--n-jobs N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from lib.posterior import build_prior_slices, get_theta_hat
from run_bootstrap import _bootstrap_iter, _build_animal_maps

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
CFG = yaml.safe_load((ROOT / "config" / "reproduction.yaml").read_text())
HPRS = [30, 45, 60, 75, 90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=CFG["parallel"]["n_jobs"])
    args = ap.parse_args()
    b = CFG["bootstrap"]
    n_iter, seed, ss_h, ss_c = b["n_iter"], b["seed"], b["sample_size_hhl"], b["sample_size_chl"]
    g = CFG["sigmoid_grid"]
    x0 = np.arange(g["threshold_db_min"], g["threshold_db_max"] + g["threshold_step_db"],
                   g["threshold_step_db"], dtype=float)
    kg = np.round(np.linspace(g["gain_min"], g["gain_max"], g["gain_levels"]), 2)

    df_mi_pdf = pd.read_parquet(ART / "df_mi_pdf.parquet").set_index(["hpr", "xi", "beta", "x0", "k"])
    df_mi_stats = pd.read_parquet(ART / "df_mi_stats.parquet").set_index(["hpr", "beta", "xi"])
    prior = build_prior_slices(df_mi_pdf, HPRS)

    stg = pd.read_parquet(ART / "HHL_stg.parquet")
    med = stg.groupby("neuron-id")["threshold"].median()         # across-context median, per neuron
    grp_of = dict(zip(stg["neuron-id"], stg["group"]))
    sh = stg[stg.hpr.isin(HPRS)].copy()
    th = get_theta_hat(sh, x0, kg)
    sh = sh.set_index(["group", "neuron-id", "hpr"])
    sh["x0"], sh["k"] = th["x0"].to_numpy(), th["k"].to_numpy()
    theta = sh.reset_index()

    edges = [-np.inf] + list(med.quantile([.2, .4, .6, .8]).values) + [np.inf]
    rows = []
    for i in range(5):
        ids = med[(med > edges[i]) & (med <= edges[i + 1])].index
        tr = theta[theta["neuron-id"].isin(ids)].copy()
        groups = sorted(tr["group"].unique().tolist())
        abg, _, nbga = _build_animal_maps(tr, groups)
        parts = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_bootstrap_iter)(j, tr, groups, prior, df_mi_stats, seed, ss_h, ss_c, abg, nbga)
            for j in range(n_iter))
        d = pd.concat([p for p in parts if p is not None], ignore_index=True)
        row = {"quintile": i + 1,
               "thr_lo": round(float(med[ids].min()), 1), "thr_hi": round(float(med[ids].max()), 1),
               "n_NE": sum(grp_of[n] == "NE" for n in ids), "n_SH": sum(grp_of[n] == "SH" for n in ids)}
        for h in HPRS:
            a = d[(d.hpr == h) & (d.group == "SH")].set_index("boot_iter")["norm_avg_utility"]
            bb = d[(d.hpr == h) & (d.group == "NE")].set_index("boot_iter")["norm_avg_utility"]
            delta = (a - bb).dropna()
            row[f"pr_ne_gt_sh_hpr{h}"] = round(float((delta < 0).mean()), 3)
            row[f"median_delta_hpr{h}"] = round(float(delta.median()), 3)
            # 95% bootstrap CI on delta utility (same convention as Panel E1/E2)
            row[f"ci_lo_hpr{h}"] = round(float(delta.quantile(0.025)), 3)
            row[f"ci_hi_hpr{h}"] = round(float(delta.quantile(0.975)), 3)
        rows.append(row)
        print(f"# quintile {i+1} ({row['thr_lo']}-{row['thr_hi']} dB): "
              + " ".join(f"H{h}={row[f'pr_ne_gt_sh_hpr{h}']:.2f}" for h in HPRS), flush=True)

    out = ROOT / "data" / "derived" / "tables" / "threshold_quantile_profile.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"# Wrote {out}")


if __name__ == "__main__":
    main()
