#!/usr/bin/env python
"""Hierarchical animal-balanced bootstrap of the population posterior.

For each cohort we draw `n_iter` resamples (with replacement) of animals,
then per-animal balanced neuron draws, refit the posterior over (beta, xi),
take the MAP, and aggregate posterior summaries.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from lib.posterior import (
    build_prior_slices,
    build_map_stats,
    calc_logpost_all,
    get_df_maxl,
    get_theta_hat,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "reproduction.yaml"
ARTIFACTS = PROJECT_ROOT / "data" / "artifacts"

ANIMAL_RE = re.compile(r"^([A-Z]+\d+[a-z]?)u")


def _animal_of(neuron_id: str) -> str:
    m = ANIMAL_RE.match(str(neuron_id))
    return m.group(1) if m else "UNK"


def _build_animal_maps(theta_reset, groups):
    animals_by_group = {}
    neurons_by_group = {}
    neurons_by_group_animal = {}
    for g in groups:
        sub = theta_reset[theta_reset["group"] == g]
        nids = sub["neuron-id"].unique().tolist()
        neurons_by_group[g] = np.array(nids, dtype=object)
        n_by_animal = {}
        for nid in nids:
            n_by_animal.setdefault(_animal_of(nid), []).append(nid)
        animals_by_group[g] = np.array(list(n_by_animal.keys()), dtype=object)
        neurons_by_group_animal[g] = n_by_animal
    return animals_by_group, neurons_by_group, neurons_by_group_animal


def _bootstrap_iter(boot_iter, theta_reset, groups, prior_slices, df_mi_stats,
                    seed, sample_size_hhl, sample_size_chl,
                    animals_by_group, neurons_by_group_animal):
    rng = np.random.default_rng(seed + boot_iter)
    boot_rows = []
    for g in groups:
        ss = sample_size_hhl if g in ("NE", "SH") else sample_size_chl
        animals = animals_by_group[g]
        n_an = len(animals)
        sampled_animals = rng.choice(animals, size=n_an, replace=True)
        base = ss // n_an
        rem = ss % n_an
        parts = []
        for i, an in enumerate(sampled_animals):
            n_draws = base + (1 if i < rem else 0)
            if n_draws <= 0:
                continue
            pool = neurons_by_group_animal[g].get(an, [])
            if not pool:
                continue
            parts.append(rng.choice(np.array(pool, dtype=object), size=n_draws, replace=True))
        sampled = np.concatenate(parts)
        gd = theta_reset[theta_reset["group"] == g]
        for n in sampled:
            boot_rows.append(gd[gd["neuron-id"] == n])

    df_boot = pd.concat(boot_rows, ignore_index=True).set_index(["group", "neuron-id", "hpr"])
    logpost = calc_logpost_all(df_boot.reset_index().set_index(["group", "hpr"]), prior_slices)
    df_maxl = get_df_maxl(logpost)

    groups_in = df_maxl.index.get_level_values("group").unique()
    hprs = sorted(set(theta_reset["hpr"]))
    hhl = [g for g in ("NE", "SH") if g in groups_in]
    chl = [g for g in ("EP", "PP") if g in groups_in]
    out = []
    for grp_set in (hhl, chl):
        if not grp_set:
            continue
        st = build_map_stats(df_maxl, grp_set, hprs, df_mi_stats)
        if st is not None:
            st = st.reset_index()
            st["boot_iter"] = boot_iter
            out.append(st)
    if not out:
        return None
    return pd.concat(out, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True,
                        choices=["HHL", "CHL", "HHL_low", "HHL_high"])
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size-hhl", type=int, default=None)
    parser.add_argument("--sample-size-chl", type=int, default=None)
    parser.add_argument("--hprs", default="30,45,60,75,90")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    n_jobs = args.n_jobs or cfg["parallel"]["n_jobs"]
    n_iter = args.n_iter or cfg["bootstrap"]["n_iter"]
    ss_h = args.sample_size_hhl or cfg["bootstrap"]["sample_size_hhl"]
    ss_c = args.sample_size_chl or cfg["bootstrap"]["sample_size_chl"]
    split_db = float(cfg["bootstrap"].get("threshold_split_db", 70))
    hprs = [int(h) for h in args.hprs.split(",")]

    grid = cfg["sigmoid_grid"]
    x0_grid = np.arange(grid["threshold_db_min"], grid["threshold_db_max"] + grid["threshold_step_db"],
                        grid["threshold_step_db"], dtype=float)
    k_grid = np.round(np.linspace(grid["gain_min"], grid["gain_max"], grid["gain_levels"]), 2)

    df_mi_pdf = pd.read_parquet(ARTIFACTS / "df_mi_pdf.parquet").set_index(["hpr", "xi", "beta", "x0", "k"])
    df_mi_stats = pd.read_parquet(ARTIFACTS / "df_mi_stats.parquet").set_index(["hpr", "beta", "xi"])

    if args.cohort.startswith("HHL"):
        df_stg = pd.read_parquet(ARTIFACTS / "HHL_stg.parquet")
        if args.cohort in ("HHL_low", "HHL_high"):
            # Per-neuron median adapted threshold over ALL HPRs (incl. uniform -1),
            # split at split_db: low < cut, high >= cut. Median is taken before
            # restricting to the analysis HPRs.
            med = df_stg.groupby("neuron-id")["threshold"].median()
            keep = med[med < split_db].index if args.cohort == "HHL_low" else med[med >= split_db].index
            df_stg = df_stg[df_stg["neuron-id"].isin(keep)]
        df_stg = df_stg[df_stg["hpr"].isin(hprs)]
        groups = sorted(df_stg["group"].unique().tolist())
    else:
        df_stg = pd.read_parquet(ARTIFACTS / "CHL_stg.parquet")
        df_stg = df_stg[df_stg["hpr"].isin(hprs)]
        groups = sorted(df_stg["group"].unique().tolist())

    print(f"# Cohort {args.cohort}: {df_stg['neuron-id'].nunique()} neurons, "
          f"groups={groups}, hprs={hprs}")

    th = get_theta_hat(df_stg, x0_grid, k_grid)
    df_stg = df_stg.set_index(["group", "neuron-id", "hpr"])
    df_stg["x0"] = th["x0"].to_numpy()
    df_stg["k"] = th["k"].to_numpy()
    theta_reset = df_stg.reset_index()

    prior_slices = build_prior_slices(df_mi_pdf, hprs)
    abg, _, nbga = _build_animal_maps(theta_reset, groups)

    print(f"# Running {n_iter} bootstrap iterations on {n_jobs} workers")
    parts = Parallel(n_jobs=n_jobs, backend="loky", verbose=2)(
        delayed(_bootstrap_iter)(
            i, theta_reset, groups, prior_slices, df_mi_stats,
            args.seed, ss_h, ss_c, abg, nbga,
        ) for i in range(n_iter)
    )
    parts = [p for p in parts if p is not None]
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"avg_utility": "avg_utility",
                            "norm_avg_utility": "norm_avg_utility"})
    df["mean_fr"] = df["mean_fr"].astype(float)
    out = Path(args.output) if args.output else ARTIFACTS / f"{args.cohort}_bootstrap.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"# Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
