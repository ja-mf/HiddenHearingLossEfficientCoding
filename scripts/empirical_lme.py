#!/usr/bin/env python
"""Empirical mixed-effects group comparisons (Table S1).
    PYTHONPATH=scripts python scripts/empirical_lme.py
"""
import warnings
from pathlib import Path

import pandas as pd

from lib.empirical import (
    prepare_hhl_stg, prepare_chl_stg, prepare_chl_rlf, prepare_hhl_rlf,
    compute_mean_spike_rate, run_empirical_tests,
)

ART = Path(__file__).resolve().parent.parent / "data" / "artifacts"
OUT_CSV = ART / "empirical_lme.csv"
COLS = ["id", "family", "claim", "statistic", "chi2", "df",
        "p_raw", "p_fdr", "main_beta", "main_p", "n_obs", "converged"]


def main():
    warnings.simplefilter("ignore")
    print("Loading artifacts...", flush=True)
    hhl_stg = prepare_hhl_stg()
    chl_stg = prepare_chl_stg()
    chl_rlf = prepare_chl_rlf()
    chl_rates = compute_mean_spike_rate(chl_rlf)
    hhl_rlf = prepare_hhl_rlf()
    print(f"  HHL stg: {hhl_stg['neuron-id'].nunique()} neurons, "
          f"animals={sorted(hhl_stg['animal'].unique())}", flush=True)

    res = run_empirical_tests(hhl_stg, chl_stg, chl_rates, hhl_rlf, chl_rlf)
    for c in COLS:
        if c not in res.columns:
            res[c] = pd.NA
    res = res[COLS].sort_values("p_fdr").reset_index(drop=True)

    with pd.option_context("display.max_colwidth", 50, "display.width", 170):
        print(res.to_string(index=False))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(res)} rows -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
