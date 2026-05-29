#!/usr/bin/env python
"""Generate the statistical table CSVs from the shipped artifacts.

Writes:
  data/derived/tables/empirical_tests.csv     -- Table S1 (16-test panel, per-family BH)
  data/derived/tables/bootstrap_contrasts.csv -- Table S2 (OP contrasts)

    PYTHONPATH=scripts python scripts/reproduce_tables.py
"""
import warnings
from pathlib import Path

import pandas as pd

from lib.paths import DERIVED, DATA
from lib.empirical import (
    prepare_hhl_stg, prepare_chl_stg, prepare_chl_rlf, prepare_hhl_rlf,
    compute_mean_spike_rate, run_empirical_tests,
)

ART = DATA / "artifacts"
OUT = DERIVED / "tables"
NU = "norm_avg_utility"


def _col(df, var, hpr, g):
    return df[(df.hpr == hpr) & (df.group == g)].set_index("boot_iter")[var]


def _summ(d, negative):
    pr = float((d < 0).mean()) if negative else float((d > 0).mean())
    return float(d.median()), float(d.quantile(0.025)), float(d.quantile(0.975)), pr


def contrast(df, var, hpr, gA, gB, negative):
    return _summ((_col(df, var, hpr, gA) - _col(df, var, hpr, gB)).dropna(), negative)


def transition(df, var, h0, h1):
    d = ((_col(df, var, h1, "SH") - _col(df, var, h0, "SH")) -
         (_col(df, var, h1, "NE") - _col(df, var, h0, "NE"))).dropna()
    return _summ(d, negative=False)


def dod(lo, hi, var, hpr):
    d = ((_col(hi, var, hpr, "NE") - _col(lo, var, hpr, "NE")) -
         (_col(hi, var, hpr, "SH") - _col(lo, var, hpr, "SH"))).dropna()
    return _summ(d, negative=False)


def empirical_table() -> pd.DataFrame:
    res = run_empirical_tests(
        prepare_hhl_stg(), prepare_chl_stg(),
        compute_mean_spike_rate(prepare_chl_rlf()), prepare_hhl_rlf(), prepare_chl_rlf(),
    )
    res = res.rename(columns={"p_fdr": "p_BH"})
    cols = ["id", "family", "claim", "statistic", "chi2", "df", "p_raw", "p_BH",
            "main_beta", "main_p", "n_obs", "converged"]
    for c in cols:
        if c not in res.columns:
            res[c] = pd.NA
    return res[cols].sort_values("p_BH").reset_index(drop=True)


def bootstrap_table() -> pd.DataFrame:
    hhl = pd.read_parquet(ART / "HHL_bootstrap.parquet")
    lo = pd.read_parquet(ART / "HHL_low_bootstrap.parquet")
    hi = pd.read_parquet(ART / "HHL_high_bootstrap.parquet")
    rows = [
        ("HD1", "Utility (SH-NE)", 30, "All units", *contrast(hhl, NU, 30, "SH", "NE", True)),
        ("HH1", "Entropy (SH-NE)", 30, "All units", *contrast(hhl, "entropy", 30, "SH", "NE", False)),
        ("HH2", "Entropy (SH-NE)", 45, "All units", *contrast(hhl, "entropy", 45, "SH", "NE", False)),
        ("HH3", "Entropy (SH-NE)", 60, "All units", *contrast(hhl, "entropy", 60, "SH", "NE", False)),
        ("HH4", "Entropy (SH-NE)", 75, "All units", *contrast(hhl, "entropy", 75, "SH", "NE", False)),
        ("HH5", "Entropy (SH-NE)", 90, "All units", *contrast(hhl, "entropy", 90, "SH", "NE", False)),
        ("HF1", "Mean FR (SH-NE)", 30, "All units", *contrast(hhl, "mean_fr", 30, "SH", "NE", False)),
        ("HF2", "Mean FR (SH-NE)", 45, "All units", *contrast(hhl, "mean_fr", 45, "SH", "NE", False)),
        ("HF3", "Mean FR (SH-NE)", 60, "All units", *contrast(hhl, "mean_fr", 60, "SH", "NE", False)),
        ("HF4", "Mean FR (SH-NE)", 75, "All units", *contrast(hhl, "mean_fr", 75, "SH", "NE", False)),
        ("HF5", "Mean FR (SH-NE)", 90, "All units", *contrast(hhl, "mean_fr", 90, "SH", "NE", True)),
        ("TX1", "Utility transition 30->45", 45, "All units", *transition(hhl, NU, 30, 45)),
        ("TX2", "Utility transition 30->60", 60, "All units", *transition(hhl, NU, 30, 60)),
        ("TX3", "Utility transition 30->75", 75, "All units", *transition(hhl, NU, 30, 75)),
        ("TX4", "Utility transition 30->90", 90, "All units", *transition(hhl, NU, 30, 90)),
        ("HE1", "Utility (SH-NE)", 30, "Low-threshold", *contrast(lo, NU, 30, "SH", "NE", True)),
        ("HE2", "Utility (SH-NE)", 45, "Low-threshold", *contrast(lo, NU, 45, "SH", "NE", True)),
        ("HE3", "Utility (SH-NE)", 60, "Low-threshold", *contrast(lo, NU, 60, "SH", "NE", True)),
        ("HE4", "Utility (SH-NE)", 75, "Low-threshold", *contrast(lo, NU, 75, "SH", "NE", True)),
        ("HE6", "Utility (SH-NE)", 90, "Low-threshold", *contrast(lo, NU, 90, "SH", "NE", True)),
        ("S1", "Utility (SH-NE)", 30, "High-threshold", *contrast(hi, NU, 30, "SH", "NE", True)),
        ("S2", "Utility (SH-NE)", 45, "High-threshold", *contrast(hi, NU, 45, "SH", "NE", True)),
        ("S3", "Utility (SH-NE)", 60, "High-threshold", *contrast(hi, NU, 60, "SH", "NE", True)),
        ("S4", "Utility (SH-NE)", 75, "High-threshold", *contrast(hi, NU, 75, "SH", "NE", True)),
        ("S5", "Utility (SH-NE)", 90, "High-threshold", *contrast(hi, NU, 90, "SH", "NE", False)),
        ("HE5", "Degree of divergence", 60, "Stratified", *dod(lo, hi, NU, 60)),
    ]
    return pd.DataFrame(rows, columns=["id", "contrast", "hpr", "population",
                                       "median", "ci_lo", "ci_hi", "pr_dir"])


def main():
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    emp = empirical_table()
    boot = bootstrap_table()
    emp.to_csv(OUT / "empirical_tests.csv", index=False)
    boot.to_csv(OUT / "bootstrap_contrasts.csv", index=False)
    print(f"# Wrote {OUT/'empirical_tests.csv'} ({len(emp)} rows)")
    print(f"# Wrote {OUT/'bootstrap_contrasts.csv'} ({len(boot)} rows)")
    print("\nTable S2 (bootstrap contrasts):")
    with pd.option_context("display.width", 140):
        print(boot.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
