#!/usr/bin/env python
"""Render Tables S1, S2, S3 from the generated CSVs."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

TABLES = Path(__file__).resolve().parents[1] / "data" / "derived" / "tables"


def _hr(title, width=96):
    print(f"\n=== {title} " + "=" * max(0, width - len(title) - 5))


def _fmt(value, places=3, signed=False):
    quantum = Decimal("1").scaleb(-places)
    text = format(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP), f".{places}f")
    if signed and not text.startswith("-"):
        text = "+" + text
    return text


def _s1_flag(row):
    if row.id == "E3":
        return "exp"
    if row.p_BH < 0.05:
        return "**"
    if row.p_BH < 0.10:
        return "*"
    return ""


def _support_flag(value):
    if value >= 0.95:
        return "**"
    if value >= 0.85:
        return "*"
    return ""


def _threshold_flag(value):
    if value >= 0.99:
        return "***"
    if value >= 0.95:
        return "**"
    return ""


def render_s1():
    df = pd.read_csv(TABLES / "empirical_tests.csv").sort_values("p_BH")
    _hr("Table S1: Empirical LME (per-family BH-corrected)")
    print(f"{'ID':<7}{'Claim':<45}{'Statistic':<28}{'p_raw':>8}{'p_BH':>8}  flag")
    for _, r in df.iterrows():
        flag = _s1_flag(r)
        print(f"{r.id:<7}{r.claim[:44]:<45}{r.statistic[:27]:<28}"
              f"{_fmt(r.p_raw):>8}{_fmt(r.p_BH):>8}  {flag}")


def render_s2():
    df = pd.read_csv(TABLES / "bootstrap_contrasts.csv").sort_values("pr_dir", ascending=False)
    _hr("Table S2: Optimization-prior contrasts (3000 hierarchical bootstrap reps)")
    print(f"{'ID':<6}{'Contrast':<28}{'HPR':>5} {'Population':<16}"
          f"{'Median':>8}  {'95% CI':<20}{'Pr_dir':>10}")
    for _, r in df.iterrows():
        ci = f"[{_fmt(r.ci_lo, 2, True)}, {_fmt(r.ci_hi, 2, True)}]"
        pr = f"{_fmt(r.pr_dir)}{_support_flag(r.pr_dir)}"
        print(f"{r.id:<6}{r.contrast[:27]:<28}{int(r.hpr):>5} {r.population:<16}"
              f"{_fmt(r['median'], 2, True):>8}  {ci:<20}{pr:>10}")


def render_s3():
    df = pd.read_csv(TABLES / "threshold_quantile_profile.csv")
    _hr("Table S3: Threshold-quintile Pr(NE > SH) on normalized utility")
    hprs = [30, 45, 60, 75, 90]
    head = f"{'Bin (dB SPL)':<14}{'n NE/SH':>10}  " + " ".join(f"HPR{h:>3}" for h in hprs)
    print(head)
    for _, r in df.iterrows():
        binlab = f"{r.thr_lo:.0f}--{r.thr_hi:.0f}"
        cells = " ".join(
            f"{_fmt(r[f'pr_ne_gt_sh_hpr{h}'])}{_threshold_flag(r[f'pr_ne_gt_sh_hpr{h}']):<3}"
            for h in hprs
        )
        print(f"{binlab:<14}{int(r.n_NE)}/{int(r.n_SH):<7}  {cells}")


def main():
    render_s1()
    render_s2()
    render_s3()
    print()


if __name__ == "__main__":
    main()
