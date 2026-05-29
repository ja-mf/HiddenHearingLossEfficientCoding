#!/usr/bin/env python
"""Compute the optimization-prior table over (hpr, beta, xi, x0, k).

Outputs:
  data/artifacts/df_mi_pdf.parquet   -- prior table U, pdf, mi, mu_rt, std_rt
  data/artifacts/df_mi_stats.parquet -- per-(hpr, beta, xi) summaries

Caching: joblib.Memory at .cache/ for the inner Monte-Carlo step.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Memory
from sklearn.metrics import mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "reproduction.yaml"
ARTIFACTS = PROJECT_ROOT / "data" / "artifacts"
CACHE_DIR = PROJECT_ROOT / ".cache" / "mi"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(str(CACHE_DIR), verbose=0)


def _logistsig(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _pmf(levels):
    hp, lp = 0.8 / 5, 0.2 / 20
    return {
        "90": np.vstack((levels, np.array([lp] * 20 + [hp] * 5))).T,
        "75": np.vstack((levels, np.array([lp] * 15 + [hp] * 5 + [lp] * 5))).T,
        "60": np.vstack((levels, np.array([lp] * 10 + [hp] * 5 + [lp] * 10))).T,
        "45": np.vstack((levels, np.array([lp] * 5 + [hp] * 5 + [lp] * 15))).T,
        "30": np.vstack((levels, np.array([hp] * 5 + [lp] * 20))).T,
        "-1": np.vstack((levels, np.array([1 / 25] * 25))).T,
    }


def _contextualise_stim(x, pmf_arr):
    """Bin stimulus levels relative to the PMF's high-probability plateau.

    0 = below the plateau, 1 = inside it, 2 = above it. The plateau is read
    from the PMF itself (levels carrying the maximum probability), so it tracks
    the true high-probability region.
    """
    prob = pmf_arr[:, 1]
    plateau = pmf_arr[prob == prob.max(), 0]
    lo, hi = float(plateau.min()), float(plateau.max())
    return np.where(x < lo, 0, np.where(x > hi, 2, 1)).astype(int)


@memory.cache
def _mi_table(hprs, x0_grid, k_grid, n_draws, seed):
    """Monte-Carlo MI over (hpr, x0, k) with a single sequential RNG. 
    Returns a DataFrame indexed by (hpr, x0, k).
    """
    levels = np.array(x0_grid, dtype=float)
    pmfs = _pmf(levels)
    rng = np.random.default_rng(seed)
    n_total = len(hprs) * len(x0_grid) * len(k_grid)
    print(f"  Computing MI over {n_total} (hpr, x0, k) cells (sequential RNG, seed={seed})")
    rows = []
    for h in hprs:
        pmf = pmfs[str(int(h))]
        for x0 in x0_grid:
            for k in k_grid:
                xt = rng.choice(pmf[:, 0], n_draws, p=pmf[:, 1])
                rt = _logistsig(xt, x0, k)
                ct = _contextualise_stim(xt, pmf)
                c_xy = np.histogram2d(rt, ct, [2, len(np.unique(ct))])[0]
                mi = mutual_info_score(None, None, contingency=c_xy)
                rows.append((int(h), float(x0), float(k), float(mi),
                             float(rt.mean()), float(rt.std())))
    df = pd.DataFrame(rows, columns=["hpr", "x0", "k", "mi", "mu_rt", "std_rt"])
    return df.set_index(["hpr", "x0", "k"]).sort_index()


def _expand_with_priors(mi_df, betas, xis):
    nb = len(betas)
    nx = len(xis)
    df = pd.concat([mi_df] * nb)
    df["beta"] = np.repeat(betas, len(mi_df))
    df = pd.concat([df] * nx)
    df["xi"] = np.repeat(xis, len(df) // nx)
    df["U"] = df["mi"] - df["xi"] * df["mu_rt"]
    df["pdf"] = np.exp(df["beta"] * df["U"])
    df = df.reset_index().set_index(["hpr", "xi", "beta"])
    z = df.groupby(["hpr", "xi", "beta"])["pdf"].transform("sum")
    df["pdf"] = df["pdf"] / z
    df = df.reset_index().set_index(["hpr", "xi", "beta", "x0", "k"])
    return df[["U", "pdf", "mi", "mu_rt", "std_rt"]]


def _calc_stats(df_mi):
    stats = pd.DataFrame()
    stats["avg_utility"] = df_mi.groupby(["hpr", "beta", "xi"]).apply(
        lambda r: float(np.sum(r["U"] * r["pdf"]))
    )
    Umax = df_mi.groupby(["hpr", "xi"])["U"].max().rename("Umax")
    U0 = stats.xs(0, level="beta")["avg_utility"].rename("U0")
    stats = stats.join(Umax, on=["hpr", "xi"]).join(U0, on=["hpr", "xi"])
    stats["norm_avg_utility"] = (stats["avg_utility"] - stats["U0"]) / (stats["Umax"] - stats["U0"])
    stats["entropy"] = df_mi.groupby(["hpr", "beta", "xi"]).apply(
        lambda r: float(-(r["pdf"] * np.log(r["pdf"] + 1e-12)).sum())
    )
    stats["mean_fr"] = df_mi.groupby(["hpr", "beta", "xi"]).apply(
        lambda r: float(np.sum(r["mu_rt"] * r["pdf"]))
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hprs", default="30,45,60,75,90")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--out-dir", default=None,
                        help="Write df_mi_pdf/df_mi_stats here instead of data/artifacts/")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    grid = cfg["sigmoid_grid"]
    x0_grid = np.arange(grid["threshold_db_min"], grid["threshold_db_max"] + grid["threshold_step_db"],
                        grid["threshold_step_db"], dtype=float)
    k_grid = np.round(np.linspace(grid["gain_min"], grid["gain_max"], grid["gain_levels"]), 2)

    mi_cfg = cfg["mi_grid"]
    n_draws = int(mi_cfg["n_draws"])
    seed = int(mi_cfg["seed"])
    nb = int(mi_cfg["n_beta"])
    nx = int(mi_cfg["n_xi"])
    betas = np.round(np.linspace(0.0, float(mi_cfg["beta_max"]), nb), 2)
    xis = np.round(np.linspace(0.0, float(mi_cfg["xi_max"]), nx), 4)

    hprs = tuple(int(h) for h in args.hprs.split(","))

    print(f"# Stage 1: MI Monte-Carlo (cached) [n_draws={n_draws}, seed={seed}]")
    mi_df = _mi_table(hprs, tuple(x0_grid.tolist()), tuple(k_grid.tolist()), n_draws, seed)
    print(f"  -> {len(mi_df)} (hpr, x0, k) rows")

    print(f"# Stage 2: Expanding with priors over {nb} betas x {nx} xis")
    df_mi_pdf = _expand_with_priors(mi_df, betas, xis)
    print(f"  -> {len(df_mi_pdf)} rows")

    print(f"# Stage 3: Computing stats")
    df_mi_stats = _calc_stats(df_mi_pdf)
    print(f"  -> {len(df_mi_stats)} stat rows")

    out_root = Path(args.out_dir) if args.out_dir else ARTIFACTS
    out_root.mkdir(parents=True, exist_ok=True)
    out_pdf = out_root / "df_mi_pdf.parquet"
    out_stats = out_root / "df_mi_stats.parquet"
    df_mi_pdf.reset_index().to_parquet(out_pdf, index=False, compression="zstd")
    df_mi_stats.reset_index().to_parquet(out_stats, index=False, compression="zstd")
    print(f"# Wrote {out_pdf}")
    print(f"# Wrote {out_stats}")


if __name__ == "__main__":
    main()
