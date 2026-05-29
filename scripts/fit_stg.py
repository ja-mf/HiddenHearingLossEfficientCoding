#!/usr/bin/env python
"""Fit sigmoidal threshold-gain parameters to rate-level functions in parallel."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "reproduction.yaml"
ARTIFACTS = PROJECT_ROOT / "data" / "artifacts"

VARIANT_SUFFIX = {"60dB": "a", "75dB": "b"}

# Brute-force grid for sigmoid threshold (dB) and gain
_X0_RANGE = (24.0, 96.01, 0.01)
_K_RANGE = (-0.5, 0.51, 0.01)

def _vectorized_brute(lvls, spks_norm, *, dtype=np.float32, chunk=500):
    """Exhaustive grid search over (x0, k) using vectorised numpy."""
    x0_grid = np.arange(*_X0_RANGE, dtype=dtype)
    k_grid = np.arange(*_K_RANGE, dtype=dtype)
    lvls = np.asarray(lvls, dtype=dtype)
    y = np.asarray(spks_norm, dtype=dtype)
    best_mse = np.inf
    best = (0.0, 0.0)
    for s in range(0, len(x0_grid), chunk):
        xc = x0_grid[s:s + chunk]
        pred = 1.0 / (1.0 + np.exp(-k_grid[None, :, None] * (lvls[None, None, :] - xc[:, None, None])))
        mse = np.mean((y[None, None, :] - pred) ** 2, axis=-1)
        flat = mse.argmin()
        i, j = np.unravel_index(flat, mse.shape)
        m = float(mse[i, j])
        if m < best_mse:
            best_mse = m
            best = (float(xc[i]), float(k_grid[j]))
    return best[0], best[1], best_mse


_FITTERS = { "vbrute": lambda l, s: _vectorized_brute(l, s, dtype=np.float64, chunk=200), }


def _fit_one(neuron_id, hpr, lvls, spks_norm, x0_grid, k_grid, fitter="vbrute"):
    fn = _FITTERS[fitter]
    x0_fit, k_fit, mse = fn(lvls, spks_norm)
    x0 = float(x0_grid[np.abs(x0_grid - x0_fit).argmin()])
    k = float(k_grid[np.abs(k_grid - k_fit).argmin()])
    return {
        "neuron-id": neuron_id,
        "hpr": int(hpr),
        "threshold": float(x0_fit),
        "gain": float(k_fit),
        "mse": float(mse),
        "x0": x0,
        "k": k,
    }


def fit_cohort(
    df_rlfs: pd.DataFrame,
    *,
    min_firing_rate: float,
    max_mean_mse: float,
    n_jobs: int,
    x0_grid: np.ndarray,
    k_grid: np.ndarray,
    fitter: str = "vbrute",
) -> pd.DataFrame:
    df = df_rlfs.copy()
    variants = df["spl_variant"].unique()
    if len(variants) > 1:
        suffix = df["spl_variant"].map(VARIANT_SUFFIX).fillna(df["spl_variant"])
        df["neuron_id"] = df["neuron_id"].astype(str) + suffix.astype(str)

    mean_rates = df.groupby(["neuron_id", "hpr"])["spks"].sum() / 25.0
    low = mean_rates[mean_rates < min_firing_rate].index.get_level_values("neuron_id").unique()
    df = df[~df["neuron_id"].isin(low)]
    print(f"  After firing-rate filter (>={min_firing_rate}): "
          f"{df['neuron_id'].nunique()} neuron-variants")

    tasks = []
    for (nid, hpr), grp in df.groupby(["neuron_id", "hpr"]):
        gs = grp.sort_values("lvl")
        lvls = gs["lvl"].to_numpy(float)
        spks = gs["spks"].to_numpy(float)
        rng = spks.max() - spks.min()
        if rng < 1e-10:
            continue
        spks_norm = (spks - spks.min()) / rng
        tasks.append((nid, int(hpr), lvls, spks_norm))

    print(f"  Fitting {len(tasks)} (neuron, hpr) pairs with n_jobs={n_jobs} fitter={fitter}")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_fit_one)(nid, hpr, lvls, spks_norm, x0_grid, k_grid, fitter)
        for nid, hpr, lvls, spks_norm in tasks
    )

    meta = df[["neuron_id", "group"]].drop_duplicates("neuron_id").set_index("neuron_id")["group"].to_dict()
    rows = []
    for r in results:
        rows.append({
            "group": meta.get(r["neuron-id"], ""),
            "neuron-id": r["neuron-id"],
            "hpr": r["hpr"],
            "threshold": r["threshold"],
            "gain": r["gain"],
            "mse": r["mse"],
            "x0": r["x0"],
            "k": r["k"],
        })
    df_fits = pd.DataFrame(rows)

    mean_mse = df_fits.groupby("neuron-id")["mse"].mean()
    keep = set(mean_mse[mean_mse <= max_mean_mse].index)
    print(f"  After MSE filter (mean<={max_mean_mse}): {len(keep)}/{df_fits['neuron-id'].nunique()} neurons retained")
    df_fits = df_fits[df_fits["neuron-id"].isin(keep)].copy()

    # Degenerate-fit guard: drop (neuron, hpr) pairs whose |gain| is below
    # numerical floor. With ~0 slope the sigmoid is flat and the threshold
    # is mathematically undefined (any x0 yields identical MSE), so the
    # value reported is purely a tie-breaker artifact. Including these rows
    # pollutes per-neuron median threshold (and the 85 dB SPL split) and
    # LMM threshold inputs. Dropping is consistent with how some neurons
    # already lack a fitted (neuron, hpr) pair when the rate has no
    # measurable range (the rng<1e-10 guard a few lines above).
    GAIN_EPS = 1e-3
    degenerate = df_fits["gain"].abs() < GAIN_EPS
    n_deg = int(degenerate.sum())
    if n_deg:
        print(f"  Dropping {n_deg}/{len(df_fits)} degenerate (|gain|<{GAIN_EPS}) fits (flat RIF, threshold undefined)")
        df_fits = df_fits[~degenerate].copy()

    return df_fits[["group", "neuron-id", "hpr", "threshold", "gain", "x0", "k"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", choices=["HHL", "CHL"])
    parser.add_argument("--rlfs", default=None, help="Override input rlfs parquet")
    parser.add_argument("--output", default=None, help="Override output parquet")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    n_jobs = args.n_jobs or cfg.get("parallel", {}).get("n_jobs", 8)
    max_mean_mse = float(cfg["quality_filter"]["max_mean_mse"])
    g = cfg["sigmoid_grid"]
    x0_grid = np.arange(g["threshold_db_min"], g["threshold_db_max"] + g["threshold_step_db"], g["threshold_step_db"], dtype=float)
    k_grid = np.round(np.linspace(g["gain_min"], g["gain_max"], g["gain_levels"]), 2)

    min_fr = float(cfg.get("quality_filter", {}).get("min_firing_rate", 0.0))

    in_path = Path(args.rlfs) if args.rlfs else ARTIFACTS / f"rlfs_{args.cohort}.parquet"
    out_path = Path(args.output) if args.output else ARTIFACTS / f"{args.cohort}_stg.parquet"

    df = pd.read_parquet(in_path)
    print(f"# Loaded {len(df)} RLF rows from {in_path}")
    fits = fit_cohort(df, min_firing_rate=min_fr, max_mean_mse=max_mean_mse,
                      n_jobs=n_jobs, x0_grid=x0_grid, k_grid=k_grid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.to_parquet(out_path, index=False)
    print(f"# Wrote {len(fits)} rows -> {out_path}")


if __name__ == "__main__":
    main()
