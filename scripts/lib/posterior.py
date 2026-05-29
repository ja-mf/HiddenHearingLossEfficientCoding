"""Posterior inference utilities used by the bootstrap pipeline."""

from collections import Counter

import numpy as np
import pandas as pd


def find_nearest(arr, val):
    a = np.asarray(arr)
    return float(a.flat[np.abs(a - val).argmin()])


def get_theta_hat(df_stg, x0_grid, k_grid):
    df = df_stg.copy()
    df["x0"] = df["threshold"].apply(lambda v: find_nearest(x0_grid, v))
    df["k"] = df["gain"].apply(lambda v: find_nearest(k_grid, v))
    return df[["x0", "k"]]


def build_prior_slices(df_mi_pdf, hprs):
    """Returns dict[hpr] -> {'log_pdf', 'beta_vals', 'xi_vals', 'lookup'}."""
    if not df_mi_pdf.index.is_monotonic_increasing:
        df_mi_pdf = df_mi_pdf.sort_index()

    slices = {}
    for h in hprs:
        slc = df_mi_pdf.xs(h, level="hpr")["pdf"].unstack("x0").unstack("k")
        cols = slc.columns.to_list()
        lookup = {(float(x0), round(float(k), 3)): i for i, (x0, k) in enumerate(cols)}
        log_pdf = np.log(np.clip(slc.to_numpy(), 1e-300, None)).astype(np.float64)
        slices[h] = {
            "log_pdf": np.ascontiguousarray(log_pdf),
            "beta_vals": slc.index.get_level_values("beta").to_numpy(np.float64),
            "xi_vals": slc.index.get_level_values("xi").to_numpy(np.float64),
            "lookup": lookup,
        }
    return slices


def _logpost_single(log_pdf, col_indices, counts):
    contrib = log_pdf[:, col_indices] @ counts
    contrib -= contrib.max()
    return np.exp(contrib)


def calc_logpost_all(df_theta_hat, prior_slices):
    df = df_theta_hat.reset_index()
    theta_x0 = df["x0"].to_numpy(np.float64)
    theta_k = df["k"].to_numpy(np.float64)
    theta_g = df["group"].to_numpy()
    theta_h = df["hpr"].to_numpy()

    dfs = []
    for group, hpr in set(zip(theta_g, theta_h)):
        info = prior_slices.get(hpr)
        if info is None:
            continue
        mask = (theta_g == group) & (theta_h == hpr)
        x0_v = theta_x0[mask]
        k_v = np.round(theta_k[mask], 3)
        keys = list(zip(x0_v.astype(float), k_v.astype(float)))
        kc = Counter(keys)
        col_idx, counts = [], []
        for k, c in kc.items():
            if k in info["lookup"]:
                col_idx.append(info["lookup"][k])
                counts.append(c)
        if not col_idx:
            continue
        post = _logpost_single(info["log_pdf"], np.array(col_idx, np.int64), np.array(counts, np.float64))
        idx = pd.MultiIndex.from_arrays(
            [np.full(len(post), group), np.full(len(post), hpr),
             info["beta_vals"], info["xi_vals"]],
            names=["group", "hpr", "beta", "xi"],
        )
        dfs.append(pd.DataFrame({0: post}, index=idx))
    return pd.concat(dfs) if dfs else pd.DataFrame()


def get_df_maxl(logpost):
    idxmax = logpost.groupby(["group", "hpr"])[0].idxmax()
    return pd.DataFrame(idxmax.to_list(),
                        columns=["group", "hpr", "betamax", "ximax"]
                        ).set_index(["group", "hpr"])


def build_map_stats(df_maxl, groups, hprs, df_mi_stats):
    rows = []
    for g in groups:
        for h in hprs:
            if (g, h) not in df_maxl.index:
                continue
            b, x = df_maxl.loc[(g, h), ["betamax", "ximax"]].astype(float).to_numpy()
            if (h, b, x) in df_mi_stats.index:
                s = df_mi_stats.loc[(h, b, x)]
                rows.append((g, h, b, x, s["avg_utility"], s["norm_avg_utility"],
                             s["entropy"], s["mean_fr"]))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["group", "hpr", "beta", "xi",
                                       "avg_utility", "norm_avg_utility",
                                       "entropy", "mean_fr"]
                        ).set_index(["group", "hpr", "beta", "xi"])
