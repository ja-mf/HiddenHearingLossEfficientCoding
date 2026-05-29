"""Empirical mixed-effects group comparisons (Table S1).

Reads per-neuron sigmoid fits (data/artifacts/{HHL,CHL}_stg.parquet) and rate-level
functions (data/artifacts/rlfs_{HHL,CHL}.parquet). 16 tests across 5 families;
Benjamini-Hochberg FDR is applied per family. Interaction models report the
joint Wald chi-square on the group x HPR terms; main-effect models report the
group coefficient.
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats

ART = Path(__file__).resolve().parents[1].parent / "data" / "artifacts"

EXCLUDE_ANIMALS = {"SH1"}           # only 8 neurons, lacks uniform condition
EXCLUDE_SIGMA_UNIF = {"SH1", "SH2"}
HPRS_ALL = [30, 45, 60, 75, 90]


# %% Loaders + animal extraction
def load_stg(cohort: str) -> pd.DataFrame:
    return pd.read_parquet(ART / f"{cohort}_stg.parquet")


def load_rlf(cohort: str) -> pd.DataFrame:
    return pd.read_parquet(ART / f"rlfs_{cohort}.parquet")


def extract_animal_hhl(neuron_id: str) -> str:
    """NE1u003 -> NE1, SH6au012 -> SH6a"""
    m = re.match(r"^([A-Z]+\d+[a-z]?)u", neuron_id)
    return m.group(1) if m else neuron_id


def extract_animal_num_chl(neuron_id: str) -> str:
    """EP1u003 -> 1, PP2u005 -> 2"""
    m = re.match(r"^[A-Z]+(\d+)", neuron_id)
    return m.group(1) if m else neuron_id


def bh_correct(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals
    ranked = pvals.rank(method="first")
    corrected = pvals * n / ranked
    corrected = corrected.sort_values(ascending=False).cummin()
    return corrected.clip(upper=1.0).reindex(pvals.index)


def prepare_hhl_stg():
    df = load_stg("HHL")
    df["animal"] = df["neuron-id"].apply(extract_animal_hhl)
    return df[~df["animal"].isin(EXCLUDE_ANIMALS)].copy()


def prepare_chl_stg():
    df = load_stg("CHL")
    df["animal"] = df["neuron-id"].apply(extract_animal_num_chl)
    return df


def prepare_chl_rlf():
    df = load_rlf("CHL")
    df = df[df["spl_variant"] == "60dB"].copy()
    df["animal"] = df["animal_id"]
    return df


def prepare_hhl_rlf():
    df = load_rlf("HHL")
    df = df[df["spl_variant"] == "60dB"].copy()
    df["animal"] = df["neuron_id"].apply(extract_animal_hhl)
    return df[~df["animal"].isin(EXCLUDE_ANIMALS)].copy()


# %% Derived metrics
def compute_mean_spike_rate(rlf_df):
    rates = (
        rlf_df.groupby(["group", "animal", "neuron_id", "hpr"])["spks"]
        .mean()
        .reset_index()
    )
    rates.columns = ["group", "animal", "neuron_id", "hpr", "mean_rate"]
    return rates


def compute_sigma_unif(stg_df: pd.DataFrame) -> pd.DataFrame:
    needed_hprs = {-1, 30, 45, 60, 75}
    sub = stg_df[stg_df["hpr"].isin(needed_hprs)].copy()
    piv = sub.pivot_table(index="neuron-id", columns="hpr", values="threshold", aggfunc="first")
    piv = piv.dropna(subset=list(needed_hprs))
    t_unif = piv[-1]
    sigma = np.sqrt(sum((piv[h] - t_unif) ** 2 for h in [30, 45, 60, 75]))
    result = sigma.reset_index()
    result.columns = ["neuron-id", "sigma_unif"]
    return result


def compute_fi_per_neuron(rlf_df, lvl_lo=None, lvl_hi=None):
    df = rlf_df.copy()
    if lvl_lo is not None:
        df = df[df["lvl"] >= lvl_lo]
    if lvl_hi is not None:
        df = df[df["lvl"] <= lvl_hi]
    agg = (
        df.groupby(["group", "animal", "neuron_id", "hpr", "lvl"])["spks"]
        .mean().reset_index().rename(columns={"spks": "mean_spks"})
    )
    agg = agg.sort_values(["neuron_id", "hpr", "lvl"])
    results = []
    for (nid, hpr), sub in agg.groupby(["neuron_id", "hpr"]):
        sub = sub.sort_values("lvl")
        r = sub["mean_spks"].values
        if len(r) < 2:
            continue
        dr = np.diff(r)
        r_mid = (r[:-1] + r[1:]) / 2.0
        r_mid = np.maximum(r_mid, 0.5)
        fi_per_level = dr ** 2 / r_mid
        results.append({
            "group": sub["group"].iloc[0], "animal": sub["animal"].iloc[0],
            "neuron_id": nid, "hpr": hpr, "total_fi": float(fi_per_level.sum()),
        })
    fi_df = pd.DataFrame(results)
    fi_df["log_fi"] = np.log(fi_df["total_fi"] + 1)
    return fi_df


def compute_spontaneous_rate(rlf_df, max_lvl=30):
    sub = rlf_df[rlf_df["lvl"] <= max_lvl].copy()
    return (
        sub.groupby(["group", "animal", "neuron_id", "hpr"])["spks"]
        .mean().reset_index().rename(columns={"spks": "spont_rate"})
    )


def compute_dynamic_range(rlf_df):
    g = rlf_df.groupby(["group", "animal", "neuron_id", "hpr"])["spks"]
    merged = g.max().reset_index().rename(columns={"spks": "max_spks"}).merge(
        g.min().reset_index().rename(columns={"spks": "min_spks"}),
        on=["group", "animal", "neuron_id", "hpr"])
    merged["rate_range"] = merged["max_spks"] - merged["min_spks"]
    return merged[["group", "animal", "neuron_id", "hpr", "rate_range"]]


def compute_gain_deviation(stg_df):
    df = stg_df.copy()
    medians = df.groupby(["group", "hpr"])["gain"].median().rename("median_gain")
    df = df.merge(medians, on=["group", "hpr"])
    df["gain_dev"] = (df["gain"] - df["median_gain"]).abs()
    return df[["neuron-id", "hpr", "group", "animal", "gain_dev"]]


def compute_threshold_deviation(stg_df):
    df = stg_df.copy()
    medians = df.groupby(["group", "hpr"])["threshold"].median().rename("median_thresh")
    df = df.merge(medians, on=["group", "hpr"])
    df["thresh_dev"] = (df["threshold"] - df["median_thresh"]).abs()
    return df[["neuron-id", "hpr", "group", "animal", "thresh_dev"]]


# %% LME fitting
def fit_lme_simple(df, formula, groups_col="animal"):
    try:
        md = smf.mixedlm(formula, df, groups=df[groups_col])
        mdf = None
        for method in [None, "lbfgs", "powell", "cg"]:
            try:
                kw = {"reml": True}
                if method is not None:
                    kw["method"] = method
                mdf = md.fit(**kw)
                if mdf is not None:
                    break
            except Exception:
                continue
        if mdf is None:
            return None
        fe_names = [n for n in mdf.params.index
                    if n != "Intercept" and "Group" not in n and "Var" not in n]
        if not fe_names:
            return None
        gn = fe_names[0]
        return {
            "beta": mdf.params[gn], "SE": mdf.bse[gn], "t": mdf.tvalues[gn], "p": mdf.pvalues[gn],
            "sigma2_animal": (mdf.cov_re.iloc[0, 0] if hasattr(mdf.cov_re, "iloc") else float(mdf.cov_re)),
            "fe_name": gn, "n_obs": int(mdf.nobs), "converged": mdf.converged,
        }
    except Exception as e:
        print(f"  WARNING: LME fit failed: {e}", file=sys.stderr)
        return None


def fit_lme_interaction(df, dep_var, group_var, hpr_var="hpr", groups_col="animal"):
    df = df.copy()
    df["C_hpr"] = pd.Categorical(df[hpr_var])
    df["C_group"] = df[group_var]
    formula = f"{dep_var} ~ C_group * C_hpr"
    try:
        md = smf.mixedlm(formula, df, groups=df[groups_col])
        mdf = None
        for method in [None, "lbfgs", "powell", "cg"]:
            try:
                kw = {"reml": True}
                if method is not None:
                    kw["method"] = method
                mdf = md.fit(**kw)
                if mdf is not None:
                    break
            except Exception:
                continue
        if mdf is None:
            return None
        params, pvalues, tvalues, bse = mdf.params, mdf.pvalues, mdf.tvalues, mdf.bse
        interaction_terms = [n for n in params.index if ":" in n]
        main_group_terms = [n for n in params.index
                            if "C_group" in n and ":" not in n and n != "Intercept"]
        if interaction_terms:
            try:
                wt = mdf.wald_test_terms(scalar=True)
                interaction_p = interaction_F = None
                for term_name in wt.table.index:
                    if ":" in str(term_name):
                        interaction_p = wt.table.loc[term_name, "P>chi2"]
                        interaction_F = wt.table.loc[term_name, "statistic"]
                        break
                if interaction_p is None:
                    raise ValueError("fallback")
            except Exception:
                idx = [list(params.index).index(t) for t in interaction_terms]
                R = np.zeros((len(idx), len(params)))
                for i, j in enumerate(idx):
                    R[i, j] = 1.0
                beta_r = R @ params.values
                cov_r = R @ mdf.cov_params() @ R.T
                chi2 = float(beta_r @ np.linalg.solve(cov_r, beta_r))
                interaction_p = float(sp_stats.chi2.sf(chi2, len(idx)))
                interaction_F = chi2 / len(idx)
        else:
            interaction_p = interaction_F = np.nan
        main_beta = params[main_group_terms[0]] if main_group_terms else np.nan
        main_p = pvalues[main_group_terms[0]] if main_group_terms else np.nan
        return {
            "interaction_p": interaction_p,
            "interaction_chi2": (interaction_F * len(interaction_terms) if interaction_terms else np.nan),
            "interaction_df": len(interaction_terms),
            "main_group_beta": main_beta, "main_group_p": main_p,
            "sigma2_animal": (mdf.cov_re.iloc[0, 0] if hasattr(mdf.cov_re, "iloc") else float(mdf.cov_re)),
            "n_obs": int(mdf.nobs), "converged": mdf.converged,
            "interaction_details": {t: {"beta": params[t], "SE": bse[t], "p": pvalues[t], "t": tvalues[t]}
                                    for t in interaction_terms},
        }
    except Exception as e:
        print(f"  WARNING: Interaction LME fit failed: {e}", file=sys.stderr)
        return None


# %% Test battery
def _interaction_row(df, dep, group, cid, family, claim):
    r = fit_lme_interaction(df, dep, group, "hpr", "animal")
    if not r:
        return None
    return {
        "id": cid, "family": family, "claim": claim,
        "statistic": f"chi2={r['interaction_chi2']:.2f}, df={r['interaction_df']}",
        "chi2": r["interaction_chi2"], "df": r["interaction_df"], "p_raw": r["interaction_p"],
        "main_beta": r["main_group_beta"], "main_p": r["main_group_p"],
        "n_obs": r["n_obs"], "converged": r["converged"],
    }


def _simple_row(df, formula, cid, family, claim):
    r = fit_lme_simple(df, formula, "animal")
    if not r:
        return None
    return {
        "id": cid, "family": family, "claim": claim,
        "statistic": f"beta={r['beta']:.4f}, SE={r['SE']:.4f}",
        "chi2": np.nan, "df": np.nan, "p_raw": r["p"],
        "beta": r["beta"], "SE": r["SE"], "t": r["t"],
        "n_obs": r["n_obs"], "converged": r["converged"],
    }


def run_empirical_tests(hhl_stg, chl_stg, chl_rlf_rates, hhl_rlf, chl_rlf):
    """Run all 16 empirical LME tests; per-family BH. Returns DataFrame."""
    ep_stg = chl_stg[chl_stg["group"] == "EP"].copy()
    pp_stg = chl_stg[chl_stg["group"] == "PP"].copy()
    sh_stg = hhl_stg[hhl_stg["group"] == "SH"].copy()
    ne_stg = hhl_stg[hhl_stg["group"] == "NE"].copy()

    hhl_fi_total = compute_fi_per_neuron(hhl_rlf)
    hhl_fi_quiet = compute_fi_per_neuron(hhl_rlf, lvl_lo=24, lvl_hi=50)
    chl_fi_total = compute_fi_per_neuron(chl_rlf)
    hhl_spont = compute_spontaneous_rate(hhl_rlf, max_lvl=30)
    chl_spont = compute_spontaneous_rate(chl_rlf, max_lvl=30)
    hhl_dr = compute_dynamic_range(hhl_rlf)
    hhl_gain_dev = compute_gain_deviation(hhl_stg[hhl_stg["hpr"].isin(HPRS_ALL)])
    hhl_thresh_dev = compute_threshold_deviation(hhl_stg[hhl_stg["hpr"].isin(HPRS_ALL)])

    R = []
    H = lambda d: d[d["hpr"].isin(HPRS_ALL)].copy()

    # HHL-core
    R.append(_interaction_row(H(hhl_stg), "gain", "group", "E1", "HHL-core", "Gain x HPR (NE vs SH)"))
    # E3 sigma_unif (exploratory)
    df_sigma = hhl_stg[~hhl_stg["animal"].isin(EXCLUDE_SIGMA_UNIF)].copy()
    sigma_df = compute_sigma_unif(df_sigma)
    info = hhl_stg[["neuron-id", "animal", "group"]].drop_duplicates(subset="neuron-id")
    sigma_df = sigma_df.merge(info, on="neuron-id", how="left")
    sigma_df = sigma_df[~sigma_df["animal"].isin(EXCLUDE_SIGMA_UNIF)]
    R.append(_simple_row(sigma_df, "sigma_unif ~ group", "E3", "HHL-core", "sigma_Unif NE < SH (exploratory)"))
    R.append(_interaction_row(H(hhl_stg), "threshold", "group", "E4", "HHL-core", "Threshold x HPR (NE vs SH)"))
    R.append(_interaction_row(H(hhl_spont), "spont_rate", "group", "E_SR1", "HHL-core", "Spontaneous rate x HPR (NE vs SH)"))
    # FI
    R.append(_interaction_row(H(hhl_fi_total), "log_fi", "group", "E_FI1", "FI", "log(total_FI) x HPR (NE vs SH)"))
    R.append(_simple_row(hhl_fi_quiet[hhl_fi_quiet["hpr"] == 30].copy(), "log_fi ~ group", "E_FI2", "FI", "log(FI_24-50dB) at HPR30 (NE vs SH)"))
    R.append(_interaction_row(H(hhl_dr), "rate_range", "group", "E_DR1", "FI", "Dynamic range x HPR (NE vs SH)"))
    # Spread
    R.append(_interaction_row(hhl_gain_dev.copy(), "gain_dev", "group", "E_SP1", "Spread", "Gain deviation x HPR (NE vs SH)"))
    R.append(_interaction_row(hhl_thresh_dev.copy(), "thresh_dev", "group", "E_SP2", "Spread", "Threshold deviation x HPR (NE vs SH)"))
    # CHL within-subject
    R.append(_simple_row(H(chl_stg), "threshold ~ group", "E5", "CHL", "Threshold EP > PP (within-subject)"))
    R.append(_simple_row(H(chl_rlf_rates), "mean_rate ~ group", "E6", "CHL", "Spike rate EP > PP (within-subject)"))
    R.append(_interaction_row(H(chl_stg), "gain", "group", "E7", "CHL", "Gain x HPR (EP vs PP)"))
    R.append(_interaction_row(H(chl_fi_total), "log_fi", "group", "E_FI3", "CHL", "log(total_FI) x HPR (EP vs PP)"))
    R.append(_simple_row(H(chl_spont), "spont_rate ~ group", "E_SR2", "CHL", "Spontaneous rate EP vs PP (all HPRs)"))

    # Cross-cohort (re-namespace animals so EP1 != SH1 etc.)
    def _cross(stg1, stg2, g1, g2, cid, claim):
        d1, d2 = stg1.copy(), stg2.copy()
        d1["xgroup"], d2["xgroup"] = g1, g2
        d1["xanimal"] = g1 + "_" + d1["animal"].astype(str)
        d2["xanimal"] = g2 + "_" + d2["animal"].astype(str)
        cols = ["neuron-id", "hpr", "threshold", "xgroup", "xanimal"]
        comb = pd.concat([d1[d1["hpr"].isin(HPRS_ALL)][cols], d2[d2["hpr"].isin(HPRS_ALL)][cols]], ignore_index=True)
        comb = comb.rename(columns={"xgroup": "group", "xanimal": "animal"})
        return _interaction_row(comb, "threshold", "group", cid, "Cross-cohort", claim)

    R.append(_cross(ep_stg, sh_stg, "EP", "SH", "E8", "Threshold x HPR (EP vs SH)"))
    R.append(_cross(ep_stg, ne_stg, "EP", "NE", "E9", "Threshold x HPR (EP vs NE)"))
    R.append(_cross(pp_stg, sh_stg, "PP", "SH", "E10", "Threshold x HPR (PP vs SH)"))

    res = pd.DataFrame([r for r in R if r is not None])
    res["p_fdr"] = np.nan
    for fam, grp in res.groupby("family"):
        res.loc[grp.index, "p_fdr"] = bh_correct(res.loc[grp.index, "p_raw"])
    return res
