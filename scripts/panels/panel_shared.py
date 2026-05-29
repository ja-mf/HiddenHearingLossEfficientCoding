
from functools import lru_cache
from pathlib import Path

import gc
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from numba import jit


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "reproduction.yaml"


def _load_reproduction_config():
    try:
        return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def get_threshold_split_db():
    """Two-population split (dB SPL), single source of truth: config/reproduction.yaml
    bootstrap.threshold_split_db. Defaults to 85 if config absent or key missing."""
    cfg = _load_reproduction_config()
    return float(cfg.get("bootstrap", {}).get("threshold_split_db", 85))


levels = np.arange(24, 97, 3)
x0s = levels.copy()
ks = np.round(np.linspace(-0.5, 0.5, 25), 2)
hprs1 = [30, 45, 60, 75]
lxlim, hxlim = float(x0s.min()), float(x0s.max())
dos = 3
despineopts = dict(trim=True, offset=dos)

# ── HHL group colors  (NE: noise-exposed, SH: sham) ─────────────────────────
group_colours    = sns.color_palette("Set1", n_colors=3)[1:]   # [0]≈#377eb8 blue, [1]≈#4daf4a green
HHL_GROUP_COLORS = {"NE": group_colours[0], "SH": group_colours[1]}

# ── CHL group colors  (EP: ear-plugged, PP: post-plug) ───────────────────────
# Collision: PP (#009E73 teal-green) ≈ SH (~#4daf4a green)
#            EP (#D55E00 burnt-orange) ≈ HHL_LOW (#ff7f0e) both orange
# _chl_ep, _chl_pp = "#D55E00", "#009E73"   # burnt-orange / teal-green  [current Okabe-Ito]
# _chl_ep, _chl_pp = "#CC79A7", "#E69F00"   # pink-violet / amber        [A — colorblind-safe; pair w/ LOW="#56B4E9"]
# _chl_ep, _chl_pp = "#E63946", "#7B5EA7"   # rose-red / muted-violet    [B — LOW stays orange; HIGH stays red]
_chl_ep, _chl_pp = "#F4845F", "#1D7A8A"   # coral / deep-teal-blue     [C — pair w/ LOW="#9B59B6"]
CHL_GROUP_COLORS = {"EP": _chl_ep, "PP": _chl_pp}

# ── HHL threshold subgroup colors  (low / high, Fig 5 panel E) ───────────────
# Collision: low (#ff7f0e orange) ≈ EP (#D55E00 burnt-orange) with current CHL
# HHL_LOW_COLOR,  HHL_HIGH_COLOR = "#ff7f0e", "#d62728"    # orange / dark-red        [current; ok w/ CHL B]
# HHL_LOW_COLOR, HHL_HIGH_COLOR = "#56B4E9", "#d62728"   # sky-blue / dark-red      [A — pair w/ CHL A]
HHL_LOW_COLOR, HHL_HIGH_COLOR = "#9B59B6", "#9B2226"   # purple / deep-burgundy   [C — pair w/ CHL C]
PANEL_RCPARAMS = {"text.usetex": True, "axes.labelpad": 0, "figure.dpi": 200}


def apply_panel_rcparams(*, usetex=True, figure_dpi=200, axes_labelpad=0):
    from matplotlib import pyplot as plt

    plt.rcParams.update({
        "text.usetex": usetex,
        "axes.labelpad": axes_labelpad,
        "figure.dpi": figure_dpi,
    })


def sf_letterannotation(subfigure, pos, letter):
    subfigure.text(pos[0], pos[1], f"\\textbf {letter}", ha="left", va="bottom", size=14, weight="extra bold")


def letter_annotation(ax, xoffset, yoffset, letter):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=14, weight="extra bold")


def figure_title(number, text):
    return f"Figure {number}: {text}"


def make_hpr_palette(hprs):
    hprs = list(hprs)
    palette_hprs = sns.color_palette("flare", len(hprs))
    return palette_hprs, dict(zip(hprs, palette_hprs))


def logistsig(x, x0, k, rsp=0, rmax=1):
    return rsp + (rmax - rsp) / (1 + np.exp(-k * (x - x0)))


def pmf_singledistr_gerbil():
    hp, lp = 0.8 / 5, 0.2 / 20
    return {
        "90": np.vstack((levels, np.array([lp] * 20 + [hp] * 5))).T,
        "75": np.vstack((levels, np.array([lp] * 15 + [hp] * 5 + [lp] * 5))).T,
        "60": np.vstack((levels, np.array([lp] * 10 + [hp] * 5 + [lp] * 10))).T,
        "45": np.vstack((levels, np.array([lp] * 5 + [hp] * 5 + [lp] * 15))).T,
        "30": np.vstack((levels, np.array([hp] * 5 + [lp] * 20))).T,
        "-1": np.vstack((levels, np.array([1 / 25] * 25))).T,
    }


def load_example_stimlevels(stimlevels_path: Path):
    """Load the example stimulus-level matrix (repeats, samples) shipped at stimlevels_path."""
    return np.load(stimlevels_path)


def find_nearest(a, a0):
    a = np.asarray(a)
    return a.flat[np.abs(a - a0).argmin()]


def get_theta_hat(df_stg):
    df = df_stg.copy()
    df["x0"] = df.apply(lambda d: float(find_nearest(x0s, d["threshold"])), axis=1)
    df["k"] = df.apply(lambda d: float(find_nearest(ks, d["gain"])), axis=1)
    return df[["x0", "k"]]


@jit(nopython=True, fastmath=True, cache=True)
def _compute_logpost_single(log_pdf, col_indices, counts):
    n_rows = log_pdf.shape[0]
    logpost = np.empty(n_rows, dtype=np.float64)

    for i in range(n_rows):
        acc = 0.0
        for j in range(len(col_indices)):
            acc += log_pdf[i, col_indices[j]] * counts[j]
        logpost[i] = acc

    max_val = logpost.max()
    for i in range(n_rows):
        logpost[i] = np.exp(logpost[i] - max_val)

    return logpost


def _build_prior_slices(df_mi_pdf, hprs_list):
    if not df_mi_pdf.index.is_monotonic_increasing:
        df_mi_pdf = df_mi_pdf.sort_index()

    prior_slices = {}
    for h in hprs_list:
        try:
            slice_series = df_mi_pdf.xs(h, level="hpr")["pdf"]
        except (KeyError, TypeError):
            slice_series = df_mi_pdf.xs(h, level="hpr")
            if isinstance(slice_series, pd.DataFrame):
                slice_series = slice_series["pdf"]

        slice_df = slice_series.unstack("x0").unstack("k")
        if slice_df.empty:
            raise KeyError(f"No MI prior entries available for hpr={h}")

        cols = slice_df.columns.to_list()
        lookup = {(float(x0), round(float(k), 3)): idx for idx, (x0, k) in enumerate(cols)}
        log_pdf = np.log(np.clip(slice_df.to_numpy(), 1e-300, None)).astype(np.float64)

        prior_slices[h] = {
            "log_pdf": np.ascontiguousarray(log_pdf),
            "beta_vals": slice_df.index.get_level_values("beta").to_numpy().astype(np.float64),
            "xi_vals": slice_df.index.get_level_values("xi").to_numpy().astype(np.float64),
            "lookup": lookup,
            "n_beta_xi": len(slice_df),
        }

    return prior_slices


def _calc_logpost_fast(theta_x0, theta_k, theta_group, theta_hpr, prior_slices):
    results = {}
    unique_pairs = set(zip(theta_group, theta_hpr))

    for group, hpr in unique_pairs:
        mask = (theta_group == group) & (theta_hpr == hpr)
        x0_vals = theta_x0[mask]
        k_vals = np.round(theta_k[mask], 3)

        if len(x0_vals) == 0:
            continue

        info = prior_slices.get(hpr)
        if info is None:
            continue

        keys = list(zip(x0_vals.astype(float), k_vals.astype(float)))
        key_counts = Counter(keys)

        col_indices = []
        counts = []
        for key, count in key_counts.items():
            if key in info["lookup"]:
                col_indices.append(info["lookup"][key])
                counts.append(count)

        if not col_indices:
            continue

        col_indices = np.array(col_indices, dtype=np.int64)
        counts = np.array(counts, dtype=np.float64)
        post = _compute_logpost_single(info["log_pdf"], col_indices, counts)
        results[(group, hpr)] = (post, info["beta_vals"], info["xi_vals"])

    return results


def _calc_logpost_all(df_theta_hat, df_mi_pdf=None, cache_path=None, use_cache=False, save=False, prior_slices=None):
    if use_cache:
        if cache_path is None:
            raise ValueError("cache_path must be provided when use_cache=True")
        p = Path(cache_path)
        if not p.exists():
            raise FileNotFoundError(f"Cache not found: {p}")
        if p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(p)
        return pd.read_pickle(p)

    if prior_slices is None:
        if df_mi_pdf is None:
            raise ValueError("df_mi_pdf is required when prior_slices is not provided")
        hprs_in_data = df_theta_hat.index.get_level_values("hpr").unique().tolist()
        prior_slices = _build_prior_slices(df_mi_pdf, hprs_in_data)

    df_reset = df_theta_hat.reset_index()
    theta_x0 = df_reset["x0"].to_numpy().astype(np.float64)
    theta_k = df_reset["k"].to_numpy().astype(np.float64)
    theta_group = df_reset["group"].to_numpy()
    theta_hpr = df_reset["hpr"].to_numpy()

    results = _calc_logpost_fast(theta_x0, theta_k, theta_group, theta_hpr, prior_slices)

    dfs = []
    for (group, hpr), (post, beta_vals, xi_vals) in results.items():
        idx = pd.MultiIndex.from_arrays(
            [
                np.full(len(post), group),
                np.full(len(post), hpr),
                beta_vals,
                xi_vals,
            ],
            names=["group", "hpr", "beta", "xi"],
        )
        dfs.append(pd.DataFrame({0: post}, index=idx))

    df_lp = pd.concat(dfs) if dfs else pd.DataFrame()

    if save and cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        parquet_path = p.with_suffix(".parquet")
        df_lp.to_parquet(parquet_path)

    return df_lp


def get_df_maxl_from_logpost(logpost):
    idxmax = logpost.groupby(["group", "hpr"])[0].idxmax()
    return pd.DataFrame(idxmax.to_list(), columns=["group", "hpr", "betamax", "ximax"]).set_index(["group", "hpr"])


def build_map_stats(df_maxl, groups_present, hprs_list, df_mi_stats):
    rows = []
    for g in groups_present:
        for h in hprs_list:
            if (g, h) in df_maxl.index:
                b, e = df_maxl.loc[(g, h), ["betamax", "ximax"]].astype(float).values.tolist()
                if (h, b, e) in df_mi_stats.index:
                    stats_row = df_mi_stats.loc[(h, b, e)]
                    rows.append(
                        (
                            g,
                            h,
                            b,
                            e,
                            stats_row["avg_utility"],
                            stats_row["norm_avg_utility"],
                            stats_row["entropy"],
                            stats_row["mean_fr"],
                        )
                    )
    if not rows:
        raise ValueError("No rows could be built for MAP stats from df_maxl")
    df = pd.DataFrame(
        rows,
        columns=["group", "hpr", "beta", "xi", "avg_utility", "norm_avg_utility", "entropy", "mean_fr"],
    )
    return df.set_index(["group", "hpr", "beta", "xi"])


def _normalize_xi(d):
    rename = {}
    if "etamax" in d.columns:
        rename["etamax"] = "ximax"
    if rename:
        d = d.rename(columns=rename)
    return d


def _load_mi_pdf(path: Path) -> pd.DataFrame:
    try:
        d = pd.read_parquet(path, columns=["hpr", "xi", "beta", "x0", "k", "pdf"])
    except Exception:
        try:
            d = pd.read_parquet(path, columns=["hpr", "eta", "beta", "x0", "k", "pdf"])
        except Exception:
            d = pd.read_parquet(path)
    d = _normalize_xi(d)
    if "hpr" in d.columns:
        d = d.set_index(["hpr", "xi", "beta", "x0", "k"])
    if "pdf" in d.columns:
        d = d[["pdf"]]
    return d.sort_index()


def _load_mi_stats(path: Path) -> pd.DataFrame:
    d = _normalize_xi(pd.read_parquet(path))
    if "hpr" in d.columns:
        d = d.set_index(["hpr", "beta", "xi"])
    return d.sort_index()


def _load_stg(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    if "group" in d.columns:
        d = d.set_index(["group", "neuron-id", "hpr"])
    return d


def clear_fig5_hpr90_cache():
    load_fig5_hpr90_context.cache_clear()


@lru_cache(maxsize=None)
def load_fig5_hpr90_context(project_root: str, setting_tag: str = '', subset_setting_tag: str | None = None):
    project_root_path = Path(project_root).resolve()
    data_derived = project_root_path / "data" / "derived"
    dfpkls = data_derived / "df_pkls"
    hprs5 = (30, 45, 60, 75, 90)

    mi_path = project_root_path / "data" / "artifacts" / "df_mi_pdf.parquet"
    mi_stats_path = project_root_path / "data" / "artifacts" / "df_mi_stats.parquet"
    hhl_all_path = project_root_path / "data" / "artifacts" / "HHL_bootstrap.parquet"
    chl_all_path = project_root_path / "data" / "artifacts" / "CHL_bootstrap.parquet"
    hhl_low_boot_path = project_root_path / "data" / "artifacts" / "HHL_low_bootstrap.parquet"
    hhl_high_boot_path = project_root_path / "data" / "artifacts" / "HHL_high_bootstrap.parquet"

    hhl_stg_all_path = project_root_path / "data" / "artifacts" / "HHL_stg.parquet"
    chl_stg_all_path = project_root_path / "data" / "artifacts" / "CHL_stg.parquet"

    required = [
        mi_path,
        mi_stats_path,
        hhl_all_path,
        chl_all_path,
        hhl_low_boot_path,
        hhl_high_boot_path,
        hhl_stg_all_path,
        chl_stg_all_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Figure 5 inputs:\n" + "\n".join(missing))

    df_mi_pdf = _load_mi_pdf(mi_path)
    df_mi_stats = _load_mi_stats(mi_stats_path)

    betas = np.sort(df_mi_stats.index.get_level_values("beta").unique().astype(float))
    xis = np.sort(df_mi_stats.index.get_level_values("xi").unique().astype(float))

    df_hhl = _normalize_xi(pd.read_parquet(hhl_all_path))
    df_chl = _normalize_xi(pd.read_parquet(chl_all_path))
    df_map_stats_hhl_low = _normalize_xi(pd.read_parquet(hhl_low_boot_path))
    df_map_stats_hhl_high = _normalize_xi(pd.read_parquet(hhl_high_boot_path))

    hhl_stg_all = _load_stg(hhl_stg_all_path)
    chl_stg_all = _load_stg(chl_stg_all_path)

    df_sig_thresh_gain = hhl_stg_all.reset_index()[["group", "hpr", "threshold", "gain"]]
    df_sig_thresh_gain = df_sig_thresh_gain[df_sig_thresh_gain["hpr"].isin(hprs5)].copy()

    theta_hhl = get_theta_hat(hhl_stg_all)
    theta_chl = get_theta_hat(chl_stg_all)
    df_theta_all = pd.concat([theta_hhl, theta_chl]).sort_index()

    prior_slices_all = _build_prior_slices(df_mi_pdf, hprs5)
    logpost_all_neurons = _calc_logpost_all(
        df_theta_all.reset_index().set_index(["group", "hpr"]),
        prior_slices=prior_slices_all,
    )
    df_maxl = get_df_maxl_from_logpost(logpost_all_neurons)

    panel_b_hyperparams = {}
    panel_b_heatmaps = {}
    panel_b_vmax = {}
    for hpr in hprs5:
        vmax_values = []
        for group in ["SH", "NE"]:
            sel = df_maxl.loc[group, hpr]
            if isinstance(sel, pd.Series):
                beta_idx = float(sel["betamax"])
                xi_idx = float(sel["ximax"])
            else:
                beta_idx = float(sel["betamax"].mean())
                xi_idx = float(sel["ximax"].mean())
            beta_idx = float(betas[(np.abs(betas - beta_idx)).argmin()])
            xi_idx = float(xis[(np.abs(xis - xi_idx)).argmin()])
            panel_b_hyperparams[(group, hpr)] = (beta_idx, xi_idx)

            pdf = df_mi_pdf.loc[(hpr, xi_idx, beta_idx), "pdf"]
            panel_b_heatmaps[(group, hpr)] = pdf.unstack("x0").sort_index(ascending=False)
            vmax_values.append(float(pdf.max()))
        panel_b_vmax[hpr] = max(vmax_values)

    hhl_stg_all_reset = hhl_stg_all.reset_index()
    # Per-neuron median adapted threshold over ALL HPRs (incl. uniform -1).
    # Split value comes from config/reproduction.yaml::bootstrap.threshold_split_db
    # so that this and run_bootstrap.py share a single source of truth.
    split_db = get_threshold_split_db()
    thr_med = hhl_stg_all_reset.groupby("neuron-id")["threshold"].median()
    low_ids = set(thr_med[thr_med < split_db].index)
    high_ids = set(thr_med[thr_med >= split_db].index)
    hhl_stg_low_df = hhl_stg_all_reset[hhl_stg_all_reset["neuron-id"].isin(low_ids)].copy().set_index(
        ["group", "neuron-id", "hpr"]
    )
    hhl_stg_high_df = hhl_stg_all_reset[hhl_stg_all_reset["neuron-id"].isin(high_ids)].copy().set_index(
        ["group", "neuron-id", "hpr"]
    )

    del df_mi_pdf
    gc.collect()

    return {
        "df_mi_stats": df_mi_stats,
        "betas": betas,
        "xis": xis,
        "df_hhl": df_hhl,
        "df_chl": df_chl,
        "df_map_stats_HHL_low": df_map_stats_hhl_low,
        "df_map_stats_HHL_high": df_map_stats_hhl_high,
        "df_map_stats_all_neurons_HHL": df_hhl,
        "df_map_stats_all_neurons_CHL": df_chl,
        "df_sig_thresh_gain": df_sig_thresh_gain,
        "prior_slices_all": prior_slices_all,
        "logpost_all_neurons": logpost_all_neurons,
        "df_maxl": df_maxl,
        "panel_b_hyperparams": panel_b_hyperparams,
        "panel_b_heatmaps": panel_b_heatmaps,
        "panel_b_vmax": panel_b_vmax,
        "hhl_stg_all": hhl_stg_all,
        "chl_stg_all": chl_stg_all,
        "hhl_stg_low_df": hhl_stg_low_df,
        "hhl_stg_high_df": hhl_stg_high_df,
    }


# --- SVG metadata helper -------------------------------------------------
import json as _json
from pathlib import Path as _Path

_CAPTIONS_PATH = _Path(__file__).resolve().parents[2] / "data" / "captions" / "figure_captions.json"

def _load_captions():
    try:
        return _json.loads(_CAPTIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

_CAPTIONS = _load_captions()

def save_figure(fig, stem, figure_number=None):
    """Save figure as SVG (figures/svg/) and PDF (figures/pdf/)."""
    from pathlib import Path as _PPath
    root = _PPath(__file__).resolve().parents[2]
    svg_dir = root / 'figures' / 'svg'
    pdf_dir = root / 'figures' / 'pdf'
    svg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta = _svg_caption_metadata(figure_number) if figure_number is not None else {}
    fig.savefig(svg_dir / f'{stem}.svg', metadata=meta)
    fig.savefig(pdf_dir / f'{stem}.pdf')


def _svg_caption_metadata(figure_number):
    cap = _CAPTIONS.get(str(figure_number), "")
    title = f"Figure {figure_number}"
    return {
        "Title": title,
        "Description": cap,
        "Format": "image/svg+xml",
        "Type": "Image",
    }
