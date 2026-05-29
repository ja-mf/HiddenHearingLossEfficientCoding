# %% preamble
import os
import shutil
from pathlib import Path


import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata

from .panel_shared import (
    CHL_GROUP_COLORS,
    HHL_GROUP_COLORS,
    HHL_LOW_COLOR,
    HHL_HIGH_COLOR,
    _calc_logpost_all,
    apply_panel_rcparams,
    build_map_stats,
    despineopts,
    figure_title,
    get_df_maxl_from_logpost,
    get_theta_hat,
    ks,
    load_fig5_hpr90_context,
    lxlim,
    make_hpr_palette,
    sf_letterannotation,
    x0s,
)
from .panel_shared import _svg_caption_metadata, save_figure  # noqa: E402

apply_panel_rcparams(usetex=bool(shutil.which('latex')))


# Helpers
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DERIVED = PROJECT_ROOT / 'data' / 'derived'
DFPKLS = DATA_DERIVED / 'df_pkls'
OUTPUT_DIR = PROJECT_ROOT / 'figures' / 'svg'


def _df_reset(df):
    if df is None:
        return None
    return df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()


def _ci_bounds(ci_level):
    a = (1.0 - float(ci_level) / 100.0) / 2.0
    return a, 1.0 - a


def _extract_be_means(df_stats, groups):
    d = _df_reset(df_stats)
    if d is None:
        return None
    rename = {}
    if 'betamax' in d.columns:
        rename['betamax'] = 'beta'
    if 'ximax' in d.columns:
        rename['ximax'] = 'xi'
    if rename:
        d = d.rename(columns=rename)
    req = {'group', 'hpr', 'beta', 'xi'}
    if not req.issubset(set(d.columns)):
        return None
    d = d[d['group'].isin(groups)]
    if d.empty:
        return None
    return d.groupby(['group', 'hpr'])[['beta', 'xi']].mean().reset_index()


def _draw_trajectory(ax, be_df, groups, colors, label_side=None):
    label_side = label_side or {}
    for g in groups:
        gd = be_df[be_df['group'] == g]
        pts = []
        for h in hprs1:
            r = gd[gd['hpr'] == h]
            if len(r) == 0:
                continue
            pts.append((h, float(r['beta'].iloc[0]), float(r['xi'].iloc[0])))
        if len(pts) < 2:
            continue

        c = colors[g]
        x = np.array([p[1] for p in pts], dtype=float)
        y = np.array([p[2] for p in pts], dtype=float)

        ax.plot(x, y, color=c, lw=1.7, alpha=0.72, zorder=4)
        for i in range(len(pts) - 1):
            ax.annotate(
                '', xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle='->', color=c, lw=0.8, shrinkA=0, shrinkB=0),
                zorder=5,
            )
        ax.scatter(x, y, color=c, s=18, edgecolors='white', linewidths=0.5, zorder=6)

        side = label_side.get(g, 'right')
        if side == 'left':
            dx, ha = -0.08, 'right'
        else:
            dx, ha = 0.08, 'left'
        for h, xb, yb in pts:
            ax.text(xb + dx, yb + 0.02, str(int(h)), fontsize=6, color=c, ha=ha, va='center', zorder=7)


def _draw_c_legend_floating(anchor_ax):
    anchor_ax.axis('off')
    from matplotlib.lines import Line2D

    hhl_handles = [
        Line2D([0], [0], color=hhl_palette[1], lw=1.8, marker='o', markersize=3, label='Sham-exposed'),
        Line2D([0], [0], color=hhl_palette[0], lw=1.8, marker='o', markersize=3, label='Noise-exposed'),
    ]
    chl_handles = [
        Line2D([0], [0], color=chl_palette[0], lw=1.8, marker='o', markersize=3, label='Earplugged'),
        Line2D([0], [0], color=chl_palette[1], lw=1.8, marker='o', markersize=3, label='Post plug'),
    ]

    leg1 = anchor_ax.legend(
        handles=hhl_handles,
        title='Hidden hearing loss',
        loc='upper left',
        frameon=False,
        fontsize='x-small',
        title_fontsize='x-small',
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.0,
    )
    anchor_ax.add_artist(leg1)

    anchor_ax.legend(
        handles=chl_handles,
        title='Conductive hearing loss',
        loc='upper left',
        frameon=False,
        fontsize='x-small',
        title_fontsize='x-small',
        bbox_to_anchor=(0.0, 0.52),
        borderaxespad=0.0,
    )


def _bootstrap_group_contrast(df_boot, g_ctrl='SH', g_case='NE', metric='norm_avg_utility'):
    d = _df_reset(df_boot)
    req = {'boot_iter', 'group', 'hpr', metric}
    if d is None or not req.issubset(set(d.columns)):
        return None
    piv = d.pivot_table(index=['boot_iter', 'hpr'], columns='group', values=metric)
    if g_ctrl not in piv.columns or g_case not in piv.columns:
        return None
    out = (piv[g_ctrl] - piv[g_case]).rename('contrast').reset_index()
    return out


def _center_group_contrast(df_center, g_ctrl='SH', g_case='NE', metric='norm_avg_utility'):
    d = _df_reset(df_center)
    req = {'group', 'hpr', metric}
    if d is None or not req.issubset(set(d.columns)):
        return None
    piv = d.pivot_table(index='hpr', columns='group', values=metric, aggfunc='mean')
    if g_ctrl not in piv.columns or g_case not in piv.columns:
        return None
    return (piv[g_ctrl] - piv[g_case]).rename('center')


def _load_stg(path):
    d = pd.read_parquet(path)
    if 'group' in d.columns:
        d = d.set_index(['group', 'neuron-id', 'hpr'])
    return d


def _compute_hhl_population_center(path_stg, prior_slices):
    d = _load_stg(path_stg) if not isinstance(path_stg, pd.DataFrame) else path_stg.copy()
    if isinstance(d.index, pd.MultiIndex) and 'hpr' in d.index.names:
        try:
            d = d.query('hpr != -1')
        except Exception:
            pass
    theta = get_theta_hat(d)
    logpost = _calc_logpost_all(theta, prior_slices=prior_slices)
    df_maxl_pop = get_df_maxl_from_logpost(logpost)
    stats_pop = build_map_stats(df_maxl_pop, ['NE', 'SH'], hprs1, df_mi_stats)
    return _df_reset(stats_pop)


def _animal_from_neuron(nid):
    import re
    m = re.match(r'^([A-Z]+\d+[a-z]?)u', str(nid))
    return m.group(1) if m else str(nid)


def _animal_sort_key(a):
    import re
    m = re.match(r'^([A-Z]+)(\d+)([a-z]?)$', str(a))
    if not m:
        return (str(a), 10**9, '')
    return (m.group(1), int(m.group(2)), m.group(3))


# settings

HPRS5 = [30, 45, 60, 75, 90]
hprs1 = HPRS5.copy()
hprs = HPRS5.copy()

FIGURE_OUTPUT_STEM = 'figure5'

MI_PATH = PROJECT_ROOT / 'data/artifacts' / 'df_mi_pdf.parquet'
MI_STATS_PATH = PROJECT_ROOT / 'data/artifacts' / 'df_mi_stats.parquet'

HHL_ALL_PATH = PROJECT_ROOT / 'data/artifacts' / 'HHL_bootstrap.parquet'
CHL_ALL_PATH = PROJECT_ROOT / 'data/artifacts' / 'CHL_bootstrap.parquet'
HHL_LOW_BOOT_PATH = PROJECT_ROOT / 'data/artifacts' / 'HHL_low_bootstrap.parquet'
HHL_HIGH_BOOT_PATH = PROJECT_ROOT / 'data/artifacts' / 'HHL_high_bootstrap.parquet'

HHL_STG_ALL_PATH = PROJECT_ROOT / 'data/artifacts' / 'HHL_stg.parquet'
CHL_STG_ALL_PATH = PROJECT_ROOT / 'data/artifacts' / 'CHL_stg.parquet'

required = [
    MI_PATH,
    MI_STATS_PATH,
    HHL_ALL_PATH,
    CHL_ALL_PATH,
    HHL_LOW_BOOT_PATH,
    HHL_HIGH_BOOT_PATH,
    HHL_STG_ALL_PATH,
    CHL_STG_ALL_PATH,
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError('Missing required Figure 5 inputs:\n' + '\n'.join(missing))

CTX = load_fig5_hpr90_context(str(PROJECT_ROOT))

df_mi_stats = CTX['df_mi_stats']
betas = CTX['betas']
xis = CTX['xis']
df_hhl = CTX['df_hhl']
df_chl = CTX['df_chl']
df_map_stats_HHL_low = CTX['df_map_stats_HHL_low']
df_map_stats_HHL_high = CTX['df_map_stats_HHL_high']
df_map_stats_all_neurons_HHL = CTX['df_map_stats_all_neurons_HHL']
df_map_stats_all_neurons_CHL = CTX['df_map_stats_all_neurons_CHL']
df_map_stats_all = df_hhl
df_map_stats_all_CHL = df_chl
df_sig_thresh_gain = CTX['df_sig_thresh_gain']
_prior_slices_all = CTX['prior_slices_all']
logpost_all_neurons = CTX['logpost_all_neurons']
df_maxl = CTX['df_maxl']
logpost_all = None
PANEL_B_HYPERPARAMS = CTX['panel_b_hyperparams']
PANEL_B_HEATMAPS = CTX['panel_b_heatmaps']
PANEL_B_VMAX = CTX['panel_b_vmax']
hhl_stg_all = CTX['hhl_stg_all']
chl_stg_all = CTX['chl_stg_all']
hhl_stg_low_df = CTX['hhl_stg_low_df']
hhl_stg_high_df = CTX['hhl_stg_high_df']


# %% figure
PANEL_D_CI_LEVEL = 95.0
PANEL_E_CI_LEVEL = 95.0
PANEL_STYLE = {
    'a_map_lw': 1.0,
    'd_jitter': 1.45,
    'd_err': dict(markersize=3.6, linewidth=2.1, capsize=2.9, elinewidth=1.7),
    'e1_jitter': 1.35,
    'e2_jitter': 1.35,
    'e3_jitter': 0.175,  # in quintile-index units (x runs over Q1..Q5)
    'e_err': dict(markersize=3.6, linewidth=2.1, capsize=2.6, elinewidth=1.7),
}
LOW_POP_COLOR  = HHL_LOW_COLOR
HIGH_POP_COLOR = HHL_HIGH_COLOR
FIG5_FIGNUM = 95

PANEL_B_CBAR_WIDTH = 0.006
PANEL_B_CBAR_HEIGHT_FRAC = 1
PANEL_B_CBAR_PAD = 0.0075
PANEL_E_FLOAT_LEGEND_ANCHOR = (0.905, 0.31)

hhl_palette = [HHL_GROUP_COLORS['NE'], HHL_GROUP_COLORS['SH']]
chl_palette = [CHL_GROUP_COLORS['EP'], CHL_GROUP_COLORS['PP']]
palette_hprs, hprs_colours = make_hpr_palette(hprs1)
thgplotopts = dict(
    ylabel=None,
    yticks=[0, 0.5],
    ylim=[-0.03, 0.5],
    frame_on=True,
    yticklabels=[],
    xticks=hprs1,
    xticklabels=hprs1,
    xlabel='Threshold',#\ndB SPL',
)

# plt.close(FIG5_FIGNUM)
with plt.ioff():
    fig = plt.figure(num=FIG5_FIGNUM, layout='compressed', figsize=(9.8, 5.2), dpi=200)
sf_LR = fig.subfigures(1, 2, width_ratios=[1, 1])
sf_L_outer = sf_LR[0].subfigures(2, 1, height_ratios=[0.04, 1.0], hspace=0.0)
sf_R_outer = sf_LR[1].subfigures(2, 1, height_ratios=[0.03, 1.0], hspace=0.0)

# Left side: A (marginals) + B (threshold/gain with OP background) + C (optimality space)
sf_L = sf_L_outer[1].subfigures(3, 1, height_ratios=[0.95, 1.1, 1], hspace=0.07)
sf_A = sf_L[0].subfigures(1, 2, width_ratios=[0.07, 1], wspace=0.0)
axA_label = sf_A[0].subplots(1, 1)
axsA = sf_A[1].subplots(
    2, 5,
    gridspec_kw=dict(
        left=0.02,
        right=0.995,
        bottom=0.06,
        top=0.86,
        wspace=0.06,
        hspace=0.08,
    ),
)
axsB = sf_L[1].subplots(
    2, 5,
    gridspec_kw=dict(
        left=0.12,
        right=0.995,
        bottom=0.14,
        top=0.80,
        wspace=0.01,
        hspace=0.02,
    ),
)
axsC = sf_L[2].subplots(1, 3, width_ratios=[1, 1, 0.75])
axA_label.axis('off')
axA_label.text(0.5, 0.5, 'Marginal posterior', rotation=90, ha='center', va='center')
plt.setp(axsB, box_aspect=1)

# Right side: D (raw stats), E (twopop contrasts)
sf_R = sf_R_outer[1].subfigures(2, 1, height_ratios=[1.0, 1.0], hspace=0.04)
axsD = sf_R[0].subplots(2, 3)
sf_E = sf_R[1].subfigures(1, 3, width_ratios=[1, 1, 1.45], wspace=0.0)
axsE = np.array([
    sf_E[0].subplots(1, 1),
    sf_E[1].subplots(1, 1),
    sf_E[2].subplots(1, 1),
], dtype=object)


# Panel A

def plot_logpost_marginal(hpr, hyperparam, ax, logpost_data=None):
    if logpost_data is None:
        logpost_data = logpost_all_neurons if logpost_all_neurons is not None else logpost_all

    if 'boot_iter' in logpost_data.index.names:
        tmp = logpost_data.xs(0, level='boot_iter').query(
            f'hpr == {hpr} and (group == "NE" or group == "SH")'
        ).groupby(['group', hyperparam]).apply(np.sum, axis=0)
    else:
        tmp = logpost_data.query(f'hpr == {hpr} and (group == "NE" or group == "SH")').reset_index()
        tmp = tmp.groupby(['group', hyperparam])[0].sum()

    tmp = tmp.groupby('group', group_keys=False).apply(lambda r: r / r.max())

    try:
        maxlpp = tmp.groupby('group').apply(lambda s: s.droplevel('group').idxmax())
        if hasattr(maxlpp, 'iloc') and hasattr(maxlpp, 'columns'):
            maxlpp = maxlpp.iloc[:, 0]
    except Exception:
        maxlpp = tmp.groupby('group').idxmax()

        if logpost_all is not None and 'boot_iter' in logpost_all.index.names:
            boot = logpost_all.reset_index()
            boot = boot[(boot['hpr'] == hpr) & (boot['group'].isin(['NE', 'SH']))]
            if len(boot) > 0:
                boot_m = boot.groupby(['boot_iter', 'group', hyperparam])[0].sum().reset_index(name='m')
                boot_m['m'] = boot_m['m'] / boot_m.groupby(['boot_iter', 'group'])['m'].transform('max')
                for g, gg in boot_m.groupby('group'):
                    piv = gg.pivot(index='boot_iter', columns=hyperparam, values='m').sort_index(axis=1)
                    if piv.shape[1] > 1:
                        x = piv.columns.to_numpy(dtype=float)
                        ql = np.nanpercentile(piv.to_numpy(), 2.5, axis=0)
                        qh = np.nanpercentile(piv.to_numpy(), 97.5, axis=0)
                        ax.fill_between(x, ql, qh, color=HHL_GROUP_COLORS.get(g, 'gray'), alpha=0.14, linewidth=0)

    sns.lineplot(data=tmp.reset_index(), x=hyperparam, y=0, ax=ax,
                 hue='group', legend=False, palette=HHL_GROUP_COLORS)

    for g, mlpp in maxlpp.items():
        ax.axvline(x=mlpp, ymax=0.98, color=HHL_GROUP_COLORS[g],
                   linestyle='--', linewidth=PANEL_STYLE['a_map_lw'])
    ax.set(yticks=[], ylabel='', xlabel='')
    ax.grid(False)


for i, ax in enumerate(axsA[0, :]):
    plot_logpost_marginal(hprs[i], 'beta', ax)
    ax.set_xlim(0, 6)
    ax.set_xticks([0, 6])
    ax.set_xticklabels(['', ''])
    sns.despine(ax=ax, **despineopts, left=True)
    ax.set_title(f'HPR {hprs[i]}', fontsize='medium')

for i, ax in enumerate(axsA[1, :]):
    plot_logpost_marginal(hprs[i], 'xi', ax)
    ax.set_xlim(0, 6)
    ax.set_xticks([0, 6])
    if i == 0:
        ax.set_xticklabels(['0', '6'])
    else:
        ax.set_xticklabels(['', '6'])
    sns.despine(ax=ax, **despineopts, left=True)

# Panel B

def get_beta_xi_from_maxl(group, hpr):
    return PANEL_B_HYPERPARAMS[(group, hpr)]


def get_vmax_cbar(group, hpr):
    return PANEL_B_VMAX[hpr]


def plot_thg_heatmap(df, group, hpr, ax, vmax):
    tmp = PANEL_B_HEATMAPS[(group, hpr)]
    ax.imshow(tmp, aspect='auto', extent=[lxlim, max(x0s), min(ks), max(ks)],
              cmap='Greys', vmin=0, vmax=vmax)

    c = HHL_GROUP_COLORS[group]
    ax.scatter(df['threshold'], df['gain'], color=c, alpha=0.4, marker='o', s=4.75)
    ax.add_patch(plt.Rectangle((hpr - 6, -0.025), 12, -0.05, transform=ax.transData,
                               color=hprs_colours[hpr], clip_on=False))


for i, (hpr, df_hpr) in enumerate(df_sig_thresh_gain[df_sig_thresh_gain['hpr'].isin(hprs1)].groupby('hpr')):
    vmax = max(get_vmax_cbar(group, hpr) for group, _ in df_hpr.groupby('group'))
    for j, (group, df_hpr_grp) in enumerate(df_hpr.groupby('group')):
        plot_thg_heatmap(df_hpr_grp, group, hpr, axsB[(j + 1) % 2, i], vmax)


def _style_panel_b():
    sm = axsB[0, -2].get_images()[0]
    vmin, vmax = sm.get_clim()

    cax = fig.add_axes([0, 0, PANEL_B_CBAR_WIDTH, PANEL_B_CBAR_HEIGHT_FRAC])
    cax.set_in_layout(False)

    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_label('$U(\\theta)$', rotation=270, labelpad=-5)
    cb.set_ticks([vmin, vmax])
    cb.ax.set_yticklabels(['0', 'maxU'])
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    cb.ax.tick_params(axis='y', which='both', left=False, right=True,
                      labelleft=False, labelright=True, length=0, pad=1,
                      labelsize='xx-small')
    cb.ax.yaxis.label.set_size('x-small')

    plt.setp(axsB, **thgplotopts)
    plt.setp(axsB[0, :], xlabel=None, xticklabels=[])
    plt.setp(axsB[0, 0], yticklabels=[0, 0.5])
    plt.setp(axsB[1, 0], yticklabels=[0, 0.5])
    for ax in axsB.flat:
        ax.tick_params(axis='both', labelsize='xx-small', pad=1)
        ax.xaxis.label.set_size('x-small')
        ax.yaxis.label.set_size('x-small')
    axsB[0, 0].set_ylabel('Sham-exposed\nGain', labelpad=-3, fontsize='x-small')
    axsB[1, 0].set_ylabel('Noise-exposed\nGain', labelpad=-3, fontsize='x-small')
    return cax


panel_b_cax = _style_panel_b()


# Panel C

_df = df_mi_stats.reset_index().set_index(['beta', 'xi'])[['norm_avg_utility', 'entropy', 'hpr', 'mean_fr']]
_df2 = _df.reset_index().query('hpr == 45')


def interpolate_countours(ax, xlab, ylab, zlab, vmin, vmax, title=None,
                          clevels=None, manual_pos_clvls=None):
    xi = np.linspace(_df2[xlab].min(), _df2[xlab].max(), 200)
    yi = np.linspace(_df2[ylab].min(), _df2[ylab].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((_df2[xlab], _df2[ylab]), _df2[zlab], (Xi, Yi), method='linear')
    ax.imshow(Zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower',
              aspect='auto', cmap='Grays', vmin=vmin, vmax=vmax + 1)

    manual_levels = []
    manual_positions = []
    if manual_pos_clvls:
        manual_levels, manual_positions = manual_pos_clvls
        if len(manual_levels) != len(manual_positions):
            raise ValueError('manual_pos_clvls mismatch: levels and positions length differ')

    auto_levels_input = clevels if clevels is not None else []
    all_levels = sorted(set(auto_levels_input) | set(manual_levels))
    cs0 = ax.contour(Xi, Yi, Zi, levels=all_levels, colors='#333', linewidths=0.3)

    if manual_levels:
        ax.clabel(cs0, levels=manual_levels, manual=manual_positions, inline=True, fontsize=4)
    rem = [lv for lv in cs0.levels if lv not in manual_levels]
    if rem:
        ax.clabel(cs0, levels=rem, inline=True, fontsize=4)

    if title:
        ax.set_title(title, fontsize='medium')
    return ax


def draw_panel_c():
    ax_hhl, ax_chl, ax_leg_anchor = axsC

    # Utility background only
    for axc in (ax_hhl, ax_chl):
        interpolate_countours(
            axc, 'beta', 'xi', 'norm_avg_utility', 0, 1,
            clevels=[0.1, 0.4, 0.6, 0.8],
            manual_pos_clvls=([0.1], [(2.5, 1.45)]),
            title=None,
        )
        axc.set_xlabel('$\\beta$', labelpad=-5)
        axc.set_xlabel('Optimization hyperparam.', labelpad=-6, fontsize='x-small')
        axc.set_ylabel('Metabolic trade-off', labelpad=-4, fontsize='x-small')
        axc.set(yticks=[0, 6], xticks=[0, 6])
        axc.set_box_aspect(1)

    # Use all-neuron centers when available
    c_hhl = df_map_stats_all_neurons_HHL if df_map_stats_all_neurons_HHL is not None else df_map_stats_all
    c_chl = df_map_stats_all_neurons_CHL if df_map_stats_all_neurons_CHL is not None else df_map_stats_all_CHL

    be_hhl = _extract_be_means(c_hhl, ['NE', 'SH'])
    be_chl = _extract_be_means(c_chl, ['EP', 'PP'])

    if be_hhl is not None:
        _draw_trajectory(
            ax_hhl, be_hhl, ['SH', 'NE'],
            {'SH': hhl_palette[1], 'NE': hhl_palette[0]},
            label_side={'SH': 'left', 'NE': 'right'},
        )
    if be_chl is not None:
        _draw_trajectory(
            ax_chl, be_chl, ['EP', 'PP'],
            {'EP': chl_palette[0], 'PP': chl_palette[1]},
            label_side={'EP': 'left', 'PP': 'right'},
        )

    _draw_c_legend_floating(ax_leg_anchor)


draw_panel_c()


# Panel D (as current: hierarchical bootstrap intervals, full-data centered)

def _ops_stats_centered(df_boot, df_center, groups, palette, axs, ci_level=95.0):
    metrics = ['norm_avg_utility', 'mean_fr', 'entropy']
    ycfg = {
        'norm_avg_utility': dict(ylim=[0, 0.85], ylabel='Utility'),
        'mean_fr': dict(ylim=[0, 0.85], ylabel='p(spike)'),
        'entropy': dict(ylim=[5.75, 6.55], yticks=[6, 6.3], ylabel='Entropy'),
    }
    qlo, qhi = _ci_bounds(ci_level)

    data_boot = _df_reset(df_boot)
    data_boot = data_boot[data_boot['group'].isin(groups)]

    data_center = None
    if df_center is not None:
        data_center = _df_reset(df_center)
        data_center = data_center[data_center['group'].isin(groups)]

    if len(groups) == 2:
        offsets = np.array([-PANEL_STYLE['d_jitter'], PANEL_STYLE['d_jitter']], dtype=float)
    else:
        offsets = np.linspace(-PANEL_STYLE['d_jitter'], PANEL_STYLE['d_jitter'], len(groups))
    col_map = {g: palette[i] for i, g in enumerate(groups)}

    for mi, metric in enumerate(metrics):
        ax = axs[mi]
        for gi, g in enumerate(groups):
            boot_g = data_boot[data_boot['group'] == g]
            if boot_g.empty:
                continue

            q = boot_g.groupby('hpr')[metric].quantile([qlo, qhi]).unstack()
            q.columns = ['qlo', 'qhi']

            if data_center is not None and len(data_center) > 0:
                ctr = data_center[data_center['group'] == g].groupby('hpr')[metric].mean()
            else:
                ctr = boot_g.groupby('hpr')[metric].mean()

            hpr_common = [h for h in hprs1 if (h in ctr.index and h in q.index)]
            if not hpr_common:
                continue

            x = np.array(hpr_common, dtype=float) + offsets[gi]
            y = ctr.loc[hpr_common].to_numpy(dtype=float)
            ylo = q.loc[hpr_common, 'qlo'].to_numpy(dtype=float)
            yhi = q.loc[hpr_common, 'qhi'].to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(0, y - ylo), np.maximum(0, yhi - y)])

            ax.errorbar(
                x, y, yerr=yerr,
                color=col_map[g], marker='o', linestyle='none',
                **PANEL_STYLE['d_err'],
            )

        ax.set(xlabel='', **{k: v for k, v in ycfg[metric].items() if k in ['ylim', 'yticks']})
        ax.set_xticks(hprs1)
        if mi > 0:
            ax.set(yticklabels=[])
        ax.set_ylabel(ycfg[metric]['ylabel'], fontsize='small')
def draw_panel_d(ci_level=95.0):
    _ops_stats_centered(df_hhl, df_map_stats_all_neurons_HHL, ['NE', 'SH'], hhl_palette, axsD[0, :], ci_level=ci_level)
    _ops_stats_centered(df_chl, df_map_stats_all_neurons_CHL, ['EP', 'PP'], chl_palette, axsD[1, :], ci_level=ci_level)
    for a in axsD[0, :]:
        a.set_xticklabels([])
    for a in axsD[1, :]:
        a.set_xticks(hprs1)
        a.set_xticklabels(hprs1)
        a.set_xlabel('HPR')
    for a in axsD.flat:
        sns.despine(ax=a, **despineopts)

    _ax_d1 = axsD[0, 0]
    _ax_d1.text(29, 0.8, '*', ha='center', va='bottom', fontsize='medium')
    _ax_d3 = axsD[0, 2]
    _ax_d3.text(30, 6.5, '**', ha='center', va='bottom', fontsize='medium')


draw_panel_d(ci_level=PANEL_D_CI_LEVEL)


# Panel E (HHL twopop, SH-NE convention)

def _draw_e3_animal_composition_legacy(ax, ax_leg, hhl_stg_low_df, hhl_stg_high_df):
    """E3 legacy: per-animal low/high sample composition (barplot).

    Retained for transparency / supplementary use. Not called from draw_panel_e;
    the active E3 is _draw_e3_threshold_quintile_profile (Table S3).
    """
    low_stg = hhl_stg_low_df.reset_index().copy()
    high_stg = hhl_stg_high_df.reset_index().copy()
    low_counts = low_stg[['group', 'neuron-id']].drop_duplicates()
    low_counts['animal'] = low_counts['neuron-id'].apply(_animal_from_neuron)
    low_counts['population'] = 'low'
    high_counts = high_stg[['group', 'neuron-id']].drop_duplicates()
    high_counts['animal'] = high_counts['neuron-id'].apply(_animal_from_neuron)
    high_counts['population'] = 'high'
    comp = pd.concat([low_counts, high_counts], ignore_index=True)
    comp = comp.groupby(['group', 'animal', 'population']).size().rename('n').reset_index()

    animals_order = sorted(comp['animal'].unique(), key=_animal_sort_key)
    sns.barplot(data=comp, x='animal', y='n', hue='population', order=animals_order,
                palette={'low': LOW_POP_COLOR, 'high': HIGH_POP_COLOR}, ax=ax)
    ax.set_xlabel('Animal')
    ax.set_ylabel('n neurons')
    ax.set_xlim(-0.5, len(animals_order) - 0.5)
    ax.set_xticks(np.arange(len(animals_order)))
    ax.set_xticklabels(animals_order)
    ax.tick_params(axis='x', rotation=90, labelsize='xx-small')
    ax.tick_params(axis='y', labelsize='xx-small')
    ax.xaxis.label.set_size('x-small')
    ax.yaxis.label.set_size('x-small')
    if ax.legend_ is not None:
        ax.legend_.remove()

    from matplotlib.lines import Line2D
    pop_handles = [
        Line2D([0], [0], color=HIGH_POP_COLOR, lw=1.6, marker='s', markersize=4, label='High threshold'),
        Line2D([0], [0], color=LOW_POP_COLOR, lw=1.6, marker='s', markersize=4, label='Low threshold'),
    ]
    ax_leg.axis('off')
    ax_leg.legend(handles=pop_handles, labels=['High threshold', 'Low threshold'],
                  frameon=False, ncol=1, loc='center', fontsize='x-small')
    sns.despine(ax=ax, **despineopts)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 60, 120, 180])


def _draw_e3_threshold_quintile_profile(ax, hprs_panel=(30, 45, 75, 90),
                                        sig_markers=None, ylim_for_markers=(-0.70, 0.85)):
    """E3 active: Δ̄ utility (SH − NE) per across-context threshold quintile,
    one series per HPR, with 95% bootstrap CIs (Table S3, threshold-resolved view).

    Reads data/derived/tables/threshold_quantile_profile.csv. Regenerate via
    scripts/threshold_quantile_profile.py if the canonical artifacts change.
    Errorbar idiom matches Panel E1/E2 (points + vertical bars, no connecting line).
    Legend lives inside the plot (upper-right, 2-column) so axsE3 fills the
    full panel-E row height like E1 and E2.
    `sig_markers` is an iterable of (quintile, hpr, text) tuples placed above
    the upper CI of the matching cell.
    """
    csv = PROJECT_ROOT / 'data' / 'derived' / 'tables' / 'threshold_quantile_profile.csv'
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run scripts/threshold_quantile_profile.py")
    df = pd.read_csv(csv).sort_values('quintile')
    x_base = df['quintile'].to_numpy(dtype=float)

    drawn = list(hprs_panel)
    n_series = len(drawn)
    if n_series > 1:
        offsets = np.linspace(-PANEL_STYLE['e3_jitter'], PANEL_STYLE['e3_jitter'], n_series)
    else:
        offsets = np.array([0.0])

    drawn_ok = []
    series_offset = {}
    for i, h in enumerate(drawn):
        med_col = f'median_delta_hpr{h}'
        lo_col = f'ci_lo_hpr{h}'
        hi_col = f'ci_hi_hpr{h}'
        if med_col not in df.columns or lo_col not in df.columns or hi_col not in df.columns:
            continue
        y = df[med_col].to_numpy(dtype=float)
        ylo = df[lo_col].to_numpy(dtype=float)
        yhi = df[hi_col].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0, y - ylo), np.maximum(0, yhi - y)])
        c = hprs_colours[h]
        ax.errorbar(
            x_base + offsets[i], y, yerr=yerr,
            color=c, marker='o', linestyle='none', label=f'HPR {h}',
            **PANEL_STYLE['e_err'],
        )
        drawn_ok.append(h)
        series_offset[h] = float(offsets[i])

    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f'Q{int(q)}' for q in x_base])
    ax.set_xlabel('Threshold quintile (low $\\rightarrow$ high)')
    ax.tick_params(axis='x', labelsize='xx-small')
    ax.tick_params(axis='y', labelsize='xx-small')
    ax.xaxis.label.set_size('x-small')
    ax.yaxis.label.set_size('x-small')

    # Significance markers (caller supplies cells; placed above each CI bracket)
    if sig_markers:
        yspan = ylim_for_markers[1] - ylim_for_markers[0]
        for q, h, txt in sig_markers:
            if h not in series_offset:
                continue
            row = df[df['quintile'] == q]
            if row.empty:
                continue
            row = row.iloc[0]
            x_pos = float(q) + series_offset[h]
            y_pos = float(row[f'ci_hi_hpr{h}']) - 0.42 * yspan
            ax.text(x_pos, y_pos, txt, ha='center', va='bottom', fontsize='medium')

    sns.despine(ax=ax, **despineopts)

    # Legend inside the plot, upper-right, 2x2
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=hprs_colours[h], lw=0, marker='o',
               markersize=4, markeredgecolor='white', markeredgewidth=0.5,
               label=f'HPR {h}')
        for h in drawn_ok
    ]
    ax.legend(
        handles=handles, frameon=False, ncol=2, loc='upper right',
        fontsize='xx-small', handlelength=0.8, columnspacing=0.8,
        handletextpad=0.3, borderaxespad=0.4,
    )


def draw_panel_e(ci_level=95.0):
    # Bootstrap tables from preamble
    if df_map_stats_HHL_low is None or df_map_stats_HHL_high is None:
        for ax in axsE:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Two-pop data unavailable', ha='center', va='center', transform=ax.transAxes)
        return

    qlo, qhi = _ci_bounds(ci_level)

    # Full-data centers for low/high populations
    center_low = _compute_hhl_population_center(hhl_stg_low_df, _prior_slices_all)
    center_high = _compute_hhl_population_center(hhl_stg_high_df, _prior_slices_all)

    c_low = _center_group_contrast(center_low, g_ctrl='SH', g_case='NE')
    c_high = _center_group_contrast(center_high, g_ctrl='SH', g_case='NE')

    b_low = _bootstrap_group_contrast(df_map_stats_HHL_low, g_ctrl='SH', g_case='NE')
    b_high = _bootstrap_group_contrast(df_map_stats_HHL_high, g_ctrl='SH', g_case='NE')

    # E1: SH-NE utility by HPR for low/high populations
    ax = axsE[0]
    offsets = {'low': -PANEL_STYLE['e1_jitter'], 'high': PANEL_STYLE['e1_jitter']}
    pop_cols = {'low': LOW_POP_COLOR, 'high': HIGH_POP_COLOR}

    e1_lows, e1_highs = [], []
    for pop, bdf, ctr in [('low', b_low, c_low), ('high', b_high, c_high)]:
        q = bdf.groupby('hpr')['contrast'].quantile([qlo, qhi]).unstack()
        q.columns = ['qlo', 'qhi']
        hpr_common = [h for h in hprs1 if (h in q.index and h in ctr.index)]
        x = np.array(hpr_common, dtype=float) + offsets[pop]
        y = ctr.loc[hpr_common].to_numpy(dtype=float)
        ylo = q.loc[hpr_common, 'qlo'].to_numpy(dtype=float)
        yhi = q.loc[hpr_common, 'qhi'].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0, y - ylo), np.maximum(0, yhi - y)])
        ax.errorbar(x, y, yerr=yerr, color=pop_cols[pop], marker='o', linestyle='none',
                    label=pop, **PANEL_STYLE['e_err'])
        if len(ylo):
            e1_lows.append(float(np.min(ylo)))
            e1_highs.append(float(np.max(yhi)))

    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(hprs1)
    ax.set_xticklabels(hprs1)
    ax.set_xlabel('HPR')
    # ax.set_ylabel(r'$\Delta U_{\mathrm{HHL}} = U_{\mathrm{SH}} - U_{\mathrm{NE}}$', fontsize='small')
    ax.set_title(r'$\Delta U_{\mathrm{SH-NE}}$', fontsize='medium')
    if e1_lows and e1_highs:
        lo = min(min(e1_lows), 0.0)
        hi = max(max(e1_highs), 0.0)
        pad = max(0.03, 0.08 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    _e1_low_x = 30 + offsets['low']
    _e1_low_hi = b_low.groupby('hpr')['contrast'].quantile(qhi).loc[30]
    _e1_yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.text(_e1_low_x, _e1_low_hi + 0.12 * _e1_yspan, '**', ha='center', va='bottom', fontsize='medium')
    sns.despine(ax=ax, **despineopts)

    # E2: per-group high-low difference: Delta_h^g = U_high^g - U_low^g
    ax = axsE[1]

    # Bootstrap group-wise deltas
    b_low_full = _df_reset(df_map_stats_HHL_low)
    b_high_full = _df_reset(df_map_stats_HHL_high)
    m = b_high_full.merge(
        b_low_full,
        on=['boot_iter', 'group', 'hpr'],
        suffixes=('_high', '_low')
    )
    m['delta_group'] = m['norm_avg_utility_high'] - m['norm_avg_utility_low']

    # Full-data centers by group
    c_low_full = center_low.pivot_table(index='hpr', columns='group', values='norm_avg_utility', aggfunc='mean')
    c_high_full = center_high.pivot_table(index='hpr', columns='group', values='norm_avg_utility', aggfunc='mean')

    e2_lows, e2_highs = [], []
    for g, col, dx in [('SH', hhl_palette[1], -PANEL_STYLE['e2_jitter']), ('NE', hhl_palette[0], PANEL_STYLE['e2_jitter'])]:
        q = m[m['group'] == g].groupby('hpr')['delta_group'].quantile([qlo, qhi]).unstack()
        q.columns = ['qlo', 'qhi']
        ctr = (c_high_full[g] - c_low_full[g]).rename('center')
        hpr_common = [h for h in hprs1 if (h in q.index and h in ctr.index)]
        x = np.array(hpr_common, dtype=float) + dx
        y = ctr.loc[hpr_common].to_numpy(dtype=float)
        ylo = q.loc[hpr_common, 'qlo'].to_numpy(dtype=float)
        yhi = q.loc[hpr_common, 'qhi'].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0, y - ylo), np.maximum(0, yhi - y)])
        ax.errorbar(x, y, yerr=yerr, color=col, marker='o', linestyle='none',
                    **PANEL_STYLE['e_err'])
        if len(ylo):
            e2_lows.append(float(np.min(ylo)))
            e2_highs.append(float(np.max(yhi)))

    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(hprs1)
    ax.set_xticklabels(hprs1)
    ax.set_xlabel('HPR')
    # ax.set_ylabel(r'$\Delta U_h^{g} = U_{h,\mathrm{high}}^{g} - U_{h,\mathrm{low}}^{g}$', fontsize='small')
    ax.set_title(r'$\Delta U_{\mathrm{high-low}}$', fontsize='medium')
    if e2_lows and e2_highs:
        lo = min(min(e2_lows), 0.0)
        hi = max(max(e2_highs), 0.0)
        pad = max(0.03, 0.08 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    sns.despine(ax=ax, **despineopts)

    # E3: threshold-resolved Δ̄U_{SH-NE} per quintile per HPR (Table S3).
    _draw_e3_threshold_quintile_profile(
        axsE[2],
        hprs_panel=(30, 45, 75, 90),
        sig_markers=[(2, 30, '**'), (3, 30, '***')],
        ylim_for_markers=(-0.70, 0.85),
    )
    axsE[2].set_title(r'$\mathrm{Pr}(\mathrm{NE}>\mathrm{SH})$', fontsize='medium')


draw_panel_e(ci_level=PANEL_E_CI_LEVEL)

# Panel E post-draw: E1, E2, E3 all share the same ΔU y-scale and ticks.
_ax_e1, _ax_e2, _ax_e3 = axsE
for _ax in (_ax_e1, _ax_e2, _ax_e3):
    _ax.set(ylim=(-0.70, 0.85), yticks=[-0.5, 0, 0.5])
    _ax.yaxis.set_major_formatter(ScalarFormatter())
    sns.despine(ax=_ax, **despineopts)
_ax_e2.tick_params(labelleft=False)
_ax_e3.tick_params(labelleft=False)


# Annotations and save

fig.canvas.draw()
fig.set_layout_engine('none')

_fig_w = fig.bbox.width
_fig_h = fig.bbox.height

def _fig_x_mid(sf):
    return 0.5 * (sf.bbox.x0 + sf.bbox.x1) / _fig_w

def _fig_y_gap_mid(sf_upper, sf_lower):
    return 0.5 * (sf_upper.bbox.y0 + sf_lower.bbox.y1) / _fig_h

def _fig_y_mid(sf):
    return 0.5 * (sf.bbox.y0 + sf.bbox.y1) / _fig_h

fig.text(_fig_x_mid(sf_L_outer[0]), _fig_y_mid(sf_L_outer[0]), r'\textbf{A} Hyperparameter inference', ha='center', va='center')
fig.text(_fig_x_mid(sf_L[1]), _fig_y_gap_mid(sf_L[0], sf_L[1]), r'\textbf{B} Threshold-gain maps', ha='center', va='center')
fig.text(_fig_x_mid(sf_L[2]), _fig_y_gap_mid(sf_L[1], sf_L[2]), r'\textbf{C} Optimality space dynamics ($\tilde{U}$)', ha='center', va='center')
fig.text(_fig_x_mid(sf_R_outer[0]), _fig_y_mid(sf_R_outer[0]), r'\textbf{D} Dynamics of OPs statistics', ha='center', va='center')
fig.text(_fig_x_mid(sf_R[1]), _fig_y_gap_mid(sf_R[0], sf_R[1]), r'\textbf{E} Sub-population contrasts (HHL)', ha='center', va='center')

_renderer = fig.canvas.get_renderer()
_b_ref_top = axsB[0, -1].get_window_extent(_renderer)
_b_ref_bottom = axsB[1, -1].get_window_extent(_renderer)
_b_height = PANEL_B_CBAR_HEIGHT_FRAC * _b_ref_top.height / _fig_h
_b_mid_y = 0.5 * (_b_ref_top.y0 + _b_ref_bottom.y1) / _fig_h
_b_y0 = _b_mid_y - 0.5 * _b_height
_b_x0 = _b_ref_top.x1 / _fig_w + PANEL_B_CBAR_PAD
panel_b_cax.set_position([_b_x0, _b_y0, PANEL_B_CBAR_WIDTH, _b_height])

for _ax in axsA[0, :]:
    _ax.set_xlabel(r'$\beta$', labelpad=-4)
for _ax in axsA[1, :]:
    _ax.set_xlabel(r'$\xi$', labelpad=-12)

save_figure(fig, FIGURE_OUTPUT_STEM, 5)

