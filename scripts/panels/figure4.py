# %% preamble
import os
from pathlib import Path
from typing import Any


import matplotlib

matplotlib.rcParams.update({'text.usetex': True, 'axes.labelpad': 0, 'figure.dpi': 200})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Annotation
from scipy.interpolate import griddata

from .panel_shared import (
    apply_panel_rcparams,
    figure_title,
    hprs1,
    ks,
    levels,
    load_example_stimlevels,
    logistsig,
    lxlim,
    make_hpr_palette,
    pmf_singledistr_gerbil,
    sf_letterannotation,
    x0s,
)
from .panel_shared import _svg_caption_metadata, _normalize_xi, save_figure  # noqa: E402

apply_panel_rcparams()

# panel 3 v2 (with OPs instead of OPs stats, as I added a new panel w OPs stats.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DFPKLS = PROJECT_ROOT / 'data' / 'derived' / 'df_pkls'
OUTPUT_DIR = PROJECT_ROOT / 'figures' / 'svg'
MI_PATH = PROJECT_ROOT / 'data/artifacts' / 'df_mi_pdf.parquet'
MI_STATS_PATH = PROJECT_ROOT / 'data/artifacts' / 'df_mi_stats.parquet'
STIMLEVELS_PATH = PROJECT_ROOT / 'data' / 'artifacts' / 'example_stimlevels.npy'

palette_ct = sns.color_palette('Set3', n_colors=6)[3:]
yticks_hpr = [0.01, 0.16]
dos = 4
mult = 4


def despine_ax(ax, *, fig=None, left=False, bottom=False):
    sns.despine(ax=ax, fig=fig, left=left, bottom=bottom, trim=True, offset=dos)


def load_df_mi() -> pd.DataFrame:
    d = _normalize_xi(pd.read_parquet(MI_PATH))
    if 'hpr' in d.columns:
        d = d.set_index(['hpr', 'xi', 'beta', 'x0', 'k'])
    return d.sort_index()


def load_df_mi_stats() -> pd.DataFrame:
    d = _normalize_xi(pd.read_parquet(MI_STATS_PATH))
    if 'hpr' in d.columns:
        d = d.set_index(['hpr', 'beta', 'xi'])
    return d.sort_index()


prob = pmf_singledistr_gerbil()
stimlevels = load_example_stimlevels(STIMLEVELS_PATH)
df_mi = load_df_mi()
df_mi_stats = load_df_mi_stats()
palette_hprs, hprs_colours = make_hpr_palette(hprs1)
nchanges = 25

# seq
seq = stimlevels[:nchanges, 3]
stim = stimlevels[:nchanges, 3]
xax = np.arange(len(stim))
bHPR = np.where(seq <= 36)
iHPR = np.where((seq > 36) & (seq < 54))
aHPR = np.where(seq >= 54)

# sigmoid / response
x0sv = np.linspace(24, 96, 100)
resp = logistsig(stim, 48, 0.2, 0, 1)
resp0i = np.where(resp < 0.5)[0]
resp1i = np.where(resp >= 0.5)[0]

# Panel E only: use corrected HPR90 MI table and 5-HPR layout
PANEL_E_HPRS = [30, 45, 60, 75, 90]
panel_e_palette = sns.color_palette('flare', len(PANEL_E_HPRS))
panel_e_hpr_colours = dict(zip(PANEL_E_HPRS, panel_e_palette))
stats_cmap = 'Greys'
yticks = [min(ks), 0, max(ks)]
stats_labs = {'mi': 'MI', 'mu_rt': '$\\mu_r$', 'std_rt': '$\\sigma_r$'}


def load_panel_e_df_mi() -> Any:
    mi_path = PROJECT_ROOT / 'data' / 'artifacts' / 'df_mi_pdf.parquet'
    if not mi_path.exists():
        return df_mi

    d = _normalize_xi(pd.read_parquet(mi_path))
    if 'hpr' in d.columns:
        d = d.set_index(['hpr', 'xi', 'beta', 'x0', 'k'])
    return d.sort_index()


df_mi_panel_e = load_panel_e_df_mi()
stats_maxs = {
    'mi': df_mi_panel_e.mi.max(),
    'mu_rt': df_mi_panel_e.mu_rt.max(),
    'std_rt': df_mi_panel_e.std_rt.max(),
}
panel_e_heatmaps = {}
for _hpr in PANEL_E_HPRS:
    for _stat in ['mi', 'mu_rt']:
        _df_heat = df_mi_panel_e.loc[(_hpr, 0, 0), _stat]
        panel_e_heatmaps[(_hpr, _stat)] = (
            _df_heat.reset_index().pivot(index='k', columns='x0', values=_stat).sort_index(ascending=False)
        )

# Optimization priors panel
op_hpr, op_stat = 45, 'pdf'
op_cmap = LinearSegmentedColormap.from_list('fade_to_color', ['white', hprs_colours[op_hpr]])
# df_mi = df_mi.sort_index()
betaxis = [(0, 0), (2, 1),
            (2, 0), (1, 6)]
_beta_vals = df_mi.index.get_level_values('beta').unique().to_numpy()
_xi_vals = df_mi.index.get_level_values('xi').unique().to_numpy()
nearest_betaxis = [
    (
        _beta_vals[np.abs(_beta_vals - beta).argmin()],
        _xi_vals[np.abs(_xi_vals - xi).argmin()],
    )
    for beta, xi in betaxis
]
op_surfaces = {}
for _beta, _xi in nearest_betaxis:
    _df_op = df_mi.loc[(op_hpr, _xi, _beta), op_stat]
    op_surfaces[(_beta, _xi)] = (
        _df_op.reset_index().pivot(index='k', columns='x0', values=op_stat).sort_index(ascending=False)
    )

# OP distribution characteristics
_df = df_mi_stats.reset_index().set_index(['beta', 'xi'])[['norm_avg_utility', 'entropy', 'hpr', 'mean_fr']]
_df2 = _df.reset_index().query('hpr == 45')
contour_xi = np.linspace(_df2['beta'].min(), _df2['beta'].max(), 200)
contour_yi = np.linspace(_df2['xi'].min(), _df2['xi'].max(), 200)
contour_extent = [contour_xi.min(), contour_xi.max(), contour_yi.min(), contour_yi.max()]
contour_Xi, contour_Yi = np.meshgrid(contour_xi, contour_yi)
contour_surfaces = {
    'norm_avg_utility': griddata((_df2['beta'], _df2['xi']), _df2['norm_avg_utility'], (contour_Xi, contour_Yi), method='linear'),
    'mean_fr': griddata((_df2['beta'], _df2['xi']), _df2['mean_fr'], (contour_Xi, contour_Yi), method='linear'),
    'entropy': griddata((_df2['beta'], _df2['xi']), _df2['entropy'], (contour_Xi, contour_Yi), method='linear'),
}
utility_points = {
    'beta': _df.reset_index().groupby(['hpr', 'beta']).mean().reset_index().query('hpr == 45'),
    'xi': _df.reset_index().groupby(['hpr', 'xi']).mean().reset_index().query('hpr == 45'),
}


def plot_stat_heatmap(hpr, stat, ax):
    tmp = panel_e_heatmaps[(hpr, stat)]
    hm = ax.imshow(tmp, aspect='auto', extent=[lxlim, max(x0s), min(ks), max(ks)],
                   cmap=stats_cmap, vmin=0, vmax=stats_maxs[stat])
    ax.set(yticklabels=[], xticklabels=[])
    ax.tick_params(left=False, bottom=False, right=False, top=False)

    x_start = tmp.columns.min()
    ax.add_patch(plt.Rectangle((hpr - 6, ks.min()), 12, -0.08, transform=ax.transData,
                               color=panel_e_hpr_colours[hpr], clip_on=False))
    ax.grid(False)
    #ax.set(xticks=hprs1+[max(levels)])

#    if hpr == max(hprs1):
#        f.colorbar(hm,label=stats_labs[stat])
    return hm, x_start


def add_heatmap_cbar(ax_heatmap, fig, pos, label, ticks):
    ax_cbar_mi = fig.add_axes(pos)
    cbar = plt.colorbar(ax_heatmap.get_images()[0], cax=ax_cbar_mi,
                        orientation='horizontal')

    cbar.set_label(label, labelpad=-10, loc='center',fontsize='small')  # loc can be 'left', 'center', or 'right'
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.tick_top()

    # cbar.set_ticks([])
    #cbar.ax.text(-0.1, 0.5, f'{ticks[0]}', ha='left', va='bottom', transform=cbar.ax.transAxes)
    #cbar.ax.text(1.05, 0.5, f'{ticks[1]}', ha='left', va='bottom', transform=cbar.ax.transAxes)
    # cbar.ax.text(1, 1.2, f'{ticks[1]}', ha='right', va='bottom', transform=cbar.ax.transAxes)

    # cbar.ax.xaxis.set_label_position('top')
    cbar.set_ticks([cbar.vmin, cbar.vmax])
    cbar.set_ticklabels(ticks)
    return cbar


def plot_OP(beta, xi, ax):
    tmp = op_surfaces[(beta, xi)]
    hm = ax.imshow(tmp, aspect='auto', extent=[lxlim, max(x0s), min(ks), max(ks)],
                   cmap=op_cmap, vmin=0, vmax=0.005)
    ax.set(yticklabels=[], xticklabels=[], box_aspect=1)
    ax.grid(False)
    ax.tick_params(left=False, bottom=False, right=False, top=False)
    text = f'$({round(beta)}, {round(xi)})$'

    if (xi, beta) == (0, 0):
        text = '$(\\beta,\\xi)$\n' + text

    ax.text(1, 0, text, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='bottom',
            fontsize='small', color='black')
    #ax.set(xticks=hprs1+[max(levels)])
    return hm


def interpolate_countours(ax, surface_key, vmin, vmax, title=None):
    Zi = contour_surfaces[surface_key]

    ax.imshow(Zi, extent=contour_extent, origin='lower', aspect='auto', cmap=op_cmap, vmin=vmin, vmax=vmax)
    cmap_contour = sns.color_palette('Grays', n_colors=20)[-5:]
    cs0 = ax.contour(contour_Xi, contour_Yi, Zi, levels=5,#[0.1, 0.4, 0.8],
                     colors='#333', linewidths=0.5)
    ax.clabel(cs0, inline=True, fontsize=8)

    # import matplotlib.patheffects as pe
    # for txt in cs0.axes.texts:
    #     txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground='gray')])
    if title:
        ax.set_title(title, fontsize='small')
    return ax, cmap_contour


def plot_utility(ax, x, group, cmap):
    df = utility_points[group]
    ax.scatter(df[x], df['norm_avg_utility'], c=df[group], cmap=cmap, s=3, vmin=0, vmax=6)


def add_scalar_cbar(ax, parent, pos, label, ticks, cmap='Greys'):
    cax = parent.add_axes(pos)
    sm = ScalarMappable(cmap=cmap)
    sm.set_array([min(ticks), max(ticks)])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks(ticks)
    return cbar
    #cbar.set(label=label, ticks=ticks)


# %% figure

fig = plt.figure(layout='compressed', figsize=(2 * mult, 1.1 * mult))#, figsize=(10, 4))
subfigs = fig.subfigures(2, 1, height_ratios=[1, 1.5])#, wspace=0.07)
axsTop = subfigs[0].subplots(1, 4)
plt.setp(axsTop, box_aspect=1)

panel_e_cbar_top = [0.79, 0.835, 0.17, 0.018]
panel_e_cbar_bottom = [0.79, 0.405, 0.17, 0.018]
PANEL_E_SUBPLOTS_KW = {
    'left': 0.070,
    'right': 0.980,
    'bottom': 0.09,
    'top': 0.88,
    'wspace': 0.08,
    'hspace': 0.16,
}
PANEL_E_SHIFT_X = 0.0
PANEL_E_SHIFT_Y = 0.0
PANEL_E_SCALE_W = 1.0
PANEL_E_SCALE_H = 1.0

# stim prob distr
ax = axsTop[0]
(m1, s1, b1) = ax.stem(levels[:5], prob[str(45)][:5, 1], linefmt='grey', label='bHPR')
(m2, s2, b2) = ax.stem(levels[5:10], prob[str(45)][5:10, 1], linefmt='grey', label='HPR')
(m3, s3, b3) = ax.stem(levels[10:], prob[str(45)][10:, 1], linefmt='grey', label='aHPR')
plt.setp([m1], color=palette_ct[0], markersize=3)
plt.setp([m2], color=palette_ct[1], markersize=3)
plt.setp([m3], color=palette_ct[2], markersize=3)
plt.setp([b1, b2, b3], color='white')
ax.legend(frameon=False, loc='center right', bbox_to_anchor=(1.2, 0.5), handletextpad=0)
#ax.set_yscale('log')

axsTop[0].set(ylim=[0, 0.165], xticks=[30, 45, 60, 75, 96], yticks=yticks_hpr, xlabel='dB SPL', ylabel='Probability', title='Stimulus distribution')
axsTop[0].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
axsTop[0].grid(False)
despine_ax(axsTop[0])


#[a.tick_params(axis='x',which='both') for a in ax]
#ax[0].set(yticklabels=[0,0.04,0.08,0.16], ylabel='Probability')

# seq
(m1, _, _) = axsTop[1].stem(xax[bHPR], seq[bHPR], linefmt='grey')
(m2, _, _) = axsTop[1].stem(xax[iHPR], seq[iHPR], linefmt='grey')
(m3, _, _) = axsTop[1].stem(xax[aHPR], seq[aHPR], linefmt='grey')
plt.setp([m1], color=palette_ct[0], markersize=3)
plt.setp([m2], color=palette_ct[1], markersize=3)
plt.setp([m3], color=palette_ct[2], markersize=3)

axsTop[1].set(ylim=[20, 100], yticks=[30, 45, 60, 75], xticks=[0, 10], xticklabels=[None, '500ms'], title='Stimulus sequence $s_t$\n\\small{$c_t=\\{\\mathrm{bHPR, HPR, aHPR}\\}^N$}', ylabel='dB SPL')
despine_ax(axsTop[1])
axsTop[1].grid(False)

# sigmoid
ax = axsTop[2]
#ax.plot(x0sv,sigmoid(x0sv,48,0.07,0,1))
ax.plot(x0sv, logistsig(x0sv, 48, 0.2, 0, 1), color='black')

ax.annotate('', xy=(24, 0.5), xycoords='data', xytext=(48, 0.5), textcoords='data', arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", ls='dashed', color='gray'), color='#ccc')
ax.annotate('', xy=(60, 0.8), xycoords='data', xytext=(47, 0.2), textcoords='data', arrowprops=dict(arrowstyle="-", ls='dashed', color='gray'), color='#ccc')
ax.annotate('$x_0$', xy=(32, 0.55), textcoords='data')
ax.annotate('$k$', xy=(57, 0.4), textcoords='data')
ax.set(xlim=[24, 96], xticks=[24, 96], yticks=[0, 1], ylabel='p(spike)', title='Model neuron\n $\\theta=\\{x_0,k\\}$')
ax.set_xlabel('$x$', labelpad=-3)
ax.grid(False)
despine_ax(ax)

# response
ax = axsTop[3]
ax.scatter(resp0i, resp[resp0i], marker='o', facecolors='white', edgecolors='black')
ax.scatter(resp1i, resp[resp1i], c='black', marker='o')
ax.plot(np.arange(len(stim)), logistsig(stim, 48, 0.2, 0, 1), color='gray')
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')
ax.set(yticks=[0, 0.5, 1], xticks=[0, 10], xticklabels=[None, '500ms'], ylabel='$r$', title='Response $r_t=\\{0,1\\}$\n$\\fullmoon:0, \\newmoon: 1$')
despine_ax(ax)

# bottom panels
axsBottom = subfigs[1].subfigures(1, 3, width_ratios=[1.75, 1, 1.75])

axOPS = axsBottom[0].subplots(2, len(PANEL_E_HPRS), gridspec_kw=PANEL_E_SUBPLOTS_KW)#,width_ratios=[20,20,20,20,1])
axOPST = axsBottom[1].subplots(3, 2, height_ratios=[1, 3, 3])
axOPST_cbar = axOPST[0, :]
[ax.set_visible(False) for ax in axOPST_cbar]
axOPST = axOPST[1:, :]
plt.setp(axOPS, box_aspect=1)
axsBottom[0].suptitle('Response statistics', x=0.45)

for i, hpr in enumerate(PANEL_E_HPRS):
    for j, stat in enumerate(['mi', 'mu_rt']):
        plot_stat_heatmap(hpr, stat, axOPS[j, i])

for ax in axOPS.flatten():
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + PANEL_E_SHIFT_X,
        pos.y0 + PANEL_E_SHIFT_Y,
        pos.width * PANEL_E_SCALE_W,
        pos.height * PANEL_E_SCALE_H,
    ])

[a.set_ylabel('$k$', labelpad=-10) for a in axOPS[:, 0]]
last_col = len(PANEL_E_HPRS) - 1
for i in range(len(PANEL_E_HPRS)):
    axOPS[1, i].set_xlabel('$x_0$', labelpad=-10 if i in (0, last_col) else 0)

axOPS[0, 0].set(yticks=[0.5], yticklabels=[0.5])
axOPS[1, 0].set(yticks=[-0.5], yticklabels=[-0.5], xticks=[24], xticklabels=[24])
for ax in axOPS[:, 0]:
    ax.tick_params(axis='y', labelsize='small', pad=-2)
axOPS[1, last_col].set(xticks=[96], xticklabels=[96])

# ax_cbar_mi = axsBottom[0].add_axes([0.92, 0.55, 0.015, 0.3])
# plt.colorbar(axOPS[0,3].get_images()[0], cax=ax_cbar_mi)

cbar = add_heatmap_cbar(axOPS[0, last_col], axsBottom[0], panel_e_cbar_top, '$I(c; r)$', [0, 0.65])
cbar = add_heatmap_cbar(axOPS[1, last_col], axsBottom[0], panel_e_cbar_bottom, '$\\mu_r$', [0, 1])

axsBottom[0].text(
    panel_e_cbar_top[0] - 0.015,
    panel_e_cbar_top[1] + panel_e_cbar_top[3] / 2,
    'Mutual information',
    transform=axsBottom[0].transSubfigure,
    ha='right',
    va='center',
)
axsBottom[0].text(
    panel_e_cbar_bottom[0] - 0.015,
    panel_e_cbar_bottom[1] + panel_e_cbar_bottom[3] / 2,
    'p(spike)',
    transform=axsBottom[0].transSubfigure,
    ha='right',
    va='center',
)

#cax=axOPS[0,-1],label='$I$ [bits]')
# plt.colorbar(axOPS[1,3].get_images()[0], cax=axOPS[1,-1], label='$\mu_r$ [prob]')
# plt.setp(axOPS[:,-1], box_aspect=20)
# axOPS[0,0].text(-1, 0.5, 'Mutual information \n $I(c_t,r_t)$', transform=axOPS[0,0].transAxes, horizontalalignment='center')
# axOPS[1,0].text(-1, 0.5, 'Mean p(spike)\n $\mu_r$', transform=axOPS[1,0].transAxes, horizontalalignment='center')

axsBottom[1].suptitle('Optimization priors\n $p(\\theta|\\beta,\\xi)$')

for ax, (beta, xi) in zip(axOPST.flatten(), nearest_betaxis):
    plot_OP(beta, xi, ax)
[ax.set_ylabel('$k$', labelpad=-5) for ax in axOPST[:, 0]]
[ax.set_xlabel('$x_0$', labelpad=-5) for ax in axOPST[-1, :]]

cbar = add_heatmap_cbar(axOPST[0, 1], axsBottom[1], [0.65, 0.7, 0.3, 0.02], '', [0, 'max $P$'])


axs = axsBottom[2].subplots(2, 3)

ax = axs[0, :]
interpolate_countours(ax[0], 'norm_avg_utility', 0, 1, title='Utility')
interpolate_countours(ax[1], 'mean_fr', 0, 0.6, title='Firing rate')
interpolate_countours(ax[2], 'entropy', 4.5, 6.5, title='Entropy')
[a.set(xticks=[0, 6]) for a in ax]
[a.set(yticks=[0, 6]) for a in ax]
# [a.tick_params(left=False,bottom=False,right=False,top=False) for a in ax]

[a.set_xlabel('$\\beta$', labelpad=-5) for a in ax]
ax[0].set(yticks=[0, 6])
ax[0].set_ylabel('$\\xi$', labelpad=-5)

# bottom panels
ax = axs[1, :]

for i, x in enumerate(['entropy', 'mean_fr']):
    for group, cmap in [('beta', 'Greys'), ('xi', 'Reds')]:
        plot_utility(ax[i], x, group, cmap)
    ax[i].set(xlabel='Entropy' if i == 0 else 'Firing rate',
              ylabel='Utility' if i == 0 else None, yticks=[0, 1],
              xticks=[5, 6.5] if i == 0 else [0, 0.5])

ax[2].remove()
despine_ax(ax[0])
despine_ax(ax[1])
plt.setp(axs, box_aspect=1)
plt.setp(axs[0, 1:], yticklabels=[])
plt.setp(axs[1, 1], yticklabels=[])

for a in axs.flatten():
    a.tick_params(axis='x', labelsize='xx-small')
for a in axs.flatten():
    a.tick_params(axis='y', labelsize='xx-small')


cbar_beta = add_scalar_cbar(axs[1, 0], axsBottom[2], [0.71, 0.13, 0.015, 0.25], '', [0, 6])
cbar_xi = add_scalar_cbar(axs[1, 0], axsBottom[2], [0.76, 0.13, 0.015, 0.25], '', [0, 6], cmap='Reds')
cbar_beta.set_ticks([])

# axsBottom[2].text(0.2, 1.1, '$\langle\xi\rangle_\beta$', ha='left', va='bottom', transform=cbar_xi.ax.transAxes)
# axsBottom[2].text(0.2, 1.1, '$\langle\beta\rangle_\xi$', ha='left', va='bottom', transform=cbar_beta.ax.transAxes)

axsBottom[2].text(0.2, 1.1, '$\\bar{\\xi}$', ha='left', va='bottom', transform=cbar_xi.ax.transAxes)
axsBottom[2].text(0.2, 1.1, '$\\bar{\\beta}$', ha='left', va='bottom', transform=cbar_beta.ax.transAxes)


# cbar_xi
axsBottom[2].suptitle('OPs distr. characteristics')
#sns.despine(**despineopts,ax=axOPST[1])
#sns.despine(**despineopts,ax=axOPST[0])
#axOPST[0].set_xlabel('Entropy', labelpad=-6)
#axOPST[1].set_xlabel('Entropy', labelpad=-6)

#cbbeta = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=betapal),ax=axOPST,shrink=.3,anchor=(5.4,0.7))
#cbeta = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=etapal),ax=axOPST,shrink=.3,anchor=(0,0.1))
#cbbeta.ax.set_ylabel('$\\beta$',labelpad=-3)
#cbeta.ax.set_ylabel('$\\xi$',labelpad=-3)
#plt.setp([cbbeta.ax, cbeta.ax], yticklabels=[0,6])


sf_letterannotation(subfigs[0], (0.01, 0.9), 'A')
sf_letterannotation(subfigs[0], (0.27, 0.9), 'B')
sf_letterannotation(subfigs[0], (0.52, 0.9), 'C')
sf_letterannotation(subfigs[0], (0.75, 0.9), 'D')
sf_letterannotation(axsBottom[0], (0.01, 0.98), 'E')
sf_letterannotation(axsBottom[1], (0, 0.98), 'F')
sf_letterannotation(axsBottom[2], (0, 0.98), 'G')

# subfigs[0].patch.set_edgecolor('black')
# subfigs[0].patch.set_linewidth(2)

# arrows in the middle of top panel
for i in range(3):
    b1, b2 = axsTop[i].get_position(), axsTop[i + 1].get_position()
    arrowlen = 0.005
    if i == 0:
        offset = -0.065
    elif i == 1:
        offset = -0.02
    elif i == 2:
        offset = 0.02

    x0, x1 = (b1.x1 + arrowlen) + offset, (b2.x0 - arrowlen) + offset
    y = b1.y1 - 0.05
    subfigs[0].add_artist(Annotation('', xy=(x1, y), xytext=(x0, y),
                              arrowprops=dict(arrowstyle='->', lw=1.5),
                              xycoords='figure fraction'))
# fig.show()
#


# Annotation(
#         '', xy=(x_end, y), xytext=(x_start, y),
#         arrowprops=dict(arrowstyle='->', lw=1.5),
#         xycoords='figure fraction'
#     )

save_figure(fig, 'figure4', 4)

