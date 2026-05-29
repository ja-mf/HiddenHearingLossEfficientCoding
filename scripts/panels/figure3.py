# %% preamble
import os
from pathlib import Path


import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from .panel_shared import (
    CHL_GROUP_COLORS,
    apply_panel_rcparams,
    despineopts,
    figure_title,
    letter_annotation,
    levels,
    make_hpr_palette,
)
from .panel_shared import _svg_caption_metadata, save_figure  # noqa: E402

from scipy.interpolate import splrep, splev
from scipy.ndimage import gaussian_filter1d


apply_panel_rcparams()


# HPR90 setting-aligned data source (same workflow family as Figure 5)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / 'figures' / 'svg'

hprs_plot = [30, 45, 60, 75, 90]
hprs = [*hprs_plot, -1]
hprs1 = hprs_plot.copy()
palette_hprs, hprs_colours = make_hpr_palette(hprs_plot)
palette = [*palette_hprs, '#000']


def _load_rlfs_chl():
    preferred = PROJECT_ROOT / 'data/artifacts' / 'rlfs_CHL.parquet'
    fallback = PROJECT_ROOT / 'data/artifacts' / 'rlfs_CHL.parquet'
    path = preferred if preferred.exists() else fallback
    df = pd.read_parquet(path)
    df = df.rename(columns={'neuron_id': 'neuron-id'})
    df = df[['group', 'neuron-id', 'hpr', 'lvl', 'spks']].copy()
    return df.set_index(['group', 'neuron-id', 'hpr', 'lvl']).sort_index()


def _load_stg_chl():
    stg_path = PROJECT_ROOT / 'data/artifacts' / 'CHL_stg.parquet'
    df = pd.read_parquet(stg_path)
    return df.set_index(['group', 'neuron-id', 'hpr']).query('hpr != -1')


rlfs_chl = _load_rlfs_chl()
_stg_ids = set(_load_stg_chl().reset_index()['neuron-id'].astype(str).unique())
rlfs_chl = rlfs_chl.reset_index()
rlfs_chl = rlfs_chl[rlfs_chl['neuron-id'].astype(str).isin(_stg_ids)]
rlfs_chl = rlfs_chl.set_index(['group', 'neuron-id', 'hpr', 'lvl']).sort_index()


def _safe_norm(r):
    den = max(r['spks']) - min(r['spks'])
    if den <= 0:
        return pd.Series(np.zeros(len(r), dtype=float), index=r.index)
    return (r['spks'] - min(r['spks'])) / den


rlfs_norm_chl = rlfs_chl.groupby(['neuron-id', 'hpr']).apply(
    _safe_norm
).droplevel([0, 1])
rlfs_avg_chl = rlfs_chl.groupby(['group', 'hpr', 'lvl']).mean()


def get_groupavg_smooth_fi(rlfs_avg_df):
    dl_us = 0.1
    lvl_us = np.arange(levels.min(), levels.max() + dl_us, dl_us)
    rows = []
    for (group, hpr), d in rlfs_avg_df.reset_index().groupby(['group', 'hpr']):
        lv = d['lvl'].to_numpy(dtype=float)
        sp = d['spks'].to_numpy(dtype=float)
        order = np.argsort(lv)
        lv = lv[order]
        sp = sp[order]

        tck = splrep(lv, sp, s=0)
        srlf = gaussian_filter1d(splev(lvl_us, tck, der=0), 6 / dl_us)
        fi = np.gradient(srlf, dl_us) ** 2 / np.maximum(srlf, 1e-12)

        rows.append(pd.DataFrame({
            'group': group,
            'hpr': int(hpr),
            'lvl': lvl_us,
            'srlf': srlf,
            'fi': fi,
        }))

    out = pd.concat(rows, ignore_index=True)
    return out.set_index(['group', 'hpr', 'lvl']).sort_index()


srlfs_fi_chl = get_groupavg_smooth_fi(rlfs_avg_chl)
df_sig_thresh_gain_chl = _load_stg_chl()

# %% Panel 2 CHL: avg rlfs fi thgd

m=1
fig = plt.figure(layout='none',figsize=(9.8*m,4.6*m))
w_ratio, h_ratio = [5,0.5,5,1.5, 5,1,5,1,5,1,5,1,5,1], [1,1,1,1.5]
gs = gridspec.GridSpec(4,14, width_ratios = w_ratio, height_ratios=h_ratio,figure=fig,
                       wspace=0,hspace=0.1,
                       left=0.06, right=0.98, top=0.94)
no_ticks = dict(xticklabels=[],yticklabels=[],frame_on=False,xlabel=None,ylabel=None,xticks=hprs[:-1])

rlfs_opts = dict(xlim=[24,96], xticks=hprs[:-1], frame_on=False)
matplotlib.rcParams.update({"axes.grid" : True, "grid.alpha": 0.5,'axes.prop_cycle': matplotlib.cycler(color=["gray", "#e94cdc", "0.7"]) })
axGRP= []
axGRP.append(fig.add_subplot(gs[0,0], ylim=[0,11], yticks=[0,5.5,11], xticklabels=[], **rlfs_opts, xlabel=None))
axGRP.append(fig.add_subplot(gs[0,2], ylim=[0,11], yticks=[0,5.5,11], xticklabels=[], yticklabels=[], ylabel=None, xlabel=None, **rlfs_opts))
axGRP.append(fig.add_subplot(gs[1,0], **rlfs_opts))
axGRP.append(fig.add_subplot(gs[1,2], yticklabels=[], **rlfs_opts))
[a.tick_params(left=False,bottom=False) for a in axGRP]
#axGRP[1].margins(x=0.5)

# group panels
a = sns.lineplot(data=rlfs_avg_chl.query('group == "EP"'), x='lvl', y='spks', hue='hpr', palette=palette, ax=axGRP[0], hue_order=hprs, legend=False, errorbar=None)
a.set(ylabel='spk/s',xlabel=None, title='Ear-plugged')
axGRP[0].set_yticklabels(['', '5.5', '11'])

a=sns.lineplot(data=rlfs_avg_chl.query('group == "PP"'), x='lvl', y='spks', hue='hpr', palette=palette, ax=axGRP[1], hue_order=hprs, legend=False, errorbar=None)
a.set(ylabel=None, xlabel=None, title='Post-plug')

data=srlfs_fi_chl.reset_index()
data.hpr = data.hpr.astype(int)
data.set_index(['group','hpr','lvl'], inplace=True)

a = sns.lineplot(data=data.loc['EP'].reset_index(), x='lvl', y='fi', hue='hpr', palette=palette, ax=axGRP[2], hue_order=hprs, legend=False, errorbar=None)
a.set(ylabel='1/dB$^2$',xlabel='dB SPL', ylim=[0,0.012], yticks=[0.006,0.012],yticklabels=['0.006','0.012'])

#.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

a = sns.lineplot(data=data.loc['PP'].reset_index(), x='lvl', y='fi', hue='hpr', palette=palette, ax=axGRP[3], hue_order=hprs, legend=False, errorbar=None)
a.set(ylabel=None, xlabel='dB SPL',ylim=[0,0.012], yticks=[0.006,0.012])

# axGRP[-1].legend([matplotlib.lines.Line2D([0], [0], color=c, lw=plt.rcParams['lines.linewidth']) for c in palette],
#        hprs[:-1]+['Unif'],bbox_to_anchor=(1,-0.7),frameon=False, title='HPR')


# compact animal-level summary for threshold and gain across HPRs

cgs = gs[2:4,0:3].subgridspec(3, 2, width_ratios=[2, 1], height_ratios=[0.6, 1, 1], hspace=0.15, wspace=0.05)
axC_thr = fig.add_subplot(cgs[1,0])
axC_gain = fig.add_subplot(cgs[2,0])
axC_leg = fig.add_subplot(cgs[1:3, 1])
axC_leg.set(xlim=(0,1), ylim=(0,1))
axC_leg.axis('off')

df_anim_chl = df_sig_thresh_gain_chl.reset_index().copy()
df_anim_chl['animal'] = df_anim_chl['neuron-id'].astype(str).str.extract(r'^([A-Z]{2}\d+[ab]?)u')
df_anim_chl = (
    df_anim_chl.groupby(['group', 'animal', 'hpr'], as_index=False)[['threshold', 'gain']]
    .mean()
)
df_anim_chl['hpr_str'] = df_anim_chl['hpr'].astype(int).astype(str)
_hpr_order_str = [str(h) for h in hprs_plot]
_group_palette = CHL_GROUP_COLORS.copy()

def _compact_metric_panel(ax, metric, ylabel, ylim, yticks):
    sns.boxplot(
        data=df_anim_chl,
        x='hpr_str', y=metric, hue='group',
        order=_hpr_order_str, hue_order=['EP', 'PP'],
        palette=_group_palette,
        width=0.72, dodge=True, fliersize=0, linewidth=0.85,
        boxprops={'alpha': 0.22},
        medianprops={'linewidth': 1.05},
        whiskerprops={'linewidth': 0.85},
        capprops={'linewidth': 0.85},
        ax=ax,
    )
    sns.stripplot(
        data=df_anim_chl,
        x='hpr_str', y=metric, hue='group',
        order=_hpr_order_str, hue_order=['EP', 'PP'],
        palette=_group_palette,
        dodge=True, jitter=0.16, size=2.8, alpha=1.0,
        edgecolor='none', ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set(ylabel=ylabel, ylim=ylim, yticks=yticks)
    ax.grid(True, axis='y', alpha=0.35)
    sns.despine(ax=ax, **despineopts)
    ax.tick_params(axis='x', bottom=False)
    ax.xaxis.set_ticks(range(len(_hpr_order_str)))
    ax.xaxis.set_ticklabels(_hpr_order_str)

_compact_metric_panel(axC_thr, 'threshold', 'Threshold', [60, 85], [60, 70, 80])
axC_thr.text(0.5, 1.08, 'Animal summary', transform=axC_thr.transAxes, ha='center', va='bottom', fontsize=9)
axC_thr.set_xlabel(None)
axC_thr.set_xticklabels([])
axC_thr.tick_params(labelbottom=False, bottom=False)
axC_thr.spines['bottom'].set_visible(False)

_compact_metric_panel(axC_gain, 'gain', 'Gain', [0, 0.3], [0, 0.1, 0.2, 0.3])
axC_gain.set_xlabel('HPR')
axC_gain.tick_params(axis='x', bottom=True)

for _ax, _shrink in [(axC_thr, 0.92), (axC_gain, 0.92)]:
    _p = _ax.get_position()
    _new_h = _p.height * _shrink
    _new_y = _p.y0 + (_p.height - _new_h) * 0.5
    _ax.set_position([_p.x0, _new_y, _p.width, _new_h])

_hpr_handles = [matplotlib.lines.Line2D([0], [0], color=c, lw=plt.rcParams['lines.linewidth']) for c in palette]
leg1 = axC_leg.legend(_hpr_handles, hprs[:-1]+['Unif'], loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False, title='HPR')
axC_leg.add_artist(leg1)

_group_handles = [
    matplotlib.lines.Line2D([0], [0], color=_group_palette['EP'], lw=plt.rcParams['lines.linewidth']),
    matplotlib.lines.Line2D([0], [0], color=_group_palette['PP'], lw=plt.rcParams['lines.linewidth']),
]
axC_leg.legend(_group_handles, ['EP', 'PP'], loc='lower center', bbox_to_anchor=(0.45, -0.225), frameon=False)


# axProp = fig.add_subplot(gs[4,0:2])
# bins = [-np.inf,-0.02,0.02,np.inf]
# labels = ['Negative gain','Low modulation','Positive gain']
# bar_opts = dict(hue='group', x='rif-type', stat='percent',common_norm=False,
#                 element='bars',multiple='dodge',shrink=0.7, )#palette=sns.color_palette("Set2"))
#
# df_sig_thresh_gain_chl = pd.read_pickle('df_pkls/CHL-allunits-newfit-stg.pkl').query('hpr != -1')
# df_sig_thresh_gain_chl['rif-type'] = pd.cut(df_sig_thresh_gain_chl['gain'],bins = bins , labels = labels)
#
# sns.histplot(data=df_sig_thresh_gain_chl, ax=axProp, **bar_opts,legend=False,hue_order=['EP','PP'],palette=group_colours_chl)
# #sns.move_legend(axProp, loc="upper left", bbox_to_anchor=(0,1.2), frameon=False, title=None,fontsize='x-small',reverse=True)
#
# axProp.set(xlabel=None,ylim=[0,90],title='RIF-type proportion')#,yticks=[0,20,40,60]);
# sns.despine(ax=axProp, **despineopts)
# plt.sca(axProp); plt.xticks(rotation=45, fontsize='x-small',ha='right', rotation_mode='anchor');  plt.grid(axis='x')




# hpr panels
for i, hpr in enumerate(hprs[:-1]):
    i=i+2
    # avg rlf
    ax, data = fig.add_subplot(gs[0,i*2]), rlfs_avg_chl.query(f'hpr == {hpr}')
    if hpr == 30:
        letter_annotation(ax,-.1,1.2,'B')
    ax.set_title(f'HPR c. {hpr}')
    opts = dict(x='lvl', y='spks', ax=ax)
    sns.lineplot(data=data.loc['EP'].reset_index(), color=CHL_GROUP_COLORS['EP'], errorbar=None, **opts)
    sns.lineplot(data=data.loc['PP'].reset_index(), color=CHL_GROUP_COLORS['PP'], errorbar=None, **opts)

    ax.set(ylim=[0, 11], yticks=[0, 5.5, 11],  **no_ticks)
    ax.tick_params(left=False,bottom=False)
    ax.grid(True,alpha=0.5)
    if hpr == 30:
        #ax.set(yticklabels=[None,11,22])
        #ax.set_ylabel('spk/s')
        pass

    ax2 = fig.add_subplot(gs[1,i*2])
    data = srlfs_fi_chl.query(f'hpr == {hpr}')
    opts = dict(x='lvl', y='fi', ax=ax2)
    sns.lineplot(data=data.loc['EP'].reset_index(), color=CHL_GROUP_COLORS['EP'], errorbar=None, **opts)
    sns.lineplot(data=data.loc['PP'].reset_index(), color=CHL_GROUP_COLORS['PP'], errorbar=None, **opts)

    # fisher info
    ax2.set(ylim=[0,0.012],yticks=[0.006,0.012], **no_ticks)
    ax2.tick_params(left=False,bottom=False)
    ax2.grid(True,alpha=0.5)
    if hpr == 30:
        #ax2.set(yticklabels=[0.02],ylabel='1/d$B^2$')
        pass
    ax.sharex(ax2)
    ax_joint = fig.add_subplot(gs[3,i*2])
    ax_joint.set_box_aspect(1.0)
    ax_marg_x = fig.add_subplot(gs[2,i*2],sharex=ax2)
    ax_marg_y = fig.add_subplot(gs[3,i*2+1])
    ax_marg_x.set(xticklabels=[],xlabel=None)
    ax_marg_y.set(yticklabels=[],ylabel=None)

    # density threshold
    data = df_sig_thresh_gain_chl.query(f'hpr == {hpr}').reset_index()
    sns.kdeplot(data=data.query('group == "EP"'),ax=ax_marg_x,x='threshold',color=CHL_GROUP_COLORS['EP'],cut=0)
    sns.kdeplot(data=data.query('group == "PP"'),ax=ax_marg_x,x='threshold',color=CHL_GROUP_COLORS['PP'],cut=0)
    ax_marg_x.set(ylim=[0,0.028], yticks=[0, 0.014, 0.028], **no_ticks)
    ax_marg_x.grid(True,alpha=0.5)
    if hpr==30:
        ax_marg_x.set(yticklabels=['', '0.014', ''])
        pass

    if hpr == hprs_plot[-1]:
        ax_marg_x.set(ylabel='Density')
        ax_marg_x.yaxis.set_label_position("right")
    #    ax_marg_x.set(yticklabels=[0,0.02])
    #    ax_marg_x.yaxis.tick_right()

    ax_marg_x.tick_params(left=False,bottom=False,right=False)

    # density gain
    sns.kdeplot(data=data.query('group == "EP"'),ax=ax_marg_y,y='gain',color=CHL_GROUP_COLORS['EP'],cut=True)
    sns.kdeplot(data=data.query('group == "PP"'),ax=ax_marg_y,y='gain',color=CHL_GROUP_COLORS['PP'],cut=True)
    #ax_marg_y.set(**no_ticks)
    ax_marg_y.set(xticks=[],yticks=[0],xlabel=None,ylabel=None,yticklabels=[],xticklabels=[],ylim=[-0.03,0.5],frame_on=False)
    ax_marg_y.tick_params(left=False,bottom=False,right=False,top=False)
    ax_marg_y.grid(True,alpha=0.5)

    # joint plot
    data = df_sig_thresh_gain_chl.query(f'hpr == {hpr}').reset_index()
    sns.scatterplot(data=data.query('group == "EP"'),ax=ax_joint,x='threshold',y='gain', color=CHL_GROUP_COLORS['EP'],s=6)
    sns.scatterplot(data=data.query('group == "PP"'),ax=ax_joint,x='threshold',y='gain', color=CHL_GROUP_COLORS['PP'],s=6)
    thgplotopts = dict(ylabel=None,yticks=[0,0.5],ylim=[-0.03,0.5],frame_on=True,yticklabels=[],
                 xticks=hprs[:-1],xticklabels=hprs[:-1],xlabel='Threshold\ndB SPL')
    ax_joint.set(**thgplotopts)
    ax_joint.set_xlabel('dB SPL')
    ax_joint.tick_params(left=False,bottom=False,right=False,top=False)
    ax_joint.grid(True,alpha=0.5)
    if hpr==30:
        ax_joint.set(yticklabels=[0,0.5])
        ax_joint.set_ylabel('Gain',labelpad=-1)


letter_annotation(axGRP[0],-.4,1.15,'A')
letter_annotation(axC_thr,-.3,1.1,'C')

save_figure(fig, 'figure3', 3)


if __name__ == '__main__':
    pass
