import os
from pathlib import Path


import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from .panel_shared import (
    apply_panel_rcparams,
    despineopts,
    dos,
    figure_title,
    letter_annotation,
    levels,
    load_example_stimlevels,
    logistsig,
    make_hpr_palette,
    pmf_singledistr_gerbil,
    x0s,
)
from .panel_shared import _svg_caption_metadata, save_figure  # noqa: E402
from scipy.optimize import curve_fit


apply_panel_rcparams()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PICS_DIR = PROJECT_ROOT / 'data' / 'artifacts' / 'pics'
OUTPUT_DIR = PROJECT_ROOT / 'figures' / 'svg'
RAWDATA_PATH = PROJECT_ROOT / 'data' / 'artifacts'

disp_time = 1000
nchanges = disp_time // 40
fs = 44100
nsamples = int(fs * 0.04)

hprs_bottom = [30, 45, 60, 75, 90]
palette_hprs_bottom, hpr_color = make_hpr_palette(hprs_bottom)
palette_hprs = palette_hprs_bottom
palette = [*palette_hprs, '#000']


prob = pmf_singledistr_gerbil()
stimlevels = load_example_stimlevels(PROJECT_ROOT / 'data' / 'artifacts' / 'example_stimlevels.npy')


def _load_rlfs_hhl_setting():
    preferred = PROJECT_ROOT / 'data/artifacts' / 'rlfs_HHL.parquet'
    fallback = PROJECT_ROOT / 'data/artifacts' / 'rlfs_HHL.parquet'
    path = preferred if preferred.exists() else fallback
    df = pd.read_parquet(path)
    df = df.rename(columns={'neuron_id': 'neuron-id'})
    df = df[['group', 'neuron-id', 'hpr', 'lvl', 'spks']]
    df = df[df['group'].isin(['NE', 'SH'])]
    # Collapse duplicate spl variants so each neuron/hpr/lvl contributes once
    df = df.groupby(['group', 'neuron-id', 'hpr', 'lvl'], as_index=False)['spks'].mean()
    return df


def _safe_norm(s):
    den = s.max() - s.min()
    if den <= 0:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - s.min()) / den


def _fit_sigmoid_table(df_norm):
    rows = []
    for (group, uid, hpr), d in df_norm.groupby(['group', 'neuron-id', 'hpr']):
        if int(hpr) == -1:
            continue
        x = d['lvl'].to_numpy(dtype=float)
        y = d['spks'].to_numpy(dtype=float)
        if np.allclose(y, y[0]):
            continue
        try:
            popt, _ = curve_fit(
                lambda xx, x0, k: logistsig(xx, x0, k),
                x,
                y,
                p0=[50.0, 0.1],
                bounds=([24.0, -0.5], [96.0, 0.5]),
                maxfev=20000,
            )
            yhat = logistsig(x, popt[0], popt[1])
            mse = float(np.mean((y - yhat) ** 2))
            rows.append((group, uid, int(hpr), float(popt[0]), float(popt[1]), mse))
        except Exception:
            continue
    if not rows:
        raise RuntimeError('No sigmoid fits available for figure1 sample selection')
    return pd.DataFrame(rows, columns=['group', 'neuron-id', 'hpr', 'x0', 'k', 'mse'])


def _pick_clean_sample(fit_df):
    need = set(hprs_bottom)
    cand = []
    for (group, uid), sub in fit_df.groupby(['group', 'neuron-id']):
        if group != 'NE':
            continue
        hs = set(sub['hpr'].astype(int).tolist())
        if need.issubset(hs):
            sub = sub.sort_values('hpr')
            ks = sub['k'].to_numpy(dtype=float)
            x0s_local = sub['x0'].to_numpy(dtype=float)
            if np.any(ks <= 0):
                continue
            x0_range = float(sub['x0'].max() - sub['x0'].min())
            k_range = float(sub['k'].max() - sub['k'].min())
            delta_x0 = float(x0s_local[-1] - x0s_local[0])
            trend = float(np.corrcoef(np.arange(len(x0s_local), dtype=float), x0s_local)[0, 1])
            cand.append((
                group,
                uid,
                float(sub['mse'].mean()),
                float(sub['mse'].max()),
                x0_range,
                k_range,
                delta_x0,
                trend,
            ))
    if cand:
        s = pd.DataFrame(cand, columns=['group', 'neuron-id', 'mean_mse', 'max_mse', 'x0_range', 'k_range', 'delta_x0', 'trend'])
        # Prefer clean, clearly shifting thresholds from HPR30 to HPR90
        pool = s[(s['delta_x0'] >= 8.0) & (s['trend'] > 0.5)]
        if len(pool) == 0:
            pool = s
        s = pool.sort_values(['mean_mse', 'max_mse', 'delta_x0', 'x0_range'], ascending=[True, True, False, False])
        return s.iloc[0]['group'], s.iloc[0]['neuron-id']

    # Fallback: maximize HPR coverage then minimize MSE
    rows = []
    for (group, uid), sub in fit_df.groupby(['group', 'neuron-id']):
        rows.append((group, uid, int(sub['hpr'].nunique()), float(sub['mse'].mean())))
    s = pd.DataFrame(rows, columns=['group', 'neuron-id', 'n_hpr', 'mean_mse'])
    s = s.sort_values(['n_hpr', 'mean_mse'], ascending=[False, True])
    return s.iloc[0]['group'], s.iloc[0]['neuron-id']


def _add_image(ax, path, rect, alpha=1.0):
    from PIL import Image
    img = np.array(Image.open(path))
    ax_img = ax.add_axes(rect)
    ax_img.imshow(img, alpha=alpha)
    ax_img.axis('off')
    return ax_img


def draw_hpr_distr(ax, hpr, idx):
    m1, s1, _ = ax.stem(levels, prob[str(hpr)][:, 1])
    plt.setp([m1, s1], color=palette_hprs_bottom[idx])
    plt.setp([m1], markersize=3)
    ax.grid(True, alpha=0.5)
    ax.set(
        ylim=[0, 0.165],
        xticks=[24, 30, 45, 60, 75, 96],
        yticks=[0.01, 0.16],
        yticklabels=[],
        xticklabels=[],
    )
    if hpr == 30:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.set(ylabel='Probability')
    sns.despine(ax=ax, trim=True, offset=[dos, dos])
    ax.grid(which='major', axis='y')


def draw_rlf_w_fits(ax, data, fit_df, nofit=False):
    fitfunc = logistsig
    neuronid = data['neuron-id'].iloc[0]
    group = data['group'].iloc[0]
    hpr = int(data['hpr'].iloc[0])

    sns.scatterplot(data=data, x='lvl', y='spks', ax=ax, color='k', s=8)
    if nofit:
        sns.lineplot(data=data, x='lvl', y='spks', ax=ax, color='k', errorbar=None)

    fit_row = fit_df.query('group == @group and `neuron-id` == @neuronid and hpr == @hpr')
    if len(fit_row) and not nofit:
        x0, k = fit_row[['x0', 'k']].iloc[0].to_numpy(dtype=float)
        xvec = np.linspace(min(x0s), max(x0s))
        sns.lineplot(x=xvec, y=fitfunc(xvec, x0, k), ax=ax, color=palette[0], errorbar=None)
        ax.scatter(x0, k, marker='$\\bowtie$', s=200, color=hpr_color[hpr])

    ax.set(
        xticks=[24, 30, 45, 60, 75, 96],
        yticks=[0, 0.5, 1],
        ylabel='spk/s',
        xticklabels=[24, '', '', '', '', 96],
        yticklabels=[0, 11, 22],
    )
    ax.grid(True, alpha=0.5)
    ax.set_xlabel('dB SPL')

    if not nofit:
        y_max = ax.get_ylim()[1]
        ax.axhspan(
            ymin=-y_max / 16,
            ymax=0,
            xmin=(int(hpr) - 4) / 68 - 24 / 68,
            xmax=(int(hpr) + 4) / 68 - 24 / 68,
            color=hpr_color[hpr],
            alpha=1,
            zorder=1,
        )
        ax.set(ylim=[-0.05, 1])

    sns.despine(ax=ax, **despineopts)
    if hpr != 30:
        ax.tick_params(left=False)
        ax.spines['left'].set_visible(False)
        ax.set(ylabel=None, yticklabels=[])
    if hpr == hprs_bottom[-1]:
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.set(ylim=[-0.05, 1], yticks=[0, 0.5, 1], yticklabels=[0, 0.5, 1])
        sns.despine(ax=ax2, left=True, **despineopts)
        ax2.spines['right'].set_visible(True)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(left=False, bottom=False, right=True, labelright=True)
        ax2.set_ylabel('1/dB$^2$')
        ax.tick_params(left=False)


def figure1():
    # data for sample neuron selection/fitting in lower panel
    df_rlf = _load_rlfs_hhl_setting()
    df_norm = df_rlf.copy()
    df_norm['spks'] = df_rlf.groupby(['neuron-id', 'hpr'])['spks'].transform(_safe_norm)
    fit_df = _fit_sigmoid_table(df_norm)
    sample_group, sample_unit = _pick_clean_sample(fit_df)

    fig = plt.figure(layout='compressed')
    sf = fig.subfigures(2, 1, height_ratios=[1.3, 2])
    sfU = sf[0].subfigures(1, 3, width_ratios=[1.8, 1, 1.6])

    # Upper-left: acoustic context + delivered stimulus
    ax0 = sfU[0].subplot_mosaic([['ac', '.'], ['ac', '.']], gridspec_kw=dict(width_ratios=[2, 1]))

    draw_hpr_distr(ax0['ac'], 45, 1)
    ax0['ac'].set(
        yticklabels=[0.01, 0.16],
        ylabel='Probability',
        xlabel='dB SPL',
        ylim=[0, 0.2],
        xticks=[30, 45, 60, 75],
        xticklabels=[30, 45, 60, 75],
        box_aspect=1,
    )
    ax0['ac'].yaxis.set_major_formatter(ScalarFormatter())
    ax0['ac'].yaxis.set_minor_formatter(ScalarFormatter())
    sns.despine(ax=ax0['ac'], **despineopts)
    ax0['ac'].set_title('Acoustic context')
    ax0['ac'].grid(False)
    ax0['ac'].text(36, 0.175, 'HPR')

    ax_seq = sfU[0].add_axes([0.55, 0.55, 0.4, 0.15])
    m1, s1, _ = ax_seq.stem(stimlevels[:nchanges, 3])
    plt.setp([m1, s1], color=palette_hprs[1])
    plt.setp(m1, markersize=3)
    ax_seq.set(ylim=[10, 100], yticks=[24, 96], xticks=[], xticklabels=[])
    ax_seq.tick_params(labelsize='x-small')
    ax_seq.set_title('Delivered stimulus', fontsize='small')
    sns.despine(**despineopts, bottom=True, ax=ax_seq)

    ax_sig = sfU[0].add_axes([0.55, 0.4, 0.4, 0.15])
    sample_stim = np.array([])
    for k in stimlevels[:nchanges, 3]:
        sample_stim = np.append(sample_stim, np.random.normal(scale=k, size=nsamples))
    ax_sig.plot(np.linspace(0, disp_time, len(sample_stim)), sample_stim, linewidth=0.1, color='gray')
    ax_sig.set(ylim=[-320, 320], yticks=[], xticks=[0, 500], xticklabels=[None, '500ms'])
    ax_sig.tick_params(labelsize='x-small')
    ax_sig.spines['left'].set_visible(False)
    sns.despine(left=True, trim=True, offset=[-5, 3], bottom=False, ax=ax_sig)
    ax_sig.patch.set_alpha(0)

    # Upper-middle: sample RIF (no fit line)
    axRIF = sfU[1].subplots()
    df_rif = df_norm.query('group == @sample_group and `neuron-id` == @sample_unit and hpr == 45').copy()
    draw_rlf_w_fits(axRIF, df_rif, fit_df, nofit=True)
    axRIF.set(title='Rate-intensity\nfunction', yticklabels=[0, 11, 22], ylabel='spk/s', ylim=[0, 1.1])
    axRIF.grid(False)
    axRIF.tick_params(left=True)
    sns.despine(ax=axRIF, **despineopts)

    # Upper-right: experiments
    sfU[2].suptitle('Experiments')
    experiment = sfU[2].subfigures(1, 2)
    _add_image(experiment[1], str(PICS_DIR / 'mouse_experiment.png'), [0, 0, 1, 1], alpha=0.3)

    gs = 0.2
    gerb1 = _add_image(experiment[0], str(PICS_DIR / 'gerbil.png'), [0.5, 0.2, gs, gs])
    gerb2 = _add_image(experiment[0], str(PICS_DIR / 'gerbil.png'), [0.5, 0.4, gs, gs])
    gerb3 = _add_image(experiment[0], str(PICS_DIR / 'gerbil.png'), [0.5, 0.6, gs, gs])
    gerb4 = _add_image(experiment[1], str(PICS_DIR / 'gerbil.png'), [0.6, 0.2, gs, gs])

    labs = ['Ear-plugging', 'Noise-exposure', 'Sham-exposure']
    for lab, gerb in zip(labs, [gerb1, gerb2, gerb3]):
        gerb.text(-2, .9, lab, fontsize='small', transform=gerb.transAxes)

    gerb4.text(-2.5, .5, 'Controls (SH)\n\nNoise-exposed (NE)\n\nEar-plugged (EP)\n\nPost-plug (PP)',
               fontsize='small', transform=gerb4.transAxes)
    experiment[0].suptitle('Procedure', fontsize='medium', fontweight='heavy')
    experiment[1].suptitle('Recordings', fontsize='medium')

    xw, yw = 0.6, 0.5
    gerb1.text(xw + 0.6, yw, '1 week', transform=gerb1.transAxes, fontsize='x-small')
    gerb1.annotate('', xy=(xw + .5, yw + .5), xycoords='axes fraction', xytext=(xw + .5 + 2, yw + .5),
                   arrowprops=dict(arrowstyle='<-'))

    xw, yw = 0.6, 2.5
    gerb1.text(xw + 0.6, yw, '3 weeks', transform=gerb1.transAxes, fontsize='x-small')
    gerb1.annotate('', xy=(xw + .5, yw + .5), xycoords='axes fraction', xytext=(xw + .5 + 2, yw + .5),
                   arrowprops=dict(arrowstyle='<-'))

    # Lower half: tested distributions + sample RIFs (now includes HPR90)
    titles = [f'HPR at {hpr} dB' for hpr in hprs_bottom]
    axp = sf[1].subplot_mosaic([
        ['30p', '45p', '60p', '75p', '90p'],
        ['30f', '45f', '60f', '75f', '90f'],
    ])

    for i, hpr in enumerate(hprs_bottom):
        ax = axp[f'{hpr}p']
        draw_hpr_distr(ax, hpr, i)
        ax.set(title=titles[i], box_aspect=0.8)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False)
        if hpr != 30:
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)

    df_sample = df_norm.query('group == @sample_group and `neuron-id` == @sample_unit and hpr in @hprs_bottom')
    for hpr, data in df_sample.groupby('hpr'):
        h = int(hpr)
        ax = axp[f'{h}f']
        draw_rlf_w_fits(ax, data, fit_df, nofit=False)
        ax.set(box_aspect=0.8)

    sf[1].suptitle('Tested distributions and sample single-unit RIFs')

    letter_annotation(ax0['ac'], -.3, 1.1, 'A')
    letter_annotation(axRIF, -.5, 1.2, 'B')
    letter_annotation(gerb3, -1, 2.6, 'C')
    letD = sf[1].add_axes([.05, 0.95, 0, 0])
    letD.axis('off')
    letter_annotation(letD, 0, 0, 'D')

    save_figure(fig, 'figure1', 1)



figure1()
