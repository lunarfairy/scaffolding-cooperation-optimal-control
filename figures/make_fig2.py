"""Figure fig2: regenerates fig2 of the manuscript from the inputs in ../figure_data.
Run from this directory: python make_fig2.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use('Agg')
from fig_style import apply_figure_style, panel_letter
apply_figure_style(sizes=(8, 7, 6))
import os
import json
import gzip
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

COL = dict(window='#1f4e9c', twophase='#e08214', enc='#7f7f7f', excl='#8c510a', conc='#5e3c99',
           oracle='#35978f', recruit='#d6604d', retain='#1f4e9c')
R_ = "../figure_data/"
# ABM screening sweep of the two-phase family (50,000 games per schedule)
dfs = pd.read_csv(R_ + "abm_two_phase_sweep.csv")
# reduced-model exclusion-window length L = T-1-s* on the kappa grid
GP = pd.read_csv(R_ + "ipa_exclusion_window_length_L.csv", index_col='kappa')
GP.columns = [int(c) for c in GP.columns]
# reduced-model tie values (costates) along fixed schedules
cs = json.load(open("../figure_data/ipa_costate.json"))
lam = pd.DataFrame(cs['lam'])

# ── Figure ────────────────────────────────────────────────────────────────────

def arr(v):
    return np.array([np.nan if x is None else x for x in v])

def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35),
                            gridspec_kw=dict(wspace=0.42))
    ax = axs[0]
    cm = plt.get_cmap('viridis')
    ks = [0.5, 0.625, 0.75, 0.875, 1.0]
    for i, k in enumerate(ks):
        d = dfs[(dfs.kappa == k) & (dfs['T'] == 15)].sort_values('s')
        colr = cm(i / (len(ks) - 1) * 0.9)
        ax.plot(d.s, d.final, color=colr, lw=1.1, label=f'$\\kappa={k:g}$')
        ax.fill_between(d.s, d.final - 2 * d.final_se, d.final + 2 * d.final_se,
                        color=colr, alpha=0.25, lw=0)
        b = d.loc[d.final.idxmax()]
        ax.plot(b.s, b.final, 'o', color=colr, ms=3.5)
    ax.set_xlabel('Last conciliating decision $s$')
    ax.set_ylabel('Final cooperation')
    ax.set_title('Two-phase schedules ($T=15$)', loc='left')
    ax.legend(frameon=False, fontsize=5.2, loc='lower left', ncol=2)

    ax = axs[1]
    for T, ls, colr in [(8, ':', '#9ecae1'), (15, '-', COL['window']),
                        (25, '--', '#3182bd'), (40, '-.', '#08306b')]:
        g_ = GP[T]
        ax.plot(g_.index, g_.values, ls=ls, color=colr, lw=1.1,
                label=f'reduced model, $T={T}$')
    for T, mk in [(8, '^'), (15, 'o'), (25, 'v')]:
        d = dfs[dfs['T'] == T]
        b = d.loc[d.groupby('kappa').final.idxmax()]
        ax.plot(b.kappa, T - 1 - b.s, mk, mfc='none', color='k', ms=3.5,
                label=f'ABM, $T={T}$')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_yticks([0, 1, 3, 5, 10, 20, 40])
    ax.set_yticklabels(['0', '1', '3', '5', '10', '20', '40'])
    ax.set_xlabel('Defector responsiveness $\\kappa$')
    ax.set_ylabel('Exclusion decisions $L=T-1-s^*$')
    ax.set_title('Window length is\nhorizon-invariant', loc='left')
    ax.legend(frameon=False, fontsize=5.5, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, -0.30), columnspacing=0.8)

    ax = axs[2]
    for T, ls in [(15, '-'), (25, '--'), (40, ':')]:
        r = lam[(lam.policy == 'conciliate') & (lam.kappa == 0.75) &
                (lam['T'] == T)].iloc[0]
        tau = T - np.arange(1, T)
        ax.plot(tau, 16 * arr(r.lam1), ls=ls, color=COL['retain'], lw=1.1)
        ax.plot(tau, 16 * arr(r.lam0), ls=ls, color=COL['recruit'], lw=1.1)
    ax.set_xlim(0.5, 14.5)
    ax.invert_xaxis()
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xticks([14, 10, 6, 2])
    ax.set_xlabel('Decisions remaining $T-t$')
    ax.set_ylabel('Marginal value of a C–D tie\n(final cooperators per tie)')
    ax.set_title('$\\kappa=0.75$, conciliate throughout', loc='left')
    ax.text(13.5, -0.006, 'curves for $T=15$, 25\nand 40 coincide',
            fontsize=5.2, color='0.35')
    hh = [Line2D([], [], color=COL['retain'],
                 label='tied pairs ($\\bar\\Lambda_1$)'),
          Line2D([], [], color=COL['recruit'],
                 label='untied pairs ($\\bar\\Lambda_0$)'),
          Line2D([], [], color='k', ls='-', label='$T=15$'),
          Line2D([], [], color='k', ls='--', label='$T=25$'),
          Line2D([], [], color='k', ls=':', label='$T=40$')]
    ax.legend(handles=hh, frameon=False, fontsize=7.2, loc='lower left')
    for i, a in enumerate(axs):
        panel_letter(a, 'abc'[i])
    return fig

plt.close('all')
f2 = fig2()
f2.savefig("fig2_two_phase.png", dpi=300, bbox_inches='tight')
f2.savefig("fig2_two_phase.pdf", bbox_inches='tight')
