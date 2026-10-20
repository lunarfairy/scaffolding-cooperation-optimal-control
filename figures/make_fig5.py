"""Figure fig5: regenerates fig5 of the manuscript from the inputs in ../figure_data.
Run from this directory: python make_fig5.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use('Agg')
from fig_style import apply_figure_style, panel_letter
apply_figure_style(sizes=(8, 7, 6))
import json, gzip, os, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations

H = pd.read_csv('../figure_data/human_tie_state_gaps.csv')
CV = pd.read_csv('../figure_data/human_tie_state_by_round.csv')

COL = dict(window='#1f4e9c', twophase='#e08214', enc='#7f7f7f', excl='#8c510a', conc='#5e3c99',
           oracle='#35978f', retain='#1f4e9c', recruit='#d6604d', planner='#1f4e9c', static='#7f7f7f', other='#80b1d3')

# apply_figure_style equivalent
mpl.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'xtick.labelsize': 5.5,
    'ytick.labelsize': 5.5,
    'legend.fontsize': 5.5,
})

def panel_letter(ax, letter):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontweight='bold', fontsize=8, va='top')

def fig5():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1, 1.25]))
    for ax, var, yl, ttl in [(axs[0], 'streak', 'Current defection streak (rounds)', 'Tied defectors lapsed recently'),
                              (axs[1], 'lo_past', 'Defector\'s cooperation rate\nbefore round $t$', 'Tie state records past behaviour')]:
        for grp, ls in [('Planner', '-'), ('Static', '--')]:
            for tied, cname, mk in [(1, 'retain', 'o'), (0, 'recruit', 's')]:
                d = CV[(CV.grp == grp) & (CV.tied == tied)]
                ax.plot(d.r + 1, d[var], ls=ls, marker=mk, ms=2.5, lw=1.0 if grp == 'Planner' else 0.8,
                        color=COL[cname], alpha=1 if grp == 'Planner' else 0.55)
        ax.set_xlabel('Round $t$')
        ax.set_ylabel(yl)
        ax.set_title(ttl, loc='left')
        ax.set_xticks([2, 5, 8, 11, 14])
    hs = [Line2D([], [], color=COL['retain'], marker='o', ms=3, label='tied C–D pair'),
          Line2D([], [], color=COL['recruit'], marker='s', ms=3, label='untied C–D pair'),
          Line2D([], [], color='k', ls='-', label='planner conditions'),
          Line2D([], [], color='k', ls='--', alpha=0.55, label='static networks')]
    axs[0].legend(handles=hs, frameon=False, loc='upper left', fontsize=5.5)
    ax = axs[2]
    sets = ['Planner (GraphNet+Enc.)', 'Other dynamic', 'Static']
    labs = ['GraphNet +\nencouragement', 'other\nrewiring', 'static']
    vars_ = [('next', 'next-round\ncooperation', COL['window']), ('lo_past', 'past\ncooperation rate', COL['oracle'])]
    hp = H[H.weight == 'pair']
    for j, (v, lab, colr) in enumerate(vars_):
        for i, sname in enumerate(sets):
            r = hp[(hp.set == sname) & (hp['var'] == v)].iloc[0]
            x = i + (j - 0.5) * 0.28
            ax.errorbar(x, r['diff'], yerr=[[r['diff'] - r['lo']], [r['hi'] - r['diff']]],
                        fmt='o' if j == 0 else 'D', color=colr, ms=3.5, capsize=2, lw=1,
                        label=lab.replace('\n', ' ') if i == 0 else None)
    ax.axhline(0, color='0.6', lw=0.6, zorder=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs)
    ax.set_ylabel('Tied minus untied defectors')
    ax.set_title('Sorting appears only when ties are managed', loc='left')
    ax.legend(frameon=False, loc='upper right', fontsize=5.5)
    ax.margins(x=0.1)
    for i, a in enumerate(axs):
        panel_letter(a, 'abc'[i])
    return fig

plt.close('all')
f5 = fig5()
a0 = f5.axes[0]
h = a0.get_legend().legend_handles
a0.get_legend().remove()
a0.legend(handles=h, frameon=False, loc='lower right', fontsize=5.5)
f5.savefig("fig5_human_data.png", dpi=300, bbox_inches='tight')
f5.savefig("fig5_human_data.pdf", bbox_inches='tight')