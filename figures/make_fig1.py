"""Figure fig1: regenerates fig1 of the manuscript from the inputs in ../figure_data.
Run from this directory: python make_fig1.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use('Agg')
from fig_style import apply_figure_style, panel_letter
import json, os, glob, gzip, shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Figure-specific typography
def apply_figure_style(sizes=(8, 7, 6), style='default'):
    mpl.rcParams.update({
        'font.size': sizes[2],
        'axes.titlesize': sizes[1],
        'axes.labelsize': sizes[1],
        'xtick.labelsize': sizes[2],
        'ytick.labelsize': sizes[2],
        'legend.fontsize': sizes[2],
        'figure.titlesize': sizes[0],
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def panel_letter(ax, letter, x=-0.13, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='right')

apply_figure_style(sizes=(8, 7, 6))

Aa = json.load(open('../figure_data/abm_trajectories.json'))
acc = json.load(open('../figure_data/ipa_acc.json'))

# Load ABM results for capital constraint ablation
RA = json.load(gzip.open("../figure_data/rev_abm_results.json.gz"))

traj = pd.DataFrame([r for r in Aa if r['kind'] == 'traj'])
iacc = pd.DataFrame(acc)

cap = pd.DataFrame(RA['capacc'])

E_rows = []
for r in acc:
    E_rows.append(dict(depth=r['depth'], kappa=r['kappa'], planner=r['planner'], ipa=r['coop_node_final']))
ia = pd.DataFrame(E_rows)

ab = cap.pivot_table(index=['planner', 'kappa'], columns='capital_rule', values='final').reset_index().rename(columns={True: 'abm_cap', False: 'abm_nocap'})
E = ia.merge(ab, on=['planner', 'kappa'])
E['err_cap'] = E.ipa - E.abm_cap
E['err_nocap'] = E.ipa - E.abm_nocap

COL = dict(window='#1f4e9c', twophase='#e08214', enc='#7f7f7f', excl='#8c510a', conc='#5e3c99',
           oracle='#35978f', retain='#1f4e9c', recruit='#d6604d', planner='#1f4e9c', static='#7f7f7f', other='#80b1d3')


def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(wspace=0.4, width_ratios=[1.35, 1, 1]))
    ax = axs[0]
    order = ['exclusion', 'conciliation', 'switch_T-2', 'encouragement', 'static', 'random', 'neutral', 'maxconn']
    names = ['exclusion', 'conciliation', 'two-phase $s=T-2$', 'encouragement', 'static', 'random (30%)', 'neutral', 'max. connectivity']
    for j, (d, cap_, mk, colr, lab) in enumerate([(1, True, '^', '#9ecae1', '$m=1$, capital rule on'),
                                                    (3, True, 'o', COL['window'], '$m=3$, capital rule on'),
                                                    (3, False, 'D', COL['recruit'], '$m=3$, capital rule off')]):
        e = E[E.depth == d]
        for i, p in enumerate(order):
            x = e[e.planner == p]
            col = 'err_cap' if cap_ else 'err_nocap'
            ax.scatter(np.full(len(x), i) + (j - 1) * 0.22, x[col], marker=mk, s=9, color=colr,
                       label=lab if i == 0 else None, zorder=3)
    ax.axhline(0, color='0.6', lw=0.6)
    ax.axvline(3.5, color='0.8', lw=0.6, ls=':')
    ax.set_xticks(range(8))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=5.5)
    ax.set_ylabel('Reduced model minus ABM\n(final cooperation)')
    ax.set_title('Residual error is the capital rule', loc='left')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')
    ax.text(1.5, -0.016, 'choice-conditioned', ha='center', fontsize=5.5, color='0.4')
    ax.text(5.5, -0.016, 'choice-agnostic', ha='center', fontsize=5.5, color='0.4')
    ax.set_ylim(-0.02, 0.072)
    for ax2, k in zip(axs[1:], [1.0, 0.5]):
        for p, colr in [('exclusion', COL['excl']), ('conciliation', COL['conc']), ('encouragement', COL['enc'])]:
            a_ = traj[(traj.planner == p) & (traj.kappa == k)].iloc[0]
            ax2.plot(range(1, 16), a_.coop, color=colr, lw=1.2, label=p)
            q = iacc[(iacc.planner == p) & (iacc.kappa == k) & (iacc.depth == 3)].iloc[0]
            ax2.plot(range(1, 16), q['coop'], color=colr, lw=1, ls='--')
        ax2.set_xlabel('Round')
        ax2.set_title('$\\kappa=%g$' % k, loc='left')
        ax2.set_xticks([1, 5, 10, 15])
    axs[1].set_ylabel('Fraction cooperating')
    h, l = axs[1].get_legend_handles_labels()
    h += [Line2D([], [], color='k', lw=1.2), Line2D([], [], color='k', ls='--', lw=1)]
    l += ['agent-based model', 'reduced model ($m=3$)']
    axs[1].legend(h, l, frameon=False, fontsize=5.3, loc='lower right')
    for i, a in enumerate(axs):
        panel_letter(a, 'abc'[i])
    return fig


plt.close('all')
f1 = fig1()
f1.savefig('fig1_reduced_model.png', dpi=300, bbox_inches='tight')
f1.savefig('fig1_reduced_model.pdf', bbox_inches='tight')
plt.close(f1)