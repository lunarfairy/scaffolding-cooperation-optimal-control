"""Fig. 6: where decoupling pays (gain against kappa, robustness conditions, gain over the encouragement planner).
Run from this directory python make_fig6.py
"""
import os, sys, json, gzip
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fig_style import apply_figure_style, panel_letter
apply_figure_style(sizes=(8, 7, 6))
COL = dict(window='#1f4e9c', twophase='#e08214', enc='#7f7f7f', excl='#8c510a', conc='#5e3c99',
           oracle='#35978f', retain='#1f4e9c', recruit='#d6604d')
R_ = "../figure_data/"
SUM = pd.read_csv(R_ + "fresh_window_vs_twophase.csv")
def fig6():
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.3),gridspec_kw=dict(wspace=0.55,width_ratios=[1,1.35,1]))
    ax=axs[0]; m=SUM[SUM.tag=='main']
    for T,mk,colr in [(15,'o',COL['window']),(8,'^','#6baed6'),(25,'v','#08306b')]:
        d=m[m['T']==T].sort_values('kappa'); ax.errorbar(d.kappa,d.gain,yerr=2*d.gain_se,marker=mk,color=colr,ms=3.5,lw=1 if T==15 else 0,label=f'$T={T}$',capsize=1.5)
    e=SUM[SUM.model=='refit'].sort_values('kappa'); ax.errorbar(e.kappa,e.gain,yerr=2*e.gain_se,fmt='D',color=COL['recruit'],ms=3.5,label='re-estimated model')
    ax.axhline(0,color='0.6',lw=0.6,zorder=0); ax.set_xlabel('Defector responsiveness $\\kappa$'); ax.set_ylabel('Window minus best two-phase\n(final cooperation)')
    ax.set_title('Decoupling pays at intermediate $\\kappa$',loc='left'); ax.legend(frameon=False,fontsize=5.5,loc='upper left')
    ax=axs[1]; r=SUM[SUM.tag.isin(['main','robust'])&(SUM['T']==15)&SUM.kappa.isin([0.5,0.75,1.0])]
    conds=['base','p0=0.15','p0=0.5','targeted','n=24','n=24,deg','nocap','sd=0']; lab=['baseline','$p_0=0.15$','$p_0=0.5$','targeted\nrouting','$n=24$','$n=24$,\nsame degree','no capital\nrule','$\\sigma_\\theta=0$']
    for k,mk,colr,off in [(0.5,'s',COL['excl'],-0.2),(0.75,'^',COL['twophase'],0),(1.0,'o',COL['window'],0.2)]:
        ys=[r[(r.cond==c_)&(r.kappa==k)].gain.values[0] for c_ in conds]; ses=[r[(r.cond==c_)&(r.kappa==k)].gain_se.values[0] for c_ in conds]
        ax.errorbar(np.arange(len(conds))+off,ys,yerr=2*np.array(ses),fmt=mk,color=colr,ms=3.5,capsize=1.2,label=f'$\\kappa={k:g}$')
    ax.axhline(0,color='0.6',lw=0.6,zorder=0); ax.set_xticks(range(len(conds))); ax.set_xticklabels(lab,fontsize=5); ax.set_ylabel('Window minus best two-phase')
    ax.set_title('Gain survives most perturbations',loc='left'); ax.legend(frameon=False,fontsize=5.5,loc='upper right')
    ax=axs[2]; d=SUM[SUM.enc.notna()&(SUM['T']==15)]
    mm=d[d.tag=='main'].sort_values('kappa')
    ax.errorbar(mm.kappa,mm.gain_enc,yerr=2*mm.gain_enc_se,marker='o',color=COL['window'],ms=3.5,lw=1,label='published model')
    ee=d[d.model=='refit'].sort_values('kappa'); ax.errorbar(ee.kappa,ee.gain_enc,yerr=2*ee.gain_enc_se,fmt='D',color=COL['recruit'],ms=3.5,label='re-estimated model')
    ax.axhline(0,color='0.6',lw=0.6,zorder=0); ax.set_xlabel('Defector responsiveness $\\kappa$'); ax.set_ylabel('Window minus encouragement')
    ax.set_title('Distilled planner recruits too late',loc='left'); ax.legend(frameon=False,fontsize=5.5,loc='upper right'); ax.set_ylim(0,0.045)
    for i,a in enumerate(axs): panel_letter(a,'abc'[i])
    return fig

plt.close('all'); f6 = fig6(); ax = f6.axes[1]
ax.set_xticklabels(['baseline','$p_0=0.15$','$p_0=0.5$','targeted routing','$n=24$','$n=24$, same degree','no capital rule','$\\sigma_\\theta=0$'], rotation=40, ha='right', fontsize=5.5)
f6.axes[0].set_title('Gain is largest at $0.75\\leq\\kappa\\leq1$', loc='left')
f6.savefig("fig6_scope.png", dpi=300, bbox_inches='tight'); f6.savefig("fig6_scope.pdf", bbox_inches='tight')
