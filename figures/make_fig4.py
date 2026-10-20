"""Figure fig4: regenerates fig4 of the manuscript from the inputs in ../figure_data.
Run from this directory: python make_fig4.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use('Agg')
from fig_style import apply_figure_style, panel_letter
import json
import gzip
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

apply_figure_style(sizes=(8,7,6))

COL=dict(window='#1f4e9c', twophase='#e08214', enc='#7f7f7f', excl='#8c510a', conc='#5e3c99', oracle='#35978f', retain='#1f4e9c', recruit='#d6604d', planner='#1f4e9c', static='#7f7f7f', other='#80b1d3')

RA=json.load(gzip.open("../figure_data/rev_abm_results.json.gz"))

iv=[]
for r in RA['tieiv']:
    for key,lab in (('cut','retain (tied pair)'),('add','recruit (untied pair)')):
        x=r[key]; iv.append(dict(kappa=r['kappa'],policy=r['policy'],t0=r['t0'],tie=lab,n=x['n'],theta_D=x['thD'],streak_D=x['stD'],next_coop_ctrl=x['aD1_ctrl'],
            d_next_coop=x['dDef'][0],d_next_coop_se=x['dDef'][1],d_coop_partner=x['dCoop'][0],d_final_cooperators=x['dFinal16'][0],d_final_cooperators_se=x['dFinal16'][1]))
IV=pd.DataFrame(iv)

ivb=[]
for r in RA['tieiv']:
    for key,lab in (('cut','retain'),('add','recruit')):
        for b in r[key]['by_theta']: ivb.append(dict(kappa=r['kappa'],policy=r['policy'],t0=r['t0'],tie=lab,theta_lo=b['lo'],theta_hi=b['hi'],n=b['n'],d_final_cooperators=b['dFinal16'][0],se=b['dFinal16'][1]))

tsrows=[]
for ts in RA['tiestats']:
    for rw in ts['rounds']:
        d=dict(kappa=ts['kappa'],policy=ts['policy'],t=rw['t'])
        for k,v in rw.items():
            if k!='t': d[k]=v[0]; d[k+'_se']=v[1]
        tsrows.append(d)
TS=pd.DataFrame(tsrows)

def diff(s,x,y,k='final'):
    key=f'{k}|{x}|{y}' if f'{k}|{x}|{y}' in s['diffs'] else f'{k}|{y}|{x}'
    m,se=s['diffs'][key]; return (m if key==f'{k}|{x}|{y}' else -m), se

tab=[]
for s in RA['reeval']:
    st=s['setting']
    if s['tag']!='main' or st['T']!=15 or st['kappa'] not in (0.5,0.75,1.0): continue
    pol=s['policies']
    def best(pref, post=None):
        c=[n for n in pol if n.startswith(pref) and (post is None or (n.endswith('post')==post)) and '_enc' not in n]
        b=max(c,key=lambda n:pol[n]['final']); return b,pol[b]['final']
    P=best('P'); Wb=best('W'); Lo=best('Loracle'); Lp=best('Lpost'); Do=best('D',False); Dp=best('D',True)
    pr=max([n for n in pol if n.startswith('prob')],key=lambda n:pol[n]['final'])
    row=dict(kappa=st['kappa'],coupled_time=P,decoupled_time=Wb,coupled_oracle=Lo,coupled_post=Lp,decoupled_oracle=Do,decoupled_post=Dp,probation=(pr,pol[pr]['final']),enc=pol['enc']['final'])
    row['d_info_coupled']=diff(s,Lo[0],P[0]); row['d_info_decoupled']=diff(s,Do[0],Wb[0]); row['d_post_decoupled']=diff(s,Dp[0],Wb[0])
    row['d_dec_noinfo']=diff(s,Wb[0],P[0]); row['d_dec_oracle']=diff(s,Do[0],Lo[0])
    row['ccdd_enc']=diff(s,'enc_stdCCDD','enc'); bw=[n for n in pol if n.endswith('_encCCDD')][0]; row['ccdd_win']=diff(s,bw.replace('_encCCDD',''),bw)
    tab.append(row)

def fig4():
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.6),gridspec_kw=dict(wspace=0.45,hspace=0.75))
    for ax,(k,pol,ttl) in zip(axs[0,:2],[(1.0,'W4,15','$\\kappa=1$: retained ties keep value'),(0.75,'W0,14','$\\kappa=0.75$: late recruitment hurts')]):
        d=IV[(IV.kappa==k)&(IV.policy==pol)]
        for lab,cname,mk in [('retain (tied pair)','retain','o'),('recruit (untied pair)','recruit','s')]:
            x=d[d.tie==lab]; ax.errorbar(x.t0,x.d_final_cooperators,yerr=2*x.d_final_cooperators_se,color=COL[cname],marker=mk,ms=3.5,lw=1.2,capsize=1.5,label=('keep an existing tie' if cname=='retain' else 'add a new tie'))
        ax.axhline(0,color='0.6',lw=0.6,zorder=0); ax.set_xlabel('Decision round $t_0$'); ax.set_title(ttl,loc='left')
        ax.set_xticks([3,5,7,10,13]); ax.margins(x=0.08)
    axs[0,0].set_ylabel('Value of one C\u2013D tie\n(final-round cooperators)'); axs[0,0].legend(frameon=False,loc='lower left')
    ax=axs[0,2]; b=pd.DataFrame(ivb); b=b[(b.kappa==1.0)&(b.policy=='W4,15')&(b.t0==13)]
    labs=['<\u22126','\u22126,\u22124','\u22124,\u22122','\u22122,0','\u22650']; xs=np.arange(5)
    for off,cname,lab in [(-0.15,'retain','retain'),(0.15,'recruit','recruit')]:
        x=b[b.tie==lab]; ax.errorbar(xs+off,x.d_final_cooperators,yerr=2*x.se,fmt='o' if lab=='retain' else 's',color=COL[cname],ms=3.5,capsize=1.5,lw=1)
    ax.axhline(0,color='0.6',lw=0.6,zorder=0); ax.set_xticks(xs); ax.set_xticklabels(labs); ax.set_xlabel('Defector disposition $\\theta$ (bin)')
    ax.set_title('Gap persists within disposition bins',loc='left'); ax.set_ylim(-0.045,0.03)
    for ax,r in zip(axs[1,:2],[tab[2],tab[1]]):
        cats=['time only','oracle $\\theta$','inferred $\\theta$']
        cpl=[r['coupled_time'][1],r['coupled_oracle'][1],r['coupled_post'][1]]
        dec=[r['decoupled_time'][1],r['decoupled_oracle'][1],r['decoupled_post'][1]]
        xs=np.arange(3)
        ax.plot(xs,dec,'o-',color=COL['window'],ms=4,lw=1,label='recruit and retain decoupled')
        ax.plot(xs,cpl,'s-',color=COL['twophase'],ms=4,lw=1,label='coupled (conciliate or exclude)')
        ax.axhline(r['enc'],color=COL['enc'],ls=':',lw=0.9); ax.text(2.1,r['enc'],'encouragement',fontsize=5.5,color=COL['enc'],va='bottom',ha='right')
        ax.set_xticks(xs); ax.set_xticklabels(cats); ax.set_xlim(-0.3,2.3); ax.margins(y=0.15)
        ax.set_title('$\\kappa=%g$: structure beats information'%r['kappa'],loc='left'); ax.set_ylabel('Final cooperation')
    axs[1,0].legend(frameon=False,loc='lower right',fontsize=5.5)
    ax=axs[1,2]; t=TS[(TS.kappa==1.0)&(TS.policy=='W4,15')]
    ax.plot(t.t,t.an_t,'o-',color=COL['retain'],ms=3,lw=1.1,label='tied defectors')
    ax.plot(t.t,t.an_u,'s-',color=COL['recruit'],ms=3,lw=1.1,label='untied defectors')
    ax.set_xlabel('Round $t$'); ax.set_ylabel('Realised cooperation in $t+1$'); ax.set_title('Tie state predicts conversion',loc='left'); ax.legend(frameon=False,loc='center right')
    ax.set_ylim(0,0.7)
    for i,a in enumerate(axs.flat): panel_letter(a,'abcdef'[i])
    return fig

_f=fig4
def fig4b():
    fig=_f(); axs=np.array(fig.axes).reshape(2,3)
    lg=axs[1,0].get_legend(); h,l=axs[1,0].get_legend_handles_labels(); lg.remove()
    fig.legend(h,l,frameon=False,loc='lower center',bbox_to_anchor=(0.37,-0.035),ncol=2,fontsize=6)
    axs[1,2].set_xticks([1,4,7,10,13])
    return fig

import shutil
plt.close('all')
f4=fig4b()
f4.savefig("fig4_mechanism.png",dpi=300,bbox_inches='tight')
f4.savefig('fig4_mechanism.pdf', bbox_inches='tight')
plt.close(f4)