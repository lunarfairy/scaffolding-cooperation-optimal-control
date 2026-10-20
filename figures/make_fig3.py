"""Fig. 3: converged reduced-model controls, window landscapes and fresh-game comparison.
Run from this directory python make_fig3.py
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
RA = json.load(gzip.open("../figure_data/rev_abm_results.json.gz"))
dfw = pd.read_csv(R_ + "abm_window_screening_grid.csv")              # ABM window screening grid
O3 = pd.read_csv(R_ + "reduced_model_optimum_preselected_abm.csv")   # pre-selected reduced-model optima in the ABM
BC = json.load(open(R_ + "reduced_model_best_controls_T15.json"))    # converged controls, shape (14, 3 pair types, 2 add/delete)
def bestu(k, T=15, obj='final'):
    return np.array(BC[str(float(k))])

def fig3():
    fig=plt.figure(figsize=(7.2,4.4)); gs=GridSpec(2,3,figure=fig,wspace=0.8,hspace=0.6)
    a1=fig.add_subplot(gs[0,0]); a2=fig.add_subplot(gs[1,0])
    for k,colr,ls in [(1.0,COL['window'],'-'),(0.75,COL['twophase'],'--'),(0.5,COL['excl'],':')]:
        u=bestu(k); t=np.arange(1,15)
        a1.step(t,u[:,1,0],where='mid',color=colr,ls=ls,lw=1.3,label=f'$\\kappa={k:g}$'); a2.step(t,1-u[:,1,1],where='mid',color=colr,ls=ls,lw=1.3)
    a1.set_ylabel('Recruit $u^+_{\\mathrm{CD}}(t)$'); a2.set_ylabel('Retain $1-u^-_{\\mathrm{CD}}(t)$'); a2.set_xlabel('Decision round $t$')
    a1.set_title('Converged reduced-model controls',loc='left'); a1.legend(frameon=False,fontsize=5.5,loc='center right')
    for a in (a1,a2): a.set_ylim(-0.08,1.08); a.set_xticks([1,4,7,10,13])
    for pos,k,ttl in [(gs[0,1],1.0,'$\\kappa=1$: recruit early, never cut'),(gs[0,2],0.75,'$\\kappa=0.75$: cut only at the end')]:
        ax=fig.add_subplot(pos); g=dfw[(dfw.kappa==k)&(dfw['T']==15)]
        M=np.full((15,15),np.nan)
        for _,r in g.iterrows(): M[int(r.td)-1,int(r.ta)]=r.final
        im=ax.imshow(M,origin='lower',cmap='viridis',aspect='auto',extent=[-0.5,14.5,0.5,15.5],vmin=np.nanmax(M)-0.06,vmax=np.nanmax(M))
        ax.plot(np.arange(0,15),np.arange(1,16),'w--',lw=0.7)
        bw=SUM[(SUM.tag=='main')&(SUM.kappa==k)&(SUM['T']==15)].best_window.values[0]; ta,td=map(int,bw[1:].split(','))
        ax.plot(ta,td,'*',color='w',ms=8,mec='k',mew=0.4)
        bp=int(SUM[(SUM.tag=='main')&(SUM.kappa==k)&(SUM['T']==15)].best_tp.values[0][1:]); ax.plot(bp,bp+1,'o',mfc='none',mec='w',ms=5,mew=1)
        ax.set_xlabel('Last recruiting decision $t_a$'); ax.set_ylabel('First cutting decision $t_d$'); ax.set_title(ttl,loc='left')
        ax.set_yticks([1,5,10,15]); ax.set_yticklabels(['1','5','10','never'])
        cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.02); vmax_=np.nanmax(M); cb.set_ticks([np.ceil((vmax_-0.06)*100)/100,np.floor(vmax_*100)/100]); cb.set_label('Final cooperation',fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax=fig.add_subplot(gs[1,1:])
    names=[('P0','exclusion',COL['excl'],'v'),('P14','conciliation',COL['conc'],'^'),('bestP','best two-phase',COL['twophase'],'s'),('enc','encouragement',COL['enc'],'D'),
           ('bestW','best window',COL['window'],'o'),('ipa','reduced-model optimum',COL['window'],'*'),('Do','decoupled oracle',COL['oracle'],'P')]
    for j,k in enumerate([0.5,0.75,1.0]):
        s=[x for x in RA['reeval'] if x['tag']=='main' and x['setting']['kappa']==k and x['setting']['T']==15][0]; pol=s['policies']
        row=SUM[(SUM.tag=='main')&(SUM.kappa==k)&(SUM['T']==15)].iloc[0]
        vals={'P0':pol['P0']['final'],'P14':pol['P14']['final'],'bestP':row.tp,'enc':row.enc,'bestW':row.win,'ipa':O3[(O3.obj=='final')&(O3['T']==15)&(O3.kappa==k)].abm_final.values[0],
              'Do':max(pol[n]['final'] for n in pol if n.startswith('D') and not n.endswith('post'))}
        for i,(key,lab,colr,mk) in enumerate(names):
            ax.plot(j*8+i,vals[key],mk,color=colr,ms=4.5 if mk!='*' else 7,label=lab if j==0 else None,mfc=colr if key!='ipa' else 'w',mec=colr)
        ax.text(j*8+3,0.875,f'$\\kappa={k:g}$',ha='center',fontsize=6.5)
    ax.set_xticks([]); ax.set_ylabel('Final cooperation (fresh games)'); ax.set_ylim(0.56,0.89)
    ax.set_title('Fresh-game comparison (200,000 games per policy, common random numbers)',loc='left')
    ax.legend(frameon=False,fontsize=5.3,ncol=4,loc='lower center',bbox_to_anchor=(0.5,-0.32))
    panel_letter(a1,'a'); panel_letter(fig.axes[2],'b'); panel_letter(fig.axes[4],'c'); panel_letter(ax,'d')
    return fig

plt.close('all'); f3 = fig3()
f3.savefig("fig3_optimal_control.png", dpi=300, bbox_inches='tight'); f3.savefig("fig3_optimal_control.pdf", bbox_inches='tight')
