"""Human-data analysis of tie state (main-text Fig. 5, Supplementary Table S11).

Uses the public node- and edge-level files of McKee et al. (2023), https://osf.io/8ahkg (CC-BY 4.0),
extracted to data/osf by prepare_data.py. Writes results/human_tie_state_gaps.csv,
results/human_tie_state_gaps_round_stratified.csv and results/human_tie_state_by_round.csv.
This file is an ordered export of the analysis steps used for the manuscript.
Run from the 06_code directory: python human_data_tie_state.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
# ---- step 1: Loading OSF human cooperation and edge data ----
coop=pd.concat([pd.read_csv(f"data/osf/{p}_cooperation_data.csv") for p in ['baseline','evaluation','validation']])
edges=pd.concat([pd.read_csv(f"data/osf/{p}_graph_struct_data.csv", usecols=['Group','Condition','Round','Edge','Edge_Type','Prev_Status','Recommendation']) for p in ['baseline','evaluation','validation']])
print(coop.Condition.value_counts()/16/15); print(coop.columns.tolist()); print(coop.Round.min(), coop.Round.max(), edges.Round.min(), edges.Round.max())
print(coop.Cooperation.isna().sum(), edges.Edge_Type.value_counts())

# ---- step 2: Building human action and tie panels per group ----
from itertools import combinations
pairs=np.array(list(combinations(range(16),2)))
groups={}
for (g,cond),dg in coop.groupby(['Group','Condition']):
    a=np.full((15,16),np.nan); 
    for r,p,c in dg[['Round','Player','Cooperation']].values: a[int(r),int(p)]=c
    obs=~np.isnan(a)
    eg=edges[edges.Group==g]
    A=np.zeros((15,16,16),bool); typ=np.full((15,120),'',dtype=object)
    for r,e,et,ps in eg[['Round','Edge','Edge_Type','Prev_Status']].values:
        i,j=pairs[int(e)]; A[int(r),i,j]=A[int(r),j,i]=bool(ps); typ[int(r),int(e)]=et
    # infer missing actions from edge types
    af_=a.copy()
    for it in range(3):
        for r in range(15):
            for e,(i,j) in enumerate(pairs):
                et=typ[r,e]
                if et=='C-C': af_[r,i]=1 if np.isnan(af_[r,i]) else af_[r,i]; af_[r,j]=1 if np.isnan(af_[r,j]) else af_[r,j]
                elif et=='D-D': af_[r,i]=0 if np.isnan(af_[r,i]) else af_[r,i]; af_[r,j]=0 if np.isnan(af_[r,j]) else af_[r,j]
                elif et=='C-D':
                    if np.isnan(af_[r,i]) and not np.isnan(af_[r,j]): af_[r,i]=1-af_[r,j]
                    if np.isnan(af_[r,j]) and not np.isnan(af_[r,i]): af_[r,j]=1-af_[r,i]
    groups[g]=dict(cond=cond,a=a,af=af_,obs=obs,A=A)
print(len(groups), sum(np.isnan(v['af']).sum() for v in groups.values()))
# consistency: obs actions vs edge types
bad=0;tot=0
for g,v in groups.items():
    eg=edges[edges.Group==g]
    for r,e,et in eg[['Round','Edge','Edge_Type']].values[:2000]:
        i,j=pairs[int(e)]; ai,aj=v['af'][int(r),i],v['af'][int(r),j]
        s={'C-C':2,'C-D':1,'D-D':0}[et]; tot+=1; bad+= (ai+aj)!=s
print(bad,tot)
print(pd.Series({g:v['cond'] for g,v in groups.items()}).value_counts())

# ---- step 3: Computing tied versus untied defector statistics in human data ----
rows=[]; drows=[]
for g,v in groups.items():
    a,af_,obs,A=v['a'],v['af'],v['obs'],v['A']
    for r in range(1,14):
        for j in range(16):
            if not (obs[r,j] and a[r,j]==0 and obs[r+1,j]): continue
            # streak: consecutive observed defections ending at r
            st=0; rr=r
            while rr>=0 and obs[rr,j] and a[rr,j]==0: st+=1; rr-=1
            other=[x for x in range(15) if x not in (r,r+1) and obs[x,j]]
            lo=np.mean(a[other,j]) if other else np.nan
            prev=[x for x in range(r) if obs[x,j]]; lo_past=np.mean(a[prev,j]) if prev else np.nan
            fut=[x for x in range(r+2,15) if obs[x,j]]; lo_fut=np.mean(a[fut,j]) if fut else np.nan
            coopnb=[i for i in range(16) if i!=j and af_[r,i]==1]
            tiedC=[i for i in coopnb if A[r,i,j]]
            for i in coopnb:
                rows.append((g,v['cond'],r,j,i,int(A[r,i,j]),st,a[r+1,j],lo,lo_past,lo_fut,A[r,j].sum()))
            drows.append((g,v['cond'],r,j,int(len(tiedC)>0),len(tiedC),st,a[r+1,j],lo,lo_past,lo_fut,A[r,j].sum()))
cols=['g','cond','r','j','i','tied','streak','next','lo','lo_past','lo_fut','deg']
P=pd.DataFrame(rows,columns=cols); Dd=pd.DataFrame(drows,columns=['g','cond','r','j','tied','ntied','streak','next','lo','lo_past','lo_fut','deg'])
print(P.shape, Dd.shape)
def summ(df):
    out=df.groupby(['cond','tied'])[['streak','next','lo','lo_past','lo_fut']].mean().unstack('tied')
    return out
print(summ(P).round(3).to_string()); print(summ(Dd).round(3).to_string())

# ---- step 4: Bootstrapping tied-minus-untied gaps by condition ----
cmap={'GraphNet planner':'GraphNet','Encourag. planner':'Encouragement','Random rec.':'Random','Coop. clustering':'Clustering','Neutral planner':'Neutral','Max. connectivity':'MaxConn','Static network':'Static'}
P['c']=P.cond.map(cmap); Dd['c']=Dd.cond.map(cmap)
rng_b=np.random.default_rng(1)
def gap_stats(df, var):
    # group-level: per group compute mean tied and untied (pair-weighted within group), then pooled ratio estimator
    gs=df.groupby(['g','tied'])[var].agg(['sum','count']).unstack('tied').fillna(0)
    gids=gs.index.values
    def est(idx):
        s=gs.iloc[idx]; 
        t=s[('sum',1)].sum()/max(s[('count',1)].sum(),1); u=s[('sum',0)].sum()/max(s[('count',0)].sum(),1)
        return t,u,t-u
    t,u,d=est(np.arange(len(gids)))
    bs=np.array([est(rng_b.integers(0,len(gids),len(gids)))[2] for _ in range(2000)])
    return dict(tied=t,untied=u,diff=d,lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),ngroups=len(gids),
                n_t=int(gs[('count',1)].sum()),n_u=int(gs[('count',0)].sum()))
res=[]
for name,sel in [('GraphNet',['GraphNet']),('Encouragement',['Encouragement']),('Planner (GraphNet+Enc.)',['GraphNet','Encouragement']),
                 ('Other dynamic',['Random','Clustering','Neutral','MaxConn']),('Static',['Static'])]:
    for var in ['streak','next','lo_past','lo_fut']:
        for w,df in [('pair',P),('defector',Dd)]:
            s=gap_stats(df[df.c.isin(sel)],var); s.update(set=name,var=var,weight=w); res.append(s)
H=pd.DataFrame(res)
print(H[H.weight=='pair'][['set','var','tied','untied','diff','lo','hi','ngroups','n_t','n_u']].round(3).to_string())
print(H[H.weight=='defector'][['set','var','tied','untied','diff','lo','hi','n_t','n_u']].round(3).to_string())

# ---- step 5: Round-stratified human tie-state gaps and saving tables ----
def strat_gap(df,var):
    gs=df.groupby(['r','tied'])[var].mean().unstack('tied').dropna()
    w=df[df.tied==1].groupby('r').size().reindex(gs.index).fillna(0)
    return float(((gs[1]-gs[0])*w).sum()/w.sum())
def strat_boot(df,var,B=1000):
    gids=df.g.unique(); est=strat_gap(df,var); bs=[]
    byg={g:d for g,d in df.groupby('g')}
    for _ in range(B):
        s=rng_b.choice(gids,len(gids)); d=pd.concat([byg[x] for x in s]); bs.append(strat_gap(d,var))
    return est,np.percentile(bs,2.5),np.percentile(bs,97.5)
SR=[]
for name,sel in [('Planner (GraphNet+Enc.)',['GraphNet','Encouragement']),('Other dynamic',['Random','Clustering','Neutral','MaxConn']),('Static',['Static'])]:
    for var in ['streak','next','lo_past']:
        e,l,h=strat_boot(P[P.c.isin(sel)],var,600); SR.append(dict(set=name,var=var,diff_round_strat=e,lo=l,hi=h))
SR=pd.DataFrame(SR); print(SR.round(3).to_string())
os.makedirs("results",exist_ok=True)
H.to_csv("results/human_tie_state_gaps.csv",index=False); SR.to_csv("results/human_tie_state_gaps_round_stratified.csv",index=False)
# per-round curves for figure
curve=P[P.c.isin(['GraphNet','Encouragement','Static'])].assign(grp=lambda d: np.where(d.c=='Static','Static','Planner')).groupby(['grp','r','tied'])[['streak','next','lo_past']].mean().reset_index()
curve.to_csv("results/human_tie_state_by_round.csv",index=False); print(curve.head())