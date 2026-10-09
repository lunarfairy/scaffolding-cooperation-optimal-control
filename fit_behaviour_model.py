"""Re-estimation of the behaviour model from all seven conditions of McKee et al. (2023).

Model (same functional form as the published simulator; main-text Eq. 2):
  round 1:      logit P(C) = b00 + b01 * theta_i
  rounds t>=2:  logit P(C) = beta0 + beta_s k_i + kappa_i (beta_n x_n,i + beta_r x_r,i) + theta_i,
                kappa_i = 1 if i cooperated in round t-1, kappa otherwise,
  theta_i = mu + sigma z_i, z_i ~ N(0,1), one draw per participant (random intercept), mu fixed at -0.304.
k_i is the degree in the network of round t (after the planner move of round t-1) and x_n,i the number of
neighbours who cooperated in round t-1. Only recorded participant decisions enter; decisions that the capital rule
made infeasible (capital after round t-1 below c * k_i) are excluded. The marginal likelihood is computed with
40-point Gauss-Hermite quadrature and maximised with L-BFGS-B. A group bootstrap (resampling groups within the
seven conditions) gives the interval for kappa. Acceptance probabilities are the empirical acceptance rates of
add and delete recommendations by referent action in the conditions with recorded responses.
Usage (from 06_code): see the end of this file.
"""
import os, sys, json
from itertools import combinations
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import log_expit
os.chdir(os.path.dirname(os.path.abspath(__file__)))
C_COST, MU = 0.05, -0.304
PH = ['baseline', 'evaluation', 'validation']

def build():
    coop = pd.concat([pd.read_csv(f"data/osf/{p}_cooperation_data.csv") for p in PH])
    edges = pd.concat([pd.read_csv(f"data/osf/{p}_graph_struct_data.csv",
                                   usecols=['Group', 'Round', 'Edge', 'Edge_Type', 'Prev_Status']) for p in PH])
    pairs = np.array(list(combinations(range(16), 2)))
    rows = []
    for (g, cond), dg in coop.groupby(['Group', 'Condition']):
        a = np.full((15, 16), np.nan); cap = np.full((15, 16), np.nan)
        for r, p, c, k in dg[['Round', 'Player', 'Cooperation', 'Capital']].values:
            a[int(r), int(p)] = c; cap[int(r), int(p)] = k
        obs = ~np.isnan(a)
        eg = edges[edges.Group == g]
        A = np.zeros((15, 16, 16), bool); typ = np.full((15, 120), '', dtype=object)
        for r, e, et, ps in eg[['Round', 'Edge', 'Edge_Type', 'Prev_Status']].values:
            i, j = pairs[int(e)]; A[int(r), i, j] = A[int(r), j, i] = bool(ps); typ[int(r), int(e)] = et
        af = a.copy()                       # actions of bot-replaced players inferred from edge types
        for _ in range(3):
            for r in range(15):
                for e, (i, j) in enumerate(pairs):
                    et = typ[r, e]
                    if et == 'C-C':
                        af[r, i] = 1 if np.isnan(af[r, i]) else af[r, i]; af[r, j] = 1 if np.isnan(af[r, j]) else af[r, j]
                    elif et == 'D-D':
                        af[r, i] = 0 if np.isnan(af[r, i]) else af[r, i]; af[r, j] = 0 if np.isnan(af[r, j]) else af[r, j]
                    elif et == 'C-D':
                        if np.isnan(af[r, i]) and not np.isnan(af[r, j]): af[r, i] = 1 - af[r, j]
                        if np.isnan(af[r, j]) and not np.isnan(af[r, i]): af[r, j] = 1 - af[r, i]
        for i in range(16):
            for r in range(15):
                if not obs[r, i]: continue
                if r == 0:
                    rows.append((g, cond, i, r, a[r, i], 1, 0, 0, 0, 1)); continue
                if np.isnan(af[r - 1, i]): continue
                k = A[r, i].sum()
                nb = np.where(A[r, i])[0]
                if np.isnan(af[r - 1, nb]).any(): continue
                xn = af[r - 1, nb].sum(); xr = xn / k if k > 0 else 0.0
                if not np.isnan(cap[r - 1, i]) and cap[r - 1, i] < C_COST * k - 1e-9: continue
                rows.append((g, cond, i, r, a[r, i], 0, k, xn, xr, af[r - 1, i]))
    D = pd.DataFrame(rows, columns=['g', 'cond', 'p', 'r', 'y', 'first', 'k', 'xn', 'xr', 'aprev'])
    D['pid'] = D.g.astype(str) + '_' + D.p.astype(str)
    return D

def make(D):
    pid_codes, pid = np.unique(D.pid.values, return_inverse=True)
    order = np.argsort(pid, kind='stable')
    X = {c: D[c].values[order].astype(float) for c in ['y', 'first', 'k', 'xn', 'xr', 'aprev']}
    pidx = pid[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pidx)) + 1]
    grp = D.g.values[order][starts]
    return X, starts, grp

z, w = np.polynomial.hermite_e.hermegauss(40); lw = np.log(w / w.sum())

def nll(par, X, starts, wt=None):
    b00, b01, b0, bs, bn, br, kap, lsig = par
    th = MU + np.exp(lsig) * z[:, None]                       # (Q, 1)
    kk = np.where(X['aprev'] == 1, 1.0, kap)
    eta_l = b0 + bs * X['k'] + kk * (bn * X['xn'] + br * X['xr'])
    eta = np.where(X['first'] == 1, b00 + b01 * th, eta_l + th)  # (Q, N)
    ll = np.where(X['y'] == 1, log_expit(eta), log_expit(-eta))
    lp = np.add.reduceat(ll, starts, axis=1) + lw[:, None]    # (Q, P)
    m = lp.max(0); li = m + np.log(np.exp(lp - m).sum(0))
    return -(li if wt is None else wt * li).sum()

def fit(X, starts, wt=None, x0=None):
    x0 = np.array([1.6, 0.7, 0.5, -0.25, 0.23, 3.5, 0.86, 1.17]) if x0 is None else x0
    r = minimize(nll, x0, args=(X, starts, wt), method='L-BFGS-B',
                 bounds=[(None, None)] * 6 + [(0, 3), (-3, 3)], options=dict(maxiter=2000, ftol=1e-12, gtol=1e-7))
    return r

def acceptance():
    e = pd.read_csv("data/osf/baseline_graph_struct_data.csv")
    e = e[e.Recommendation.notna() & e.Accepted.notna() & (e.Recommendation != 0)] if 'Recommendation' in e else e
    out = {}
    for val, name in ((0, 'del'), (1, 'add')):
        s = e[e.Prev_Status == (1 if name == 'del' else 0)]
        for ac in (0, 1):
            ss = s[s.Referent_Player_Coop == ac]
            out[f"{name}_{'C' if ac else 'D'}"] = (float(ss.Accepted.mean()), int(len(ss)))
    return out

def boot(start, count, D, X, starts, grp, x0):
    gs = np.unique(grp); cond_of = D.groupby('g').cond.first().loc[gs].values
    out = []
    for b in range(start, start + count):
        rng = np.random.default_rng(20261008 + b)
        samp = np.concatenate([rng.choice(gs[cond_of == c], (cond_of == c).sum()) for c in np.unique(cond_of)])
        cnt = pd.Series(samp).value_counts(); wt = pd.Series(grp).map(cnt).fillna(0).values
        out.append(fit(X, starts, wt, x0=x0).x.tolist())
    return out

if __name__ == '__main__':
    # python fit_behaviour_model.py                  point estimate -> results/behaviour_model_refit.json
    # python fit_behaviour_model.py boot START COUNT  bootstrap replicates START..START+COUNT-1 -> results/boot/
    # python fit_behaviour_model.py combine           adds the bootstrap interval to the json
    os.makedirs('results/boot', exist_ok=True)
    names = ['b00', 'b01', 'beta0', 'beta_s', 'beta_n', 'beta_r', 'kappa', 'log_sigma']
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    if mode == 'combine':
        est = json.load(open('results/behaviour_model_refit.json'))
        reps = []
        for f in sorted(os.listdir('results/boot')):
            reps += json.load(open('results/boot/' + f))
        R = np.array(reps)
        est['n_bootstrap'] = len(R)
        est['boot_95'] = {nm: [float(np.quantile(R[:, i], 0.025)), float(np.quantile(R[:, i], 0.975))] for i, nm in enumerate(names)}
        est['boot_95']['sigma'] = [float(np.exp(x)) for x in est['boot_95']['log_sigma']]
        json.dump(est, open('results/behaviour_model_refit.json', 'w'), indent=1)
        print(json.dumps(est['boot_95'], indent=1)); sys.exit()
    D = build(); X, starts, grp = make(D)
    if mode == 'boot':
        x0 = np.array([json.load(open('results/behaviour_model_refit.json'))[nm] for nm in names])
        s0, cnt = int(sys.argv[2]), int(sys.argv[3])
        json.dump(boot(s0, cnt, D, X, starts, grp, x0), open(f'results/boot/boot_{s0:04d}.json', 'w')); sys.exit()
    r = fit(X, starts)
    est = dict(zip(names, r.x.tolist())); est['sigma'] = float(np.exp(r.x[7])); est['mu_theta_fixed'] = MU
    est.update(nll=float(r.fun), n_decisions=int(len(D)), n_participants=int(len(starts)), n_groups=int(len(np.unique(grp))),
               converged=bool(r.success), decisions_by_condition=D.cond.value_counts().to_dict())
    est['acceptance'] = acceptance()
    json.dump(est, open('results/behaviour_model_refit.json', 'w'), indent=1)
    # Simulator-format input consumed by run_rev_abm.py; export the fitted values unchanged.
    simulator_params = dict(
        R1=[est['b00'], est['b01']],
        RL=[est['beta0'], est['beta_s'], est['beta_n'], est['beta_r']],
        MU_TH=est['mu_theta_fixed'], SD_TH=est['sigma'],
        PHI=[est['acceptance'][key][0] for key in ('del_D', 'del_C', 'add_D', 'add_C')],
        kappa=est['kappa'])
    json.dump(simulator_params, open('results/refit_params.json', 'w'), indent=1)
    print(json.dumps(est, indent=1))
