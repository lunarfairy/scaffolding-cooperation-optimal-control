"""Job C: agent-based evaluation of reduced-model policies and mechanism analyses.
  eval_oc   : IPA-optimal open-loop controls evaluated in the ABM (continuous and rounded)
  fbgrid    : feedback threshold policies (oracle theta / Bayesian posterior mean), linear-in-t threshold family
  fbipa     : IPA-optimised thresholds evaluated in the ABM
  base      : reference planners with recommendation-rate logging (for comparison with the encouragement table)
  myopic    : one-step value of a C-D tie and myopic responsiveness threshold kappa_m(t) along trajectories
  needle    : needle-variation (single-round flip) switching functions with common random numbers
  robust    : switch sweeps under targeted routing, other initial densities and group size
"""
import os, sys, json, time, glob
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import multiprocessing as mp
import abm_fast as af

NPROC = int(os.environ.get('NPROC', 32))
SMALL = os.environ.get('SMALL', '0') == '1'
NE = 100000 if not SMALL else 500       # evaluation games
NG = 20000 if not SMALL else 300        # grid games
NN = 200000 if not SMALL else 500       # needle games


def stats(r, n):
    return dict(final=float(r['final'].mean()), final_se=float(r['final'].std() / np.sqrt(n)),
                capital=float(r['capital'].mean()), capital_se=float(r['capital'].std() / np.sqrt(n)),
                mcoop=float(r['mcoop'].mean()), mcoop_se=float(r['mcoop'].std() / np.sqrt(n)))


def lin_thr(c0, c1, T):
    t = np.arange(1, T)
    return c0 + c1 * (t - 1) / max(T - 2, 1)


# ------------------------------------------------------------------ tasks
def t_eval_oc(a):
    rec, rounded, seed = a
    u = np.array(rec['u'])
    if rounded:
        u = (u > 0.5).astype(float)
        u[0, 2] = (1, 0); u[0, 0] = (0, 1)          # standard C-C / D-D rule in round 1 (flat direction)
    r = af.run_batched(af.typed(u), NE, seed, T=rec['T'], kappa=rec['kappa'])
    out = dict(task='eval_oc', T=rec['T'], obj=rec['obj'], init=rec['init'], kappa=rec['kappa'],
               rounded=rounded, J_ipa=rec['J'])
    out.update(stats(r, NE))
    return out


def t_fbgrid(a):
    kappa, use, c0, c1, seed = a
    T = 15
    r = af.run_batched(af.threshold_planner(lin_thr(c0, c1, T), use=use), NG, seed, T=T, kappa=kappa,
                       track_posterior=(use == 'post'))
    out = dict(task='fbgrid', kappa=kappa, use=use, c0=c0, c1=c1)
    out.update(stats(r, NG))
    return out


def t_policy(a):
    """Generic evaluation with rec-rate logging. spec: dict(kind=..., ...)."""
    spec, kappa, seed, n = a
    T = spec.get('T', 15)
    kind = spec['kind']
    tp = False
    if kind == 'typed':
        pl = af.typed(np.array(spec['u']))
    elif kind == 'thr':
        pl = af.threshold_planner(np.array(spec['thr']), use=spec['use']); tp = spec['use'] == 'post'
    elif kind == 'probation':
        pl = af.probation_planner(spec['strikes'])
    r = af.run_batched(pl, n, seed, T=T, kappa=kappa, track_posterior=tp, record_recs=True)
    out = dict(task='policy', name=spec['name'], kappa=kappa, T=T,
               coop=r['coop'].mean(0).tolist(), recs=r['recs'].tolist(),
               ecc=r['ecc'].mean(0).tolist(), ecd=r['ecd'].mean(0).tolist(), edd=r['edd'].mean(0).tolist())
    out.update(stats(r, n))
    return out


def t_needle(a):
    kappa, T, s_base, seed = a
    base = af.switch_u(T, s_base)
    res = []
    for t in range(0, T):          # t = 0 -> base policy
        u = base.copy()
        if t > 0:
            u[t - 1, 1] = (0.0, 1.0) if base[t - 1, 1, 0] == 1 else (1.0, 0.0)
        r = af.run_batched(af.typed(u), NN, seed, T=T, kappa=kappa)   # same seed = common random numbers
        res.append((r['final'], r['capital']))
    f0, c0 = res[0]
    out = dict(task='needle', kappa=kappa, T=T, s_base=s_base, J_base=float(f0.mean()), cap_base=float(c0.mean()))
    d, dse, dc, dcse = [], [], [], []
    for t in range(1, T):
        df = res[t][0] - f0; dcap = res[t][1] - c0
        d.append(float(df.mean())); dse.append(float(df.std() / np.sqrt(NN)))
        dc.append(float(dcap.mean())); dcse.append(float(dcap.std() / np.sqrt(NN)))
    out.update(dfinal=d, dfinal_se=dse, dcap=dc, dcap_se=dcse)
    return out


def t_robust(a):
    kappa, T, s, cond, seed = a
    kw = dict(T=T, kappa=kappa)
    if cond == 'targeted':
        kw['route'] = 'targeted'
    elif cond.startswith('p0='):
        kw['p0'] = float(cond[3:])
    elif cond.startswith('n='):
        kw['n'] = int(cond[2:])
    r = af.run_batched(af.typed(af.switch_u(T, s)), NG, seed, **kw)
    out = dict(task='robust', kappa=kappa, T=T, s=s, cond=cond)
    out.update(stats(r, NG))
    return out


def t_myopic(a):
    """Along trajectories of a switch policy, compute for every C-D pair the one-step effect of the tie on the
    expected number of cooperators in the next round, as a function of the responsiveness kappa' used in the
    response of the defecting endpoint. Pre-recommendation network; the pair's own tie is toggled."""
    kappa, T, s, seed = a
    rng = np.random.default_rng(seed)
    G = 4000 if not SMALL else 200
    reps = 5 if not SMALL else 1
    kgrid = np.linspace(0, 3, 301)
    b0, bs, bn, br = af.RL
    acc = {t: dict(num=np.zeros_like(kgrid), dC=0.0, npairs=0, thD=0.0, nD=0, pD=0.0, dD1=0.0) for t in range(1, T)}
    planner = af.typed(af.switch_u(T, s))
    for rep in range(reps):
        n = 16
        U = rng.random((G, n, n)) < 0.3
        A = np.triu(U, 1); A = A | A.transpose(0, 2, 1)
        theta = rng.normal(af.MU_TH, af.SD_TH, (G, n))
        dcap = np.full((G, n), af.D0)
        a_prev = None
        for t in range(1, T + 1):
            Af = A.astype(float); deg = Af.sum(-1)
            if t == 1:
                pc = af.sig(af.R1[0] + af.R1[1] * theta)
            else:
                xn = np.einsum('gij,gj->gi', Af, a_prev)
                xr = np.where(deg > 0, xn / np.maximum(deg, 1), 0.0)
                k = np.where(a_prev == 1, 1.0, kappa)
                pc = af.sig(b0 + bs * deg + k * (bn * xn + br * xr) + theta)
            a = (rng.random((G, n)) < pc).astype(float)
            a = a * (af.C * deg <= dcap)
            xn_now = np.einsum('gij,gj->gi', Af, a)
            dcap = dcap + af.B * xn_now - af.C * a * deg
            if t == T:
                break
            # --- one-step value of each C-D pair (i C, j D) at the current network
            cdm = (a[:, :, None] == 1) & (a[:, None, :] == 0)          # (g,i,j): i C, j D
            gi, ii, jj = np.nonzero(cdm)
            if len(gi):
                e = Af[gi, ii, jj]
                kD = deg[gi, jj] - e; xD = xn_now[gi, jj] - e * 1.0    # exclude own tie (partner i cooperates)
                kC = deg[gi, ii] - e; xC = xn_now[gi, ii]              # partner j defects: contributes 0
                thD = theta[gi, jj]; thC = theta[gi, ii]

                def eta(kk, xx, kap, th):
                    xr_ = np.where(kk > 0, xx / np.maximum(kk, 1), 0.0)
                    return b0 + bs * kk + kap * (bn * xx + br * xr_) + th
                dC = af.sig(eta(kC + 1, xC, 1.0, thC)) - af.sig(eta(kC, xC, 1.0, thC))
                num = np.zeros_like(kgrid)
                for q, kap in enumerate(kgrid):
                    dD = af.sig(eta(kD + 1, xD + 1, kap, thD)) - af.sig(eta(kD, xD, kap, thD))
                    num[q] = (dD + dC).sum()
                A_ = acc[t]
                A_['num'] += num; A_['dC'] += dC.sum(); A_['npairs'] += len(gi)
                A_['dD1'] += (af.sig(eta(kD + 1, xD + 1, kappa, thD)) - af.sig(eta(kD, xD, kappa, thD))).sum()
            dmask = a == 0
            A_ = acc[t]; A_['thD'] += theta[dmask].sum(); A_['nD'] += dmask.sum()
            if t > 1:
                A_['pD'] += pc[dmask].sum()
            a_prev = a
            padd, pdel = planner(dict(A=A, a=a, t=t, theta=theta, ndef=None))
            r = rng.random((G, n, n))
            add = np.triu((~A) & (r < padd), 1); dele = np.triu(A & (r < pdel), 1)
            swap = rng.random((G, n, n)) < 0.5
            aref = np.where(swap, a[:, :, None], a[:, None, :])
            acc_r = rng.random((G, n, n))
            add &= acc_r < np.where(aref == 1, af.PHI_ADD_C, af.PHI_ADD_D)
            dele &= acc_r < np.where(aref == 1, af.PHI_DEL_C, af.PHI_DEL_D)
            add = add | add.transpose(0, 2, 1); dele = dele | dele.transpose(0, 2, 1)
            A = (A | add) & ~dele
    out = dict(task='myopic', kappa=kappa, T=T, s=s, rounds=[])
    for t in range(1, T):
        A_ = acc[t]
        npairs = max(A_['npairs'], 1)
        v = A_['num'] / npairs
        idx = np.nonzero(np.diff(np.sign(v)))[0]
        km = float(np.interp(0, [v[idx[0]], v[idx[0] + 1]], [kgrid[idx[0]], kgrid[idx[0] + 1]])) if len(idx) else (float('nan') if v[-1] < 0 else 0.0)
        out['rounds'].append(dict(t=t, kappa_m=km, v_at_kappa=float(np.interp(kappa, kgrid, v)),
                                  dC=A_['dC'] / npairs, dD=A_['dD1'] / npairs, npairs_per_game=A_['npairs'] / (G * reps),
                                  mean_theta_D=A_['thD'] / max(A_['nD'], 1), frac_D=A_['nD'] / (G * reps * 16),
                                  v_curve=v[::10].tolist()))
    return out


def window_u(T, ta, td):
    """Add C-D ties in rounds t <= ta, cut C-D ties in rounds t >= td (both may be active); CC connect, DD cut."""
    u = np.zeros((T - 1, 3, 2))
    u[:, 2, 0] = 1.0; u[:, 0, 1] = 1.0
    for t in range(1, T):
        u[t - 1, 1, 0] = 1.0 if t <= ta else 0.0
        u[t - 1, 1, 1] = 1.0 if t >= td else 0.0
    return u


def t_window(a):
    kappa, T, ta, td, seed, ng = a
    r = af.run_batched(af.typed(window_u(T, ta, td)), ng, seed, T=T, kappa=kappa)
    out = dict(task='window', kappa=kappa, T=T, ta=ta, td=td)
    out.update(stats(r, ng))
    return out


def t_tiestate(a):
    """Selection encoded in network state: for C-D pairs, compare defectors on existing ties with defectors on
    absent ties (mean disposition, current defection streak, next-round cooperation probability if tied)."""
    name, u, kappa, T, seed = a
    u = np.array(u)
    rng = np.random.default_rng(seed)
    G = 4000 if not SMALL else 200
    reps = 5 if not SMALL else 1
    b0, bs, bn, br = af.RL
    acc = {t: dict(nt=0, nu=0, th_t=0.0, th_u=0.0, st_t=0.0, st_u=0.0, pc_t=0.0, pc_u=0.0) for t in range(1, T)}
    planner = af.typed(u)
    for rep in range(reps):
        n = 16
        U = rng.random((G, n, n)) < 0.3
        A = np.triu(U, 1); A = A | A.transpose(0, 2, 1)
        theta = rng.normal(af.MU_TH, af.SD_TH, (G, n))
        dcap = np.full((G, n), af.D0); streak = np.zeros((G, n))
        a_prev = None
        for t in range(1, T + 1):
            Af = A.astype(float); deg = Af.sum(-1)
            if t == 1:
                pc = af.sig(af.R1[0] + af.R1[1] * theta)
            else:
                xn = np.einsum('gij,gj->gi', Af, a_prev)
                xr = np.where(deg > 0, xn / np.maximum(deg, 1), 0.0)
                k = np.where(a_prev == 1, 1.0, kappa)
                pc = af.sig(b0 + bs * deg + k * (bn * xn + br * xr) + theta)
            a = (rng.random((G, n)) < pc).astype(float)
            a = a * (af.C * deg <= dcap)
            xn_now = np.einsum('gij,gj->gi', Af, a)
            dcap = dcap + af.B * xn_now - af.C * a * deg
            streak = np.where(a == 0, streak + 1, 0)
            if t == T:
                break
            cdm = (a[:, :, None] == 1) & (a[:, None, :] == 0)
            gi, ii, jj = np.nonzero(cdm)
            if len(gi):
                e = A[gi, ii, jj]
                kD = deg[gi, jj]; xD = xn_now[gi, jj]
                # next-round cooperation probability of the defector given its current neighbourhood (tie as is)
                xr_ = np.where(kD > 0, xD / np.maximum(kD, 1), 0.0)
                pnext = af.sig(b0 + bs * kD + kappa * (bn * xD + br * xr_) + theta[gi, jj])
                A_ = acc[t]
                A_['nt'] += e.sum(); A_['nu'] += (~e).sum()
                A_['th_t'] += theta[gi, jj][e].sum(); A_['th_u'] += theta[gi, jj][~e].sum()
                A_['st_t'] += streak[gi, jj][e].sum(); A_['st_u'] += streak[gi, jj][~e].sum()
                A_['pc_t'] += pnext[e].sum(); A_['pc_u'] += pnext[~e].sum()
            a_prev = a
            padd, pdel = planner(dict(A=A, a=a, t=t, theta=theta, ndef=None))
            r = rng.random((G, n, n))
            add = np.triu((~A) & (r < padd), 1); dele = np.triu(A & (r < pdel), 1)
            swap = rng.random((G, n, n)) < 0.5
            aref = np.where(swap, a[:, :, None], a[:, None, :])
            acc_r = rng.random((G, n, n))
            add &= acc_r < np.where(aref == 1, af.PHI_ADD_C, af.PHI_ADD_D)
            dele &= acc_r < np.where(aref == 1, af.PHI_DEL_C, af.PHI_DEL_D)
            add = add | add.transpose(0, 2, 1); dele = dele | dele.transpose(0, 2, 1)
            A = (A | add) & ~dele
    out = dict(task='tiestate', name=name, kappa=kappa, T=T, rounds=[])
    for t in range(1, T):
        A_ = acc[t]; nt = max(A_['nt'], 1); nu = max(A_['nu'], 1)
        out['rounds'].append(dict(t=t, n_tied=A_['nt'] / (G * reps), n_untied=A_['nu'] / (G * reps),
                                  theta_tied=A_['th_t'] / nt, theta_untied=A_['th_u'] / nu,
                                  streak_tied=A_['st_t'] / nt, streak_untied=A_['st_u'] / nu,
                                  pnext_tied=A_['pc_t'] / nt, pnext_untied=A_['pc_u'] / nu))
    return out


def t_pairacc(a):
    """ABM on the identical theta draws used by the reduced model (64 groups x reps)."""
    name, u, kappa, seed = a
    th = np.load('theta_draws.npy')
    reps = 500 if not SMALL else 5
    rng = np.random.default_rng(seed)
    fin, coop = [], []
    for b in range(reps // 25):
        thb = np.tile(th, (25, 1))
        r = af.simulate(af.typed(np.array(u)), thb.shape[0], rng, T=15, kappa=kappa, theta=thb)
        fin.append(r['final']); coop.append(r['coop'])
    fin = np.concatenate(fin); coop = np.concatenate(coop)
    return dict(task='pairacc', planner=name, kappa=kappa, final=float(fin.mean()), final_se=float(fin.std() / np.sqrt(len(fin))),
                coop=coop.mean(0).tolist())


def dispatch(a):
    return {'eval_oc': t_eval_oc, 'fbgrid': t_fbgrid, 'policy': t_policy, 'needle': t_needle,
            'robust': t_robust, 'myopic': t_myopic, 'window': t_window, 'tiestate': t_tiestate, 'pairacc': t_pairacc}[a[0]](a[1:])


if __name__ == '__main__':
    t0 = time.time()
    tasks = []
    seed = 100
    # IPA-optimal controls
    ocrecs = []
    for f in sorted(glob.glob('ipa_oc_*.json')):
        ocrecs += json.load(open(f))
    for rec in ocrecs:
        for rounded in (False, True):
            tasks.append(('eval_oc', rec, rounded, seed)); seed += 1
    # feedback grid
    c0s = np.arange(-4, 3.01, 0.5) if not SMALL else [-1, 0]
    c1s = np.arange(-2, 6.01, 1.0) if not SMALL else [0, 2]
    for kappa in [0.5, 0.75, 1.0]:
        for use in ['oracle', 'post']:
            for c0 in c0s:
                for c1 in c1s:
                    tasks.append(('fbgrid', kappa, use, float(c0), float(c1), seed)); seed += 1
    # IPA thresholds in the ABM
    fb = json.load(open('ipa_fb.json')) if os.path.exists('ipa_fb.json') else {}
    for key, v in fb.items():
        if not key.startswith('defector_'):
            continue
        kappa = float(key.split('_')[1])
        for use in ['oracle', 'post']:
            tasks.append(('policy', dict(kind='thr', name=f'ipa_thr_{use}', thr=v['thr'][0], use=use), kappa, seed, NE)); seed += 1
    # baselines with recommendation logging
    sweep = [r for r in json.load(open('abm_a_results.json')) if r['kind'] == 'sweep'] if os.path.exists('abm_a_results.json') else []
    for kappa in [0.5, 0.75, 1.0]:
        base = [dict(kind='typed', name='encouragement', u=af.encouragement_u(15).tolist()),
                dict(kind='typed', name='exclusion', u=af.switch_u(15, 0).tolist()),
                dict(kind='typed', name='conciliation', u=af.switch_u(15, 14).tolist()),
                dict(kind='typed', name='switch_T-2', u=af.switch_u(15, 13).tolist())]
        cand = [r for r in sweep if r['T'] == 15 and abs(r['kappa'] - kappa) < 1e-9]
        if cand:
            sbest = max(cand, key=lambda r: r['final'])['s']
            base.append(dict(kind='typed', name='switch_best', u=af.switch_u(15, sbest).tolist(), s=sbest))
        for k_ in [1, 2, 3, 5]:
            base.append(dict(kind='probation', name=f'probation_{k_}', strikes=k_))
        for spec in base:
            tasks.append(('policy', spec, kappa, seed, NE)); seed += 1
    # needle variations
    for kappa in [0.5, 0.75, 1.0]:
        for sb in [0, 13]:
            tasks.append(('needle', kappa, 15, sb, seed)); seed += 1
    # myopic analysis
    for kappa, s in [(1.0, 13), (0.75, 13), (0.5, 13), (0.5, 0), (1.0, 0)]:
        tasks.append(('myopic', kappa, 15, s, seed)); seed += 1
    # paired-theta accuracy check of the reduced model
    if os.path.exists('theta_draws.npy'):
        T = 15
        plan = {'static': af.uniform_u(T, 0, 0), 'random': af.uniform_u(T, .3, .3), 'neutral': af.neutral_u(T),
                'maxconn': af.uniform_u(T, 1, 0), 'encouragement': af.encouragement_u(T),
                'exclusion': af.switch_u(T, 0), 'conciliation': af.switch_u(T, T - 1), 'switch_T-2': af.switch_u(T, T - 2)}
        for kappa in [0.5, 0.75, 1.0]:
            for name, u in plan.items():
                tasks.append(('pairacc', name, u.tolist(), kappa, seed)); seed += 1
    # add-window x cut-window family
    for kappa in [0.5, 0.625, 0.75, 0.875, 1.0, 1.25]:
        for ta in range(0, 15):
            for td in range(1, 16):
                tasks.append(('window', kappa, 15, ta, td, seed, 30000 if not SMALL else 200)); seed += 1
    for kappa, T in [(0.75, 8), (1.0, 8), (0.75, 25), (1.0, 25)]:
        for ta in range(0, T):
            for td in range(1, T + 1):
                tasks.append(('window', kappa, T, ta, td, seed, 20000 if not SMALL else 200)); seed += 1
    # selection encoded in tie state
    for kappa in [0.5, 1.0]:
        for name, u in [('conciliation', af.switch_u(15, 14)), ('window_4_15', window_u(15, 4, 15)),
                        ('switch_T-2', af.switch_u(15, 13)), ('exclusion', af.switch_u(15, 0))]:
            tasks.append(('tiestate', name, u.tolist(), kappa, 15, seed)); seed += 1
    # robustness sweeps
    for cond in ['targeted', 'p0=0.15', 'p0=0.5', 'n=24']:
        for kappa in [0.5, 0.75, 1.0]:
            for s in range(15):
                tasks.append(('robust', kappa, 15, s, cond, seed)); seed += 1
    parts = os.environ.get('C_PARTS', '')
    if parts:
        keep = set(parts.split(','))
        tasks = [a for a in tasks if a[0] in keep or (a[0] == 'policy' and (('fbipa' in keep and a[1]['name'].startswith('ipa_thr')) or ('base' in keep and not a[1]['name'].startswith('ipa_thr'))))]
    pri = {'pairacc': 0, 'myopic': 0, 'tiestate': 0, 'needle': 1, 'policy': 2, 'eval_oc': 3, 'fbgrid': 4, 'window': 5, 'robust': 6}
    tasks.sort(key=lambda a: pri[a[0]])
    print('tasks', len(tasks), flush=True)
    with mp.Pool(NPROC) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(dispatch, tasks, chunksize=1)):
            res.append(r)
            if i % 50 == 0:
                print(i, round(time.time() - t0, 1), flush=True)
                json.dump(res, open(os.environ.get('C_OUT', 'abm_c_results.json'), 'w'))
    json.dump(res, open(os.environ.get('C_OUT', 'abm_c_results.json'), 'w'))
    print('done', len(res), round(time.time() - t0, 1), flush=True)
