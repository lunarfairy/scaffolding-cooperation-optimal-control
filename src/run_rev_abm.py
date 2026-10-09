"""Revision experiments in the agent-based model (ABM).

Stage 1 (screening and diagnostics)
  grid      : recruit/retain windows under robustness conditions, the re-estimated model at kappa = 0.808 / 0.910,
              the nested decoupled disposition-threshold family (oracle) and a widened coupled linear-threshold family
  needle1   : one-sided perturbations of the C-D recruitment (u+) and deletion (u-) controls, one round at a time,
              with common random numbers (CRN)
  tieiv     : interventional tie values: cut one random tied C-D pair / add one random untied C-D pair after the
              planner move of round t0 (paired with an untouched control game, CRN)
  tiestats  : tie-state statistics, pair- and defector-weighted, with realised next-round actions
  capacc    : paired-disposition ABM with and without the capital constraint (for the reduced-model error budget)
  ref       : vectorised simulator versus the loop-based reference simulator (sim_pilot.py), per planner
  myopic    : myopic threshold under the re-estimated model
Stage 2 (selection-free confirmation)
  reeval    : finalists of every screening grid re-simulated on fresh seeds (never used for selection), all policies
              of a setting on the same seeds (CRN) so that paired differences have small standard errors.

Output: rev_abm_results.json (summaries only).  Environment: NPROC, SMALL=1 for a smoke test.
"""
import os, sys, json, time, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import multiprocessing as mp
import abm_fast as af
from run_abm_c import window_u, t_myopic

NPROC = int(os.environ.get('NPROC', 56))
SMALL = os.environ.get('SMALL', '0') == '1'
NSCREEN = 10000 if not SMALL else 200
CH = 20000 if not SMALL else 200               # games per re-evaluation chunk
NCH_MAIN = 10 if not SMALL else 2              # 200,000 games per main-setting policy
NCH_ROB = 5 if not SMALL else 1                # 100,000 games per robustness-setting policy
NEEDLE_CH = 50000 if not SMALL else 200
NEEDLE_NCH = 8 if not SMALL else 1             # 400,000 games per needle point
IV_CH = 50000 if not SMALL else 300
IV_NCH = 4 if not SMALL else 1
DELTA = 0.25
INF = 99.0

PUB = dict(R1=af.R1, RL=af.RL, MU_TH=af.MU_TH, SD_TH=af.SD_TH,
           PHI=(af.PHI_DEL_D, af.PHI_DEL_C, af.PHI_ADD_D, af.PHI_ADD_C))
EMP = json.load(open('../results/refit_params.json'))   # re-estimated model of this study (fit_behaviour_model.py)


def set_model(model):
    P = EMP if model == 'emp' else PUB
    af.R1 = tuple(P['R1']); af.RL = tuple(P['RL']); af.MU_TH = P['MU_TH']; af.SD_TH = P['SD_TH']
    af.PHI_DEL_D, af.PHI_DEL_C, af.PHI_ADD_D, af.PHI_ADD_C = P['PHI']
    af.LOG_PRIOR = -0.5 * ((af.TH_GRID - af.MU_TH) / af.SD_TH) ** 2


COND = {'base': {}, 'targeted': dict(route='targeted'), 'p0=0.15': dict(p0=0.15), 'p0=0.5': dict(p0=0.5),
        'n=24': dict(n=24), 'n=24,deg': dict(n=24, p0=0.3 * 15 / 23), 'sd=0': dict(sd_th=0.0),
        'nocap': dict(capital_rule=False)}


def planner_of(spec):
    k = spec['kind']
    if k == 'typed':
        return af.typed(np.array(spec['u'])), False
    if k == 'thr2':
        ca = -np.inf if spec['ca'] is None else spec['ca']
        cd = np.inf if spec['cd'] is None else spec['cd']
        return af.window_threshold_planner(spec['ta'], ca, spec['td'], cd, use=spec['use']), spec['use'] == 'post'
    if k == 'thr':
        return af.threshold_planner(np.array(spec['thr']), use=spec['use']), spec['use'] == 'post'
    if k == 'probation':
        return af.probation_planner(spec['strikes']), False
    if k == 'dthr':
        return af.dec_threshold_planner(spec['thrA'], spec['thrR'], use=spec['use']), spec['use'] == 'post'
    raise ValueError(k)


def sim(spec, setting, ng, seed, **extra):
    set_model(setting.get('model', 'pub'))
    pl, tp = planner_of(spec)
    kw = dict(COND[setting.get('cond', 'base')]); kw.update(extra)
    rng = np.random.default_rng(seed)
    out = []
    left = ng
    b = 0
    while left > 0:
        g = min(4000, left)
        if 'intervene' in kw:
            kw['intervene'] = dict(kw['intervene'], seed=kw['intervene']['seed'] + 7919 * b)
        out.append(af.simulate(pl, g, rng, T=setting['T'], kappa=setting['kappa'], track_posterior=tp, **kw))
        left -= g; b += 1
    return out


def cat(outs, key):
    return np.concatenate([o[key] for o in outs])


# ------------------------------------------------------------------ tasks
def t_grid(a):
    tag, setting, name, spec, seed, ng = a
    o = sim(spec, setting, ng, seed)
    f, m, c = cat(o, 'final'), cat(o, 'mcoop'), cat(o, 'capital')
    return dict(task='grid', tag=tag, setting=setting, name=name, spec=spec, n=ng,
                final=float(f.mean()), final_se=float(f.std() / np.sqrt(ng)),
                mcoop=float(m.mean()), mcoop_se=float(m.std() / np.sqrt(ng)), capital=float(c.mean()))


def t_chunk(a):
    """All policies of one re-evaluation set on the same seed (CRN); per-game arrays returned for pairing."""
    tag, setting, pols, seed, ng = a
    res = {}
    for name, spec in pols:
        o = sim(spec, setting, ng, seed)
        res[name] = dict(final=cat(o, 'final').astype(np.float32), mcoop=cat(o, 'mcoop').astype(np.float32),
                         capital=cat(o, 'capital').astype(np.float32), bind=float(np.mean([x['bind'].mean() for x in o])),
                         bind_max=float(np.max([x['bind'].max() for x in o])))
    return dict(task='chunk', tag=tag, setting=setting, res=res)


def t_needle1(a):
    setting, bname, u0, t, seed, ng = a
    u0 = np.array(u0)
    base = sim(dict(kind='typed', u=u0.tolist()), setting, ng, seed)
    fb = cat(base, 'final'); cb = cat(base, 'capital')
    out = dict(task='needle1', setting=setting, base=bname, t=t, n=ng)
    for ci, cname in ((0, 'add'), (1, 'del')):
        u = u0.copy()
        v = u[t - 1, 1, ci]
        step = -DELTA if v > 0.5 else DELTA
        u[t - 1, 1, ci] = v + step
        p = sim(dict(kind='typed', u=u.tolist()), setting, ng, seed)
        df = cat(p, 'final') - fb
        out[cname] = dict(step=step, s1=float(df.sum()), s2=float((df ** 2).sum()))
    return out


def t_tieiv(a):
    setting, pname, u, t0, seed, ng = a
    spec = dict(kind='typed', u=u)
    arms = {}
    for mode in ('none', 'cut', 'add'):
        o = sim(spec, setting, ng, seed, intervene=dict(t0=t0, seed=seed + 77, mode=mode))
        iv = {k: np.concatenate([x['iv'][k] for x in o]) for k in o[0]['iv'] if k not in ('t0', 'mode')}
        iv['final'] = cat(o, 'final')
        arms[mode] = iv
    keep = {}
    c = arms['none']
    for key in ('cut', 'add'):
        keep[key] = dict(ok=c[key + '_ok'], thD=c[key + '_thD'].astype(np.float32), stD=c[key + '_stD'].astype(np.float32),
                         aD1_ctrl=c[key + '_aD1'].astype(np.float32), aD1_trt=arms[key][key + '_aD1'].astype(np.float32),
                         aC1_ctrl=c[key + '_aC1'].astype(np.float32), aC1_trt=arms[key][key + '_aC1'].astype(np.float32),
                         fin_ctrl=c['final'].astype(np.float32), fin_trt=arms[key]['final'].astype(np.float32))
    return dict(task='tieiv', setting=setting, policy=pname, t0=t0, arrays=keep)


def t_tiestats(a):
    setting, pname, u, seed, ng = a
    T = setting['T']
    o = sim(dict(kind='typed', u=u), setting, ng, seed, tiestats=list(range(1, T)))
    rows = []
    for t in range(1, T):
        r = {k: np.concatenate([x['tiestats'][t][k] for x in o]) for k in o[0]['tiestats'][t]}
        e = r['tied'].astype(bool); h = r['d_hasC'].astype(bool)
        def m(x, msk):
            return float(x[msk].mean()) if msk.any() else float('nan')
        rows.append(dict(t=t, n_tied=int(e.sum()), n_untied=int((~e).sum()),
                         th_t=m(r['thD'], e), th_u=m(r['thD'], ~e), st_t=m(r['stD'], e), st_u=m(r['stD'], ~e),
                         pw_t=m(r['p_with'], e), pwo_t=m(r['p_without'], e), p_u=m(r['p_with'], ~e),
                         an_t=m(r['a_next_pair'], e), an_u=m(r['a_next_pair'], ~e),
                         nd_C=int(h.sum()), nd_noC=int((~h).sum()), dth_C=m(r['d_th'], h), dth_noC=m(r['d_th'], ~h),
                         dst_C=m(r['d_st'], h), dst_noC=m(r['d_st'], ~h), dan_C=m(r['a_next_def'], h), dan_noC=m(r['a_next_def'], ~h)))
    return dict(task='tiestats', setting=setting, policy=pname, seed=seed, n=ng, rounds=rows)


def t_capacc(a):
    name, u, kappa, cap, seed = a
    set_model('pub')
    th = np.load('theta_draws.npy')
    reps = 500 if not SMALL else 5
    rng = np.random.default_rng(seed)
    fin, binds = [], []
    for b in range(max(reps // 25, 1)):
        thb = np.tile(th, (25, 1))
        r = af.simulate(af.typed(np.array(u)), thb.shape[0], rng, T=15, kappa=kappa, theta=thb, capital_rule=cap)
        fin.append(r['final']); binds.append(r['bind'])
    fin = np.concatenate(fin)
    return dict(task='capacc', planner=name, kappa=kappa, capital_rule=cap, final=float(fin.mean()),
                final_se=float(fin.std() / np.sqrt(len(fin))), bind=float(np.mean(binds)), bind_max=float(np.max(binds)))


def t_ref(a):
    name, seed, ng = a
    set_model('pub')
    import sim_pilot as sp
    ref = sp.run(name, n_games=ng, seed=seed)
    f = np.array([g['final'] for g in ref])
    return dict(task='ref', planner=name, n=ng, s1=float(f.sum()), s2=float((f ** 2).sum()))


def t_fastref(a):
    name, seed, ng = a
    set_model('pub')
    m = {'static': af.uniform_u(15, 0, 0), 'random': af.uniform_u(15, .3, .3), 'neutral': af.neutral_u(15),
         'maxconn': af.uniform_u(15, 1, 0), 'encouragement': af.encouragement_u(15)}.get(name)
    if name == 'exclusion':
        m = af.encouragement_u(15); m[:, 1] = (0, 1)
    if name == 'always_conciliate':
        m = af.encouragement_u(15); m[:, 1] = (1, 0)
    r = af.run_batched(af.typed(m), ng, seed)
    f = r['final']
    return dict(task='fastref', planner=name, n=ng, s1=float(f.sum()), s2=float((f ** 2).sum()))


def t_myo(a):
    model, kappa, s, seed = a
    set_model(model)
    out = t_myopic((kappa, 15, s, seed))
    out['model'] = model
    out['task'] = 'myo'
    out.pop('rounds_full', None)
    return out


def dispatch(a):
    return {'grid': t_grid, 'chunk': t_chunk, 'needle1': t_needle1, 'tieiv': t_tieiv, 'tiestats': t_tiestats,
            'capacc': t_capacc, 'ref': t_ref, 'fastref': t_fastref, 'myo': t_myo}[a[0]](a[1:])


# ------------------------------------------------------------------ policy helpers
def W(T, ta, td):
    return dict(kind='typed', u=window_u(T, ta, td).tolist())


def P2(T, s):
    return dict(kind='typed', u=af.switch_u(T, s).tolist())


def enc_std_ccdd():
    u = af.encouragement_u(15); u[:, 2] = (1, 0); u[:, 0] = (0, 1)
    return dict(kind='typed', u=u.tolist())


def window_enc_ccdd(ta, td):
    u = window_u(15, ta, td); e = af.encouragement_u(15); u[:, 2] = e[:, 2]; u[:, 0] = e[:, 0]
    return dict(kind='typed', u=u.tolist())


def run_pool(tasks, label):
    t0 = time.time()
    res = []
    if NPROC <= 1:
        it = map(dispatch, tasks)
        pool = None
    else:
        pool = mp.Pool(NPROC)
        it = pool.imap_unordered(dispatch, tasks, chunksize=1)
    if True:
        for i, r in enumerate(it):
            res.append(r)
            if i % 200 == 0:
                print(label, i, len(tasks), round(time.time() - t0, 1), flush=True)
    if pool is not None:
        pool.close(); pool.join()
    print(label, 'done', len(res), round(time.time() - t0, 1), flush=True)
    return res


def paired_summary(chunks):
    """Merge chunk outputs of one re-evaluation set; means, SEs and all paired differences (CRN)."""
    names = list(chunks[0]['res'].keys())
    arr = {nm: {k: np.concatenate([c['res'][nm][k] for c in chunks]) for k in ('final', 'mcoop', 'capital')} for nm in names}
    bind = {nm: float(np.mean([c['res'][nm]['bind'] for c in chunks])) for nm in names}
    bmax = {nm: float(np.max([c['res'][nm]['bind_max'] for c in chunks])) for nm in names}
    n = len(arr[names[0]]['final'])
    pol = {}
    for nm in names:
        d = arr[nm]
        pol[nm] = dict(final=float(d['final'].mean()), final_se=float(d['final'].std() / np.sqrt(n)),
                       final_sd=float(d['final'].std()),
                       mcoop=float(d['mcoop'].mean()), mcoop_se=float(d['mcoop'].std() / np.sqrt(n)),
                       capital=float(d['capital'].mean()), capital_se=float(d['capital'].std() / np.sqrt(n)),
                       bind=bind[nm], bind_max=bmax[nm])
    diffs = {}
    for x, y in itertools.permutations(names, 2):
        if x < y:
            for k in ('final', 'mcoop'):
                dd = arr[x][k].astype(np.float64) - arr[y][k]
                diffs[f'{k}|{x}|{y}'] = (float(dd.mean()), float(dd.std() / np.sqrt(n)))
    return dict(n=n, policies=pol, diffs=diffs)


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    T0 = time.time()
    seed = 900000
    old = json.load(open('abm_c_results.json'))
    olda = json.load(open('abm_a_results.json'))
    grid_old = [r for r in old if r['task'] == 'window']
    sweep_old = [r for r in olda if r['kind'] == 'sweep']
    out = dict(meta=dict(NSCREEN=NSCREEN, CH=CH, NCH_MAIN=NCH_MAIN, NCH_ROB=NCH_ROB, DELTA=DELTA))

    # ---------------- stage 1
    tasks = []
    kap3 = [0.5, 0.75, 1.0]
    # robustness window grids (all windows incl. the two-phase diagonal)
    for cond in ['targeted', 'p0=0.15', 'p0=0.5', 'n=24', 'n=24,deg', 'sd=0', 'nocap']:
        for k in kap3:
            st = dict(model='pub', cond=cond, kappa=k, T=15)
            for ta in (range(15) if not SMALL else [0, 4, 14]):
                for td in (range(1, 16) if not SMALL else [1, 5, 15]):
                    tasks.append(('grid', 'robust', st, f'W{ta},{td}', W(15, ta, td), seed, NSCREEN)); seed += 1
    # re-estimated model at the ends of the kappa CI
    for k in [0.808, 0.910]:
        st = dict(model='emp', cond='base', kappa=k, T=15)
        for ta in (range(15) if not SMALL else [0, 4]):
            for td in (range(1, 16) if not SMALL else [5, 15]):
                tasks.append(('grid', 'emp', st, f'W{ta},{td}', W(15, ta, td), seed, NSCREEN)); seed += 1
    # nested decoupled oracle family
    TA = [0, 1, 2, 3, 4, 5, 6, 8, 10, 14]; TD = [1, 6, 9, 10, 11, 12, 13, 14, 15]
    CA = [None, -6, -5, -4, -3, -2, -1, 0]; CDv = [None, -6, -5, -4, -3, -2, -1, 0]
    if SMALL:
        TA, TD, CA, CDv = [0, 4], [14, 15], [None, -3], [None, -3]
    for k in kap3:
        st = dict(model='pub', cond='base', kappa=k, T=15)
        for ta in TA:
            for ca in (CA if ta > 0 else [None]):
                for td in TD:
                    for cd in (CDv if td < 15 else [None]):
                        spec = dict(kind='thr2', ta=ta, ca=ca, td=td, cd=cd, use='oracle')
                        tasks.append(('grid', 'thr2', st, f'D{ta},{ca},{td},{cd}', spec, seed, NSCREEN)); seed += 1
    # widened coupled linear thresholds (oracle and posterior)
    c0s = np.arange(-8, 3.01, 0.5) if not SMALL else [-4, 0]
    c1s = np.arange(-4, 8.01, 1.0) if not SMALL else [0]
    for k in kap3:
        st = dict(model='pub', cond='base', kappa=k, T=15)
        for use in ['oracle', 'post']:
            for c0 in c0s:
                for c1 in c1s:
                    thr = (c0 + c1 * (np.arange(1, 15) - 1) / 13).tolist()
                    tasks.append(('grid', 'thr', st, f'L{use},{c0:g},{c1:g}', dict(kind='thr', thr=thr, use=use), seed, NSCREEN)); seed += 1
    # one-sided perturbations
    bases = [(1.0, 'P14', af.switch_u(15, 14)), (1.0, 'W4,15', window_u(15, 4, 15)),
             (0.75, 'P9', af.switch_u(15, 9)), (0.75, 'W0,14', window_u(15, 0, 14))]
    for k, bn, u0 in bases:
        st = dict(model='pub', cond='base', kappa=k, T=15)
        for t in range(1, 15):
            for c in range(NEEDLE_NCH):
                tasks.append(('needle1', st, bn, u0.tolist(), t, 300000 + 1000 * c + t + int(100 * k), NEEDLE_CH))
    # interventional tie values
    ivp = [(1.0, 'W4,15', window_u(15, 4, 15)), (1.0, 'P14', af.switch_u(15, 14)),
           (0.75, 'W0,14', window_u(15, 0, 14)), (0.5, 'P0', af.switch_u(15, 0))]
    for k, pn, u in ivp:
        st = dict(model='pub', cond='base', kappa=k, T=15)
        for t0 in [3, 5, 7, 10, 13]:
            for c in range(IV_NCH):
                tasks.append(('tieiv', st, pn, u.tolist(), t0, seed, IV_CH)); seed += 1
    # tie-state statistics (10 independent chunks of 2,000 games for game-clustered SEs)
    for k, pn, u in ivp + [(0.5, 'P14', af.switch_u(15, 14))]:
        st = dict(model='pub', cond='base', kappa=k, T=15)
        for c in range(10):
            tasks.append(('tiestats', st, pn, u.tolist(), seed, 2000 if not SMALL else 100)); seed += 1
    # capital-constraint ablation on the paired disposition draws
    T = 15
    plan = {'static': af.uniform_u(T, 0, 0), 'random': af.uniform_u(T, .3, .3), 'neutral': af.neutral_u(T),
            'maxconn': af.uniform_u(T, 1, 0), 'encouragement': af.encouragement_u(T),
            'exclusion': af.switch_u(T, 0), 'conciliation': af.switch_u(T, T - 1), 'switch_T-2': af.switch_u(T, T - 2)}
    for k in kap3:
        for nm, u in plan.items():
            for cap in (True, False):
                tasks.append(('capacc', nm, u.tolist(), k, cap, seed)); seed += 1
    # reference simulator
    for nm in ['static', 'random', 'neutral', 'maxconn', 'encouragement', 'exclusion', 'always_conciliate']:
        for c in range(10 if not SMALL else 1):
            tasks.append(('ref', nm, 50000 + 100 * c + len(nm), 5000 if not SMALL else 50))
        for c in range(4 if not SMALL else 1):
            tasks.append(('fastref', nm, 60000 + 100 * c + len(nm), 50000 if not SMALL else 200))
    # myopic threshold under the re-estimated model
    for s in (13, 0):
        tasks.append(('myo', 'emp', EMP['kappa'], s, seed)); seed += 1
    pri = {'ref': 0, 'tieiv': 1, 'needle1': 2, 'tiestats': 3, 'capacc': 3, 'myo': 3, 'fastref': 4, 'grid': 5}
    tasks.sort(key=lambda a: pri[a[0]])
    print('stage1 tasks', len(tasks), flush=True)
    res1 = run_pool(tasks, 'S1')

    # ---- summarise stage 1 (small outputs)
    grid = [r for r in res1 if r['task'] == 'grid']
    out['grid'] = [{k: v for k, v in r.items() if k != 'spec'} | ({'spec': r['spec']} if r['tag'] in ('thr2',) else {}) for r in grid]
    nd = {}
    for r in [r for r in res1 if r['task'] == 'needle1']:
        key = (r['setting']['kappa'], r['base'], r['t'])
        d = nd.setdefault(key, dict(n=0, add=[0, 0, 0], dele=[0, 0, 0]))
        d['n'] += r['n']
        d['add'][0] += r['add']['s1']; d['add'][1] += r['add']['s2']; d['add'][2] = r['add']['step']
        d['dele'][0] += r['del']['s1']; d['dele'][1] += r['del']['s2']; d['dele'][2] = r['del']['step']
    out['needle1'] = []
    for (k, bn, t), d in sorted(nd.items()):
        row = dict(kappa=k, base=bn, t=t, n=d['n'])
        for c in ('add', 'dele'):
            s1, s2, step = d[c]; m = s1 / d['n']; sd = np.sqrt(max(s2 / d['n'] - m * m, 0))
            row[c] = dict(step=step, dJ=m, se=sd / np.sqrt(d['n']), slope=m / step, slope_se=sd / np.sqrt(d['n']) / abs(step))
        out['needle1'].append(row)
    # interventions: merge chunks, effects overall and by disposition bin / streak
    ivs = {}
    for r in [r for r in res1 if r['task'] == 'tieiv']:
        key = (r['setting']['kappa'], r['policy'], r['t0'])
        ivs.setdefault(key, []).append(r['arrays'])
    bins = [-np.inf, -6, -4, -2, 0, np.inf]
    out['tieiv'] = []
    for (k, pn, t0), lst in sorted(ivs.items()):
        row = dict(kappa=k, policy=pn, t0=t0)
        for key in ('cut', 'add'):
            A_ = {f: np.concatenate([x[key][f] for x in lst]) for f in lst[0][key]}
            ok = A_['ok'].astype(bool)
            sgn = 1.0 if key == 'add' else -1.0          # value of having the tie
            dA = sgn * (A_['aD1_trt'] - A_['aD1_ctrl'])[ok]; dC = sgn * (A_['aC1_trt'] - A_['aC1_ctrl'])[ok]
            dF = sgn * (A_['fin_trt'] - A_['fin_ctrl'])[ok].astype(np.float64)
            th = A_['thD'][ok]; stv = A_['stD'][ok]
            def ms(x):
                return [float(x.mean()), float(x.std() / np.sqrt(max(len(x), 1)))] if len(x) else [float('nan'), float('nan')]
            rr = dict(n=int(ok.sum()), frac_ok=float(ok.mean()), thD=float(th.mean()), stD=float(stv.mean()),
                      aD1_ctrl=float(A_['aD1_ctrl'][ok].mean()), dDef=ms(dA), dCoop=ms(dC), dFinal=ms(dF), dFinal16=ms(16 * dF),
                      by_theta=[], by_streak=[])
            for lo, hi in zip(bins[:-1], bins[1:]):
                mk = (th >= lo) & (th < hi)
                rr['by_theta'].append(dict(lo=float(lo), hi=float(hi), n=int(mk.sum()), dDef=ms(dA[mk]), dFinal16=ms(16 * dF[mk])))
            for sv in [1, 2, 3, 5, 100]:
                mk = (stv <= sv) & (stv > {1: 0, 2: 1, 3: 2, 5: 3, 100: 5}[sv])
                rr['by_streak'].append(dict(max=sv, n=int(mk.sum()), dDef=ms(dA[mk]), dFinal16=ms(16 * dF[mk])))
            row[key] = rr
        out['tieiv'].append(row)
    # tie-state statistics: mean and SE over the 10 chunks
    tsr = {}
    for r in [r for r in res1 if r['task'] == 'tiestats']:
        tsr.setdefault((r['setting']['kappa'], r['policy']), []).append(r['rounds'])
    out['tiestats'] = []
    for (k, pn), lst in sorted(tsr.items()):
        rows = []
        for ti in range(len(lst[0])):
            keys = [kk for kk in lst[0][ti] if kk != 't']
            row = dict(t=lst[0][ti]['t'])
            for kk in keys:
                v = np.array([l[ti][kk] for l in lst], float)
                row[kk] = [float(np.nanmean(v)), float(np.nanstd(v) / np.sqrt(len(v)))]
            rows.append(row)
        out['tiestats'].append(dict(kappa=k, policy=pn, rounds=rows))
    out['capacc'] = [r for r in res1 if r['task'] == 'capacc']
    rf = {}
    for r in [r for r in res1 if r['task'] in ('ref', 'fastref')]:
        d = rf.setdefault((r['planner'], r['task']), [0, 0, 0]); d[0] += r['n']; d[1] += r['s1']; d[2] += r['s2']
    out['ref'] = []
    for nm in sorted(set(k[0] for k in rf)):
        row = dict(planner=nm)
        for kind in ('ref', 'fastref'):
            n, s1, s2 = rf[(nm, kind)]; m = s1 / n
            row[kind] = [m, float(np.sqrt(max(s2 / n - m * m, 0)) / np.sqrt(n)), n]
        out['ref'].append(row)
    out['myopic_emp'] = [r for r in res1 if r['task'] == 'myo']
    json.dump(out, open('rev_abm_results.json', 'w'))
    print('stage1 saved', round(time.time() - T0, 1), flush=True)

    # ---------------- stage 2: selection-free re-evaluation with CRN
    sets = []     # (tag, setting, [(name, spec)], nchunks)
    def top(rows, key, k):
        return sorted(rows, key=lambda r: -r[key])[:k]
    main = [(0.5, 15), (0.625, 15), (0.75, 8), (0.75, 15), (0.75, 25), (0.875, 15), (1.0, 8), (1.0, 15), (1.0, 25), (1.25, 15)]
    thr2 = [r for r in grid if r['tag'] == 'thr2']
    thrl = [r for r in grid if r['tag'] == 'thr']
    for k, T in main:
        st = dict(model='pub', cond='base', kappa=k, T=T)
        g = [r for r in grid_old if r['kappa'] == k and r['T'] == T]
        sw = [r for r in sweep_old if r['kappa'] == k and r['T'] == T]
        pols = {}
        for r in top(g, 'final', 5):
            pols[f"W{r['ta']},{r['td']}"] = W(T, r['ta'], r['td'])
        for r in top([r for r in g if r['ta'] < r['td']], 'final', 2):
            pols[f"W{r['ta']},{r['td']}"] = W(T, r['ta'], r['td'])
        for r in top(g, 'mcoop', 2):
            pols[f"W{r['ta']},{r['td']}"] = W(T, r['ta'], r['td'])
        for r in top(sw, 'final', 3) + top(sw, 'mcoop', 1):
            pols[f"P{r['s']}"] = P2(T, r['s'])
        pols['P0'] = P2(T, 0); pols[f'P{T - 1}'] = P2(T, T - 1)
        if T == 15:
            pols['enc'] = dict(kind='typed', u=af.encouragement_u(15).tolist())
            if k in kap3:
                pols['enc_stdCCDD'] = enc_std_ccdd()
                bw = top(g, 'final', 1)[0]
                pols[f"W{bw['ta']},{bw['td']}_encCCDD"] = window_enc_ccdd(bw['ta'], bw['td'])
                for strikes in (1, 3):
                    pols[f'prob{strikes}'] = dict(kind='probation', strikes=strikes)
                # nested decoupled oracle finalists (and their posterior twins), best coupled linear thresholds
                tk = top([r for r in thr2 if r['setting']['kappa'] == k], 'final', 6)
                for r in tk:
                    sp_ = dict(r['spec']); pols[r['name']] = sp_
                    sq = dict(sp_); sq['use'] = 'post'; pols[r['name'] + 'post'] = sq
                for use in ('oracle', 'post'):
                    rl = top([r for r in thrl if r['setting']['kappa'] == k and r['name'].startswith(f'L{use}')], 'final', 2)
                    for r in rl:
                        c0, c1 = map(float, r['name'].split(',')[1:])
                        thr = (c0 + c1 * (np.arange(1, 15) - 1) / 13).tolist()
                        pols[r['name']] = dict(kind='thr', thr=thr, use=use)
        sets.append(('main', st, list(pols.items()), NCH_MAIN))
    st = dict(model='emp', cond='base', kappa=EMP['kappa'], T=15)
    pols = {f'W{a},{b}': W(15, a, b) for a, b in [(2, 12), (1, 12), (10, 12), (0, 12), (2, 13)]}
    pols.update({f'P{x}': P2(15, x) for x in (9, 10, 11, 0, 14)})
    pols['enc'] = dict(kind='typed', u=af.encouragement_u(15).tolist()); pols['enc_stdCCDD'] = enc_std_ccdd()
    sets.append(('emp_main', st, list(pols.items()), NCH_MAIN))
    rob = [r for r in grid if r['tag'] in ('robust', 'emp')]
    for key in sorted(set((r['setting']['model'], r['setting']['cond'], r['setting']['kappa']) for r in rob)):
        model, cond, k = key
        st = dict(model=model, cond=cond, kappa=k, T=15)
        g = [r for r in rob if (r['setting']['model'], r['setting']['cond'], r['setting']['kappa']) == key]
        pols = {}
        for r in top(g, 'final', 3):
            pols[r['name']] = W(15, *map(int, r['name'][1:].split(',')))
        diag = [r for r in g if (lambda ta, td: td == ta + 1)(*map(int, r['name'][1:].split(',')))]
        for r in top(diag, 'final', 2):
            ta, td = map(int, r['name'][1:].split(',')); pols[f'P{ta}'] = P2(15, ta)
        pols['P0'] = P2(15, 0); pols['P14'] = P2(15, 14)
        pols['enc'] = dict(kind='typed', u=af.encouragement_u(15).tolist())
        sets.append(('robust' if model == 'pub' else 'emp', st, list(pols.items()), NCH_ROB))
    tasks = []
    for si, (tag, st, pols, nch) in enumerate(sets):
        for c in range(nch):
            tasks.append(('chunk', f'{si}', st, pols, 5000000 + 1000 * si + c, CH))
    tasks.sort(key=lambda a: -len(a[3]))
    print('stage2 tasks', len(tasks), flush=True)
    res2 = run_pool(tasks, 'S2')
    out['reeval'] = []
    for si, (tag, st, pols, nch) in enumerate(sets):
        ch = [r for r in res2 if r['tag'] == f'{si}']
        s = paired_summary(ch)
        s.update(tag=tag, setting=st, specs={nm: sp_ for nm, sp_ in pols if sp_['kind'] != 'typed'})
        out['reeval'].append(s)
    json.dump(out, open('rev_abm_results.json', 'w'))
    print('ALL DONE', round(time.time() - T0, 1), flush=True)
