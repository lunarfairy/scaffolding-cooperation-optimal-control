"""Agent-based analyses under the behaviour model re-estimated in this study (fit_behaviour_model.py).

Stage 1 (screen): all 225 recruit/retain windows W(ta, td) (which contain the 15 two-phase schedules
pi_s = W(s, s+1)) and the encouragement planner, 30,000 games each, at the estimated kappa and at the two ends
of its bootstrap interval; common random numbers within each setting. Myopic one-step tie values along exclusion
(pi_0) and conciliation (pi_14) trajectories at the estimated kappa.
Stage 2 (fresh): finalists (best five windows by final cooperation, best two by mean cooperation, best three
two-phase schedules, exclusion, conciliation, encouragement) on 200,000 fresh games (10 chunks of 20,000), CRN.
Usage (from 06_code/src, after fit_behaviour_model.py):
   python run_refit_abm.py work STAGE I N     worker I of N for stage 'screen' or 'fresh'
   python run_refit_abm.py combine STAGE       merge worker outputs (combine screen before running fresh)
Outputs in ../results/refit_abm/.
"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here); sys.path.insert(0, here)
import abm_fast as af
OUT = '../results/refit_abm'; os.makedirs(OUT, exist_ok=True)
FIT = json.load(open('../results/behaviour_model_refit.json'))
af.R1 = (FIT['b00'], FIT['b01']); af.RL = (FIT['beta0'], FIT['beta_s'], FIT['beta_n'], FIT['beta_r'])
af.MU_TH = FIT['mu_theta_fixed']; af.SD_TH = FIT['sigma']
acc = FIT['acceptance']
af.PHI_DEL_D, af.PHI_DEL_C, af.PHI_ADD_D, af.PHI_ADD_C = acc['del_D'][0], acc['del_C'][0], acc['add_D'][0], acc['add_C'][0]
af.LOG_PRIOR = -0.5 * ((af.TH_GRID - af.MU_TH) / af.SD_TH) ** 2
K_HAT = FIT['kappa']; K_LO, K_HI = FIT['boot_95']['kappa']
KAPPAS = {'hat': K_HAT, 'lo': K_LO, 'hi': K_HI}
T = 15; NSCREEN = 30000; CH = 20000; NCH = 10

def window_u(ta, td):
    u = np.zeros((T - 1, 3, 2)); u[:, 2, 0] = 1.0; u[:, 0, 1] = 1.0
    for t in range(1, T):
        u[t - 1, 1, 0] = 1.0 if t <= ta else 0.0
        u[t - 1, 1, 1] = 1.0 if t >= td else 0.0
    return u

def policy(name):
    if name == 'enc': return af.encouragement_u(T)
    ta, td = map(int, name[1:].split(','))
    return window_u(ta, td)

def run(name, kappa, ng, seed):
    r = af.run_batched(af.typed(policy(name)), ng, seed, batch=4000, T=T, kappa=kappa)
    return r['final'].astype(np.float32), r['mcoop'].astype(np.float32), r['capital'].astype(np.float32)

def screen_tasks():
    names = [f'W{ta},{td}' for ta in range(0, T) for td in range(1, T + 1)] + ['enc']
    tasks = [('screen', kk, nm) for kk in KAPPAS for nm in names]
    tasks += [('myopic', 'hat', s) for s in (0, T - 1)]
    return tasks

def fresh_tasks():
    fin = json.load(open(f'{OUT}/finalists.json'))
    return [('fresh', kk, c) for kk in fin for c in range(NCH)]

def do(task):
    kind, kk, x = task
    kappa = KAPPAS[kk]
    if kind == 'screen':
        f, m, c = run(x, kappa, NSCREEN, 1000 + list(KAPPAS).index(kk))
        return dict(kind=kind, k=kk, name=x, final=float(f.mean()), final_se=float(f.std() / np.sqrt(len(f))),
                    mcoop=float(m.mean()), capital=float(c.mean()))
    if kind == 'myopic':
        import run_abm_c as rc
        out = rc.t_myopic((kappa, T, x, 777 + x)); out['k'] = kk
        return out
    fin = json.load(open(f'{OUT}/finalists.json'))[kk]
    seed = 500000 + 1000 * list(KAPPAS).index(kk) + x
    res = {nm: [a.tolist() for a in run(nm, kappa, CH, seed)] for nm in fin}
    return dict(kind=kind, k=kk, chunk=x, res=res)

def combine_screen():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith('screen_w'): rows += json.load(open(f'{OUT}/{f}'))
    scr = [r for r in rows if r.get('kind') == 'screen']; myo = [r for r in rows if r.get('task') == 'myopic']
    json.dump(scr, open(f'{OUT}/screen.json', 'w')); json.dump(myo, open(f'{OUT}/myopic.json', 'w'))
    fin = {}
    for kk in KAPPAS:
        S = {r['name']: r for r in scr if r['k'] == kk}
        win = sorted([n for n in S if n != 'enc'], key=lambda n: -S[n]['final'])
        tp = sorted([f'W{s},{s + 1}' for s in range(T)], key=lambda n: -S[n]['final'])
        wm = sorted([n for n in S if n != 'enc'], key=lambda n: -S[n]['mcoop'])
        sel = list(dict.fromkeys(win[:5] + wm[:2] + tp[:3] + ['W0,1', f'W{T - 1},{T}', 'enc']))
        fin[kk] = sel
    json.dump(fin, open(f'{OUT}/finalists.json', 'w'), indent=1)
    print(json.dumps(fin, indent=1))

def combine_fresh():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith('fresh_w'): rows += json.load(open(f'{OUT}/{f}'))
    out = {}
    for kk in KAPPAS:
        ch = sorted([r for r in rows if r['k'] == kk], key=lambda r: r['chunk'])
        names = list(ch[0]['res'])
        arr = {nm: np.array([np.concatenate([np.array(c['res'][nm][i]) for c in ch]) for i in range(3)]) for nm in names}
        n = arr[names[0]].shape[1]
        pol = {nm: dict(final=float(a[0].mean()), final_se=float(a[0].std() / np.sqrt(n)), final_sd=float(a[0].std()),
                        mcoop=float(a[1].mean()), mcoop_se=float(a[1].std() / np.sqrt(n)), capital=float(a[2].mean())) for nm, a in arr.items()}
        tps = [nm for nm in names if nm.startswith('W') and int(nm[1:].split(',')[1]) == int(nm[1:].split(',')[0]) + 1]
        wins = [nm for nm in names if nm.startswith('W')]
        bw = max(wins, key=lambda nm: pol[nm]['final']); bt = max(tps, key=lambda nm: pol[nm]['final'])
        def pdiff(x, y, i=0):
            d = arr[x][i] - arr[y][i]; return float(d.mean()), float(d.std() / np.sqrt(n))
        bm = max(['W0,1', f'W{T - 1},{T}', 'enc'] + wins, key=lambda nm: pol[nm]['mcoop'])
        out[kk] = dict(kappa=KAPPAS[kk], n=n, policies=pol, best_window=bw, best_twophase=bt,
                       gain=pdiff(bw, bt), gain_enc=pdiff(bw, 'enc'), best_by_mean=bm,
                       excl_minus_window_mean=pdiff('W0,1', bw, 1))
    json.dump(out, open(f'{OUT}/fresh_summary.json', 'w'), indent=1)
    print(json.dumps({k: dict(best_window=v['best_window'], win=v['policies'][v['best_window']]['final'], best_tp=v['best_twophase'],
                              tp=v['policies'][v['best_twophase']]['final'], gain=v['gain'], gain_enc=v['gain_enc'],
                              enc=v['policies']['enc']['final'], excl=v['policies']['W0,1']['final'], best_by_mean=v['best_by_mean'],
                              excl_minus_window_mean=v['excl_minus_window_mean']) for k, v in out.items()}, indent=1))

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'work':
        stage, i, N = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        tasks = screen_tasks() if stage == 'screen' else fresh_tasks()
        res = [do(t) for j, t in enumerate(tasks) if j % N == i]
        json.dump(res, open(f'{OUT}/{stage}_w{i:02d}.json', 'w'))
    elif mode == 'combine':
        combine_screen() if sys.argv[2] == 'screen' else combine_fresh()
