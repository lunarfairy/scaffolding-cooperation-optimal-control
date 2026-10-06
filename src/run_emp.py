"""Robustness: optimal C-D schedules under the behaviour model re-estimated from all seven human
conditions of McKee et al. (2023) (pooled M1 fit of a companion analysis; parameters in emp_params.json)."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, multiprocessing as mp
import abm_fast as af
from run_abm_c import window_u, stats

PAR = json.load(open('emp_params.json'))
NG = int(os.environ.get('NG', 30000)); NE = int(os.environ.get('NE', 100000))


def setp():
    af.R1 = tuple(PAR['R1']); af.RL = tuple(PAR['RL']); af.MU_TH = PAR['MU_TH']; af.SD_TH = PAR['SD_TH']
    af.PHI_DEL_D, af.PHI_DEL_C, af.PHI_ADD_D, af.PHI_ADD_C = PAR['PHI']


def run(a):
    setp()
    kind, name, u, seed, ng, kap = a
    r = af.run_batched(af.typed(np.array(u)), ng, seed, T=15, kappa=kap)
    out = dict(kind=kind, name=name, kappa=kap, n=ng); out.update(stats(r, ng))
    out['coop'] = r['coop'].mean(0).tolist()
    return out


if __name__ == '__main__':
    t0 = time.time(); K = PAR['kappa']
    tasks = []; seed = 7000
    for s in range(15):
        tasks.append(('twophase', f's={s}', af.switch_u(15, s).tolist(), seed, NG, K)); seed += 1
    for ta in range(15):
        for td in range(1, 16):
            tasks.append(('window', f'{ta},{td}', window_u(15, ta, td).tolist(), seed, NG, K)); seed += 1
    for nm, u in [('encouragement', af.encouragement_u(15)), ('exclusion', af.switch_u(15, 0)), ('conciliation', af.switch_u(15, 14)),
                  ('static', af.uniform_u(15, 0, 0))]:
        tasks.append(('baseline', nm, u.tolist(), seed, NE, K)); seed += 1
    with mp.Pool(int(os.environ.get('NPROC', 48))) as pool:
        res = pool.map(run, tasks, chunksize=1)
    # winner's-curse-free re-evaluation with fresh seeds
    tp = sorted([r for r in res if r['kind'] == 'twophase'], key=lambda r: -r['final'])[:2]
    wi = sorted([r for r in res if r['kind'] == 'window'], key=lambda r: -r['final'])[:3]
    re_tasks = []
    for r in tp:
        s = int(r['name'][2:]); re_tasks.append(('re_twophase', r['name'], af.switch_u(15, s).tolist(), seed, NE, K)); seed += 1
    for r in wi:
        ta, td = map(int, r['name'].split(',')); re_tasks.append(('re_window', r['name'], window_u(15, ta, td).tolist(), seed, NE, K)); seed += 1
    with mp.Pool(int(os.environ.get('NPROC', 48))) as pool:
        res += pool.map(run, re_tasks, chunksize=1)
    json.dump(dict(params=PAR, results=res), open('emp_results.json', 'w'))
    print('done', len(res), round(time.time() - t0, 1))
