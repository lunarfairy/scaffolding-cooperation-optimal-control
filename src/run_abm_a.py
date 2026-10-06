"""Job A: (1) fast-vs-reference simulator check, (2) planner trajectories for IPA accuracy,
(3) two-phase switch sweep over (kappa, T, s). Parallel over tasks with multiprocessing."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import multiprocessing as mp
import abm_fast as af

NG_SWEEP = int(os.environ.get('NG_SWEEP', 50000))
NG_TRAJ = int(os.environ.get('NG_TRAJ', 50000))
NPROC = int(os.environ.get('NPROC', 96))


def planner_set(T):
    return {
        'static': af.uniform_u(T, 0.0, 0.0),
        'random': af.uniform_u(T, 0.3, 0.3),
        'neutral': af.neutral_u(T),
        'maxconn': af.uniform_u(T, 1.0, 0.0),
        'encouragement': af.encouragement_u(T),
        'exclusion': af.switch_u(T, 0),
        'conciliation': af.switch_u(T, T - 1),
        'switch_T-2': af.switch_u(T, T - 2),
    }


def task_traj(args):
    name, kappa, T, seed = args
    u = planner_set(T)[name]
    r = af.run_batched(af.typed(u), NG_TRAJ, seed, T=T, kappa=kappa)
    return dict(kind='traj', planner=name, kappa=kappa, T=T,
                coop=r['coop'].mean(0).tolist(), coop_se=(r['coop'].std(0) / np.sqrt(NG_TRAJ)).tolist(),
                ecc=r['ecc'].mean(0).tolist(), ecd=r['ecd'].mean(0).tolist(), edd=r['edd'].mean(0).tolist(),
                pay=r['pay'].mean(0).tolist(), capital=float(r['capital'].mean()),
                bind=r['bind'].tolist())


def task_sweep(args):
    kappa, T, s, seed = args
    r = af.run_batched(af.typed(af.switch_u(T, s)), NG_SWEEP, seed, T=T, kappa=kappa)
    n = NG_SWEEP
    return dict(kind='sweep', kappa=kappa, T=T, s=s,
                final=float(r['final'].mean()), final_se=float(r['final'].std() / np.sqrt(n)),
                mcoop=float(r['mcoop'].mean()), mcoop_se=float(r['mcoop'].std() / np.sqrt(n)),
                capital=float(r['capital'].mean()), capital_se=float(r['capital'].std() / np.sqrt(n)))


def task_ref(args):
    name, seed = args
    import sim_pilot as sp
    ng = 3000
    ref = sp.run(name, n_games=ng, seed=seed)
    f = np.array([g['final'] for g in ref]); cpt = np.array([g['capital'] for g in ref])
    m = {'static': af.uniform_u(15, 0, 0), 'random': af.uniform_u(15, .3, .3), 'neutral': af.neutral_u(15),
         'maxconn': af.uniform_u(15, 1, 0), 'encouragement': af.encouragement_u(15),
         'exclusion': None, 'always_conciliate': None}[name]
    if name == 'exclusion':
        m = af.encouragement_u(15); m[:, 1] = (0, 1)
    if name == 'always_conciliate':
        m = af.encouragement_u(15); m[:, 1] = (1, 0)
    r = af.run_batched(af.typed(m), 30000, seed + 1000)
    return dict(kind='ref', planner=name, ref_final=float(f.mean()), ref_final_se=float(f.std() / np.sqrt(ng)),
                fast_final=float(r['final'].mean()), fast_final_se=float(r['final'].std() / np.sqrt(30000)),
                ref_capital=float(cpt.mean()), fast_capital=float(r['capital'].mean()))


def dispatch(a):
    kind = a[0]
    return {'traj': task_traj, 'sweep': task_sweep, 'ref': task_ref}[kind](a[1:])


if __name__ == '__main__':
    t0 = time.time()
    tasks = []
    seed = 1
    for name in ['static', 'random', 'neutral', 'maxconn', 'encouragement', 'exclusion', 'always_conciliate']:
        tasks.append(('ref', name, seed)); seed += 1
    for kappa in [0.5, 0.75, 1.0]:
        for name in planner_set(15):
            tasks.append(('traj', name, kappa, 15, seed)); seed += 1
    for kappa in [0.25, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25]:
        for T in [8, 15, 25]:
            for s in range(T):
                tasks.append(('sweep', kappa, T, s, seed)); seed += 1
    # longest first
    tasks.sort(key=lambda a: -(a[2] if a[0] == 'sweep' else (40 if a[0] == 'ref' else 15)))
    with mp.Pool(NPROC) as pool:
        res = pool.map(dispatch, tasks, chunksize=1)
    with open('abm_a_results.json', 'w') as f:
        json.dump(res, f)
    print('done', len(res), 'tasks in', round(time.time() - t0, 1), 's')
