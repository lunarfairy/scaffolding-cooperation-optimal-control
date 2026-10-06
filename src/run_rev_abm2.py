"""Agent-based evaluation of the converged reduced-model controls (L-BFGS-B, all starts), the memory-depth-1
optimum, the decoupled disposition-threshold policies, and their comparators, on fresh seeds with common random
numbers (CRN) within each setting.  Reads oc_*.json, ipa_fb2.json and confirm.json (finalists from stage 2 of
run_rev_abm.py).  Output: rev_abm2_results.json."""
import os, sys, json, glob, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import abm_fast as af
from run_abm_c import window_u
import run_rev_abm as R

SMALL = os.environ.get('SMALL', '0') == '1'
CH = 20000 if not SMALL else 200
NCH = 10 if not SMALL else 1


def rounded(u):
    u = (np.array(u) > 0.5).astype(float)
    u[0, 2] = (1, 0); u[0, 0] = (0, 1)
    return u


if __name__ == '__main__':
    T0 = time.time()
    oc = [json.load(open(f)) for f in sorted(glob.glob('oc_*_d*.json'))]
    conf = json.load(open('confirm.json'))          # {"kappa|T": {"window": [ta, td], "tp": s}}
    fb2 = json.load(open('ipa_fb2.json')) if os.path.exists('ipa_fb2.json') else []
    sets = []
    keys = sorted(set((r['obj'], r['T'], r['kappa']) for r in oc if r['depth'] == 3))
    for obj, T, k in keys:
        st = dict(model='pub', cond='base', kappa=k, T=T)
        recs = sorted([r for r in oc if (r['obj'], r['T'], r['kappa'], r['depth']) == (obj, T, k, 3)], key=lambda r: -r['J'])
        pols = {}
        for r in recs:
            pols[f"ipa_{r['start']}"] = dict(kind='typed', u=r['u'])
        b = recs[0]
        pols[f"ipa_{b['start']}_rounded"] = dict(kind='typed', u=rounded(b['u']).tolist())
        if obj == 'final':
            for r in [r for r in oc if (r['obj'], r['T'], r['kappa'], r['depth']) == (obj, T, k, 1)]:
                pols[f"ipa_m1_{r['start']}"] = dict(kind='typed', u=r['u'])
        c = conf.get(f'{k}|{T}')
        if c:
            pols[f"W{c['window'][0]},{c['window'][1]}"] = R.W(T, *c['window'])
            pols[f"P{c['tp']}"] = R.P2(T, c['tp'])
        if T == 15:
            pols['enc'] = dict(kind='typed', u=af.encouragement_u(15).tolist())
        if obj == 'final' and T == 15:
            for f in [f for f in fb2 if abs(f['kappa'] - k) < 1e-9]:
                for use in ('oracle', 'post'):
                    pols[f"dthr_{f['init']}_{use}"] = dict(kind='dthr', thrA=f['thrA'], thrR=f['thrR'], use=use)
        sets.append((f'{obj}', st, list(pols.items())))
    tasks = []
    for si, (tag, st, pols) in enumerate(sets):
        for c_ in range(NCH):
            tasks.append(('chunk', f'{si}', st, pols, 7000000 + 1000 * si + c_, CH))
    tasks.sort(key=lambda a: -len(a[3]))
    print('tasks', len(tasks), flush=True)
    res = R.run_pool(tasks, 'A2')
    out = []
    for si, (tag, st, pols) in enumerate(sets):
        ch = [r for r in res if r['tag'] == f'{si}']
        s = R.paired_summary(ch)
        s.update(obj=tag, setting=st)
        out.append(s)
    json.dump(out, open('rev_abm2_results.json', 'w'))
    print('ALL DONE', round(time.time() - T0, 1), flush=True)
