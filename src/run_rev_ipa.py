"""Revision computations in the reduced model (IPA).

  oc <obj> <T> <kappa> <start> [depth]
        Box-constrained optimisation of the open-loop pair-type controls by L-BFGS-B (scipy) on exact
        autograd gradients, run to convergence (ftol 1e-13, pgtol 1e-8, at most MAXFUN evaluations).
        Starts: half, enc, window (best agent-based window), tp (best two-phase schedule), adam_old (the
        80-iteration projected-Adam solution of the original submission), r1, r2 (uniform random).
        Reports J, the projected-gradient (KKT) residual, active-set classification and the history.
  post  Second-order check on the free coordinates (finite-difference Hessian of J restricted to controls
        strictly inside (0, 1)), re-optimisation on two further disposition samples, and evaluation of every
        optimum on an independent sample of 256 disposition draws.
  costate
        Tie values along two-phase schedules for T = 8, 15, 25, 40: the switching functions divided by their
        positive weights give the mean tie value over untied (Lambda0) and tied (Lambda1) C-D pairs.
        Includes the two-phase sweep at T = 40.
  fb2   Decoupled disposition-threshold policy (separate recruitment and retention thresholds per round;
        oracle dispositions), optimised with soft thresholds.
  gradbases
        Switching functions at the bases used for the one-sided agent-based perturbations.
Environment: IPA_DEVICE (cuda), IPA_G (32), MAXFUN (300).
"""
import os, sys, json, time, glob, math
import numpy as np
import torch
import ipa
from scipy.optimize import minimize

dev = os.environ.get('IPA_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
DT = ipa.DT
G0 = int(os.environ.get('IPA_G', 32))
MAXFUN = int(os.environ.get('MAXFUN', 300))
SMALL = os.environ.get('IPA_SMALL', '0') == '1'
torch.set_num_threads(int(os.environ.get('IPA_THREADS', 4)))
STARTS = json.load(open('oc_starts.json'))
EPS = 1e-6


def theta(G, seed=12345):
    return ipa.draw_theta(G, 16, seed=seed).to(dev)


def objective(o, name):
    return {'final': o['final'], 'capital': o['capital'], 'mcoop': o['mcoop']}[name]


def window_u(T, ta, td):
    u = np.zeros((T - 1, 3, 2)); u[:, 2, 0] = 1; u[:, 0, 1] = 1
    for t in range(1, T):
        u[t - 1, 1, 0] = 1.0 if t <= ta else 0.0
        u[t - 1, 1, 1] = 1.0 if t >= td else 0.0
    return u


def switch_u(T, s):
    return window_u(T, s, s + 1)


def enc_u(T):
    import abm_fast as af
    e = af.encouragement_u(15)
    if T != 15:
        e = e[np.round(np.linspace(0, 13, T - 1)).astype(int)]
    return e


class Fun:
    def __init__(self, m, T, obj):
        self.m, self.T, self.obj, self.nev, self.hist = m, T, obj, 0, []

    def jg(self, x):
        u = torch.tensor(x.reshape(self.T - 1, 3, 2), dtype=DT, device=dev).requires_grad_(True)
        o = self.m.rollout(ipa.open_loop(u))
        J = objective(o, self.obj).mean()
        g, = torch.autograd.grad(J, u)
        self.nev += 1
        self.hist.append(float(J))
        return float(J), g.detach().cpu().numpy().ravel()

    def __call__(self, x):
        J, g = self.jg(x)
        return -J, -g


def kkt(u, g):
    """Projected-gradient residual and active-set classification for max J on [0,1]^d."""
    u = np.asarray(u).ravel(); g = np.asarray(g).ravel()
    r = np.clip(u + g, 0, 1) - u
    lo = u <= EPS; hi = u >= 1 - EPS; free = ~(lo | hi)
    viol_lo = np.maximum(g[lo], 0) if lo.any() else np.zeros(1)      # at 0 need g <= 0
    viol_hi = np.maximum(-g[hi], 0) if hi.any() else np.zeros(1)     # at 1 need g >= 0
    return dict(res_inf=float(np.abs(r).max()), res_free=float(np.abs(g[free]).max()) if free.any() else 0.0,
                viol_lo=float(viol_lo.max()), viol_hi=float(viol_hi.max()), n_free=int(free.sum()),
                n_lo=int(lo.sum()), n_hi=int(hi.sum()), gmax=float(np.abs(g).max()))


def run_oc(obj, T, kappa, start, depth=3, seed=12345, G=None, u_init=None, tag=None):
    G = G or G0
    m = ipa.IPA(theta(G, seed), T=T, kappa=kappa, depth=depth)
    key = f'{obj}|{T}|{kappa}'
    S = STARTS.get(key, {})
    if u_init is not None:
        u0 = np.array(u_init)
    elif start == 'half':
        u0 = np.full((T - 1, 3, 2), 0.5)
    elif start == 'enc':
        u0 = enc_u(T)
    elif start == 'window':
        u0 = window_u(T, *S['window'])
    elif start == 'tp':
        u0 = switch_u(T, S['tp'])
    elif start == 'adam_old':
        best = max(S['adam_old'].values(), key=lambda v: v['J'])
        u0 = np.array(best['u'])
    elif start.startswith('r'):
        u0 = np.random.default_rng(int(start[1:]) + 17).random((T - 1, 3, 2))
    else:
        raise ValueError(start)
    u0 = np.clip(u0, 0, 1)
    f = Fun(m, T, obj)
    J0, g0 = f.jg(u0.ravel())
    t0 = time.time()
    res = minimize(f, u0.ravel(), jac=True, method='L-BFGS-B', bounds=[(0, 1)] * u0.size,
                   options=dict(maxfun=MAXFUN if not SMALL else 5, maxiter=MAXFUN, ftol=1e-13, gtol=1e-8, maxcor=30))
    J, g = f.jg(res.x)
    out = dict(obj=obj, T=T, kappa=kappa, start=start, depth=depth, seed=seed, G=G, J0=J0, J=J,
               u=res.x.reshape(T - 1, 3, 2).tolist(), grad=g.reshape(T - 1, 3, 2).tolist(),
               kkt=kkt(res.x, g), kkt_start=kkt(u0, g0), nev=f.nev, nit=int(res.nit), status=int(res.status),
               message=str(res.message), hist=f.hist, sec=time.time() - t0)
    return out


def best_oc(obj, T, kappa, depth=3):
    fs = glob.glob(f'oc_{obj}_{T}_{kappa}_*_d{depth}.json')
    recs = [json.load(open(f)) for f in fs]
    return max(recs, key=lambda r: r['J']) if recs else None


def eval_J(u, T, kappa, obj, G=256, seed=2024, depth=3):
    tot = []
    with torch.no_grad():
        th = theta(G, seed)
        for c in range(0, G, 64):
            m = ipa.IPA(th[c:c + 64], T=T, kappa=kappa, depth=depth)
            o = m.rollout(ipa.open_loop(torch.tensor(np.array(u), dtype=DT, device=dev)))
            tot.append(objective(o, obj))
    v = torch.cat(tot).cpu().numpy()
    return float(v.mean()), float(v.std() / np.sqrt(len(v)))


def part_post():
    out = dict(hess=[], reseed=[], oos=[])
    # out-of-sample evaluation of every optimum (all starts)
    for f in sorted(glob.glob('oc_*_d*.json')):
        r = json.load(open(f))
        J, se = eval_J(r['u'], r['T'], r['kappa'], r['obj'], depth=r['depth'])
        Jw = None
        out['oos'].append(dict(file=f, obj=r['obj'], T=r['T'], kappa=r['kappa'], start=r['start'], depth=r['depth'],
                               J_in=r['J'], J_oos=J, J_oos_se=se))
        print('oos', f, round(J, 5), flush=True)
    # second-order check at the best cooperation optima
    for kappa in ([0.5, 0.75, 1.0] if not SMALL else [1.0]):
        b = best_oc('final', 15, kappa)
        if b is None:
            continue
        u = np.array(b['u']).ravel()
        free = np.flatnonzero((u > 1e-4) & (u < 1 - 1e-4))
        m = ipa.IPA(theta(G0), T=15, kappa=kappa, depth=3)
        f = Fun(m, 15, 'final')
        h = 1e-4
        H = np.zeros((len(free), len(free)))
        for a_, i in enumerate(free):
            up = u.copy(); up[i] += h; dn = u.copy(); dn[i] -= h
            gp = f.jg(up)[1][free]; gm = f.jg(dn)[1][free]
            H[:, a_] = (gp - gm) / (2 * h)
        H = 0.5 * (H + H.T)
        ev = np.linalg.eigvalsh(H) if len(free) else np.array([])
        idx = [np.unravel_index(i, (14, 3, 2)) for i in free]
        out['hess'].append(dict(kappa=kappa, free=[[int(t) + 1, int(s), int(c)] for t, s, c in idx],
                                u_free=u[free].tolist(), eig=ev.tolist(), grad_free=np.array(b['grad']).ravel()[free].tolist()))
        print('hess', kappa, len(free), ev[:3] if len(ev) else None, flush=True)
    # re-optimisation on two further disposition samples, warm-started at the best solution
    for kappa in ([0.5, 0.75, 1.0] if not SMALL else [1.0]):
        b = best_oc('final', 15, kappa)
        if b is None:
            continue
        for sd in (777, 999):
            r = run_oc('final', 15, kappa, 'reseed', seed=sd, u_init=b['u'])
            J, se = eval_J(r['u'], 15, kappa, 'final')
            out['reseed'].append(dict(kappa=kappa, seed=sd, J_in=r['J'], J_oos=J, J_oos_se=se, u=r['u'], kkt=r['kkt']))
            print('reseed', kappa, sd, round(r['J'], 5), flush=True)
    json.dump(out, open('ipa_post.json', 'w'))


class Recorder:
    """Wraps an open-loop control and records the C-D untied / tied masses at every planner step."""
    def __init__(self, u, m):
        self.ctrl = ipa.open_loop(u); self.m = m; self.W0 = []; self.W1 = []

    def __call__(self, t, Q):
        cur = self.m.cur
        cd = ((cur.view(-1, 1) + cur.view(1, -1)) == 1).to(DT)            # (H, H) C-D history pairs
        q0 = (Q[:, :, :, 0] * cd).sum((1, 2, 3, 4)).mean() / 2
        q1 = (Q[:, :, :, 1] * cd).sum((1, 2, 3, 4)).mean() / 2
        self.W0.append(float(q0)); self.W1.append(float(q1))
        return self.ctrl(t, Q)


def part_costate():
    pa, pd = ipa.acceptance('random')
    phip, phim = float(pa[0, 1]), float(pd[0, 1])
    KG = [0.6, 0.65, 0.7, 0.75, 0.8, 0.875, 0.95, 1.0] if not SMALL else [0.75]
    out = dict(sweep40=[], lam=[])
    # two-phase sweep at T = 40 (reduced model, 48 draws as in the original sweep)
    KG40 = [round(0.25 + 0.05 * i, 2) for i in range(26)] if not SMALL else [0.75]
    T = 40 if not SMALL else 10
    for kc in [KG40[i:i + 7] for i in range(0, len(KG40), 7)]:
        th = theta(48).repeat(len(kc), 1)
        kap = torch.tensor(kc, dtype=DT, device=dev).repeat_interleave(48)
        grp = torch.arange(len(kc), device=dev).repeat_interleave(48)
        m = ipa.IPA(th, T=T, kappa=kap, depth=3)
        for s in range(T):
            with torch.no_grad():
                o = m.rollout(ipa.open_loop(torch.tensor(switch_u(T, s), dtype=DT, device=dev)))
            fin = torch.zeros(len(kc), dtype=DT, device=dev).index_add_(0, grp, o['final']) / 48
            for i, k in enumerate(kc):
                out['sweep40'].append(dict(T=T, s=s, kappa=k, final=float(fin[i])))
        print('sweep40', kc, flush=True)
    json.dump(out, open('ipa_costate.json', 'w'))
    sw = json.load(open('ipa_sweep.json')) + out['sweep40']
    for T in ([8, 15, 25, 40] if not SMALL else [8]):
        for k in KG:
            rows = [r for r in sw if r['T'] == T and abs(r['kappa'] - k) < 1e-9]
            sstar = max(rows, key=lambda r: r['final'])['s'] if rows else T - 1
            for pol, s in (('conciliate', T - 1), ('best', sstar)):
                m = ipa.IPA(theta(G0), T=T, kappa=k, depth=3)
                u = torch.tensor(switch_u(T, s), dtype=DT, device=dev).requires_grad_(True)
                rec = Recorder(u, m)
                o = m.rollout(rec)
                J = o['final'].mean()
                g, = torch.autograd.grad(J, u)
                g = g.cpu().numpy()
                # value per physical tie: raising u+ by du adds phi+ * W0 * du expected ties (W0 = expected number of
                # untied C-D pairs, unordered), so Lambda0 = (dJ/du+) / (phi+ W0); likewise Lambda1 = -(dJ/du-) / (phi- W1)
                lam0 = [float(g[t, 1, 0] / (phip * rec.W0[t])) if rec.W0[t] > 1e-12 else None for t in range(T - 1)]
                lam1 = [float(-g[t, 1, 1] / (phim * rec.W1[t])) if rec.W1[t] > 1e-12 else None for t in range(T - 1)]
                out['lam'].append(dict(T=T, kappa=k, policy=pol, s=s, J=float(J), W0=rec.W0, W1=rec.W1, lam0=lam0, lam1=lam1,
                                       g_add=g[:, 1, 0].tolist(), g_del=g[:, 1, 1].tolist()))
                print('costate', T, k, pol, s, flush=True)
                json.dump(out, open('ipa_costate.json', 'w'))


def dec_threshold_ctrl(thrA, thrR, th, tau):
    G, n = th.shape

    def ctrl(t, Q):
        sA = torch.sigmoid((th - thrA[t - 1]) / tau)
        sR = torch.sigmoid((thrR[t - 1] - th) / tau)
        aj = sA.view(G, 1, n).expand(G, n, n); ai = sA.view(G, n, 1).expand(G, n, n)
        rj = sR.view(G, 1, n).expand(G, n, n); ri = sR.view(G, n, 1).expand(G, n, n)
        one = torch.ones(G, n, n, dtype=DT, device=dev); zero = torch.zeros_like(one)
        ua = torch.stack([torch.stack([zero, ai], -1), torch.stack([aj, one], -1)], -2)
        ud = torch.stack([torch.stack([one, ri], -1), torch.stack([rj, zero], -1)], -2)
        return ua, ud
    return ctrl


def part_fb2():
    out = []
    T = 15
    for kappa in ([0.5, 0.75, 1.0] if not SMALL else [1.0]):
        m = ipa.IPA(theta(G0), T=T, kappa=kappa, depth=3)
        th = m.theta
        ta, td = STARTS[f'final|15|{kappa}']['window']
        inits = {'window': (np.where(np.arange(1, T) <= ta, -12.0, 12.0), np.where(np.arange(1, T) >= td, 12.0, -12.0))}
        try:
            fb = json.load(open('ipa_fb.json'))[f'defector_{kappa}']['thr'][0]
            inits['coupled'] = (np.array(fb), np.array(fb))
        except Exception:
            pass
        for nm, (a0, r0) in inits.items():
            thrA = torch.tensor(a0, dtype=DT, device=dev).requires_grad_(True)
            thrR = torch.tensor(r0, dtype=DT, device=dev).requires_grad_(True)
            opt = torch.optim.Adam([thrA, thrR], lr=0.1)
            hist = []
            for it in range(150 if not SMALL else 2):
                opt.zero_grad()
                J = m.rollout(dec_threshold_ctrl(thrA, thrR, th, 0.25))['final'].mean()
                (-J).backward(); opt.step(); hist.append(float(J))
            with torch.no_grad():
                Jh = float(m.rollout(dec_threshold_ctrl(thrA, thrR, th, 0.02))['final'].mean())
            out.append(dict(kappa=kappa, init=nm, thrA=thrA.detach().cpu().numpy().tolist(),
                            thrR=thrR.detach().cpu().numpy().tolist(), J_soft=hist[-1], J_hard=Jh, hist=hist))
            print('fb2', kappa, nm, round(hist[-1], 4), round(Jh, 4), flush=True)
            json.dump(out, open('ipa_fb2.json', 'w'))


def part_gradbases():
    out = []
    th = torch.tensor(np.load('theta_draws.npy'), dtype=DT, device=dev)
    if SMALL:
        th = th[:2]
    for kappa, nm, u in [(1.0, 'P14', switch_u(15, 14)), (1.0, 'W4,15', window_u(15, 4, 15)),
                         (0.75, 'P9', switch_u(15, 9)), (0.75, 'W0,14', window_u(15, 0, 14))]:
        m = ipa.IPA(th, T=15, kappa=kappa, depth=3)
        ut = torch.tensor(u, dtype=DT, device=dev).requires_grad_(True)
        J = m.rollout(ipa.open_loop(ut))['final'].mean()
        g, = torch.autograd.grad(J, ut)
        out.append(dict(kappa=kappa, base=nm, J=float(J), grad=g.cpu().numpy().tolist()))
        print('gradbase', nm, flush=True)
    json.dump(out, open('ipa_gradbases.json', 'w'))


if __name__ == '__main__':
    part = sys.argv[1]
    t0 = time.time()
    if part == 'oc':
        obj, T, kappa, start = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
        depth = int(sys.argv[6]) if len(sys.argv) > 6 else 3
        r = run_oc(obj, T, kappa, start, depth=depth)
        json.dump(r, open(f'oc_{obj}_{T}_{kappa}_{start}_d{depth}.json', 'w'))
        print('oc', obj, T, kappa, start, depth, round(r['J0'], 5), '->', round(r['J'], 6), r['nev'], r['kkt']['res_inf'], flush=True)
    else:
        {'post': part_post, 'costate': part_costate, 'fb2': part_fb2, 'gradbases': part_gradbases}[part]()
    print('PART', part, 'done', round(time.time() - t0, 1), flush=True)
