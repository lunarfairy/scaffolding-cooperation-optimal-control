"""Job B: reduced-model (IPA) computations.
  part 'acc'    : IPA (depth 1,2,3) trajectories for the planner set on fixed theta draws (saved for ABM pairing)
  part 'sweep'  : two-phase switch family s = 0..T-1 on a kappa grid, T in {8,15,25}
  part 'oc'     : open-loop optimal control by projected gradient ascent; switching functions at optimum
  part 'sf'     : switching functions (gradients) along bang-bang switch policies + one-step (myopic) values
  part 'fb'     : feedback (disposition-threshold) optimal policy + per-node switching-function scatter
Usage: python run_ipa_b.py <part> ; env IPA_DEVICE, IPA_DTYPE, IPA_G, IPA_DEPTH
"""
import os, sys, json, time, math
import numpy as np
import torch
import ipa

dev = os.environ.get('IPA_DEVICE', 'cpu')
if os.environ.get('IPA_DTYPE', 'float64') == 'float32':
    ipa.DT = torch.float32
DT = ipa.DT
G0 = int(os.environ.get('IPA_G', 48))
DEPTH = int(os.environ.get('IPA_DEPTH', 3))
SMALL = os.environ.get('IPA_SMALL', '0') == '1'      # smoke-test mode
torch.set_num_threads(int(os.environ.get('IPA_THREADS', 8)))

KGRID = [round(0.25 + 0.05 * i, 2) for i in range(26)]    # 0.25 .. 1.50
if SMALL:
    KGRID = [0.5, 1.0]


def thetas(G, seed=12345):
    return ipa.draw_theta(G, 16, seed=seed).to(dev)


def batched_model(kgrid, G, T, depth=DEPTH, **kw):
    th = thetas(G).repeat(len(kgrid), 1)                      # common random numbers across kappa
    kap = torch.tensor(kgrid, dtype=DT, device=dev).repeat_interleave(G)
    group = torch.arange(len(kgrid), device=dev).repeat_interleave(G)
    m = ipa.IPA(th, T=T, kappa=kap, depth=depth, **kw)
    return m, group


def objective(out, name):
    if name == 'final':
        return out['final']
    if name == 'capital':
        return out['capital']
    if name == 'mcoop':
        return out['mcoop']
    raise ValueError(name)


def group_mean(v, group, ng):
    return torch.zeros(ng, dtype=v.dtype, device=v.device).index_add_(0, group, v) / (len(v) / ng)


# ---------------------------------------------------------------------- parts
def part_acc():
    import abm_fast as af
    G = 64 if not SMALL else 4
    th = thetas(G)
    np.save('theta_draws.npy', th.cpu().numpy())
    T = 15
    plan = {'static': af.uniform_u(T, 0, 0), 'random': af.uniform_u(T, .3, .3), 'neutral': af.neutral_u(T),
            'maxconn': af.uniform_u(T, 1, 0), 'encouragement': af.encouragement_u(T),
            'exclusion': af.switch_u(T, 0), 'conciliation': af.switch_u(T, T - 1), 'switch_T-2': af.switch_u(T, T - 2)}
    res = []
    for depth in ([1, 2, 3] if not SMALL else [1, 2]):
        for kappa in [0.5, 0.75, 1.0]:
            m = ipa.IPA(th, T=T, kappa=kappa, depth=depth)
            for name, u in plan.items():
                t0 = time.time()
                with torch.no_grad():
                    o = m.rollout(ipa.open_loop(torch.tensor(u, dtype=DT, device=dev)))
                res.append(dict(depth=depth, kappa=kappa, planner=name, coop=o['coop'].mean(0).tolist(),
                                coop_node_final=float(o['final'].mean()),
                                edges=o['edges'].mean(0).tolist(), capital=float(o['capital'].mean()),
                                sec=time.time() - t0))
                print('acc', depth, kappa, name, round(float(o['final'].mean()), 4), round(time.time() - t0, 1), flush=True)
    json.dump(res, open('ipa_acc.json', 'w'))


def part_sweep():
    res = []
    G = G0 if not SMALL else 4
    kc = int(os.environ.get('KCHUNK', 7))
    chunks = [KGRID[i:i + kc] for i in range(0, len(KGRID), kc)]
    for T, kg in [(T_, c_) for T_ in ([8, 15, 25] if not SMALL else [5]) for c_ in chunks]:
        m, group = batched_model(kg, G, T)
        for s in range(T):
            t0 = time.time()
            with torch.no_grad():
                o = m.rollout(ipa.open_loop(ipa.switch_u(T, s, device=dev)))
            fin = group_mean(o['final'], group, len(kg)).cpu().numpy()
            cap = group_mean(o['capital'], group, len(kg)).cpu().numpy()
            mco = group_mean(o['mcoop'], group, len(kg)).cpu().numpy()
            for i, k in enumerate(kg):
                res.append(dict(T=T, s=s, kappa=k, final=float(fin[i]), capital=float(cap[i]), mcoop=float(mco[i])))
            print('sweep', T, s, round(time.time() - t0, 1), flush=True)
            json.dump(res, open('ipa_sweep.json', 'w'))


def optimise(m, group, ng, T, obj, u0, iters, lr=0.05):
    u = u0.clone().to(dev).requires_grad_(True)               # (ng, T-1, 3, 2)
    opt = torch.optim.Adam([u], lr=lr)
    hist = []
    for it in range(iters):
        opt.zero_grad()
        o = m.rollout(ipa.open_loop(u, group))
        J = group_mean(objective(o, obj), group, ng)
        (-J.sum()).backward()
        opt.step()
        with torch.no_grad():
            u.clamp_(0.0, 1.0)
        hist.append(J.detach().cpu().numpy().tolist())
    # final evaluation + switching functions (gradient) at optimum
    u_opt = u.detach().clone().requires_grad_(True)
    o = m.rollout(ipa.open_loop(u_opt, group))
    J = group_mean(objective(o, obj), group, ng)
    g, = torch.autograd.grad(J.sum(), u_opt)
    return u_opt.detach(), J.detach(), g.detach(), hist


def part_oc():
    iters = int(os.environ.get('IPA_ITERS', 150)) if not SMALL else 3
    G = G0 if not SMALL else 4
    kset = [float(x) for x in os.environ.get('OC_KSET', '0.25,0.5,0.75,1.0,1.25').split(',')]
    jobs = [(int(a.split(':')[0]), a.split(':')[1]) for a in os.environ.get('OC_JOBS', '15:final').split(',')]
    inits_wanted = os.environ.get('OC_INITS', 'half,enc').split(',')
    if SMALL:
        kset = [0.5, 1.0]; jobs = [(5, 'final'), (5, 'capital')]
    import abm_fast as af
    fn = os.environ.get('OC_OUT', 'ipa_oc.json')
    out = []
    for T, obj in jobs:
        for k in kset:
            m, group = batched_model([k], G, T)
            enc = torch.tensor(af.encouragement_u(15), dtype=DT)
            if T != 15:
                idx = np.round(np.linspace(0, 13, T - 1)).astype(int)
                enc = enc[idx]
            inits = {'half': torch.full((1, T - 1, 3, 2), 0.5, dtype=DT), 'enc': enc.unsqueeze(0).clone()}
            for name in inits_wanted:
                t0 = time.time()
                u, J, g, hist = optimise(m, group, 1, T, obj, inits[name], iters, lr=float(os.environ.get('OC_LR', 0.08)))
                out.append(dict(T=T, obj=obj, init=name, kappa=k, J=float(J[0]), u=u[0].cpu().numpy().tolist(),
                                grad=g[0].cpu().numpy().tolist(), hist=[h[0] for h in hist]))
                print('oc', T, obj, name, k, round(float(J[0]), 4), round(time.time() - t0, 1), flush=True)
                json.dump(out, open(fn, 'w'))


def part_sf():
    """Switching functions dJ/du along bang-bang policies, and one-step (myopic) values dx(t+1)/du(t)."""
    G = G0 if not SMALL else 4
    kset_all = [0.25, 0.5, 0.75, 1.0, 1.25] if not SMALL else [0.5, 1.0]
    out = []
    for T, kset in [(T_, [k_]) for T_ in ([15, 8, 25] if not SMALL else [5]) for k_ in kset_all]:
        m, group = batched_model(kset, G, T)
        ng = len(kset)
        pols = {'exclusion': 0, 'switch_T-2': T - 2, 'conciliation': T - 1, 'switch_mid': T // 2}
        for pname, s in pols.items():
            for obj in (['final', 'capital'] if T == 15 else ['final']):
                u = ipa.switch_u(T, s, device=dev).expand(ng, T - 1, 3, 2).clone().requires_grad_(True)
                o = m.rollout(ipa.open_loop(u, group))
                J = group_mean(objective(o, obj), group, ng)
                g, = torch.autograd.grad(J.sum(), u)
                rec = dict(T=T, policy=pname, s=s, obj=obj, J=J.detach().cpu().numpy().tolist(),
                           grad=g.cpu().numpy().tolist(), kappa=kset,
                           coop=[group_mean(o['coop'][:, t].detach(), group, ng).cpu().numpy().tolist() for t in range(T)])
                if obj == 'final' and T == (15 if not SMALL else 5):
                    # one-step (myopic) value: gradient of x(t+1) w.r.t. u(t)
                    my = []
                    for t in range(1, T):
                        mt = ipa.IPA(m.theta, T=t + 1, kappa=m.kappa, depth=m.depth)
                        ut = u.detach()[:, :t].clone().requires_grad_(True)
                        ot = mt.rollout(ipa.open_loop(ut, group))
                        Jt = group_mean(ot['final'], group, ng)
                        gt, = torch.autograd.grad(Jt.sum(), ut)
                        my.append(gt[:, t - 1].cpu().numpy().tolist())
                    rec['myopic'] = my
                out.append(rec)
                print('sf', T, pname, obj, [round(float(x), 4) for x in J], flush=True)
                json.dump(out, open('ipa_sf.json', 'w'))


def part_fb():
    iters = int(os.environ.get('IPA_ITERS', 200)) if not SMALL else 3
    G = G0 if not SMALL else 4
    kset_all = [0.5, 0.75, 1.0] if not SMALL else [0.5, 1.0]
    T = 15 if not SMALL else 5
    out = {}
    scat = {}
    for variant, kset in [(v, [k]) for k in kset_all for v in ['defector', 'both']]:
        m, group = batched_model(kset, G, T)
        ng = len(kset)
        key = f'{variant}_{kset[0]}'
        thr = torch.zeros(ng, T - 1, dtype=DT, device=dev).requires_grad_(True)
        thr_c = torch.full((ng, T - 1), -6.0, dtype=DT, device=dev).requires_grad_(variant == 'both')
        params = [thr] + ([thr_c] if variant == 'both' else [])
        opt = torch.optim.Adam(params, lr=0.1)
        hist = []
        for it in range(iters):
            opt.zero_grad()
            o = m.rollout(ipa.threshold_ctrl(thr, m.theta, tau=0.25, group=group, thr_c=thr_c if variant == 'both' else None))
            J = group_mean(o['final'], group, ng)
            (-J.sum()).backward()
            opt.step()
            hist.append(J.detach().cpu().numpy().tolist())
        with torch.no_grad():
            o = m.rollout(ipa.threshold_ctrl(thr, m.theta, tau=0.25, group=group, thr_c=thr_c if variant == 'both' else None))
            J = group_mean(o['final'], group, ng)
            # hard threshold evaluation
            oh = m.rollout(ipa.threshold_ctrl(thr, m.theta, tau=0.02, group=group, thr_c=thr_c if variant == 'both' else None))
            Jh = group_mean(oh['final'], group, ng)
        out[key] = dict(kappa=kset, thr=thr.detach().cpu().numpy().tolist(),
                            thr_c=thr_c.detach().cpu().numpy().tolist(), J=J.cpu().numpy().tolist(),
                            J_hard=Jh.cpu().numpy().tolist(), hist=hist)
        print('fb', variant, [round(float(x), 4) for x in J], [round(float(x), 4) for x in Jh], flush=True)
        json.dump(out, open('ipa_fb.json', 'w'))
    # per-node switching function scatter at the clock policy s = T-2 (and s = 0)
    for (pname, s), kset in [(ps, [k]) for k in kset_all for ps in [('switch_T-2', T - 2), ('exclusion', 0)]]:
        m, group = batched_model(kset, G, T)
        c = torch.zeros(len(m.theta), 16, T - 1, dtype=DT, device=dev)
        base = ipa.switch_u(T, s, device=dev)
        c[:, :, :] = base[:, 1, 0].view(1, 1, T - 1)          # conciliation prob per defector node & round
        c.requires_grad_(True)
        n = 16

        def ctrl(t, Q, c=c):
            Gb = c.shape[0]
            cj = c[:, :, t - 1].view(Gb, 1, n).expand(Gb, n, n); ci = c[:, :, t - 1].view(Gb, n, 1).expand(Gb, n, n)
            one = torch.ones(Gb, n, n, dtype=DT, device=dev); zero = torch.zeros_like(one)
            ua = torch.stack([torch.stack([zero, ci], -1), torch.stack([cj, one], -1)], -2)
            ud = torch.stack([torch.stack([one, 1 - ci], -1), torch.stack([1 - cj, zero], -1)], -2)
            return ua, ud
        o = m.rollout(ctrl)
        Jg = o['final'].sum()
        gc, = torch.autograd.grad(Jg, c)
        # expected number of defecting nodes: weight by P(node defects at t) -> use marginal from forward pass
        scat[f'{pname}_{kset[0]}'] = dict(theta=m.theta.cpu().numpy().tolist(), kappa=m.kappa.cpu().numpy().tolist(),
                           grad=gc.cpu().numpy().tolist())
        print('scatter', pname, flush=True)
    json.dump(scat, open('ipa_scatter.json', 'w'))


if __name__ == '__main__':
    parts = sys.argv[1:] or ['acc', 'sweep', 'oc', 'sf', 'fb']
    for p in parts:
        t0 = time.time()
        {'acc': part_acc, 'sweep': part_sweep, 'oc': part_oc, 'sf': part_sf, 'fb': part_fb}[p]()
        print('PART', p, 'done in', round(time.time() - t0, 1), 's', flush=True)
