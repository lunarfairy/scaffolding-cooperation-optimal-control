"""Vectorised (batch-of-games) agent-based model of the McKee et al. (2023) network cooperation game.

Game rules, fitted bot behaviour and acceptance probabilities follow the published simulator
(Supplementary Information of McKee et al. 2023; see sim_pilot.py for the reference loop implementation).

A planner is a function  planner(ctx) -> (padd, pdel)  returning (G, n, n) recommendation
probabilities for absent / present ties. ctx carries A, a, t, theta, posterior mean, etc.
Each recommendation is delivered to one endpoint chosen uniformly at random (route='random'), or to
the endpoint most likely to accept it (route='targeted').
"""
import numpy as np

B, C, D0 = 0.10, 0.05, 1.0
MU_TH, SD_TH = -0.304, 2.410
R1 = (1.807, 0.818)
RL = (-0.010, -0.193, 0.370, 1.521)          # intercept, x_s (degree), x_n (# coop nbrs), x_r (coop share)
PHI_DEL_D, PHI_DEL_C, PHI_ADD_D, PHI_ADD_C = 0.774, 0.085, 0.287, 0.909   # (valence, referent action)

ENC_CC = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, .010], [1, .010],
                   [1, .011], [1, .028], [.991, .035], [.954, .073], [1, .108]])
ENC_CD = np.array([[.993, .048], [.973, .029], [.914, .145], [.791, .213], [.644, .318], [.594, .508],
                   [.463, .608], [.429, .745], [.366, .802], [.372, .753], [.361, .741], [.371, .774],
                   [.328, .706], [.408, .722]])
NEUTRAL = np.array([[.891, .119], [.841, .054], [.656, .084], [.642, .102], [.608, .117], [.549, .204],
                    [.545, .215], [.538, .224], [.520, .239], [.504, .213], [.532, .215], [.518, .237],
                    [.529, .232], [.522, .317]])

TH_GRID = np.linspace(-9, 9, 181)
LOG_PRIOR = -0.5 * ((TH_GRID - MU_TH) / SD_TH) ** 2


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logsig(x):
    return -np.logaddexp(0.0, -x)


# ------------------------------------------------------------------ planners
def typed(u):
    """Open-loop pair-type controls. u: array (T-1, 3, 2) with u[t-1, type, 0]=add prob, [...,1]=delete prob;
    type index 0 = DD, 1 = CD, 2 = CC (= a_i + a_j)."""
    u = np.asarray(u, float)

    def planner(ctx):
        a, t = ctx['a'], ctx['t']
        s = (a[:, :, None] + a[:, None, :]).astype(int)
        tt = min(t - 1, len(u) - 1)
        padd = u[tt, :, 0][s]
        pdel = u[tt, :, 1][s]
        return padd, pdel
    return planner


def switch_u(T, s, cd_mode=None):
    """Two-phase C-D policy: conciliate for t <= s, exclude afterwards; CC connect; DD cut."""
    u = np.zeros((T - 1, 3, 2))
    u[:, 2, 0] = 1.0            # CC add
    u[:, 0, 1] = 1.0            # DD delete
    for t in range(1, T):
        u[t - 1, 1] = (1.0, 0.0) if t <= s else (0.0, 1.0)
    return u


def encouragement_u(T=15):
    u = np.zeros((T - 1, 3, 2))
    u[:, 2] = ENC_CC[:T - 1]
    u[:, 1] = ENC_CD[:T - 1]
    u[:, 0] = (0.0, 1.0)
    return u


def uniform_u(T, add, dele):
    u = np.zeros((T - 1, 3, 2))
    u[:, :, 0] = np.asarray(add)[:, None] if np.ndim(add) else add
    u[:, :, 1] = np.asarray(dele)[:, None] if np.ndim(dele) else dele
    return u


def neutral_u(T=15):
    u = np.zeros((T - 1, 3, 2))
    u[:, :, 0] = NEUTRAL[:T - 1, 0][:, None]
    u[:, :, 1] = NEUTRAL[:T - 1, 1][:, None]
    return u


def threshold_planner(thr, use='oracle', cc_rule=True):
    """Feedback C-D policy: conciliate a C-D pair iff the defector's disposition (oracle theta, or the
    planner's posterior mean given observed history) exceeds thr[t-1]; CC connect, DD cut."""
    thr = np.asarray(thr, float)

    def planner(ctx):
        a, t = ctx['a'], ctx['t']
        est = ctx['theta'] if use == 'oracle' else ctx['post_mean']
        s = a[:, :, None] + a[:, None, :]
        # defector's estimate on a C-D pair
        dest = np.where(a[:, :, None] == 0, est[:, :, None], est[:, None, :])
        conc = dest > thr[min(t - 1, len(thr) - 1)]
        padd = np.where(s == 2, 1.0, np.where(s == 1, conc * 1.0, 0.0))
        pdel = np.where(s == 2, 0.0, np.where(s == 1, 1.0 - conc, 1.0))
        return padd, pdel
    return planner


def window_threshold_planner(ta, ca, td, cd, use='oracle'):
    """Decoupled disposition-aware C-D policy (nests the recruit/retain windows and constant coupled thresholds).
    Recruit an absent C-D pair iff t <= ta and the defector's disposition estimate exceeds ca;
    cut a present C-D pair iff t >= td and the estimate is below cd. CC connect, DD cut.
    ca = -inf and cd = +inf recover the window W(ta, td); ta = T-1, td = 1, ca = cd = c recover the coupled
    constant threshold c."""
    def planner(ctx):
        a, t = ctx['a'], ctx['t']
        est = ctx['theta'] if use == 'oracle' else ctx['post_mean']
        s = a[:, :, None] + a[:, None, :]
        dest = np.where(a[:, :, None] == 0, est[:, :, None], est[:, None, :])
        rec = (t <= ta) & (dest > ca)
        cut = (t >= td) & (dest < cd)
        padd = np.where(s == 2, 1.0, np.where(s == 1, rec * 1.0, 0.0))
        pdel = np.where(s == 2, 0.0, np.where(s == 1, cut * 1.0, 1.0))
        return padd, pdel
    return planner


def dec_threshold_planner(thrA, thrR, use='oracle'):
    """Decoupled per-round thresholds: recruit an absent C-D pair iff the defector's estimate exceeds thrA[t-1];
    cut a present C-D pair iff the estimate is below thrR[t-1]. CC connect, DD cut."""
    thrA = np.asarray(thrA, float); thrR = np.asarray(thrR, float)

    def planner(ctx):
        a, t = ctx['a'], ctx['t']
        est = ctx['theta'] if use == 'oracle' else ctx['post_mean']
        s = a[:, :, None] + a[:, None, :]
        dest = np.where(a[:, :, None] == 0, est[:, :, None], est[:, None, :])
        tt = min(t - 1, len(thrA) - 1)
        rec = dest > thrA[tt]; cut = dest < thrR[tt]
        padd = np.where(s == 2, 1.0, np.where(s == 1, rec * 1.0, 0.0))
        pdel = np.where(s == 2, 0.0, np.where(s == 1, cut * 1.0, 1.0))
        return padd, pdel
    return planner


def probation_planner(strikes=1):
    def planner(ctx):
        a, ndef = ctx['a'], ctx['ndef']
        s = a[:, :, None] + a[:, None, :]
        dc = np.where(a[:, :, None] == 0, ndef[:, :, None], ndef[:, None, :])
        len_ = dc <= strikes
        padd = np.where(s == 2, 1.0, np.where(s == 1, len_ * 1.0, 0.0))
        pdel = np.where(s == 2, 0.0, np.where(s == 1, 1.0 - len_, 1.0))
        return padd, pdel
    return planner


# ------------------------------------------------------------------ simulation
def simulate(planner, G, rng, n=16, p0=0.3, T=15, kappa=1.0, route='random', capital_rule=True,
             track_posterior=False, record_recs=False, theta=None, sd_th=None, intervene=None, tiestats=None):
    """Simulate G independent games in parallel. Returns dict of arrays.
    sd_th      : override of the disposition SD (sd_th=0 gives homogeneous dispositions at MU_TH).
    intervene  : dict(t0=round, seed=int). After the planner move of round t0, one C-D pair per game is drawn
                 (with a separate generator, so the main random stream is untouched) from the tied C-D pairs and
                 one from the untied C-D pairs; mode 'cut' deletes the tied one, 'add' adds the untied one, 'none'
                 changes nothing (control). Records the defector's and cooperator's next action and dispositions.
    tiestats   : list of rounds at which to record per-pair and per-defector tie-state statistics."""
    iu = np.triu_indices(n, 1)
    U = rng.random((G, n, n)) < p0
    A = np.triu(U, 1)
    A = A | A.transpose(0, 2, 1)
    if theta is None:
        theta = rng.normal(MU_TH, SD_TH if sd_th is None else sd_th, (G, n))
    else:
        rng.normal(0.0, 1.0, (G, n))            # keep the random stream aligned
    streak = np.zeros((G, n))
    iv = None
    ts_out = {}
    d = np.full((G, n), D0)
    ndef = np.zeros((G, n))
    logpost = np.broadcast_to(LOG_PRIOR, (G, n, len(TH_GRID))).copy() if track_posterior else None
    coop = np.zeros((G, T)); cap = np.zeros((G, T)); pay_mean = np.zeros((G, T))
    ecc = np.zeros((G, T)); ecd = np.zeros((G, T)); edd = np.zeros((G, T)); bind = np.zeros(T)
    recs = []   # per round: fraction of CD non-edges recommended add, CD edges recommended delete
    a_prev = None
    tiestats_pending = set(t + 1 for t in tiestats) if tiestats is not None else set()
    for t in range(1, T + 1):
        Af = A.astype(float)
        deg = Af.sum(-1)
        if t == 1:
            eta_base = np.full((G, n), R1[0]); slope = R1[1]
        else:
            xn = np.einsum('gij,gj->gi', Af, a_prev)
            xr = np.where(deg > 0, xn / np.maximum(deg, 1), 0.0)
            k = np.where(a_prev == 1, 1.0, kappa)
            eta_base = RL[0] + RL[1] * deg + k * (RL[2] * xn + RL[3] * xr); slope = 1.0
        pc = sig(eta_base + slope * theta)
        a = (rng.random((G, n)) < pc).astype(float)
        feasible = (C * deg <= d) if capital_rule else np.ones((G, n), bool)
        bind[t - 1] = np.mean((a == 1) & ~feasible)
        if track_posterior:
            eta_g = eta_base[:, :, None] + slope * TH_GRID[None, None, :]
            ll = np.where(a[:, :, None] == 1, logsig(eta_g), logsig(-eta_g))
            logpost += np.where(feasible[:, :, None], ll, 0.0)
        a = a * feasible
        xn_now = np.einsum('gij,gj->gi', Af, a)
        pay = B * xn_now - C * a * deg
        d = d + pay
        coop[:, t - 1] = a.mean(1); cap[:, t - 1] = d.mean(1); pay_mean[:, t - 1] = pay.mean(1)
        s = a[:, :, None] + a[:, None, :]
        Au = A[:, iu[0], iu[1]]; su = s[:, iu[0], iu[1]]
        ecc[:, t - 1] = (Au & (su == 2)).sum(1); ecd[:, t - 1] = (Au & (su == 1)).sum(1)
        edd[:, t - 1] = (Au & (su == 0)).sum(1)
        ndef += (a == 0)
        streak = np.where(a == 0, streak + 1, 0)
        if iv is not None and t == iv['t0'] + 1:
            gidx = np.arange(G)
            for key in ('cut', 'add'):
                ok = iv[key + '_ok']
                iv[key + '_aD1'] = np.where(ok, a[gidx, iv[key + '_j']], np.nan)
                iv[key + '_aC1'] = np.where(ok, a[gidx, iv[key + '_i']], np.nan)
        if tiestats is not None and t in tiestats_pending:
            # realised next-round action of defectors recorded in the previous round
            rec_ = ts_out[t - 1]
            rec_['a_next_pair'] = a[rec_['_g'], rec_['_j']]
            rec_['a_next_def'] = a[rec_['_gd'], rec_['_jd']]
        a_prev = a
        if t == T:
            break
        if tiestats is not None and t in tiestats:
            ts_out[t] = _tie_state(A, a, Af, deg, xn_now, theta, streak, kappa)
        ctx = dict(A=A, a=a, t=t, theta=theta, ndef=ndef, streak=streak)
        if track_posterior:
            lp = logpost - logpost.max(-1, keepdims=True)
            w = np.exp(lp); w /= w.sum(-1, keepdims=True)
            ctx['post_mean'] = (w * TH_GRID).sum(-1)
        padd, pdel = planner(ctx)
        r = rng.random((G, n, n))
        add = (~A) & (r < padd)
        dele = A & (r < pdel)
        add = np.triu(add, 1); dele = np.triu(dele, 1)
        if record_recs:
            cdmask = np.triu(s == 1, 1)
            ne = (cdmask & ~A).sum(); ee = (cdmask & A).sum()
            recs.append(((add & cdmask).sum() / max(ne, 1), (dele & cdmask).sum() / max(ee, 1),
                         ne / G, ee / G))
        # acceptance: recipient endpoint chosen at random; referent = other endpoint
        ai = a[:, :, None]; aj = a[:, None, :]
        if route == 'random':
            swap = rng.random((G, n, n)) < 0.5
            aref = np.where(swap, ai, aj)
        else:   # targeted: for adds ask the defector (referent = cooperator); for deletes ask the cooperator
            cd = (ai + aj) == 1
            swap = rng.random((G, n, n)) < 0.5
            aref_rand = np.where(swap, ai, aj)
            aref = np.where(cd, np.where(add, 1.0, 0.0), aref_rand)
        phi_add = np.where(aref == 1, PHI_ADD_C, PHI_ADD_D)
        phi_del = np.where(aref == 1, PHI_DEL_C, PHI_DEL_D)
        acc = rng.random((G, n, n))
        add &= acc < phi_add
        dele &= acc < phi_del
        add = add | add.transpose(0, 2, 1); dele = dele | dele.transpose(0, 2, 1)
        A = (A | add) & ~dele
        if intervene is not None and t == intervene['t0']:
            A, iv = _intervene(A, a, theta, streak, intervene)
    out = dict(coop=coop, cap=cap, pay=pay_mean, ecc=ecc, ecd=ecd, edd=edd, bind=bind,
               final=coop[:, -1], mcoop=coop.mean(1), capital=cap[:, -1])
    if record_recs:
        out['recs'] = np.array(recs)
    if iv is not None:
        out['iv'] = iv
    if tiestats is not None:
        out['tiestats'] = ts_out
    return out


def _intervene(A, a, theta, streak, spec):
    """Draw one tied and one untied C-D pair per game (separate generator) and optionally cut / add it."""
    G, n = a.shape
    rh = np.random.default_rng(spec['seed'])
    cdm = np.triu((a[:, :, None] + a[:, None, :]) == 1, 1)
    iv = dict(t0=spec['t0'], mode=spec.get('mode', 'none'))
    gidx = np.arange(G)
    A = A.copy()
    for key, mask in (('cut', cdm & A), ('add', cdm & ~A)):
        flat = mask.reshape(G, -1)
        cnt = flat.sum(1)
        ok = cnt > 0
        # choose the k-th eligible pair uniformly
        k = np.floor(rh.random(G) * np.maximum(cnt, 1)).astype(int)
        cum = np.cumsum(flat, 1)
        pos = np.argmax(cum > k[:, None], 1)
        i, j = np.divmod(pos, n)
        # orient: c = cooperator, d = defector
        ci = np.where(a[gidx, i] == 1, i, j); dj = np.where(a[gidx, i] == 1, j, i)
        iv[key + '_ok'] = ok; iv[key + '_i'] = ci; iv[key + '_j'] = dj
        iv[key + '_thD'] = theta[gidx, dj]; iv[key + '_thC'] = theta[gidx, ci]
        iv[key + '_stD'] = streak[gidx, dj]
        iv[key + '_n'] = cnt
        if spec.get('mode') == key:
            g_ = gidx[ok]
            val = (key == 'add')
            A[g_, ci[ok], dj[ok]] = val; A[g_, dj[ok], ci[ok]] = val
    return A, iv


def _tie_state(A, a, Af, deg, xn_now, theta, streak, kappa):
    """C-D pairs before the planner move: tied vs untied defecting endpoints (pair-weighted), plus a
    defector-weighted version (each current defector classified by whether it has >= 1 tie to a cooperator)."""
    b0, bs, bn, br = RL
    cdm = (a[:, :, None] == 1) & (a[:, None, :] == 0)
    g, i, j = np.nonzero(cdm)
    e = A[g, i, j]
    kD = deg[g, j]; xD = xn_now[g, j]
    xr_ = np.where(kD > 0, xD / np.maximum(kD, 1), 0.0)
    p_with = sig(b0 + bs * kD + kappa * (bn * xD + br * xr_) + theta[g, j])
    kD0 = kD - e; xD0 = xD - e
    xr0 = np.where(kD0 > 0, xD0 / np.maximum(kD0, 1), 0.0)
    p_without = sig(b0 + bs * kD0 + kappa * (bn * xD0 + br * xr0) + theta[g, j])
    dmask = a == 0
    gd, jd = np.nonzero(dmask)
    hasC = xn_now[gd, jd] > 0
    return dict(_g=g, _j=j, tied=e, thD=theta[g, j], stD=streak[g, j], p_with=p_with, p_without=p_without,
                _gd=gd, _jd=jd, d_hasC=hasC, d_th=theta[gd, jd], d_st=streak[gd, jd])


def run_batched(planner, n_games, seed, batch=2000, **kw):
    rng = np.random.default_rng(seed)
    outs = []
    left = n_games
    while left > 0:
        g = min(batch, left)
        outs.append(simulate(planner, g, rng, **kw))
        left -= g
    keys = [k for k in outs[0] if k not in ('bind', 'recs')]
    res = {k: np.concatenate([o[k] for o in outs]) for k in keys}
    res['bind'] = np.mean([o['bind'] for o in outs], 0)
    if 'recs' in outs[0]:
        res['recs'] = np.mean([o['recs'] for o in outs], 0)
    return res
