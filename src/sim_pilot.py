"""Pilot re-implementation of the McKee et al. (2023, Nat Hum Behav) network cooperation game.

Game rules, bot behaviour model and hand-crafted planner tables are taken from the
Supplementary Information (Sections A, B, D1, E4, G1; Supp. Tables 1, 3, 7-10).
The GraphNet agent itself is not reproduced; the 'encouragement' planner (its published
distillation) stands in for it.

Notation note: Supp. Table 3 swaps the primed/unprimed beta labels relative to Methods.
We use the assignment consistent with SI D3: round-1 model logit = 1.807 + 0.818*theta;
later rounds logit = -0.010 - 0.193*xs + 0.370*xn + 1.521*xr + theta.
"""
import numpy as np

# ---------------- game + bot parameters (SI Tables 1, 3) ----------------
B, C, D0, T = 0.10, 0.05, 1.0, 15
MU_TH, SD_TH = -0.304, 2.410
R1 = (1.807, 0.818)                      # round-1 intercept, theta slope
RL = (-0.010, -0.193, 0.370, 1.521)      # later rounds: intercept, xs, xn, xr
PHI = {(-1, 0): 0.774, (-1, 1): 0.085, (1, 0): 0.287, (1, 1): 0.909}  # (valence, referent action)

# ---------------- planner tables (SI Tables 7-10), rows t = 1..14 ----------------
ENC_CC = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, .010], [1, .010],
                   [1, .011], [1, .028], [.991, .035], [.954, .073], [1, .108]])
ENC_CD = np.array([[.993, .048], [.973, .029], [.914, .145], [.791, .213], [.644, .318], [.594, .508],
                   [.463, .608], [.429, .745], [.366, .802], [.372, .753], [.361, .741], [.371, .774],
                   [.328, .706], [.408, .722]])
ENC_DD = np.tile([0.0, 1.0], (14, 1))
NEUTRAL = np.array([[.891, .119], [.841, .054], [.656, .084], [.642, .102], [.608, .117], [.549, .204],
                    [.545, .215], [.538, .224], [.520, .239], [.504, .213], [.532, .215], [.518, .237],
                    [.529, .232], [.522, .317]])


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() <= 0:
        return 0.0
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


# ---------------- planners ----------------
def _typed_probs(A, a, t, cc, cd, dd):
    """Return add/delete recommendation probability matrices from pair-type tables."""
    s = a[:, None] + a[None, :]            # 2 = CC, 1 = CD, 0 = DD
    padd = np.where(s == 2, cc[t - 1, 0], np.where(s == 1, cd[t - 1, 0], dd[t - 1, 0]))
    pdel = np.where(s == 2, cc[t - 1, 1], np.where(s == 1, cd[t - 1, 1], dd[t - 1, 1]))
    return padd, pdel


def _sample(A, padd, pdel, rng):
    n = A.shape[0]
    u = rng.random((n, n))
    R = np.zeros((n, n), np.int8)
    R[(~A) & (u < padd)] = 1
    R[A & (u < pdel)] = -1
    R = np.triu(R, 1)
    return R


def make_table_planner(cc, cd, dd):
    def planner(A, a, t, mem, rng):
        padd, pdel = _typed_probs(A, a, t, cc, cd, dd)
        return _sample(A, padd, pdel, rng)
    return planner


def planner_static(A, a, t, mem, rng):
    return np.zeros(A.shape, np.int8)


def planner_random(A, a, t, mem, rng, p=0.30):
    n = A.shape[0]
    toggle = np.triu(rng.random((n, n)) < p, 1)
    R = np.zeros((n, n), np.int8)
    R[toggle & ~A] = 1
    R[toggle & A] = -1
    return R


def planner_neutral(A, a, t, mem, rng):
    padd = np.full(A.shape, NEUTRAL[t - 1, 0])
    pdel = np.full(A.shape, NEUTRAL[t - 1, 1])
    return _sample(A, padd, pdel, rng)


def planner_maxconn(A, a, t, mem, rng):
    R = np.triu((~A).astype(np.int8), 1)
    return R


def planner_clustering(A, a, t, mem, rng):
    """Centralised re-implementation of Shirado & Christakis (2020) single-bot rule (SI D1)."""
    n = A.shape[0]
    if 'focal' not in mem:
        mem['focal'] = list(rng.choice(n, 5, replace=False))
    R = np.zeros((n, n), np.int8)
    new_focal = []
    for f in mem['focal']:
        nb = np.flatnonzero(A[f])
        if a[f] == 1:
            dnb = nb[a[nb] == 0]
            if len(dnb) > 0:
                j = rng.choice(dnb)
                R[min(f, j), max(f, j)] = -1
            else:
                cand = np.flatnonzero((a == 1) & (~A[f]))
                cand = cand[cand != f]
                if len(cand) > 0:
                    j = rng.choice(cand)
                    R[min(f, j), max(f, j)] = 1
            new_focal.append(f)
        else:
            cand = [k for k in np.flatnonzero(a == 1) if k not in mem['focal'] and k not in new_focal]
            new_focal.append(rng.choice(cand) if cand else f)
    mem['focal'] = new_focal
    Rr = planner_random(A, a, t, mem, rng, p=0.05)
    R = np.where(R != 0, R, Rr).astype(np.int8)
    return np.triu(R, 1)


def planner_probation(A, a, t, mem, rng, strikes=1):
    """History-conditioned (clock-free) variant: CC connect, DD cut; a C-D edge is conciliated
    (add, keep) while the defector's cumulative defection count <= strikes, otherwise excluded."""
    ndef = mem['ndef']
    s = a[:, None] + a[None, :]
    dcount = np.where(a[:, None] == 0, ndef[:, None], ndef[None, :])  # defector's count on CD pairs
    lenient = dcount <= strikes
    padd = np.where(s == 2, 1.0, np.where(s == 1, np.where(lenient, 1.0, 0.0), 0.0))
    pdel = np.where(s == 2, 0.0, np.where(s == 1, np.where(lenient, 0.0, 1.0), 1.0))
    return _sample(A, padd, pdel, rng)


ONE = np.ones((14, 1))
PLANNERS = {
    'static': planner_static,
    'random': planner_random,
    'clustering': planner_clustering,
    'encouragement': make_table_planner(ENC_CC, ENC_CD, ENC_DD),
    'neutral': planner_neutral,
    'maxconn': planner_maxconn,
    # ablations of the encouragement planner (CC and DD rows kept unless stated)
    'exclusion': make_table_planner(ENC_CC, np.hstack([0 * ONE, ONE]), ENC_DD),
    'always_conciliate': make_table_planner(ENC_CC, np.hstack([ONE, 0 * ONE]), ENC_DD),
    'enc_noDDcut': make_table_planner(ENC_CC, ENC_CD, np.zeros((14, 2))),
    'probation': planner_probation,
}


# ---------------- one game ----------------
def play(planner_name, rng, n=16, p=0.3, kappa=1.0, w_imit=0.0, temp=0.1, budget=None, route='random', T=T, behaviour='fitted'):
    """kappa: responsiveness of last-round defectors to neighbourhood cooperation (1 = paper model).
    w_imit: probability that a player uses payoff-biased (Fermi) imitation instead of the fitted rule."""
    planner = PLANNERS[planner_name]
    A = np.triu(rng.random((n, n)) < p, 1)
    A = A | A.T
    theta = rng.normal(MU_TH, SD_TH, n)
    d = np.full(n, D0)
    mem = {'ndef': np.zeros(n)}
    coop = np.zeros(T)
    gap, nrec = [], 0
    a_prev, pay_prev = None, np.zeros(n)
    for t in range(1, T + 1):
        deg = A.sum(1)
        if t == 1:
            pc = logistic(R1[0] + R1[1] * theta)
        else:
            xn = A.astype(float) @ a_prev
            xr = np.where(deg > 0, xn / np.maximum(deg, 1), 0.0)
            k = np.where(a_prev == 1, 1.0, kappa)
            pc = logistic(RL[0] + RL[1] * deg + k * (RL[2] * xn + RL[3] * xr) + theta)
            if behaviour == 'unanimity':           # stylised LLM-elicited rule (see llm pilot)
                unan = (deg > 0) & (xn == deg)
                pc = np.where(unan, np.where(a_prev == 1, 0.9, 0.7), 0.02)
                pc = np.where(deg == 0, 0.5, pc)
        a = (rng.random(n) < pc).astype(int)
        if t > 1 and w_imit > 0:
            use = rng.random(n) < w_imit
            for i in np.flatnonzero(use):
                nb = np.flatnonzero(A[i])
                if len(nb) == 0:
                    continue
                j = rng.choice(nb)
                if rng.random() < logistic((pay_prev[j] - pay_prev[i]) / temp):
                    a[i] = a_prev[j]
                else:
                    a[i] = a_prev[i]
        a = np.where(C * deg <= d, a, 0)          # capital constraint (SI eq. 2)
        pay = B * (A.astype(float) @ a) - C * a * deg
        d = d + pay
        coop[t - 1] = a.mean()
        if 0 < a.sum() < n:
            gap.append(pay[a == 0].mean() - pay[a == 1].mean())
        mem['ndef'] = mem['ndef'] + (a == 0)
        a_prev, pay_prev = a, pay
        if t == T:
            break
        R = planner(A, a, t, mem, rng)
        idx = np.argwhere(R != 0)
        if budget is not None and len(idx) > budget:
            idx = idx[rng.choice(len(idx), budget, replace=False)]
        nrec += len(idx)
        if len(idx):
            i, j = idx[:, 0], idx[:, 1]
            val = R[i, j]
            swap = rng.random(len(i)) < 0.5
            if route == 'targeted':                # C-D edges: ask the endpoint most likely to accept
                cd = a[i] != a[j]
                d_end = np.where(a[i] == 0, i, j)
                c_end = np.where(a[i] == 0, j, i)
                # add -> ask defector (referent = cooperator); delete -> ask cooperator (referent = defector)
                ref_cd = np.where(val == 1, c_end, d_end)
                ref = np.where(cd, ref_cd, np.where(swap, i, j))
            else:
                ref = np.where(swap, i, j)         # referent = endpoint not assigned the rec
            phi = np.array([PHI[(int(v), int(a[r]))] for v, r in zip(val, ref)])
            if route == 'bilateral':               # additions need both endpoints to accept
                other = np.where(ref == i, j, i)
                phi2 = np.array([PHI[(int(v), int(a[r]))] for v, r in zip(val, other)])
                phi = np.where(val == 1, phi * phi2, phi)
            acc = rng.random(len(i)) < phi
            for ii, jj, v in zip(i[acc], j[acc], val[acc]):
                A[ii, jj] = A[jj, ii] = (v == 1)
    dens = A[np.triu_indices(n, 1)].mean()
    return dict(coop=coop, final=coop[-1], mean=coop.mean(), capital=d.mean(), gini=gini(d),
                gap=np.mean(gap) if gap else np.nan, density=dens, nrec=nrec / (T - 1))


def run(planner_name, n_games=400, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [play(planner_name, rng, **kw) for _ in range(n_games)]


def make_switch_planner(s):
    """Two-phase C-D policy: conciliate (add 1, keep) for rounds t <= s, then exclude (no add, delete).
    C-C pairs always connected; D-D pairs always cut. Clock-indexed, valid for any horizon."""
    def planner(A, a, t, mem, rng):
        st = a[:, None] + a[None, :]
        cd_add, cd_del = (1.0, 0.0) if t <= s else (0.0, 1.0)
        padd = np.where(st == 2, 1.0, np.where(st == 1, cd_add, 0.0))
        pdel = np.where(st == 2, 0.0, np.where(st == 1, cd_del, 1.0))
        return _sample(A, padd, pdel, rng)
    return planner
