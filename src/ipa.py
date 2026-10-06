"""Individual-based pair approximation (IPA) of the planner-controlled network cooperation game.

State after the actions of round t, for every ordered pair (i, j):
    Q[g, i, j, e, a, b] = P(A_ij(t) = e, a_i(t) = a, a_j(t) = b)
for G independent draws g of the quenched dispositions theta. Planner step (exact, pairwise):
    R(e'=1|e,a,b) = e (1 - u^-_{a+b} phi^-_{ab}) + (1 - e) u^+_{a+b} phi^+_{ab}.
Action step (pair closure): conditional on its own previous action a_i, the contributions of i's
other potential partners are independent categorical variables {absent, defecting nbr, cooperating nbr}
with probabilities read from the corresponding pair states; the distribution of (degree, # cooperating
neighbours) is obtained by exact dynamic programming, excluding the focal partner j.

The objective is differentiable in the controls, so switching functions (costate-weighted marginal
values of a tie of each type, i.e. the gradient of the discrete-time Hamiltonian) are obtained by
reverse-mode differentiation.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint

B, C, D0 = 0.10, 0.05, 1.0
MU_TH, SD_TH = -0.304, 2.410
R1 = (1.807, 0.818)
RL = (-0.010, -0.193, 0.370, 1.521)
PHI_DEL_D, PHI_DEL_C, PHI_ADD_D, PHI_ADD_C = 0.774, 0.085, 0.287, 0.909
DT = torch.float64


def acceptance(route='random', phi_scale_cd_add=1.0):
    """Return phi_add[a,b], phi_del[a,b] (a = action of i, b = action of j)."""
    pa = torch.zeros(2, 2, dtype=DT); pd = torch.zeros(2, 2, dtype=DT)
    pa[1, 1] = PHI_ADD_C; pd[1, 1] = PHI_DEL_C
    pa[0, 0] = PHI_ADD_D; pd[0, 0] = PHI_DEL_D
    if route == 'random':
        cd_add = 0.5 * (PHI_ADD_C + PHI_ADD_D); cd_del = 0.5 * (PHI_DEL_C + PHI_DEL_D)
    else:  # targeted: additions to the defector (referent C), deletions to the cooperator (referent D)
        cd_add = PHI_ADD_C; cd_del = PHI_DEL_D
    cd_add = min(1.0, cd_add * phi_scale_cd_add)
    pa[0, 1] = pa[1, 0] = cd_add; pd[0, 1] = pd[1, 0] = cd_del
    return pa, pd


def draw_theta(G, n, seed=0, quantile=False):
    g = torch.Generator().manual_seed(seed)
    if quantile:
        q = (torch.arange(n, dtype=DT) + 0.5) / n
        th = MU_TH + SD_TH * math.sqrt(2) * torch.erfinv(2 * q - 1)
        return th.expand(G, n).clone()
    return MU_TH + SD_TH * torch.randn(G, n, generator=g, dtype=DT)


class IPA:
    def __init__(self, theta, T=15, kappa=1.0, p0=0.3, route='random', phi_scale_cd_add=1.0,
                 use_checkpoint=True, depth=1):
        dev = theta.device
        self.depth = depth
        H = 2 ** depth
        self.H = H
        self.cur = (torch.arange(H, device=dev) & 1)
        M = torch.zeros(H, 2, H, dtype=DT, device=dev)
        for h in range(H):
            for x in range(2):
                M[h, x, ((h << 1) | x) & (H - 1)] = 1.0
        self.M = M
        self.dev = dev
        self.theta = theta.to(DT)
        self.G, self.n = theta.shape
        self.T, self.p0 = T, p0
        self.kappa = (torch.as_tensor(kappa, dtype=DT, device=dev) * torch.ones(self.G, dtype=DT, device=dev))
        pa, pd = acceptance(route, phi_scale_cd_add)
        self.pa, self.pd = pa.to(dev), pd.to(dev)
        self.ck = use_checkpoint
        n = self.n
        self.offdiag = (1 - torch.eye(n, dtype=DT, device=dev))
        K = n  # degree 0..n-1
        k = torch.arange(K + 1, dtype=DT, device=dev)[:, None]
        x = torch.arange(K + 1, dtype=DT, device=dev)[None, :]
        valid = (x <= k).to(DT)
        xr = torch.where(k > 0, x / k.clamp(min=1), torch.zeros_like(x))
        self.k_grid, self.x_grid, self.xr_grid, self.valid = k, x, xr, valid
        self.K = K
        self.stype = torch.tensor([[0, 1], [1, 2]])  # pair type index a+b

    # -------------------------------------------------------------- helpers
    def eta_table(self, kap):
        """F[g,i,a,k,x] = P(cooperate | own previous action a, degree k, x coop nbrs)."""
        kk, xx, xr = self.k_grid, self.x_grid, self.xr_grid
        base = RL[0] + RL[1] * kk
        soc = RL[2] * xx + RL[3] * xr
        kap_a = torch.stack([kap, torch.ones_like(kap)], -1).view(self.G, 1, 2, 1, 1)
        eta = base.view(1, 1, 1, *base.shape) + kap_a * soc.view(1, 1, 1, *soc.shape) \
            + self.theta.view(self.G, self.n, 1, 1, 1)
        return torch.sigmoid(eta) * self.valid

    def init_state(self):
        """Pair state Q[g,i,j,e,h_i,h_j]; h encodes the last `depth` actions, bit 0 = current action."""
        H = self.H
        m1 = torch.sigmoid(R1[0] + R1[1] * self.theta)                  # (G,n)
        ph = torch.zeros(self.G, self.n, H, dtype=DT, device=self.dev)
        ph[..., 0] = 1 - m1
        ph[..., H - 1] = m1                                            # history padded with the first action
        pe = torch.tensor([1 - self.p0, self.p0], dtype=DT, device=self.dev)
        Q = pe.view(1, 1, 1, 2, 1, 1) * ph.view(self.G, self.n, 1, 1, H, 1) * ph.view(self.G, 1, self.n, 1, 1, H)
        return Q * self.offdiag.view(1, self.n, self.n, 1, 1, 1)

    def _expand_u(self, u):
        """(.., 2, 2) control indexed by current actions -> (.., H, H) indexed by histories."""
        cur = self.cur
        return u[..., cur, :][..., :, cur]

    def planner_step(self, Q, uadd, udel):
        """uadd/udel broadcastable to (G,n,n,2,2) = recommendation probs for (a_i, a_j)."""
        pa = self._expand_u(self.pa); pd = self._expand_u(self.pd)
        ua = self._expand_u(uadd); ud = self._expand_u(udel)
        keep = 1 - ud * pd
        addp = ua * pa
        R1_ = Q[:, :, :, 1] * keep + Q[:, :, :, 0] * addp
        R0_ = Q[:, :, :, 1] * (1 - keep) + Q[:, :, :, 0] * (1 - addp)
        return torch.stack([R0_, R1_], 3)

    def action_step(self, R, kap):
        G, n, K, H = self.G, self.n, self.K, self.H
        cur = self.cur                                                 # (H,) current action of history h
        w = R.sum(dim=(3, 5))                                          # (G,i,j,h_i)
        wsafe = w.clamp(min=1e-300)
        Rc = torch.stack([R[..., cur == 0].sum(-1), R[..., cur == 1].sum(-1)], -1)   # (G,i,j,e,h_i,b)
        pi0 = Rc[:, :, :, 0].sum(-1) / wsafe                           # (G,i,j,h)
        piD = Rc[:, :, :, 1, :, 0] / wsafe
        piC = Rc[:, :, :, 1, :, 1] / wsafe
        eye = torch.eye(n, dtype=DT, device=self.dev).view(1, n, n, 1)
        pi0 = pi0 * (1 - eye) + eye; piD = piD * (1 - eye); piC = piC * (1 - eye)
        D = torch.zeros(G, n, H, n + 1, K + 1, K + 1, dtype=DT, device=self.dev)
        D[..., 0, 0] = 1.0
        excl = torch.arange(n + 1, device=self.dev)
        for kidx in range(n):
            p0 = pi0[:, :, kidx, :].reshape(G, n, H, 1, 1, 1)
            pD = piD[:, :, kidx, :].reshape(G, n, H, 1, 1, 1)
            pC = piC[:, :, kidx, :].reshape(G, n, H, 1, 1, 1)
            shK = torch.zeros_like(D); shK[..., 1:, :] = D[..., :-1, :]
            shKX = torch.zeros_like(D); shKX[..., 1:, 1:] = D[..., :-1, :-1]
            newD = p0 * D + pD * shK + pC * shKX
            mask = (excl == kidx).view(1, 1, 1, n + 1, 1, 1)
            D = torch.where(mask, D, newD)
        F = self.eta_table(kap)[:, :, cur]                             # (G,n,H,K+1,K+1)
        Dn = D[:, :, :, n]
        p_node = (Dn * F).sum((-1, -2))                                # (G,n,H)
        Dx = D[:, :, :, :n]                                            # (G,i,h,j,k,x)
        FD = torch.zeros_like(F); FD[..., :-1, :] = F[..., 1:, :]
        FC = torch.zeros_like(F); FC[..., :-1, :-1] = F[..., 1:, 1:]
        q_abs = (Dx * F.unsqueeze(3)).sum((-1, -2)).permute(0, 1, 3, 2)   # (G,i,j,h)
        q_D = (Dx * FD.unsqueeze(3)).sum((-1, -2)).permute(0, 1, 3, 2)
        q_C = (Dx * FC.unsqueeze(3)).sum((-1, -2)).permute(0, 1, 3, 2)
        qe1 = torch.where(cur.view(1, 1, 1, 1, H) == 1, q_C.unsqueeze(-1), q_D.unsqueeze(-1))  # (G,i,j,h_i,h_j)
        Pi = torch.stack([q_abs.unsqueeze(-1).expand(G, n, n, H, H), qe1], 3)   # (G,i,j,e',h_i,h_j)
        Pj = Pi.permute(0, 2, 1, 3, 5, 4)
        Pi2 = torch.stack([1 - Pi, Pi], -1)
        Pj2 = torch.stack([1 - Pj, Pj], -1)
        Qn = torch.einsum('gijeab,gijeabx,gijeaby,axc,byd->gijecd', R, Pi2, Pj2, self.M, self.M)
        Qn = Qn * self.offdiag.view(1, n, n, 1, 1, 1)
        return Qn, p_node

    def marginals(self, Q):
        """Node marginals over current action: (G,i,2)."""
        n = self.n
        mh = Q.sum(dim=(3, 5)).sum(2) / (n - 1)                        # (G,i,h)
        return torch.stack([mh[..., self.cur == 0].sum(-1), mh[..., self.cur == 1].sum(-1)], -1), mh

    def observables(self, Q):
        n = self.n
        m, _ = self.marginals(Q)
        coop = m[..., 1].mean(1)
        E1 = Q[:, :, :, 1]
        Ec = torch.stack([E1[..., self.cur == 0].sum(-1), E1[..., self.cur == 1].sum(-1)], -1)
        Ec = torch.stack([Ec[..., self.cur == 0, :].sum(-2), Ec[..., self.cur == 1, :].sum(-2)], -2)  # (G,i,j,a,b)
        a_ = torch.tensor([0.0, 1.0], dtype=DT, device=self.dev)
        pay = (Ec * (B * a_.view(1, 1, 1, 1, 2) - C * a_.view(1, 1, 1, 2, 1))).sum((1, 2, 3, 4)) / n
        ecc = Ec[..., 1, 1].sum((1, 2)) / 2; edd = Ec[..., 0, 0].sum((1, 2)) / 2
        ecd = (Ec[..., 0, 1] + Ec[..., 1, 0]).sum((1, 2)) / 2
        return coop, pay, ecc, ecd, edd

    # -------------------------------------------------------------- rollout
    def rollout(self, ctrl, record=False):
        """ctrl(t, Q) -> (uadd, udel) broadcastable to (G,n,n,2,2); t = 1..T-1."""
        Q = self.init_state()
        coops, pays, edges = [], [], []
        c, pay, ecc, ecd, edd = self.observables(Q)
        coops.append(c); pays.append(pay); edges.append(torch.stack([ecc, ecd, edd], -1))
        x_node = None
        for t in range(1, self.T):
            uadd, udel = ctrl(t, Q)
            R = self.planner_step(Q, uadd, udel)
            if self.ck and torch.is_grad_enabled():
                Qn, p_node = checkpoint(self.action_step, R, self.kappa, use_reentrant=False)
            else:
                Qn, p_node = self.action_step(R, self.kappa)
            _, mh = self.marginals(Q)
            x_node = (mh * p_node).sum(-1).mean(1)
            Q = Qn
            c, pay, ecc, ecd, edd = self.observables(Q)
            coops.append(c); pays.append(pay); edges.append(torch.stack([ecc, ecd, edd], -1))
        coop_tr = torch.stack(coops, 1); pay_tr = torch.stack(pays, 1)
        out = dict(final=x_node, final_pair=coops[-1], coop=coop_tr, capital=D0 + pay_tr.sum(1),
                   pay=pay_tr, edges=torch.stack(edges, 1), mcoop=coop_tr.mean(1))
        if record:
            out['Q_last'] = Q
        return out


# ------------------------------------------------------------------ controls
def open_loop(u, group=None):
    """u: tensor (T-1, 3, 2) shared by all batch members, or (Ng, T-1, 3, 2) with group[g] in 0..Ng-1.
    Type index 0 = DD, 1 = CD, 2 = CC; [..., 0] = add, [..., 1] = delete."""
    st = torch.tensor([[0, 1], [1, 2]], device=u.device)

    def ctrl(t, Q):
        if u.dim() == 3:
            ua = u[t - 1, :, 0][st]; ud = u[t - 1, :, 1][st]
            return ua.view(1, 1, 1, 2, 2), ud.view(1, 1, 1, 2, 2)
        ug = u[group, t - 1]                                           # (G,3,2)
        ua = ug[:, :, 0][:, st]; ud = ug[:, :, 1][:, st]               # (G,2,2)
        G = ug.shape[0]
        return ua.view(G, 1, 1, 2, 2), ud.view(G, 1, 1, 2, 2)
    return ctrl


def threshold_ctrl(thr, theta, tau=0.25, group=None, thr_c=None):
    """Feedback policy: conciliate a C-D pair iff the defector's theta exceeds thr[t-1] (soft, temperature tau).
    thr: (T-1,) or (Ng, T-1) with group index. CC pairs connected, DD pairs cut.
    Optional thr_c: additionally require the cooperator's theta to exceed thr_c[t-1]."""
    G, n = theta.shape
    dev = theta.device

    def ctrl(t, Q):
        th_t = thr[t - 1] if thr.dim() == 1 else thr[group, t - 1].view(G, 1)
        s_node = torch.sigmoid((theta - th_t) / tau)                   # (G,n)
        cj = s_node.view(G, 1, n).expand(G, n, n); ci = s_node.view(G, n, 1).expand(G, n, n)
        if thr_c is not None:
            tc = thr_c[t - 1] if thr_c.dim() == 1 else thr_c[group, t - 1].view(G, 1)
            r_node = torch.sigmoid((theta - tc) / tau)
            rj = r_node.view(G, 1, n).expand(G, n, n); ri = r_node.view(G, n, 1).expand(G, n, n)
            cj = cj * ri; ci = ci * rj
        one = torch.ones(G, n, n, dtype=DT, device=dev); zero = torch.zeros_like(one)
        # ua[..., a_i, a_j]
        ua = torch.stack([torch.stack([zero, ci], -1), torch.stack([cj, one], -1)], -2)
        ud = torch.stack([torch.stack([one, 1 - ci], -1), torch.stack([1 - cj, zero], -1)], -2)
        return ua, ud
    return ctrl


def switch_u(T, s, device='cpu'):
    u = torch.zeros(T - 1, 3, 2, dtype=DT, device=device)
    u[:, 2, 0] = 1.0; u[:, 0, 1] = 1.0
    for t in range(1, T):
        if t <= s:
            u[t - 1, 1, 0] = 1.0
        else:
            u[t - 1, 1, 1] = 1.0
    return u
