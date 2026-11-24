"""SADL's Supply Regularizer (Sun et al. 2026, "SADL: On the Demand-Supply Geometry of
Representation Collapse", arXiv 2611.21596; reference code github.com/zju3dv/SADL, MIT).

A Demand Regularizer D (SIGReg here) shapes the endpoint Z = P(H). With a BatchNorm
projector P, a component every row of H shares lies in P's Jacobian null space, so neither
D nor any loss on Z can see it, and H can drift to "one shared vector plus a sliver of
per-sample variation" while Z still looks healthy. SADL compares two scores in [0, 1]:

    demand  q = q_reg^(1-alpha) * q_exc^alpha     (detached)
              q_reg = clip(1 - D(Z) / D(0), 0, 1)  how well D is satisfied
              q_exc = F_exc(sg(Z))                 how much sample variation Z carries
    supply  k = min(F_exc(H), F_tr(H, sg(Z)))      variation H has, and passes on to Z

and penalizes only a shortfall, S = [log((q + eps) / (k + eps))]_+^2. Its gradient lands on
H directly, bypassing P.

F_exc is a normalized log det of Cov(U) over the UNcentered energy mean(U^2), so a shared
mean drags it down as much as a low-rank covariance does. F_tr is the same log det over the
regularized canonical correlations of (H, Z). This is the reference code's
`reference_excitation_free_energy` mode, with one change: the log dets are normalized by
the attainable rank min(d, N - 1) (the paper's Eq. 3/5) rather than by d, so a batch with
fewer rows than dims can still reach k = 1 and switch the penalty off.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["excitation", "transmission", "SupplyRegularizer"]


def _flat(x):
    return x.reshape(-1, x.shape[-1]).float()


def _logdet_fraction(mat, rank, snr):
    eye = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    sign, logdet = torch.linalg.slogdet(eye + snr * mat)
    if not bool((sign > 0).all()):
        raise FloatingPointError("SADL log-det matrix is not positive definite")
    return (logdet / (rank * math.log1p(snr))).clamp(0.0, 1.0)


def excitation(u, snr=9.0, eps=1e-6):
    """(..., d) rows -> (F_exc, f): the normalized log det of Cov(u) / mean(u^2), and f,
    the fraction of the energy mean(u^2) carried by sample variation (1 - f is the shared
    mean's share). F_exc is one exactly for an isotropic, zero-mean batch."""
    u = _flat(u)
    n, d = u.shape
    centered = u - u.mean(0, keepdim=True)
    cov = centered.T @ centered / n
    energy = u.square().mean().clamp_min(eps)
    rel = cov / energy
    frac = (cov.diagonal().mean() / energy).clamp(0.0, 1.0)
    return _logdet_fraction(rel, max(1, min(d, n - 1)), snr), frac


def transmission(h, z, ridge=1e-3, snr=9.0, eps=1e-6):
    """Aligned (..., d_h) and (..., d_z) rows -> F_tr: the normalized log det over the
    ridge-regularized canonical correlations of h and z. The ridge is relative to each
    side's mean variance and keeps its gradient, so rescaling h cannot raise the score."""
    h, z = _flat(h), _flat(z)
    n = h.shape[0]
    h = h - h.mean(0, keepdim=True)
    z = z - z.mean(0, keepdim=True)
    cov_h, cov_z, cov_hz = h.T @ h / n, z.T @ z / n, h.T @ z / n
    scale_h = cov_h.diagonal().mean().clamp_min(eps)
    scale_z = cov_z.diagonal().mean().clamp_min(eps)
    chol_h = torch.linalg.cholesky(
        cov_h + ridge * scale_h * torch.eye(h.shape[1], device=h.device))
    chol_z = torch.linalg.cholesky(
        cov_z + ridge * scale_z * torch.eye(z.shape[1], device=z.device))
    left = torch.linalg.solve_triangular(chol_h, cov_hz, upper=False)
    canon = torch.linalg.solve_triangular(chol_z, left.T, upper=False).T
    gram = canon @ canon.T if canon.shape[0] <= canon.shape[1] else canon.T @ canon
    return _logdet_fraction(gram, max(1, min(gram.shape[0], n - 1)), snr)


class SupplyRegularizer(nn.Module):
    """forward(h, z, demand, demand_at_zero, z_demand=None) -> (S, logs).

    `h` is the supply site (with gradient), `z` the endpoint it feeds (detached here),
    aligned row for row with `h`. `demand` is D(Z) and `demand_at_zero` D(0), the
    regularizer on an all-zero input of the same bagging. `z_demand` is the tensor D
    scores, for q_exc, when that is not `z` itself (a regularizer head after the
    projector); it need not align with `h`. Everything runs in fp32 with autocast off.
    """

    def __init__(self, alpha=0.5, ridge=1e-3, snr=9.0, eps=1e-6):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must lie in [0, 1]"
        self.alpha, self.ridge, self.snr, self.eps = float(alpha), float(ridge), float(snr), float(eps)

    def forward(self, h, z, demand, demand_at_zero, z_demand=None):
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = z.detach()
            k_exc, f_h = excitation(h, self.snr, self.eps)
            k_tr = transmission(h, z, self.ridge, self.snr, self.eps)
            with torch.no_grad():
                q_exc, f_z = excitation(z if z_demand is None else z_demand, self.snr, self.eps)
                q_reg = (1.0 - demand.detach().float()
                         / max(float(demand_at_zero), self.eps)).clamp(0.0, 1.0)
                q = q_reg.pow(1.0 - self.alpha) * q_exc.pow(self.alpha)
            k = torch.minimum(k_exc, k_tr)
            gap = F.relu(torch.log(q + self.eps) - torch.log(k + self.eps))
            loss = gap.square()
        logs = {"k_exc": k_exc, "k_tr": k_tr, "q": q, "q_reg": q_reg, "q_exc": q_exc,
                "f_h": f_h, "f_z": f_z, "active": (gap > 0).float()}
        return loss, {key: v.detach() for key, v in logs.items()}
