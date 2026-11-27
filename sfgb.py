"""Spectral Fuzzy Granular-Ball (DFGB) construction and prototype aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GranularBallStructure:
    centers: np.ndarray          # (M, F) spectral centers
    radii: np.ndarray            # (M,) 90th-percentile radii
    members: List[List[int]]     # variable indices per ball
    embeddings: np.ndarray       # (d, F) variable spectral embeddings
    relevances: np.ndarray       # (d,) target spectral coherence relevance in [0, 1]
    hard_assignments: np.ndarray # (d,) integer ball id in [0, M-1]
    rho_task: float              # task-relevance metric weight


def compute_spectral_embeddings(
    X_fit: np.ndarray,
    y_hist_fit: np.ndarray,
    n_log_bands: int = 8,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute log-band spectral signatures and target-coherence relevance on fit windows only.

    Args:
        X_fit: (N, d, L) standardized exogenous fit windows
        y_hist_fit: (N, L) standardized target fit windows
    Returns:
        embeddings: (d, n_log_bands * 2) normalized [band energy share, band target coherence]
        relevances: (d,) target-energy-weighted spectral coherence in [0, 1]
    """
    N, d, L = X_fit.shape
    # rFFT along lookback axis L
    Xf = np.fft.rfft(X_fit, axis=-1)       # (N, d, K)
    Yf = np.fft.rfft(y_hist_fit, axis=-1)  # (N, K)

    # Drop DC component (index 0) for oscillatory spectral pattern
    Xf = Xf[:, :, 1:]
    Yf = Yf[:, 1:]
    K = Xf.shape[-1]

    # Power spectra and cross-spectra averaged across fit windows
    Sxx = np.mean(np.abs(Xf) ** 2, axis=0)                      # (d, K)
    Syy = np.mean(np.abs(Yf) ** 2, axis=0)                      # (K,)
    Sxy = np.mean(Xf * np.conj(Yf)[:, None, :], axis=0)         # (d, K)

    coh = (np.abs(Sxy) ** 2) / (Sxx * Syy[None, :] + eps)       # (d, K) in [0, 1]
    coh = np.clip(coh, 0.0, 1.0)

    # Target-energy-weighted relevance per channel
    y_weights = Syy / (np.sum(Syy) + eps)                       # (K,)
    relevances = np.sum(coh * y_weights[None, :], axis=-1)      # (d,)
    relevances = np.clip(relevances, 0.0, 1.0)

    # Adaptive logarithmic frequency pooling
    n_bands = min(n_log_bands, K)
    edges = np.unique(np.round(np.geomspace(1, K + 1, num=n_bands + 1)).astype(int) - 1)
    if edges[-1] < K:
        edges = np.append(edges, K)

    energy_share = Sxx / (np.sum(Sxx, axis=-1, keepdims=True) + eps)
    band_e = []
    band_c = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        band_e.append(np.sum(energy_share[:, lo:hi], axis=-1))
        band_c.append(np.mean(coh[:, lo:hi], axis=-1))

    emb = np.concatenate([np.stack(band_e, axis=1), np.stack(band_c, axis=1)], axis=1)
    # Standardize embedding columns across channels for well-scaled distances
    emb_std = emb.std(axis=0, keepdims=True)
    emb = (emb - emb.mean(axis=0, keepdims=True)) / np.where(emb_std < eps, 1.0, emb_std)
    return emb.astype(np.float32), relevances.astype(np.float32)


def _task_dist_sq(
    emb: np.ndarray,
    rel: np.ndarray,
    center_emb: np.ndarray,
    center_rel: float,
    rho: float,
) -> np.ndarray:
    diff = emb - center_emb[None, :]
    return np.sum(diff * diff, axis=-1) + rho * (rel - center_rel) ** 2


def _jsg_score(
    indices: List[int],
    emb: np.ndarray,
    rel: np.ndarray,
    d_total: int,
    rho: float,
    eta: float = 0.35,
    lam_r: float = 0.25,
) -> Tuple[float, np.ndarray, float, float]:
    """Justifiable Spectral Granularity (JSG): coverage^eta * exp(-lam_r * dispersion)."""
    idx = np.asarray(indices, dtype=np.int64)
    c_emb = emb[idx].mean(axis=0)
    c_rel = float(rel[idx].mean())
    d2 = _task_dist_sq(emb[idx], rel[idx], c_emb, c_rel, rho)
    mean_d2 = float(np.mean(d2))
    radius = float(np.sqrt(np.quantile(d2, 0.90) + 1e-8))
    coverage = (len(indices) / float(d_total)) ** eta
    specificity = float(np.exp(-lam_r * mean_d2))
    return coverage * specificity, c_emb, c_rel, radius


def _bisect_ball(
    indices: List[int],
    emb: np.ndarray,
    rel: np.ndarray,
    rho: float,
    n_iters: int = 8,
) -> Tuple[List[int], List[int]]:
    """Deterministic farthest-seed + task-aware 2-means refinement."""
    idx = np.asarray(indices, dtype=np.int64)
    c_emb = emb[idx].mean(axis=0)
    c_rel = float(rel[idx].mean())
    d2_c = _task_dist_sq(emb[idx], rel[idx], c_emb, c_rel, rho)
    s1 = idx[int(np.argmax(d2_c))]
    d2_s1 = _task_dist_sq(emb[idx], rel[idx], emb[s1], float(rel[s1]), rho)
    s2 = idx[int(np.argmax(d2_s1))]

    c1_e, c1_r = emb[s1].copy(), float(rel[s1])
    c2_e, c2_r = emb[s2].copy(), float(rel[s2])

    left, right = [], []
    for _ in range(n_iters):
        d1 = _task_dist_sq(emb[idx], rel[idx], c1_e, c1_r, rho)
        d2 = _task_dist_sq(emb[idx], rel[idx], c2_e, c2_r, rho)
        mask = d1 <= d2
        if mask.all() or (~mask).all():
            order = np.argsort(d1 - d2)
            mid = len(idx) // 2
            left = idx[order[:mid]].tolist()
            right = idx[order[mid:]].tolist()
            break
        left = idx[mask].tolist()
        right = idx[~mask].tolist()
        c1_e, c1_r = emb[left].mean(axis=0), float(rel[left].mean())
        c2_e, c2_r = emb[right].mean(axis=0), float(rel[right].mean())

    return left, right


def build_spectral_granular_balls(
    X_fit: np.ndarray,
    y_hist_fit: np.ndarray,
    max_balls: Optional[int] = None,
    min_ball_size: Optional[int] = None,
    rho_task: float = 1.5,
    eta: float = 0.25,
    lam_r: float = 0.35,
    split_gain_tol: float = 0.01,
) -> GranularBallStructure:
    """Build spectral granular balls via JSG binary Accept/Reject split-merge."""
    d = X_fit.shape[1]
    emb, rel = compute_spectral_embeddings(X_fit, y_hist_fit)

    if max_balls is None:
        max_balls = max(2, min(16, int(round(np.sqrt(d) * 1.2))))
    if min_ball_size is None:
        min_ball_size = max(2, int(round(0.03 * d))) if d >= 16 else 1

    # Top-down binary split partitioning
    active_balls: List[List[int]] = [list(range(d))]
    frozen: List[bool] = [False]

    while len(active_balls) < max_balls:
        best_idx = -1
        best_gain = -1e9
        best_children: Tuple[List[int], List[int]] = ([], [])

        for bi, members in enumerate(active_balls):
            if frozen[bi] or len(members) < 2 * min_ball_size:
                frozen[bi] = True
                continue
            parent_jsg, _, _, _ = _jsg_score(members, emb, rel, d, rho_task, eta, lam_r)
            left, right = _bisect_ball(members, emb, rel, rho_task)
            if len(left) < min_ball_size or len(right) < min_ball_size:
                frozen[bi] = True
                continue
            j_l, _, _, _ = _jsg_score(left, emb, rel, d, rho_task, eta, lam_r)
            j_r, _, _, _ = _jsg_score(right, emb, rel, d, rho_task, eta, lam_r)
            weighted_child = (len(left) * j_l + len(right) * j_r) / float(len(members))
            rel_gain = (weighted_child - parent_jsg) / (parent_jsg + 1e-8)
            if rel_gain > split_gain_tol and rel_gain > best_gain:
                best_gain = rel_gain
                best_idx = bi
                best_children = (left, right)

        if best_idx < 0:
            break
        active_balls.pop(best_idx)
        frozen.pop(best_idx)
        active_balls.extend([best_children[0], best_children[1]])
        frozen.extend([False, False])

    # Merge pass: merge close homogeneous pairs if JSG does not degrade
    merged = True
    while merged and len(active_balls) > 2:
        merged = False
        best_pair = None
        best_merge_gain = 0.0
        for i in range(len(active_balls)):
            ji, ci_e, ci_r, _ = _jsg_score(active_balls[i], emb, rel, d, rho_task, eta, lam_r)
            for j in range(i + 1, len(active_balls)):
                jj, cj_e, cj_r, _ = _jsg_score(active_balls[j], emb, rel, d, rho_task, eta, lam_r)
                comb = active_balls[i] + active_balls[j]
                jc, _, _, _ = _jsg_score(comb, emb, rel, d, rho_task, eta, lam_r)
                sep_weighted = (len(active_balls[i]) * ji + len(active_balls[j]) * jj) / float(len(comb))
                if jc >= sep_weighted + best_merge_gain:
                    best_merge_gain = jc - sep_weighted
                    best_pair = (i, j)
        if best_pair is not None:
            i, j = best_pair
            comb = active_balls[i] + active_balls[j]
            active_balls.pop(j)
            active_balls.pop(i)
            active_balls.append(comb)
            merged = True

    centers, radii = [], []
    hard_assign = np.zeros(d, dtype=np.int64)
    for m, members in enumerate(active_balls):
        _, c_emb, _, rad = _jsg_score(members, emb, rel, d, rho_task, eta, lam_r)
        centers.append(c_emb)
        radii.append(max(rad, 0.25))
        for ch in members:
            hard_assign[ch] = m

    return GranularBallStructure(
        centers=np.asarray(centers, dtype=np.float32),
        radii=np.asarray(radii, dtype=np.float32),
        members=active_balls,
        embeddings=emb,
        relevances=rel,
        hard_assignments=hard_assign,
        rho_task=rho_task,
    )


class SpectralFuzzyGranularBallLayer(nn.Module):
    """Differentiable Dynamic Fuzzy Granular-Ball (DFGB) prototype + inconsistency layer.

    Combines:
      1. Structural spectral fuzzy membership \mu_{j,m}^{spec} from JSG granular balls
      2. Dynamic Cauchy consistency membership \mu_{t,j,m}^{dyn} in (0, 1] for spike/noise rejection
         and bounded fuzzy inconsistency degree \psi_t^{(m)} in [0, 1).

    Ablation modes:
      - mode='soft'     : structural soft fuzzy + dynamic Cauchy fuzzy membership (proposed)
      - mode='soft_stat': structural soft fuzzy only (no dynamic Cauchy rejection)
      - mode='hard'     : hard argmax ball membership (no fuzziness)
    """

    def __init__(self, structure: GranularBallStructure, mode: str = "soft"):
        super().__init__()
        assert mode in {"soft", "soft_stat", "hard"}
        self.mode = mode
        d, _ = structure.embeddings.shape
        M = structure.centers.shape[0]
        self.num_features = d
        self.num_balls = M

        # Precompute squared task-aware distance matrix D2: (d, M)
        ball_rels = np.array(
            [structure.relevances[m_idx].mean() for m_idx in structure.members],
            dtype=np.float32,
        )
        d2 = np.zeros((d, M), dtype=np.float32)
        for m in range(M):
            diff = structure.embeddings - structure.centers[m][None, :]
            d2[:, m] = np.sum(diff * diff, axis=-1) + structure.rho_task * (
                structure.relevances - ball_rels[m]
            ) ** 2

        self.register_buffer("dist_sq", torch.from_numpy(d2))  # (d, M)
        hard_one_hot = np.zeros((d, M), dtype=np.float32)
        hard_one_hot[np.arange(d), structure.hard_assignments] = 1.0
        self.register_buffer("hard_mu", torch.from_numpy(hard_one_hot))

        # Learnable structural ball temperature (initialized from granular-ball radius)
        init_log_tau = np.log(np.maximum(structure.radii, 0.35))
        self.log_tau = nn.Parameter(torch.from_numpy(init_log_tau.astype(np.float32)))

        # Learnable Cauchy tolerance scale \kappa > 0 for O(B*d*L) dynamic consistency membership
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(2.5)), dtype=torch.float32))

        # Learnable channel reliability gate initialized from target spectral coherence
        rel_clipped = np.clip(structure.relevances, 0.05, 0.95)
        init_logit = np.log(rel_clipped / (1.0 - rel_clipped))
        self.rel_logit = nn.Parameter(torch.from_numpy(init_logit.astype(np.float32)))

    def get_membership(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_spec: (d, M), rho: (d,))."""
        rho = torch.sigmoid(self.rel_logit)  # (d,)
        if self.mode == "hard":
            mu = self.hard_mu
        else:
            tau = torch.exp(self.log_tau).clamp(min=0.25, max=10.0)  # (M,)
            logits = -self.dist_sq / (2.0 * tau.unsqueeze(0) ** 2)
            mu = torch.softmax(logits, dim=-1)  # (d, M), rows sum to 1
        return mu, rho

    def forward(self, x_exog: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute denoised fuzzy ball prototypes and bounded fuzzy inconsistency degree.

        Args:
            x_exog: (B, d, L)
        Returns:
            prototypes: (B, M, L)
            inconsistency: (B, M, L) strictly bounded in [0, 1)
        """
        mu_spec, rho = self.get_membership()              # (d, M), (d,)
        w_spec = mu_spec * rho.unsqueeze(-1)              # (d, M)
        w_spec_norm = w_spec / (w_spec.sum(dim=0, keepdim=True) + 1e-6)

        # Coarse structural prototype: (B, M, L)
        proto_init = torch.einsum("dm,bdl->bml", w_spec_norm, x_exog)

        if self.mode in {"hard", "soft_stat"}:
            # Bounded inconsistency proxy for ablation parity
            mean_x2 = torch.einsum("dm,bdl->bml", w_spec_norm, x_exog * x_exog)
            var_m = (mean_x2 - proto_init * proto_init).clamp(min=0.0)
            inconsistency = var_m / (var_m + 4.0)  # bounded in [0, 1)
            return proto_init, inconsistency

        # O(B*d*L) Dynamic Cauchy consistency membership via fuzzy ball reconstruction:
        #   anchor_m = coarse prototype on softly clamped sensors
        #   x_recon_j = sum_m mu_{j,m}^{spec} * anchor_m
        #   mu_{t,j}^{dyn} = 1 / (1 + ((x_{t,j} - x_recon_{t,j}) / kappa)^2)
        x_clamped = x_exog.clamp(min=-4.0, max=4.0)
        anchor = torch.einsum("dm,bdl->bml", w_spec_norm, x_clamped)   # (B, M, L)
        x_recon = torch.einsum("dm,bml->bdl", mu_spec, anchor)         # (B, d, L)

        kappa = torch.exp(self.log_kappa).clamp(min=0.5, max=6.0)
        mu_dyn = 1.0 / (1.0 + ((x_exog - x_recon) / kappa) ** 2)       # (B, d, L) in (0, 1]

        num = torch.einsum("dm,bdl->bml", w_spec, x_exog * mu_dyn)
        den = torch.einsum("dm,bdl->bml", w_spec, mu_dyn) + 1e-6
        prototypes = num / den

        # Bounded fuzzy inconsistency degree \psi_t^{(m)} in [0, 1)
        mu_s_norm = mu_spec / (mu_spec.sum(dim=0, keepdim=True) + 1e-6)
        consistency = torch.einsum("dm,bdl->bml", mu_s_norm, mu_dyn)
        inconsistency = (1.0 - consistency).clamp(min=0.0, max=1.0)
        return prototypes, inconsistency
