"""Full-spectrum complex mixing backbone and differentiable ellipsoidal TSK interval head."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfgb import GranularBallStructure, SpectralFuzzyGranularBallLayer


class ComplexSpectralMixLayer(nn.Module):
    """Lightweight learnable complex frequency-domain filter (no manual L/M/H cutoff)."""

    def __init__(self, num_channels: int, lookback: int, hidden_dim: int = 32):
        super().__init__()
        self.num_channels = num_channels
        self.lookback = lookback
        self.n_freq = lookback // 2 + 1

        # Channel-wise complex spectral scaling (learned across all frequencies)
        self.w_real = nn.Parameter(torch.ones(num_channels, self.n_freq) * 0.5)
        self.w_imag = nn.Parameter(torch.zeros(num_channels, self.n_freq))

        # Compact cross-channel linear projection in frequency magnitude+real space
        self.proj = nn.Sequential(
            nn.Linear(num_channels * self.n_freq * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.point_head = nn.Linear(hidden_dim, 1)
        self.target_skip = nn.Linear(lookback, 1)

    def forward(
        self,
        z_seq: torch.Tensor,
        y_hist: torch.Tensor,
        channel_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            z_seq: (B, C, L) where channel 0 is target history and 1..C-1 are prototypes/raw
            y_hist: (B, L)
            channel_gate: optional (C,) in (0, 1)
        Returns:
            y_base: (B,)
            h: (B, hidden_dim)
            hf_ratio: (B,) gated high-frequency energy ratio
        """
        if channel_gate is not None:
            if channel_gate.dim() == 1:
                z_seq = z_seq * channel_gate.view(1, -1, 1)
            else:
                z_seq = z_seq * channel_gate.unsqueeze(-1)

        zf = torch.fft.rfft(z_seq, dim=-1)  # (B, C, K) complex
        xr, xi = zf.real, zf.imag
        wr = self.w_real.unsqueeze(0)
        wi = self.w_imag.unsqueeze(0)

        # Complex multiplication (xr + i xi)(wr + i wi)
        yr = xr * wr - xi * wi
        yi = xr * wi + xi * wr

        # High-frequency energy ratio (top half of non-DC spectrum) for uncertainty descriptor
        power = xr * xr + xi * xi  # (B, C, K)
        k_half = max(1, self.n_freq // 2)
        hf_energy = power[:, :, k_half:].sum(dim=(1, 2))
        tot_energy = power[:, :, 1:].sum(dim=(1, 2)) + 1e-6
        hf_ratio = hf_energy / tot_energy  # (B,)

        spec_feat = torch.cat([yr, yi], dim=-1).reshape(z_seq.size(0), -1)
        h = self.proj(spec_feat)
        y_base = self.target_skip(y_hist).squeeze(-1) + self.point_head(h).squeeze(-1)
        return y_base, h, hf_ratio


class FuzzyTSKIntervalHead(nn.Module):
    """Differentiable first-order Fuzzy Granular-Ball TSK uncertainty & interval head."""

    def __init__(
        self,
        state_dim: int,
        num_rules: int = 6,
        s_min: float = 0.05,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_rules = num_rules
        self.s_min = s_min

        # Learnable ellipsoidal granular-ball rule antecedents (centers + diagonal log-precision)
        self.centers = nn.Parameter(torch.randn(num_rules, state_dim) * 0.5)
        self.raw_prec = nn.Parameter(torch.zeros(num_rules, state_dim))

        # First-order TSK consequents: linear in [1, u_t]
        self.conseq_bias = nn.Linear(state_dim, num_rules)   # delta y_r(u_t)
        self.conseq_lower = nn.Linear(state_dim, num_rules)  # raw s_r^-(u_t)
        self.conseq_upper = nn.Linear(state_dim, num_rules)  # raw s_r^+(u_t)

        # Initialize scale consequents near unit standardized scale (~0.8)
        nn.init.zeros_(self.conseq_bias.weight)
        nn.init.zeros_(self.conseq_bias.bias)
        nn.init.constant_(self.conseq_lower.bias, 0.2)
        nn.init.constant_(self.conseq_upper.bias, 0.2)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            u: (B, q) causal uncertainty state vector
        Returns:
            delta_y: (B,)
            s_lower: (B,) strictly positive lower scale
            s_upper: (B,) strictly positive upper scale
            weights: (B, R) normalized fuzzy rule firing strengths
        """
        prec = F.softplus(self.raw_prec) + 0.1  # (R, q) > 0
        # Squared Mahalanobis diagonal distance: (B, R)
        diff = u.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, R, q)
        d2 = 0.5 * torch.sum(diff * diff * prec.unsqueeze(0), dim=-1)
        weights = torch.softmax(-d2, dim=-1)  # (B, R)

        dy_r = self.conseq_bias(u)                             # (B, R)
        sl_r = F.softplus(self.conseq_lower(u)) + self.s_min   # (B, R) > 0
        su_r = F.softplus(self.conseq_upper(u)) + self.s_min   # (B, R) > 0

        delta_y = torch.sum(weights * dy_r, dim=-1)
        s_lower = torch.sum(weights * sl_r, dim=-1)
        s_upper = torch.sum(weights * su_r, dim=-1)
        return delta_y, s_lower, s_upper, weights

    def partition_regularization(self, weights: torch.Tensor) -> torch.Tensor:
        """Prevent rule collapse: encourage balanced batch rule usage & separated centers."""
        mean_usage = weights.mean(dim=0)  # (R,)
        target = torch.full_like(mean_usage, 1.0 / self.num_rules)
        balance_loss = F.mse_loss(mean_usage, target)

        # Pairwise center separation penalty (repel collapsed centers)
        c_dist = torch.cdist(self.centers, self.centers) + torch.eye(
            self.num_rules, device=self.centers.device
        ) * 10.0
        sep_loss = torch.exp(-c_dist.min(dim=-1).values).mean()
        return balance_loss + 0.05 * sep_loss

    @torch.no_grad()
    def export_rules(self, state_names: List[str]) -> List[Dict[str, object]]:
        """Export learned IF-THEN fuzzy rule parameters."""
        prec = (F.softplus(self.raw_prec) + 0.1).cpu().numpy()
        centers = self.centers.cpu().numpy()
        wl = self.conseq_lower.weight.cpu().numpy()
        bl = self.conseq_lower.bias.cpu().numpy()
        wu = self.conseq_upper.weight.cpu().numpy()
        bu = self.conseq_upper.bias.cpu().numpy()

        rules = []
        for r in range(self.num_rules):
            antecedents = {
                name: {"center": float(centers[r, i]), "width": float((1.0 / prec[r, i]) ** 0.5)}
                for i, name in enumerate(state_names)
            }
            rules.append(
                {
                    "rule_id": r + 1,
                    "antecedents": antecedents,
                    "lower_scale_intercept": float(bl[r]),
                    "upper_scale_intercept": float(bu[r]),
                    "lower_scale_slopes": {name: float(wl[r, i]) for i, name in enumerate(state_names)},
                    "upper_scale_slopes": {name: float(wu[r, i]) for i, name in enumerate(state_names)},
                }
            )
        return rules


class FGBIPNModel(nn.Module):
    """FGB-IPN model with configurable granulation, dispersion, and TSK switches."""

    STATE_NAMES = [
        "ball_disp_mean",
        "ball_disp_max",
        "hf_energy_ratio",
        "target_vol_short",
        "target_vol_long",
    ]

    def __init__(
        self,
        num_exog: int,
        lookback: int,
        gb_structure: Optional[GranularBallStructure] = None,
        use_gb: bool = True,
        gb_mode: str = "soft",
        pca_basis: Optional[torch.Tensor] = None,
        use_dispersion: bool = True,
        use_tsk: bool = True,
        hidden_dim: int = 32,
        num_rules: int = 6,
    ):
        super().__init__()
        self.use_gb = use_gb
        self.gb_mode = gb_mode
        self.use_pca = pca_basis is not None and not use_gb
        self.use_dispersion = use_dispersion
        self.use_tsk = use_tsk
        self.lookback = lookback

        if use_gb:
            assert gb_structure is not None
            self.sfgb = SpectralFuzzyGranularBallLayer(gb_structure, mode=gb_mode)
            n_proto = gb_structure.centers.shape[0]
            self.ball_gate_logit = nn.Parameter(torch.ones(n_proto) * 1.0)
            self.psi_gate_logit = nn.Parameter(torch.zeros(n_proto))
            in_channels = 1 + n_proto
            self.register_buffer("pca_basis", torch.empty(0))
        elif self.use_pca:
            assert pca_basis is not None
            self.sfgb = None
            self.ball_gate_logit = None
            self.psi_gate_logit = None
            self.register_buffer("pca_basis", pca_basis)  # (d, M)
            in_channels = 1 + pca_basis.shape[1]
        else:
            self.sfgb = None
            self.ball_gate_logit = None
            self.psi_gate_logit = None
            self.register_buffer("pca_basis", torch.empty(0))
            in_channels = 1 + num_exog

        self.backbone = ComplexSpectralMixLayer(
            num_channels=in_channels,
            lookback=lookback,
            hidden_dim=hidden_dim,
        )

        # Running standardization buffers for uncertainty state vector (fit-only!)
        self.register_buffer("u_mean", torch.zeros(len(self.STATE_NAMES)))
        self.register_buffer("u_std", torch.ones(len(self.STATE_NAMES)))

        if use_tsk:
            self.tsk_head = FuzzyTSKIntervalHead(
                state_dim=len(self.STATE_NAMES),
                num_rules=num_rules,
            )
        else:
            self.tsk_head = None
            self.global_log_scale = nn.Parameter(torch.zeros(2))

    def _encode_exog(self, x_exog: torch.Tensor):
        if self.use_gb and self.sfgb is not None:
            proto, incon = self.sfgb(x_exog)
            gates = torch.sigmoid(self.ball_gate_logit)  # (M,)
            if self.use_dispersion and self.psi_gate_logit is not None:
                # Dynamic inconsistency-aware reliability attenuation: g_{t,m} = g_m * (1 - lambda_m * \bar{\psi}_{t,m})
                psi_bar_m = incon.mean(dim=-1)  # (B, M) in [0, 1)
                lam_m = 0.5 * torch.sigmoid(self.psi_gate_logit).unsqueeze(0)  # (1, M) in (0, 0.5)
                dyn_gates = gates.unsqueeze(0) * (1.0 - lam_m * psi_bar_m)     # (B, M)
                ones = torch.ones(x_exog.size(0), 1, device=x_exog.device, dtype=x_exog.dtype)
                ch_gate = torch.cat([ones, dyn_gates], dim=1)                  # (B, 1+M)
            else:
                ch_gate = torch.cat([torch.ones(1, device=x_exog.device, dtype=x_exog.dtype), gates])
            return proto, incon, gates, ch_gate
        elif self.use_pca:
            # Linear/cluster feature compression (PCA, VarCluster, TopM): downstream sees ONLY the M compressed features
            proto = torch.einsum("dm,bdl->bml", self.pca_basis, x_exog)
            if proto.size(1) > 1:
                var_comp = proto.var(dim=1, unbiased=False, keepdim=True)
            else:
                var_comp = proto * proto
            incon = var_comp / (var_comp + 4.0)
            return proto, incon, None, None
        else:
            var_raw = x_exog.var(dim=1, unbiased=False, keepdim=True)
            incon = var_raw / (var_raw + 4.0)
            return x_exog, incon, None, None

    def _compute_uncertainty_state(
        self,
        y_hist: torch.Tensor,
        hf_ratio: torch.Tensor,
        inconsistency: Optional[torch.Tensor],
        ball_gates: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Construct strictly pre-origin causal uncertainty state vector u_t."""
        short_len = max(4, self.lookback // 4)
        vol_short = y_hist[:, -short_len:].std(dim=-1)
        vol_long = y_hist.std(dim=-1)

        if self.use_dispersion and inconsistency is not None:
            # Bounded fuzzy inconsistency \psi_t^{(m)} in [0, 1)
            inc_recent = inconsistency[:, :, -short_len:].mean(dim=-1)
            if ball_gates is not None:
                g = ball_gates / (ball_gates.sum() + 1e-6)
                inc_mean = torch.sum(inc_recent * g.unsqueeze(0), dim=-1)
            else:
                inc_mean = inc_recent.mean(dim=-1)
            inc_max = inc_recent.max(dim=-1).values
        else:
            inc_mean = torch.zeros_like(vol_short)
            inc_max = torch.zeros_like(vol_short)

        return torch.stack([inc_mean, inc_max, hf_ratio, vol_short, vol_long], dim=-1)

    @torch.no_grad()
    def calibrate_state_norm(self, x_exog: torch.Tensor, y_hist: torch.Tensor) -> None:
        """Fit uncertainty state standardization strictly on fit split."""
        self.eval()
        exog_feat, incon, gates, ch_gate = self._encode_exog(x_exog)
        z = torch.cat([y_hist.unsqueeze(1), exog_feat], dim=1)
        _, _, hf_ratio = self.backbone(z, y_hist, channel_gate=ch_gate)
        u = self._compute_uncertainty_state(y_hist, hf_ratio, incon, gates)
        self.u_mean.copy_(u.mean(dim=0))
        u_s = u.std(dim=0)
        # Bounded [0, 1) fuzzy inconsistency & spectral ratio descriptors keep a 0.08 scale floor
        floors = torch.tensor([0.08, 0.08, 0.08, 1e-4, 1e-4], device=u_s.device, dtype=u_s.dtype)
        self.u_std.copy_(torch.maximum(u_s, floors))

    def forward(self, x_exog: torch.Tensor, y_hist: torch.Tensor) -> Dict[str, torch.Tensor]:
        exog_feat, incon, gates, ch_gate = self._encode_exog(x_exog)
        z = torch.cat([y_hist.unsqueeze(1), exog_feat], dim=1)

        y_base, _, hf_ratio = self.backbone(z, y_hist, channel_gate=ch_gate)
        u_raw = self._compute_uncertainty_state(y_hist, hf_ratio, incon, gates)
        # Smooth saturation prevents unbounded TSK scale explosion under OOD sensor bursts
        u_norm = 1.8 * torch.tanh(((u_raw - self.u_mean) / self.u_std) / 1.8)

        if self.use_tsk and self.tsk_head is not None:
            _, s_low, s_high, rule_w = self.tsk_head(u_norm)
            y_hat = y_base
        else:
            y_hat = y_base
            scales = F.softplus(self.global_log_scale) + 0.05
            s_low = scales[0].expand_as(y_hat)
            s_high = scales[1].expand_as(y_hat)
            rule_w = torch.ones(y_hat.size(0), 1, device=y_hat.device)

        return {
            "y_hat": y_hat,
            "s_lower": s_low,
            "s_upper": s_high,
            "rule_weights": rule_w,
            "inc_mean": u_raw[:, 0],
            "ball_gates": gates if gates is not None else torch.zeros(1, device=y_hat.device),
        }


def winkler_interval_loss(
    y: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    width = upper - lower
    under = F.relu(lower - y)
    over = F.relu(y - upper)
    return torch.mean(width + (2.0 / alpha) * (under + over))


def joint_fgb_loss(
    out: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    model: FGBIPNModel,
    alpha: float = 0.05,
    lam_int: float = 0.35,
    lam_gate: float = 1e-3,
    lam_rule: float = 0.05,
) -> torch.Tensor:
    y_hat = out["y_hat"]
    s_low = out["s_lower"]
    s_high = out["s_upper"]

    l_point = F.huber_loss(y_hat, y_true, delta=1.0)
    l_int = winkler_interval_loss(y_true, y_hat - s_low, y_hat + s_high, alpha=alpha)
    l_gate = out["ball_gates"].abs().mean() if model.use_gb else torch.tensor(0.0, device=y_true.device)
    l_rule = (
        model.tsk_head.partition_regularization(out["rule_weights"])
        if (model.use_tsk and model.tsk_head is not None)
        else torch.tensor(0.0, device=y_true.device)
    )
    return l_point + lam_int * l_int + lam_gate * l_gate + lam_rule * l_rule


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
