"""Interval prediction baseline implementations under the shared evaluation protocol.

Included baselines:
  - ECQR    (Jensen et al., IEEE TNNLS, 2024)
  - MQR-NN  (Hu et al., Information Sciences, 2024)
  - ACHF    (Sousa et al., Neurocomputing, 2024)
  - CAFHT   (Zhou et al., ICML, 2024)
  - DAQD-NN (Morales & Sheppard, IEEE TNNLS, 2025)
  - MSDIPN  (Jiang et al., IEEE TKDE, 2025)
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from conformal import calibrate_and_predict_intervals, compute_winkler_vector, evaluate_predictions
from data_loader import PreparedDataset, SplitData
from fgb_tsk_net import count_parameters, winkler_interval_loss


def pinball_loss(y: torch.Tensor, q_hat: torch.Tensor, tau: float) -> torch.Tensor:
    diff = y - q_hat
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))


class QuantileMLPSubNet(nn.Module):
    """Base quantile regressor member for ECQR (Jensen et al., IEEE TNNLS 2024)."""

    def __init__(self, num_channels: int, lookback: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_channels * lookback, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # [q_low, q_med, q_high]
        )

    def forward(self, x_exog: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        z = torch.cat([y_hist.unsqueeze(1), x_exog], dim=1)
        return self.net(z.reshape(z.size(0), -1))


class MonotoneQRNN(nn.Module):
    """MQR-NN (Hu et al., Information Sciences 2024):
    Enforces strict quantile monotonicity dq/dtau >= 0 via positive weight layers.
    """

    def __init__(self, num_channels: int, hidden_dim: int = 48):
        super().__init__()
        self.in_proj = nn.Linear(num_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        # Monotone tau-conditioned MLP with strictly positive weights wrt tau
        self.w_h = nn.Linear(hidden_dim, hidden_dim)
        self.w_tau1 = nn.Parameter(torch.randn(1, hidden_dim) * 0.2)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w_pos2 = nn.Parameter(torch.randn(hidden_dim, 1) * 0.2)
        self.b2 = nn.Parameter(torch.zeros(1))

    def _eval_tau(self, h: torch.Tensor, tau_val: float) -> torch.Tensor:
        tau = torch.full((h.size(0), 1), tau_val, device=h.device, dtype=h.dtype)
        # Monotone hidden activation (tanh is monotone increasing; softplus weights >= 0)
        u = torch.tanh(self.w_h(h) + tau @ F.softplus(self.w_tau1) + self.b1)
        q = u @ F.softplus(self.w_pos2) + self.b2
        return q.squeeze(-1)

    def forward(self, x_exog: torch.Tensor, y_hist: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        z = torch.cat([y_hist.unsqueeze(1), x_exog], dim=1).transpose(1, 2)  # (B, L, C)
        out, _ = self.gru(F.gelu(self.in_proj(z)))
        h = out[:, -1, :]
        ql = self._eval_tau(h, alpha / 2.0)
        qm = self._eval_tau(h, 0.5)
        qh = self._eval_tau(h, 1.0 - alpha / 2.0)
        return torch.stack([ql, qm, qh], dim=-1)


class HeteroscedasticDualNet(nn.Module):
    """ACHF (Sousa et al., Neurocomputing 2024):
    Dual-branch conditional mean + conditional residual volatility network for adaptive conformal forecasting.
    """

    def __init__(self, num_channels: int, lookback: int, hidden_dim: int = 48):
        super().__init__()
        in_dim = num_channels * lookback
        self.mean_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.vol_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_exog: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        z = torch.cat([y_hist.unsqueeze(1), x_exog], dim=1).reshape(x_exog.size(0), -1)
        mu = self.mean_net(z)
        sigma = F.softplus(self.vol_net(z)) + 0.05
        return torch.cat([mu, sigma], dim=-1)


class CAFHTNet(nn.Module):
    """CAFHT (Zhou, Lindemann & Sesia, ICML 2024):
    Patch-attention forecaster with multiplicative local volatility score normalization.
    """

    def __init__(
        self,
        num_channels: int,
        lookback: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 32,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = max(1, (lookback - patch_len) // stride + 1)
        self.in_proj = nn.Linear(num_channels * patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.q_head = nn.Linear(d_model * self.n_patches, 3)  # [q_low, q_med, q_high]

    def forward(self, x_exog: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        z = torch.cat([y_hist.unsqueeze(1), x_exog], dim=1)
        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, C, P, pl = patches.shape
        tokens = patches.permute(0, 2, 1, 3).reshape(B, P, C * pl)
        h = self.encoder(self.in_proj(tokens) + self.pos_emb[:, :P, :])
        return self.q_head(h.reshape(B, -1))


class DAQDNet(nn.Module):
    """DAQD-NN (Morales & Sheppard, IEEE TNNLS 2025):
    Dual accuracy-quality-driven neural network jointly optimizing point accuracy and smooth PI quality.
    """

    def __init__(self, num_channels: int, lookback: int, hidden_dim: int = 48):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(num_channels * lookback, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.acc_head = nn.Linear(hidden_dim, 1)
        self.qual_head = nn.Linear(hidden_dim, 2)  # [delta_low, delta_high]

    def forward(self, x_exog: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        z = torch.cat([y_hist.unsqueeze(1), x_exog], dim=1).reshape(x_exog.size(0), -1)
        h = self.trunk(z)
        yh = self.acc_head(h)
        widths = F.softplus(self.qual_head(h)) + 0.05
        return torch.cat([yh, widths], dim=-1)


class MSDIPNNet(nn.Module):
    """MSDIPN (Jiang et al., IEEE TKDE 2025):
    Multi-scale temporal pooling + cross-scale self-attention + non-crossing interval head.
    """

    def __init__(self, num_channels: int, lookback: int, d_model: int = 32):
        super().__init__()
        self.scales = [1, 2, 4]
        self.projs = nn.ModuleList([
            nn.Linear(num_channels * (lookback // s), d_model) for s in self.scales
        ])
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * len(self.scales), d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # [y_hat, raw_dl, raw_du]
        )

    def forward(self, x_exog: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        z = torch.cat([y_hist.unsqueeze(1), x_exog], dim=1)  # (B, C, L)
        B = z.size(0)
        tokens = []
        for s, proj in zip(self.scales, self.projs):
            zs = F.avg_pool1d(z, kernel_size=s, stride=s) if s > 1 else z
            tokens.append(F.gelu(proj(zs.reshape(B, -1))))
        seq = torch.stack(tokens, dim=1)  # (B, 3, d_model)
        attn_out, _ = self.attn(seq, seq, seq)
        fused = self.norm(seq + attn_out).reshape(B, -1)
        raw = self.head(fused)
        yh = raw[:, :1]
        dl = F.softplus(raw[:, 1:2]) + 0.05
        du = F.softplus(raw[:, 2:3]) + 0.05
        return torch.cat([yh, dl, du], dim=-1)


def _make_loader(split: SplitData, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(split.X_exog),
        torch.from_numpy(split.y_hist),
        torch.from_numpy(split.y_target_std),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _daqd_loss(pred: torch.Tensor, yt: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Dual accuracy + smooth coverage/sharpness quality loss (Morales & Sheppard, 2025)."""
    yh, dl, du = pred[:, 0], pred[:, 1], pred[:, 2]
    lo, hi = yh - dl, yh + du
    l_acc = F.mse_loss(yh, yt)
    # Smooth sigmoid coverage indicator (temperature s=15)
    cov_soft = torch.sigmoid(15.0 * (yt - lo)) * torch.sigmoid(15.0 * (hi - yt))
    picp_soft = cov_soft.mean()
    mpiw = (hi - lo).mean()
    cov_pen = 30.0 * F.relu((1.0 - alpha) - picp_soft) ** 2
    return l_acc + 0.35 * mpiw + cov_pen + 0.15 * winkler_interval_loss(yt, lo, hi, alpha=alpha)


def _msdipn_loss(pred: torch.Tensor, yt: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Multi-scale pinball + Winkler quality loss (Jiang et al., IEEE TKDE 2025)."""
    yh, dl, du = pred[:, 0], pred[:, 1], pred[:, 2]
    lo, hi = yh - dl, yh + du
    l_pin = (
        pinball_loss(yt, lo, alpha / 2.0)
        + pinball_loss(yt, yh, 0.5)
        + pinball_loss(yt, hi, 1.0 - alpha / 2.0)
    )
    return l_pin + 0.25 * winkler_interval_loss(yt, lo, hi, alpha=alpha)


def _achf_loss(pred: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """Gaussian heteroscedastic NLL + Huber point loss (Sousa et al., 2024)."""
    mu, sigma = pred[:, 0], pred[:, 1]
    nll = torch.mean(torch.log(sigma) + 0.5 * ((yt - mu) / sigma) ** 2)
    return 0.5 * F.huber_loss(mu, yt, delta=1.0) + 0.5 * nll


def _train_single(
    model: nn.Module,
    data: PreparedDataset,
    loss_kind: str,
    alpha: float = 0.05,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-3,
    bootstrap: bool = False,
    seed: int = 17,
) -> Tuple[nn.Module, float]:
    if bootstrap:
        rng = np.random.default_rng(seed)
        n_fit = len(data.fit.y_target)
        idx = rng.integers(0, n_fit, size=n_fit)
        fit_split = SplitData(
            X_exog=data.fit.X_exog[idx],
            y_hist=data.fit.y_hist[idx],
            y_target=data.fit.y_target[idx],
            y_target_std=data.fit.y_target_std[idx],
            origins=data.fit.origins[idx],
        )
    else:
        fit_split = data.fit

    train_loader = _make_loader(fit_split, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(data.val, batch_size=256, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def _loss(pred: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        if loss_kind == "quantile":
            return (
                pinball_loss(yt, pred[:, 0], alpha / 2.0)
                + pinball_loss(yt, pred[:, 1], 0.5)
                + pinball_loss(yt, pred[:, 2], 1.0 - alpha / 2.0)
            )
        elif loss_kind == "achf":
            return _achf_loss(pred, yt)
        elif loss_kind == "daqd":
            return _daqd_loss(pred, yt, alpha=alpha)
        elif loss_kind == "msdipn":
            return _msdipn_loss(pred, yt, alpha=alpha)
        raise ValueError(loss_kind)

    best_val = float("inf")
    best_state = None
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb, yt in train_loader:
            opt.zero_grad()
            loss = _loss(model(xb, yb), yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb, yt in val_loader:
                vls.append(float(_loss(model(xb, yb), yt).item()))
        vl = float(np.mean(vls))
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_sec


@torch.no_grad()
def _predict_raw(model: nn.Module, split: SplitData) -> Tuple[np.ndarray, float]:
    model.eval()
    loader = _make_loader(split, batch_size=256, shuffle=False)
    preds = []
    t0 = time.perf_counter()
    for xb, yb, _ in loader:
        preds.append(model(xb, yb).cpu().numpy())
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(split.y_target))
    return np.concatenate(preds, axis=0), ms


def _sequential_cqr_update(
    cal_y: np.ndarray,
    cal_lo: np.ndarray,
    cal_hi: np.ndarray,
    cal_origins: np.ndarray,
    test_y: np.ndarray,
    test_lo: np.ndarray,
    test_hi: np.ndarray,
    test_origins: np.ndarray,
    horizon: int,
    alpha: float = 0.05,
    window_size: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Causal sequential sliding-window CQR calibration (Jensen et al., IEEE TNNLS 2024)."""
    hist_scores = list(np.maximum(cal_lo - cal_y, cal_y - cal_hi).tolist())
    pending = []
    n_test = len(test_y)
    lowers = np.empty(n_test, dtype=np.float64)
    uppers = np.empty(n_test, dtype=np.float64)

    for i in range(n_test):
        orig_t = int(test_origins[i])
        still_pending = []
        for lbl_t, sc in pending:
            if lbl_t <= orig_t:
                hist_scores.append(sc)
            else:
                still_pending.append((lbl_t, sc))
        pending = still_pending

        win = np.asarray(hist_scores[-window_size:], dtype=np.float64)
        n_w = len(win)
        q_lvl = min(1.0, np.ceil((n_w + 1) * (1.0 - alpha)) / max(1, n_w))
        q_val = float(np.quantile(win, q_lvl))
        lowers[i] = test_lo[i] - q_val
        uppers[i] = test_hi[i] + q_val

        sc_i = float(max(test_lo[i] - test_y[i], test_y[i] - test_hi[i]))
        pending.append((orig_t + horizon, sc_i))

    return lowers, uppers


def run_sota_baseline(
    name: str,
    data: PreparedDataset,
    seed: int = 17,
    epochs: int = 20,
    alpha: float = 0.05,
    device: str = "cpu",
) -> Dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_ch = 1 + data.fit.X_exog.shape[1]
    L = data.lookback
    ym, ys = data.y_mean, data.y_std

    if name == "ECQR":
        # 3-member bootstrap ensemble + causal sliding CQR (Jensen et al., IEEE TNNLS 2024)
        members: List[nn.Module] = []
        train_sec = 0.0
        for m_idx in range(3):
            sub = QuantileMLPSubNet(num_ch, L)
            sub, t_s = _train_single(
                sub, data, loss_kind="quantile", alpha=alpha, epochs=epochs,
                bootstrap=True, seed=seed * 10 + m_idx,
            )
            members.append(sub)
            train_sec += t_s

        cal_preds = np.mean([_predict_raw(m, data.cal)[0] for m in members], axis=0)
        test_list = [_predict_raw(m, data.test) for m in members]
        test_preds = np.mean([p[0] for p in test_list], axis=0)
        infer_ms = float(np.sum([p[1] for p in test_list]))
        params = sum(count_parameters(m) for m in members)

        cal_lo = np.minimum(cal_preds[:, 0], cal_preds[:, 2]) * ys + ym
        cal_hi = np.maximum(cal_preds[:, 0], cal_preds[:, 2]) * ys + ym
        test_lo = np.minimum(test_preds[:, 0], test_preds[:, 2]) * ys + ym
        test_hi = np.maximum(test_preds[:, 0], test_preds[:, 2]) * ys + ym
        y_hat = test_preds[:, 1] * ys + ym

        lower, upper = _sequential_cqr_update(
            data.cal.y_target, cal_lo, cal_hi, data.cal.origins,
            data.test.y_target, test_lo, test_hi, data.test.origins,
            horizon=data.horizon, alpha=alpha,
        )

    elif name == "MQR-NN":
        model = MonotoneQRNN(num_ch)
        model, train_sec = _train_single(model, data, loss_kind="quantile", alpha=alpha, epochs=epochs, seed=seed)
        test_raw, infer_ms = _predict_raw(model, data.test)
        params = count_parameters(model)
        y_hat = test_raw[:, 1] * ys + ym
        lower = np.minimum(test_raw[:, 0], test_raw[:, 2]) * ys + ym
        upper = np.maximum(test_raw[:, 0], test_raw[:, 2]) * ys + ym

    elif name == "ACHF":
        model = HeteroscedasticDualNet(num_ch, L)
        model, train_sec = _train_single(model, data, loss_kind="achf", alpha=alpha, epochs=epochs, seed=seed)
        cal_raw, _ = _predict_raw(model, data.cal)
        test_raw, infer_ms = _predict_raw(model, data.test)
        params = count_parameters(model)

        cal_yh = cal_raw[:, 0] * ys + ym
        cal_sig = np.maximum(cal_raw[:, 1] * ys, 1e-4)
        test_yh = test_raw[:, 0] * ys + ym
        test_sig = np.maximum(test_raw[:, 1] * ys, 1e-4)
        lower, upper, _ = calibrate_and_predict_intervals(
            cal_y=data.cal.y_target, cal_yhat=cal_yh, cal_sl=cal_sig, cal_su=cal_sig,
            cal_w=np.ones((len(cal_yh), 1)), cal_origins=data.cal.origins,
            test_y=data.test.y_target, test_yhat=test_yh, test_sl=test_sig, test_su=test_sig,
            test_w=np.ones((len(test_yh), 1)), test_origins=data.test.origins,
            horizon=data.horizon, alpha=alpha, mode="global_aci",
        )
        y_hat = test_yh

    elif name == "CAFHT":
        model = CAFHTNet(num_ch, L)
        model, train_sec = _train_single(model, data, loss_kind="quantile", alpha=alpha, epochs=epochs, seed=seed)
        cal_raw, _ = _predict_raw(model, data.cal)
        test_raw, infer_ms = _predict_raw(model, data.test)
        params = count_parameters(model)

        # Multiplicative adaptive conformal quantile modulation (Zhou et al., ICML 2024)
        cal_yh = cal_raw[:, 1] * ys + ym
        cal_lo = np.minimum(cal_raw[:, 0], cal_raw[:, 2]) * ys + ym
        cal_hi = np.maximum(cal_raw[:, 0], cal_raw[:, 2]) * ys + ym
        cal_sl = np.maximum(cal_yh - cal_lo, 0.05 * ys)
        cal_su = np.maximum(cal_hi - cal_yh, 0.05 * ys)

        test_yh = test_raw[:, 1] * ys + ym
        test_lo = np.minimum(test_raw[:, 0], test_raw[:, 2]) * ys + ym
        test_hi = np.maximum(test_raw[:, 0], test_raw[:, 2]) * ys + ym
        test_sl = np.maximum(test_yh - test_lo, 0.05 * ys)
        test_su = np.maximum(test_hi - test_yh, 0.05 * ys)

        lower, upper, _ = calibrate_and_predict_intervals(
            cal_y=data.cal.y_target, cal_yhat=cal_yh, cal_sl=cal_sl, cal_su=cal_su,
            cal_w=np.ones((len(cal_yh), 1)), cal_origins=data.cal.origins,
            test_y=data.test.y_target, test_yhat=test_yh, test_sl=test_sl, test_su=test_su,
            test_w=np.ones((len(test_yh), 1)), test_origins=data.test.origins,
            horizon=data.horizon, alpha=alpha, mode="global_aci",
        )
        y_hat = test_yh

    elif name == "DAQD-NN":
        model = DAQDNet(num_ch, L)
        model, train_sec = _train_single(model, data, loss_kind="daqd", alpha=alpha, epochs=epochs, seed=seed)
        test_raw, infer_ms = _predict_raw(model, data.test)
        params = count_parameters(model)
        y_hat = test_raw[:, 0] * ys + ym
        lower = y_hat - test_raw[:, 1] * ys
        upper = y_hat + test_raw[:, 2] * ys

    elif name == "MSDIPN":
        model = MSDIPNNet(num_ch, L)
        model, train_sec = _train_single(model, data, loss_kind="msdipn", alpha=alpha, epochs=epochs, seed=seed)
        test_raw, infer_ms = _predict_raw(model, data.test)
        params = count_parameters(model)
        y_hat = test_raw[:, 0] * ys + ym
        lower = y_hat - test_raw[:, 1] * ys
        upper = y_hat + test_raw[:, 2] * ys

    else:
        raise ValueError(f"Unknown baseline: {name}")

    metrics = evaluate_predictions(
        data.test.y_target, y_hat, lower, upper, y_range=data.y_range_fit, alpha=alpha
    )
    winkler_vec = compute_winkler_vector(
        data.test.y_target, y_hat, lower, upper, alpha=alpha
    )
    return {
        "variant": name,
        "seed": seed,
        "params": params,
        "train_sec": train_sec,
        "infer_ms_per_sample": infer_ms,
        "metrics": metrics,
        "winkler_vec": winkler_vec,
    }
