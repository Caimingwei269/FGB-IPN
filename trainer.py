"""Training loops, compression basis builders, and ablation variant runners for FGB-IPN."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from conformal import calibrate_and_predict_intervals, compute_winkler_vector, evaluate_predictions
from data_loader import PreparedDataset, SplitData
from fgb_tsk_net import FGBIPNModel, count_parameters, joint_fgb_loss
from sfgb import GranularBallStructure, compute_spectral_embeddings


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loader(split: SplitData, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(split.X_exog),
        torch.from_numpy(split.y_hist),
        torch.from_numpy(split.y_target_std),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_split(
    model: FGBIPNModel,
    split: SplitData,
    y_mean: float,
    y_std: float,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    yhats, sls, sus, rws, incs = [], [], [], [], []

    t0 = time.perf_counter()
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb, yb)
        yhats.append(out["y_hat"].cpu().numpy())
        sls.append(out["s_lower"].cpu().numpy())
        sus.append(out["s_upper"].cpu().numpy())
        rws.append(out["rule_weights"].cpu().numpy())
        incs.append(out["inc_mean"].cpu().numpy())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_sample = elapsed_ms / max(1, len(split.y_target))

    # Map back to original target scale
    y_hat = np.concatenate(yhats) * y_std + y_mean
    s_low = np.concatenate(sls) * y_std
    s_high = np.concatenate(sus) * y_std
    rule_w = np.concatenate(rws, axis=0)
    model._last_inc_mean = np.concatenate(incs)
    return y_hat, s_low, s_high, rule_w, ms_per_sample


def train_model(
    model: FGBIPNModel,
    data: PreparedDataset,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    alpha: float = 0.05,
    device: str = "cpu",
) -> Tuple[FGBIPNModel, float]:
    model = model.to(device)

    # Fit uncertainty state standardization strictly on fit split
    with torch.no_grad():
        n_cal_u = min(2048, len(data.fit.y_target))
        x_fit_t = torch.from_numpy(data.fit.X_exog[:n_cal_u]).to(device)
        y_fit_t = torch.from_numpy(data.fit.y_hist[:n_cal_u]).to(device)
        model.calibrate_state_norm(x_fit_t, y_fit_t)

    train_loader = _make_loader(data.fit, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(data.val, batch_size=256, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None

    t0 = time.perf_counter()
    for ep in range(epochs):
        if ep == epochs // 2:
            with torch.no_grad():
                model.calibrate_state_norm(x_fit_t, y_fit_t)
        model.train()
        for xb, yb, yt in train_loader:
            xb, yb, yt = xb.to(device), yb.to(device), yt.to(device)
            opt.zero_grad()
            out = model(xb, yb)
            loss = joint_fgb_loss(out, yt, model, alpha=alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, yt in val_loader:
                xb, yb, yt = xb.to(device), yb.to(device), yt.to(device)
                out = model(xb, yb)
                val_losses.append(float(joint_fgb_loss(out, yt, model, alpha=alpha).item()))
        v_loss = float(np.mean(val_losses))
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_sec


def compute_fit_pca_basis(X_fit: np.ndarray, n_components: int) -> torch.Tensor:
    """Compute top-M PCA projection basis (d, M) strictly on fit windows."""
    flat = X_fit.transpose(0, 2, 1).reshape(-1, X_fit.shape[1])
    if flat.shape[0] > 20000:
        step = max(1, flat.shape[0] // 20000)
        flat = flat[::step]
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    basis = vt[:n_components].T.astype(np.float32)  # (d, M)
    return torch.from_numpy(basis)


def compute_fit_varcluster_basis(X_fit: np.ndarray, n_clusters: int, n_iters: int = 15) -> torch.Tensor:
    """Unsupervised time-domain k-means variable clustering into M clusters on fit windows only."""
    d = X_fit.shape[1]
    n_clusters = min(n_clusters, d)
    flat = X_fit.transpose(1, 0, 2).reshape(d, -1)  # (d, N*L)
    if flat.shape[1] > 4096:
        step = max(1, flat.shape[1] // 4096)
        flat = flat[:, ::step]
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    feats = flat / norms  # unit-norm variable profiles (correlation geometry)

    # Deterministic farthest-first k-means initialization
    centers = np.zeros((n_clusters, feats.shape[1]), dtype=np.float32)
    centers[0] = feats[0]
    for k in range(1, n_clusters):
        d2 = np.min([np.sum((feats - centers[j][None, :]) ** 2, axis=1) for j in range(k)], axis=0)
        centers[k] = feats[int(np.argmax(d2))]

    assign = np.zeros(d, dtype=np.int64)
    for _ in range(n_iters):
        dists = np.stack([np.sum((feats - centers[k][None, :]) ** 2, axis=1) for k in range(n_clusters)], axis=1)
        assign = np.argmin(dists, axis=1)
        for k in range(n_clusters):
            mem = np.where(assign == k)[0]
            if len(mem) > 0:
                c = feats[mem].mean(axis=0)
                centers[k] = c / (np.linalg.norm(c) + 1e-8)

    basis = np.zeros((d, n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        mem = np.where(assign == k)[0]
        if len(mem) > 0:
            basis[mem, k] = 1.0 / float(len(mem))
        else:
            basis[k % d, k] = 1.0
    return torch.from_numpy(basis)


def compute_fit_topm_basis(X_fit: np.ndarray, y_hist_fit: np.ndarray, n_select: int) -> torch.Tensor:
    """Supervised Top-M individual sensor selection by target spectral coherence on fit windows only."""
    d = X_fit.shape[1]
    n_select = min(n_select, d)
    _, rel = compute_spectral_embeddings(X_fit, y_hist_fit)
    top_idx = np.argsort(-rel)[:n_select]
    basis = np.zeros((d, n_select), dtype=np.float32)
    for m, ch in enumerate(top_idx):
        basis[int(ch), m] = 1.0
    return torch.from_numpy(basis)


def run_variant(
    variant_name: str,
    data: PreparedDataset,
    gb_struct: GranularBallStructure,
    pca_basis: torch.Tensor,
    seed: int,
    epochs: int,
    alpha: float,
    device: str,
) -> Dict[str, object]:
    set_seed(seed)
    d = data.fit.X_exog.shape[1]
    L = data.lookback
    M = len(gb_struct.members)

    configs = {
        "FGB-IPN (Full)": dict(
            use_gb=True, gb_mode="soft", proj="none", use_dispersion=True, use_tsk=True, conf_mode="fgb_aci"
        ),
        "FGB-NoPsi-TSK-ACI": dict(
            use_gb=True, gb_mode="soft", proj="none", use_dispersion=False, use_tsk=True, conf_mode="fgb_aci"
        ),
        "SoftStat-GB-IPN": dict(
            use_gb=True, gb_mode="soft_stat", proj="none", use_dispersion=True, use_tsk=True, conf_mode="fgb_aci"
        ),
        "Hard-GB-IPN": dict(
            use_gb=True, gb_mode="hard", proj="none", use_dispersion=True, use_tsk=True, conf_mode="fgb_aci"
        ),
        "VarCluster-TSK-ACI": dict(
            use_gb=False, gb_mode="soft", proj="varcluster", use_dispersion=True, use_tsk=True, conf_mode="fgb_aci"
        ),
        "TopM-Select-TSK-ACI": dict(
            use_gb=False, gb_mode="soft", proj="topm", use_dispersion=True, use_tsk=True, conf_mode="fgb_aci"
        ),
        "PCA-TSK-ACI": dict(
            use_gb=False, gb_mode="soft", proj="pca", use_dispersion=True, use_tsk=True, conf_mode="fgb_aci"
        ),
        "Raw-AllDim-TSK-ACI": dict(
            use_gb=False, gb_mode="soft", proj="none", use_dispersion=True, use_tsk=True, conf_mode="fgb_aci"
        ),
        "FGB-Global-ACI": dict(
            use_gb=True, gb_mode="soft", proj="none", use_dispersion=False, use_tsk=False, conf_mode="global_aci"
        ),
        "FGB-TSK-Static": dict(
            use_gb=True, gb_mode="soft", proj="none", use_dispersion=True, use_tsk=True, conf_mode="static"
        ),
        "Raw-Global-ACI": dict(
            use_gb=False, gb_mode="soft", proj="none", use_dispersion=False, use_tsk=False, conf_mode="global_aci"
        ),
    }
    cfg = configs[variant_name]

    proj_basis = None
    if cfg["proj"] == "pca":
        proj_basis = pca_basis
    elif cfg["proj"] == "varcluster":
        proj_basis = compute_fit_varcluster_basis(data.fit.X_exog, n_clusters=M)
    elif cfg["proj"] == "topm":
        proj_basis = compute_fit_topm_basis(data.fit.X_exog, data.fit.y_hist, n_select=M)

    model = FGBIPNModel(
        num_exog=d,
        lookback=L,
        gb_structure=gb_struct if cfg["use_gb"] else None,
        use_gb=cfg["use_gb"],
        gb_mode=cfg["gb_mode"],
        pca_basis=proj_basis,
        use_dispersion=cfg["use_dispersion"],
        use_tsk=cfg["use_tsk"],
    )
    model, train_sec = train_model(model, data, epochs=epochs, alpha=alpha, device=device)

    cal_yhat, cal_sl, cal_su, cal_w, _ = predict_split(
        model, data.cal, data.y_mean, data.y_std, device=device
    )
    test_yhat, test_sl, test_su, test_w, infer_ms = predict_split(
        model, data.test, data.y_mean, data.y_std, device=device
    )
    test_inc_mean = getattr(model, "_last_inc_mean", np.zeros_like(test_yhat))

    lower, upper, _ = calibrate_and_predict_intervals(
        cal_y=data.cal.y_target,
        cal_yhat=cal_yhat,
        cal_sl=cal_sl,
        cal_su=cal_su,
        cal_w=cal_w,
        cal_origins=data.cal.origins,
        test_y=data.test.y_target,
        test_yhat=test_yhat,
        test_sl=test_sl,
        test_su=test_su,
        test_w=test_w,
        test_origins=data.test.origins,
        horizon=data.horizon,
        alpha=alpha,
        mode=cfg["conf_mode"],
    )

    metrics = evaluate_predictions(
        data.test.y_target, test_yhat, lower, upper, y_range=data.y_range_fit, alpha=alpha
    )
    abs_err = np.abs(data.test.y_target - test_yhat)
    tot_scale = test_sl + test_su
    sp_scale = float(spearmanr(tot_scale, abs_err).statistic) if np.std(tot_scale) > 1e-6 else 0.0
    sp_psi = float(spearmanr(test_inc_mean, abs_err).statistic) if np.std(test_inc_mean) > 1e-6 else 0.0
    metrics["spearman_scale_err"] = 0.0 if np.isnan(sp_scale) else sp_scale
    metrics["spearman_psi_err"] = 0.0 if np.isnan(sp_psi) else sp_psi

    winkler_vec = compute_winkler_vector(
        data.test.y_target, test_yhat, lower, upper, alpha=alpha
    )

    rules = (
        model.tsk_head.export_rules(FGBIPNModel.STATE_NAMES)
        if (model.use_tsk and model.tsk_head is not None and variant_name == "FGB-IPN (Full)")
        else None
    )

    return {
        "variant": variant_name,
        "seed": seed,
        "params": count_parameters(model),
        "train_sec": train_sec,
        "infer_ms_per_sample": infer_ms,
        "metrics": metrics,
        "winkler_vec": winkler_vec,
        "rules": rules,
    }
