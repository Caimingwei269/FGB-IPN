"""Fuzzy-weighted delayed-feedback adaptive conformal calibration and evaluation metrics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """Compute weighted finite-sample conformal quantile for q in (0, 1)."""
    q = float(np.clip(q, 0.0, 1.0))
    if len(values) == 0:
        return 1.0
    if q >= 1.0:
        return float(np.max(values))
    if q <= 0.0:
        return float(np.min(values))

    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_sum = float(np.sum(w_sorted))
    if w_sum <= 1e-12:
        return float(np.quantile(v_sorted, q))
    cdf = np.cumsum(w_sorted) / w_sum
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])


def compute_nonconformity_scores(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    s_lower: np.ndarray,
    s_upper: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Asymmetric normalized nonconformity score E_t = max((y_hat - y)/s^-, (y - y_hat)/s^+)."""
    e_low = (y_hat - y_true) / np.maximum(s_lower, eps)
    e_high = (y_true - y_hat) / np.maximum(s_upper, eps)
    return np.maximum(e_low, e_high)


def calibrate_and_predict_intervals(
    # Calibration split (all labels occur strictly before test split)
    cal_y: np.ndarray,
    cal_yhat: np.ndarray,
    cal_sl: np.ndarray,
    cal_su: np.ndarray,
    cal_w: np.ndarray,
    cal_origins: np.ndarray,
    # Test split
    test_y: np.ndarray,
    test_yhat: np.ndarray,
    test_sl: np.ndarray,
    test_su: np.ndarray,
    test_w: np.ndarray,
    test_origins: np.ndarray,
    horizon: int,
    alpha: float = 0.05,
    mode: str = "fgb_aci",
    gamma: float = 0.005,
    window_size: int = 300,
    decay: float = 0.995,
    fuzzy_beta: float = 0.65,
    c_env: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate strictly causal prediction intervals on test split.

    Modes:
      - 'static'    : static split conformal quantile on cal only (no online ACI)
      - 'global_aci': online unclipped ACI with time-decay weights only
      - 'fgb_aci'   : online unclipped delayed-feedback ACI + fuzzy rule activation weighting
    """
    assert mode in {"static", "global_aci", "fgb_aci"}

    cal_scores = compute_nonconformity_scores(cal_y, cal_yhat, cal_sl, cal_su)
    cal_label_times = cal_origins + horizon

    # History buffers store realized (label_time, score, rule_weights)
    hist_label_times = list(cal_label_times.tolist())
    hist_scores = list(cal_scores.tolist())
    hist_weights = list(cal_w)
    max_realized_score = float(max(0.0, np.max(cal_scores))) if len(cal_scores) > 0 else 1.0

    n_test = len(test_y)
    lowers = np.empty(n_test, dtype=np.float64)
    uppers = np.empty(n_test, dtype=np.float64)
    q_trace = np.empty(n_test, dtype=np.float64)

    # Pending predictions awaiting label realization (label_time, lower, upper, y_true, score, rule_w)
    pending = []
    alpha_t = float(alpha)

    # Precompute static conformal quantile if static mode
    if mode == "static":
        n_c = len(cal_scores)
        q_level = min(1.0, np.ceil((n_c + 1) * (1.0 - alpha)) / max(1, n_c))
        q_static = float(np.quantile(cal_scores, q_level))
        q_static = max(q_static, 0.0)
        lowers = test_yhat - q_static * test_sl
        uppers = test_yhat + q_static * test_su
        q_trace.fill(q_static)
        return lowers, uppers, q_trace

    for i in range(n_test):
        orig_t = int(test_origins[i])

        # Process delayed feedback once target labels become observable (label_time <= orig_t)
        still_pending = []
        for item in pending:
            lbl_t, lo_p, hi_p, y_p, sc_p, rw_p = item
            if lbl_t <= orig_t:
                covered_p = (lo_p <= hi_p) and (lo_p <= y_p <= hi_p)
                err = 0.0 if covered_p else 1.0
                alpha_t = float(alpha_t + gamma * (alpha - err))
                if sc_p > max_realized_score:
                    max_realized_score = float(sc_p)
                hist_label_times.append(lbl_t)
                hist_scores.append(sc_p)
                hist_weights.append(rw_p)
            else:
                still_pending.append(item)
        pending = still_pending

        # Compute conformal scaling factor Q_t under boundary handling
        m = min(window_size, len(hist_scores))
        sc_win = np.asarray(hist_scores[-m:], dtype=np.float64)

        if alpha_t >= 1.0:
            # Empty prediction set \hat{I}_{t+h} = \varnothing (encoded via lo > hi: width=0, err=1)
            q_t = 0.0
            lo_i = float(test_yhat[i] + 1e-9)
            hi_i = float(test_yhat[i] - 1e-9)
        elif alpha_t <= 0.0:
            # Continuous linear excess envelope:
            # At alpha_t = 0, equals max(0, max(sc_win)); for alpha_t < -B_T/c_env, Q_t > B_T => err=0
            m_win = float(max(0.0, np.max(sc_win))) if len(sc_win) > 0 else max_realized_score
            q_t = m_win - c_env * alpha_t
            lo_i = float(test_yhat[i] - q_t * test_sl[i])
            hi_i = float(test_yhat[i] + q_t * test_su[i])
        else:
            age = np.arange(m - 1, -1, -1, dtype=np.float64)
            w_time = decay ** age

            if mode == "fgb_aci":
                rw_win = np.asarray(hist_weights[-m:], dtype=np.float64)  # (m, R)
                sim = np.clip(np.sum(rw_win * test_w[i][None, :], axis=-1), 0.0, 1.0)
                w_comb = w_time * ((1.0 - fuzzy_beta) + fuzzy_beta * sim)
            else:
                w_comb = w_time

            q_t = max(0.0, weighted_quantile(sc_win, w_comb, 1.0 - alpha_t))
            lo_i = float(test_yhat[i] - q_t * test_sl[i])
            hi_i = float(test_yhat[i] + q_t * test_su[i])

        lowers[i] = lo_i
        uppers[i] = hi_i
        q_trace[i] = q_t

        # 3. Enqueue for delayed realization at orig_t + horizon
        sc_i = float(
            compute_nonconformity_scores(
                np.array([test_y[i]]),
                np.array([test_yhat[i]]),
                np.array([test_sl[i]]),
                np.array([test_su[i]]),
            )[0]
        )
        pending.append((orig_t + horizon, lo_i, hi_i, float(test_y[i]), sc_i, test_w[i]))

    return lowers, uppers, q_trace


def compute_winkler_vector(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Compute pointwise Winkler Interval Score, handling empty intervals (lower > upper) cleanly."""
    non_empty = lower <= upper
    width = np.where(non_empty, upper - lower, 0.0)
    miss = np.where(
        non_empty,
        np.maximum(0.0, lower - y_true) + np.maximum(0.0, y_true - upper),
        np.abs(y_true - y_hat),
    )
    return width + (2.0 / alpha) * miss


def evaluate_predictions(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y_range: float,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute interval and point evaluation metrics on original target scale."""
    non_empty = lower <= upper
    covered = (non_empty & (y_true >= lower) & (y_true <= upper)).astype(np.float64)
    picp = float(np.mean(covered))
    width = np.where(non_empty, upper - lower, 0.0)
    pinaw = float(np.mean(width) / max(y_range, 1e-8))
    winkler_vec = compute_winkler_vector(y_true, y_hat, lower, upper, alpha=alpha)
    winkler = float(np.mean(winkler_vec))
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_hat)))
    return {
        "winkler": winkler,
        "picp": picp,
        "abs_cov_err": abs(picp - (1.0 - alpha)),
        "pinaw": pinaw,
        "rmse": rmse,
        "mae": mae,
    }


def paired_block_bootstrap_diff(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    block_size: int = 24,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Compute mean(loss_a - loss_b) and moving-block bootstrap 95% CI.

    Negative values favor method A (lower loss).
    """
    diff = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(diff)
    point = float(np.mean(diff))
    if n <= block_size:
        return point, point, point

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size + 1
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        boot_means[b] = float(np.mean(diff[idx]))
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return point, float(lo), float(hi)
