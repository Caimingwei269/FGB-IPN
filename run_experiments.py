"""Experiment suite runner for FGB-IPN.

Modes:
  main        : 8 benchmarks comparing FGB-IPN against 6 interval prediction baselines
  ablation    : Controlled granulation, uncertainty-channel, and calibration ablations
  stress      : Sensor-corruption stress test (0% to 40% corrupted channels)
  sensitivity : Hyperparameter sensitivity over lookback L, ball cap M_max, and rule count R
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

torch.set_num_threads(2)

from baselines import run_sota_baseline
from conformal import calibrate_and_predict_intervals, evaluate_predictions, paired_block_bootstrap_diff
from data_loader import PreparedDataset, prepare_dataset
from fgb_tsk_net import FGBIPNModel, count_parameters
from sfgb import build_spectral_granular_balls
from trainer import compute_fit_pca_basis, predict_split, run_variant, set_seed, train_model


DATASET_DEFAULTS: Dict[str, Dict[str, int]] = {
    "electricity": {"lookback": 48, "horizon": 24, "stride": 4},
    "solar":       {"lookback": 48, "horizon": 24, "stride": 8},
    "uci501":      {"lookback": 48, "horizon": 24, "stride": 4},
    "energy":      {"lookback": 48, "horizon": 24, "stride": 4},
    "sml":         {"lookback": 48, "horizon": 24, "stride": 1},
    "airquality":  {"lookback": 48, "horizon": 24, "stride": 2},
    "etth1":       {"lookback": 48, "horizon": 24, "stride": 4},
    "etth2":       {"lookback": 48, "horizon": 24, "stride": 4},
}

SOTA_BASELINES = [
    "ECQR",
    "MQR-NN",
    "ACHF",
    "CAFHT",
    "DAQD-NN",
    "MSDIPN",
]

ABLATION_VARIANTS = [
    "FGB-IPN (Full)",
    "FGB-NoPsi-TSK-ACI",
    "SoftStat-GB-IPN",
    "Hard-GB-IPN",
    "VarCluster-TSK-ACI",
    "TopM-Select-TSK-ACI",
    "PCA-TSK-ACI",
    "Raw-AllDim-TSK-ACI",
    "FGB-Global-ACI",
    "FGB-TSK-Static",
]


def _summarize_runs(runs: List[Dict[str, object]]) -> Dict[str, object]:
    m_keys = ["winkler", "picp", "abs_cov_err", "pinaw", "rmse", "mae"]
    extra_keys = [k for k in ["spearman_scale_err", "spearman_psi_err"] if k in runs[0]["metrics"]]
    all_keys = m_keys + extra_keys
    mean_m = {k: float(np.mean([r["metrics"][k] for r in runs])) for k in all_keys}
    std_m = {k: float(np.std([r["metrics"][k] for r in runs])) for k in all_keys}
    return {
        "metrics_mean": mean_m,
        "metrics_std": std_m,
        "params": int(runs[0]["params"]),
        "train_sec": float(np.mean([r["train_sec"] for r in runs])),
        "infer_ms_per_sample": float(np.mean([r["infer_ms_per_sample"] for r in runs])),
    }


def run_main_dataset(
    dataset: str,
    seeds: List[int],
    epochs: int,
    alpha: float,
    out_dir: Path,
    corrupt_frac: float = 0.0,
    noise_std: float = 0.0,
    spike_prob: float = 0.0,
) -> Dict[str, object]:
    dcfg = DATASET_DEFAULTS[dataset]
    data = prepare_dataset(
        name=dataset,
        lookback=dcfg["lookback"],
        horizon=dcfg["horizon"],
        subsample_stride=dcfg["stride"],
        corrupt_frac=corrupt_frac,
        noise_std=noise_std,
        spike_prob=spike_prob,
        corrupt_mode="test_only",
        seed=seeds[0],
    )
    gb_struct = build_spectral_granular_balls(data.fit.X_exog, data.fit.y_hist)
    pca_basis = compute_fit_pca_basis(data.fit.X_exog, n_components=len(gb_struct.members))

    print(
        f"[MAIN:{dataset}] d={data.fit.X_exog.shape[1]}+1 -> M={len(gb_struct.members)} balls "
        f"(sizes={[len(m) for m in gb_struct.members]}), "
        f"splits=({len(data.fit.y_target)}, {len(data.val.y_target)}, {len(data.cal.y_target)}, {len(data.test.y_target)})",
        flush=True,
    )

    methods = ["FGB-IPN (Full)"] + SOTA_BASELINES
    runs_by_method: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}

    for seed in seeds:
        res_fgb = run_variant(
            "FGB-IPN (Full)", data, gb_struct, pca_basis, seed=seed, epochs=epochs, alpha=alpha, device="cpu"
        )
        runs_by_method["FGB-IPN (Full)"].append(res_fgb)
        for bname in SOTA_BASELINES:
            res_b = run_sota_baseline(bname, data, seed=seed, epochs=epochs, alpha=alpha, device="cpu")
            runs_by_method[bname].append(res_b)

    full_wink = np.mean([r["winkler_vec"] for r in runs_by_method["FGB-IPN (Full)"]], axis=0)
    results = {}
    for m in methods:
        entry = _summarize_runs(runs_by_method[m])
        if m != "FGB-IPN (Full)":
            other_wink = np.mean([r["winkler_vec"] for r in runs_by_method[m]], axis=0)
            pt, lo, hi = paired_block_bootstrap_diff(full_wink, other_wink, block_size=24)
            entry["full_minus_method_winkler"] = {"mean_diff": pt, "ci95": [lo, hi]}
        results[m] = entry
        mm = entry["metrics_mean"]
        print(
            f"  {m:<16} | Wink={mm['winkler']:8.3f} | PICP={mm['picp']:.4f} | "
            f"PINAW={mm['pinaw']:.4f} | RMSE={mm['rmse']:7.3f} | Params={entry['params']:7d} | ms={entry['infer_ms_per_sample']:.3f}",
            flush=True,
        )

    payload = {
        "dataset": dataset,
        "total_dims": int(data.fit.X_exog.shape[1] + 1),
        "exog_dims": int(data.fit.X_exog.shape[1]),
        "num_balls": len(gb_struct.members),
        "ball_sizes": [len(m) for m in gb_struct.members],
        "target_name": data.target_name,
        "split_sizes": {
            "fit": len(data.fit.y_target),
            "val": len(data.val.y_target),
            "cal": len(data.cal.y_target),
            "test": len(data.test.y_target),
        },
        "corrupt": {"frac": corrupt_frac, "noise_std": noise_std, "spike_prob": spike_prob},
        "results": results,
        "rules_seed0": runs_by_method["FGB-IPN (Full)"][0]["rules"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "clean" if corrupt_frac <= 0 else f"noisy_{int(corrupt_frac*100)}"
    out_path = out_dir / f"main_{dataset}_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_ablation_dataset(
    dataset: str,
    seeds: List[int],
    epochs: int,
    alpha: float,
    out_dir: Path,
    corrupt_frac: float = 0.0,
    noise_std: float = 0.0,
    spike_prob: float = 0.0,
) -> Dict[str, object]:
    dcfg = DATASET_DEFAULTS[dataset]
    data = prepare_dataset(
        name=dataset,
        lookback=dcfg["lookback"],
        horizon=dcfg["horizon"],
        subsample_stride=dcfg["stride"],
        corrupt_frac=corrupt_frac,
        noise_std=noise_std,
        spike_prob=spike_prob,
        corrupt_mode="test_only",
        seed=seeds[0],
    )
    gb_struct = build_spectral_granular_balls(data.fit.X_exog, data.fit.y_hist)
    pca_basis = compute_fit_pca_basis(data.fit.X_exog, n_components=len(gb_struct.members))

    tag = "clean" if corrupt_frac <= 0 else f"noisy_{int(corrupt_frac*100)}"
    print(f"[ABLATION:{dataset}:{tag}] d={data.fit.X_exog.shape[1]} -> M={len(gb_struct.members)}", flush=True)

    runs_by_var: Dict[str, List[Dict[str, object]]] = {v: [] for v in ABLATION_VARIANTS}
    for seed in seeds:
        for v in ABLATION_VARIANTS:
            runs_by_var[v].append(
                run_variant(v, data, gb_struct, pca_basis, seed=seed, epochs=epochs, alpha=alpha, device="cpu")
            )

    full_wink = np.mean([r["winkler_vec"] for r in runs_by_var["FGB-IPN (Full)"]], axis=0)
    results = {}
    for v in ABLATION_VARIANTS:
        entry = _summarize_runs(runs_by_var[v])
        if v != "FGB-IPN (Full)":
            other_wink = np.mean([r["winkler_vec"] for r in runs_by_var[v]], axis=0)
            pt, lo, hi = paired_block_bootstrap_diff(full_wink, other_wink, block_size=24)
            entry["full_minus_ablation_winkler"] = {"mean_diff": pt, "ci95": [lo, hi]}
        results[v] = entry
        mm = entry["metrics_mean"]
        print(
            f"  {v:<20} | Wink={mm['winkler']:8.3f} | PICP={mm['picp']:.4f} | "
            f"PINAW={mm['pinaw']:.4f} | RMSE={mm['rmse']:7.3f} | Params={entry['params']:7d}",
            flush=True,
        )

    payload = {
        "dataset": dataset,
        "tag": tag,
        "exog_dims": int(data.fit.X_exog.shape[1]),
        "num_balls": len(gb_struct.members),
        "ball_sizes": [len(m) for m in gb_struct.members],
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"ablation_{dataset}_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_stress_curve(
    dataset: str,
    seeds: List[int],
    epochs: int,
    alpha: float,
    out_dir: Path,
) -> Dict[str, object]:
    dcfg = DATASET_DEFAULTS[dataset]
    levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    methods = ["FGB-IPN (Full)", "Hard-GB-IPN", "VarCluster-TSK-ACI", "PCA-TSK-ACI", "CAFHT", "MSDIPN"]
    curve: Dict[str, Dict[str, Dict[str, float]]] = {}

    for frac in levels:
        data = prepare_dataset(
            name=dataset,
            lookback=dcfg["lookback"],
            horizon=dcfg["horizon"],
            subsample_stride=dcfg["stride"],
            corrupt_frac=frac,
            noise_std=1.5 if frac > 0 else 0.0,
            spike_prob=0.05 if frac > 0 else 0.0,
            corrupt_mode="test_only",
            seed=seeds[0],
        )
        gb_struct = build_spectral_granular_balls(data.fit.X_exog, data.fit.y_hist)
        pca_basis = compute_fit_pca_basis(data.fit.X_exog, n_components=len(gb_struct.members))
        key = f"{int(frac * 100)}%"
        curve[key] = {}
        print(f"[STRESS:{dataset}] corruption={key}", flush=True)

        for m in methods:
            runs = []
            for seed in seeds:
                if m in {"FGB-IPN (Full)", "Hard-GB-IPN", "VarCluster-TSK-ACI", "PCA-TSK-ACI"}:
                    runs.append(run_variant(m, data, gb_struct, pca_basis, seed=seed, epochs=epochs, alpha=alpha, device="cpu"))
                else:
                    runs.append(run_sota_baseline(m, data, seed=seed, epochs=epochs, alpha=alpha, device="cpu"))
            curve[key][m] = _summarize_runs(runs)["metrics_mean"]
            mm = curve[key][m]
            print(f"  {m:<16} | Wink={mm['winkler']:8.3f} | PICP={mm['picp']:.4f} | PINAW={mm['pinaw']:.4f} | RMSE={mm['rmse']:7.3f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"stress_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "levels": levels, "curve": curve}, f, indent=2)
    return curve


def run_sensitivity(
    dataset: str,
    seeds: List[int],
    epochs: int,
    alpha: float,
    out_dir: Path,
) -> Dict[str, object]:
    dcfg = DATASET_DEFAULTS[dataset]
    sens = {"lookback": {}, "max_balls": {}, "num_rules": {}}

    def _eval_custom(data: PreparedDataset, max_balls: int, num_rules: int) -> Dict[str, object]:
        gb_struct = build_spectral_granular_balls(data.fit.X_exog, data.fit.y_hist, max_balls=max_balls)
        runs = []
        for seed in seeds:
            set_seed(seed)
            model = FGBIPNModel(
                num_exog=data.fit.X_exog.shape[1],
                lookback=data.lookback,
                gb_structure=gb_struct,
                use_gb=True,
                gb_mode="soft",
                use_dispersion=True,
                use_tsk=True,
                num_rules=num_rules,
            )
            model, tsec = train_model(model, data, epochs=epochs, alpha=alpha, device="cpu")
            cal_yh, cal_sl, cal_su, cal_w, _ = predict_split(model, data.cal, data.y_mean, data.y_std)
            tst_yh, tst_sl, tst_su, tst_w, ims = predict_split(model, data.test, data.y_mean, data.y_std)
            lo, hi, _ = calibrate_and_predict_intervals(
                data.cal.y_target, cal_yh, cal_sl, cal_su, cal_w, data.cal.origins,
                data.test.y_target, tst_yh, tst_sl, tst_su, tst_w, data.test.origins,
                horizon=data.horizon, alpha=alpha, mode="fgb_aci",
            )
            runs.append({
                "metrics": evaluate_predictions(data.test.y_target, tst_yh, lo, hi, data.y_range_fit, alpha),
                "params": count_parameters(model),
                "train_sec": tsec,
                "infer_ms_per_sample": ims,
            })
        out = _summarize_runs(runs)
        out["actual_balls"] = len(gb_struct.members)
        return out

    # 1. Lookback L sensitivity
    for L in [24, 48, 72, 96]:
        data_L = prepare_dataset(dataset, lookback=L, horizon=dcfg["horizon"], subsample_stride=dcfg["stride"])
        res = _eval_custom(data_L, max_balls=12, num_rules=6)
        sens["lookback"][str(L)] = res
        print(f"[SENS:{dataset}] L={L} -> Wink={res['metrics_mean']['winkler']:.3f}, PICP={res['metrics_mean']['picp']:.4f}, M={res['actual_balls']}", flush=True)

    # 2. Max balls M_max sensitivity
    data_base = prepare_dataset(dataset, lookback=48, horizon=dcfg["horizon"], subsample_stride=dcfg["stride"])
    for mb in [4, 8, 12, 16]:
        res = _eval_custom(data_base, max_balls=mb, num_rules=6)
        sens["max_balls"][str(mb)] = res
        print(f"[SENS:{dataset}] M_max={mb} (actual={res['actual_balls']}) -> Wink={res['metrics_mean']['winkler']:.3f}, Params={res['params']}", flush=True)

    # 3. TSK Rules R sensitivity
    for R in [3, 6, 9, 12]:
        res = _eval_custom(data_base, max_balls=12, num_rules=R)
        sens["num_rules"][str(R)] = res
        print(f"[SENS:{dataset}] R={R} -> Wink={res['metrics_mean']['winkler']:.3f}, PICP={res['metrics_mean']['picp']:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"sensitivity_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "sensitivity": sens}, f, indent=2)
    return sens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["main", "ablation", "stress", "sensitivity"])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 31, 43])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--corrupt-frac", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--spike-prob", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for ds in args.datasets:
        if args.mode == "main":
            run_main_dataset(ds, args.seeds, args.epochs, args.alpha, out_dir, args.corrupt_frac, args.noise_std, args.spike_prob)
        elif args.mode == "ablation":
            run_ablation_dataset(ds, args.seeds, args.epochs, args.alpha, out_dir, args.corrupt_frac, args.noise_std, args.spike_prob)
        elif args.mode == "stress":
            run_stress_curve(ds, args.seeds, args.epochs, args.alpha, out_dir)
        elif args.mode == "sensitivity":
            run_sensitivity(ds, args.seeds, args.epochs, args.alpha, out_dir)


if __name__ == "__main__":
    main()
