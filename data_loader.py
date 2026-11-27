"""Causal multivariate time-series dataset loaders and window preparation.

Supported datasets under `datasets/`:
  - electricity : Electricity consumption (321 dims)
  - solar       : Solar power production (137 dims)
  - uci501      : Beijing multi-site air quality (132 dims)
  - energy      : Appliances energy prediction (28 dims)
  - sml         : SML2010 indoor monitoring (22 dims)
  - airquality  : UCI AirQuality (13 dims)
  - etth1       : Electricity transformer temperature hourly 1 (7 dims)
  - etth2       : Electricity transformer temperature hourly 2 (7 dims)
"""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_SOURCES_DIR = DEFAULT_DATASETS_DIR
DEFAULT_EXTRA_DATASETS_DIR = DEFAULT_DATASETS_DIR


@dataclass
class SplitData:
    X_exog: np.ndarray        # (N_samples, d, L) standardized exogenous windows
    y_hist: np.ndarray        # (N_samples, L) standardized target history
    y_target: np.ndarray      # (N_samples,) original-scale target y_{t+h}
    y_target_std: np.ndarray  # (N_samples,) standardized target
    origins: np.ndarray       # (N_samples,) integer time index of forecast origin t


@dataclass
class PreparedDataset:
    name: str
    feature_names: List[str]
    target_name: str
    lookback: int
    horizon: int
    y_mean: float
    y_std: float
    y_range_fit: float
    fit: SplitData
    val: SplitData
    cal: SplitData
    test: SplitData


def _parse_float(val: str) -> float:
    s = val.strip()
    if not s or s.upper() == "NA" or s.lower() == "nan":
        return np.nan
    return float(s)


def _causal_ffill_matrix(mat: np.ndarray, init_fill: Optional[np.ndarray] = None) -> np.ndarray:
    """Forward-fill NaNs strictly causally along time axis 0."""
    out = mat.copy()
    T, D = out.shape
    for j in range(D):
        col = out[:, j]
        last = np.nan if init_fill is None else float(init_fill[j])
        if np.isnan(last):
            valid = np.flatnonzero(~np.isnan(col))
            last = float(col[valid[0]]) if len(valid) > 0 else 0.0
        for t in range(T):
            if np.isnan(col[t]):
                col[t] = last
            else:
                last = col[t]
    return out


def load_uci501_multisite(
    sources_dir: Path = DEFAULT_SOURCES_DIR,
    target_station: str = "Aotizhongxin",
    target_col: str = "PM2.5",
    seal_2017_holdout: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load all 12 Beijing stations (132 numeric sensors aligned hourly).

    If `seal_2017_holdout=True`, truncates before 2017-01-01 to preserve the
    sealed 2017-01..02 final confirmation holdout.
    """
    outer_zip = sources_dir / "beijing_multi_site_uci501.zip"
    sensor_cols = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    ]
    station_arrays: Dict[str, np.ndarray] = {}
    timestamps: Optional[List[Tuple[int, int, int, int]]] = None

    with zipfile.ZipFile(outer_zip) as outer:
        inner_bytes = outer.read("PRSA2017_Data_20130301-20170228.zip")

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        csv_names = sorted(n for n in inner.namelist() if n.endswith(".csv"))
        for name in csv_names:
            with inner.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                rows = list(reader)
            if not rows:
                continue
            station = rows[0]["station"].strip()
            ts = [
                (int(r["year"]), int(r["month"]), int(r["day"]), int(r["hour"]))
                for r in rows
            ]
            if timestamps is None:
                timestamps = ts
            else:
                assert ts == timestamps, f"Timestamp mismatch in {station}"
            arr = np.empty((len(rows), len(sensor_cols)), dtype=np.float64)
            for i, r in enumerate(rows):
                for j, col in enumerate(sensor_cols):
                    arr[i, j] = _parse_float(r[col])
            station_arrays[station] = arr

    assert timestamps is not None and target_station in station_arrays
    if seal_2017_holdout:
        keep_n = sum(1 for y, _, _, _ in timestamps if y < 2017)
    else:
        keep_n = len(timestamps)

    exog_cols: List[np.ndarray] = []
    exog_names: List[str] = []
    target_vec: Optional[np.ndarray] = None

    for station in sorted(station_arrays.keys()):
        arr = station_arrays[station][:keep_n]
        for j, col in enumerate(sensor_cols):
            if station == target_station and col == target_col:
                target_vec = arr[:, j].copy()
            else:
                exog_cols.append(arr[:, j])
                exog_names.append(f"{station}_{col}")

    assert target_vec is not None
    X_raw = np.stack(exog_cols, axis=1)  # (T, 131)
    return X_raw, target_vec, exog_names, f"{target_station}_{target_col}"


def load_gz_matrix_dataset(
    gz_path: Path,
    prefix: str,
    target_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load comma-separated gzipped multivariate matrix (Electricity-321, Solar-137)."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        mat = np.loadtxt(f, delimiter=",", dtype=np.float64)
    d_total = mat.shape[1]
    t_pos = d_total - 1 if target_idx < 0 else target_idx
    y = mat[:, t_pos].copy()
    exog_mask = [j for j in range(d_total) if j != t_pos]
    X = mat[:, exog_mask]
    feat_names = [f"{prefix}_ch{j+1}" for j in exog_mask]
    return X, y, feat_names, f"{prefix}_ch{t_pos+1}"


def load_sml2010(
    sources_dir: Path = DEFAULT_SOURCES_DIR,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load SML2010 indoor monitoring dataset (22 numeric sensors, target = dining room temp)."""
    zip_path = sources_dir / "SML2010UCI.zip"
    rows_all = []
    header_cols: List[str] = []
    with zipfile.ZipFile(zip_path) as z:
        for fn in ["NEW-DATA-1.T15.txt", "NEW-DATA-2.T15.txt"]:
            with z.open(fn) as f:
                lines = [ln.decode("utf-8").strip() for ln in f if ln.decode("utf-8").strip()]
            h_tokens = lines[0].lstrip("#").strip().split()
            if not header_cols:
                header_cols = h_tokens[2:]  # skip Date, Time
            for ln in lines[1:]:
                toks = ln.split()
                if len(toks) == len(h_tokens):
                    rows_all.append([float(v) for v in toks[2:]])
    mat = np.asarray(rows_all, dtype=np.float64)
    # Column 0 is 3:Temperature_Comedor_Sensor
    y = mat[:, 0].copy()
    X = mat[:, 1:]
    feat_names = header_cols[1:]
    return X, y, feat_names, header_cols[0]


def load_airquality_uci(
    sources_dir: Path = DEFAULT_SOURCES_DIR,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load UCI AirQuality (13 numeric sensors, -200 sentinel -> NaN, target = AH)."""
    zip_path = sources_dir / "AirQualityUCI.zip"
    with zipfile.ZipFile(zip_path) as z:
        with z.open("AirQualityUCI.csv") as f:
            lines = [ln.decode("utf-8", errors="ignore").strip() for ln in f if ln.decode("utf-8", errors="ignore").strip()]
    header = [h.strip() for h in lines[0].split(";") if h.strip()]
    num_cols = header[2:]  # skip Date, Time
    data_rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(";")]
        if len(parts) < 2 + len(num_cols) or not parts[0]:
            continue
        vals = []
        for p in parts[2 : 2 + len(num_cols)]:
            if not p:
                vals.append(np.nan)
            else:
                v = float(p.replace(",", "."))
                vals.append(np.nan if v <= -199.5 else v)
        data_rows.append(vals)
    mat = np.asarray(data_rows, dtype=np.float64)
    # Target is AH (last numeric column)
    t_pos = num_cols.index("AH")
    y = mat[:, t_pos].copy()
    exog_mask = [j for j in range(len(num_cols)) if j != t_pos]
    X = mat[:, exog_mask]
    feat_names = [num_cols[j] for j in exog_mask]
    return X, y, feat_names, "AH"


def load_csv_dataset(
    csv_path: Path,
    target_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = list(reader)

    # Identify numeric columns (skip date/time text column if present)
    numeric_idx: List[int] = []
    for idx, col_name in enumerate(header):
        if col_name.lower() in {"date", "time", "timestamp"}:
            continue
        try:
            _parse_float(rows[0][idx])
            numeric_idx.append(idx)
        except ValueError:
            continue

    col_names = [header[i] for i in numeric_idx]
    mat = np.empty((len(rows), len(numeric_idx)), dtype=np.float64)
    for t, r in enumerate(rows):
        for j, idx in enumerate(numeric_idx):
            mat[t, j] = _parse_float(r[idx])

    if target_col is None:
        t_pos = len(col_names) - 1
    else:
        t_pos = col_names.index(target_col)

    y = mat[:, t_pos].copy()
    exog_mask = [j for j in range(len(col_names)) if j != t_pos]
    X = mat[:, exog_mask]
    exog_names = [col_names[j] for j in exog_mask]
    return X, y, exog_names, col_names[t_pos]


def _build_windows(
    X_std: np.ndarray,
    y_std: np.ndarray,
    y_raw: np.ndarray,
    y_valid_mask: np.ndarray,
    origin_indices: np.ndarray,
    lookback: int,
    horizon: int,
) -> SplitData:
    xs, yhs, yts, yts_std, valid_origins = [], [], [], [], []
    T = len(y_raw)
    for t in origin_indices:
        if t - lookback + 1 < 0 or t + horizon >= T:
            continue
        if not y_valid_mask[t + horizon]:
            continue
        xs.append(X_std[t - lookback + 1 : t + 1].T)       # (d, L)
        yhs.append(y_std[t - lookback + 1 : t + 1])         # (L,)
        yts.append(y_raw[t + horizon])
        yts_std.append(y_std[t + horizon])
        valid_origins.append(t)

    return SplitData(
        X_exog=np.asarray(xs, dtype=np.float32),
        y_hist=np.asarray(yhs, dtype=np.float32),
        y_target=np.asarray(yts, dtype=np.float32),
        y_target_std=np.asarray(yts_std, dtype=np.float32),
        origins=np.asarray(valid_origins, dtype=np.int64),
    )


def inject_sensor_corruption(
    X_std: np.ndarray,
    start_idx: int,
    corrupt_frac: float = 0.0,
    noise_std: float = 0.0,
    spike_prob: float = 0.0,
    spike_mag: float = 6.0,
    seed: int = 42,
) -> np.ndarray:
    """Inject realistic sensor noise and transient outlier bursts into exogenous channels."""
    if corrupt_frac <= 0.0 or (noise_std <= 0.0 and spike_prob <= 0.0):
        return X_std

    rng = np.random.default_rng(seed)
    out = X_std.copy()
    T, d = out.shape
    n_corrupt = max(1, int(round(d * corrupt_frac)))
    corrupt_cols = rng.choice(d, size=n_corrupt, replace=False)

    seg = out[start_idx:, corrupt_cols]
    if noise_std > 0.0:
        seg = seg + rng.normal(0.0, noise_std, size=seg.shape)
    if spike_prob > 0.0:
        mask = rng.random(size=seg.shape) < spike_prob
        signs = rng.choice([-1.0, 1.0], size=seg.shape)
        seg = seg + mask * signs * spike_mag
    out[start_idx:, corrupt_cols] = seg
    return out


def prepare_dataset(
    name: str,
    lookback: int = 48,
    horizon: int = 24,
    split_ratios: Tuple[float, float, float, float] = (0.50, 0.15, 0.15, 0.20),
    subsample_stride: int = 1,
    corrupt_frac: float = 0.0,
    noise_std: float = 0.0,
    spike_prob: float = 0.0,
    spike_mag: float = 6.0,
    corrupt_mode: str = "test_only",
    seed: int = 42,
    sources_dir: Path = DEFAULT_SOURCES_DIR,
) -> PreparedDataset:
    """Load, causally preprocess, split with horizon gaps, and window a dataset."""
    lname = name.lower()
    if lname == "electricity":
        X_raw, y_raw, feat_names, target_name = load_gz_matrix_dataset(
            DEFAULT_EXTRA_DATASETS_DIR / "electricity.txt.gz", prefix="ECL", target_idx=-1
        )
    elif lname == "solar":
        X_raw, y_raw, feat_names, target_name = load_gz_matrix_dataset(
            DEFAULT_EXTRA_DATASETS_DIR / "solar_AL.txt.gz", prefix="SOL", target_idx=0
        )
    elif lname == "uci501":
        X_raw, y_raw, feat_names, target_name = load_uci501_multisite(
            sources_dir=sources_dir, seal_2017_holdout=True
        )
    elif lname == "energy":
        X_raw, y_raw, feat_names, target_name = load_csv_dataset(
            sources_dir / "EnergyDataAuthor.csv", target_col="Appliances"
        )
    elif lname == "sml":
        X_raw, y_raw, feat_names, target_name = load_sml2010(sources_dir=sources_dir)
    elif lname == "airquality":
        X_raw, y_raw, feat_names, target_name = load_airquality_uci(sources_dir=sources_dir)
    elif lname == "etth1":
        X_raw, y_raw, feat_names, target_name = load_csv_dataset(
            sources_dir / "ETTh1.csv"
        )
    elif lname == "etth2":
        X_raw, y_raw, feat_names, target_name = load_csv_dataset(
            sources_dir / "ETTh2.csv"
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    T, d = X_raw.shape
    r_fit, r_val, r_cal, _ = split_ratios
    t_fit_end = int(T * r_fit)
    t_val_end = int(T * (r_fit + r_val))
    t_cal_end = int(T * (r_fit + r_val + r_cal))

    # Track real (non-missing) target labels before causal forward-filling history
    y_valid_mask = ~np.isnan(y_raw)

    # Fit-only medians for safe causal initialization
    fit_x_med = np.nanmedian(X_raw[:t_fit_end], axis=0)
    fit_x_med = np.where(np.isnan(fit_x_med), 0.0, fit_x_med)
    fit_y_med = float(np.nanmedian(y_raw[:t_fit_end]))

    X_filled = _causal_ffill_matrix(X_raw, init_fill=fit_x_med)
    y_filled = _causal_ffill_matrix(y_raw[:, None], init_fill=np.array([fit_y_med]))[:, 0]

    # Standardization strictly from fit segment
    x_mean = X_filled[:t_fit_end].mean(axis=0)
    x_std = X_filled[:t_fit_end].std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    X_std = (X_filled - x_mean) / x_std

    y_fit_valid = y_raw[:t_fit_end][y_valid_mask[:t_fit_end]]
    y_mean = float(y_fit_valid.mean())
    y_std = float(y_fit_valid.std())
    if y_std < 1e-8:
        y_std = 1.0
    y_std_vec = (y_filled - y_mean) / y_std
    y_range_fit = float(np.percentile(y_fit_valid, 99) - np.percentile(y_fit_valid, 1))
    if y_range_fit < 1e-8:
        y_range_fit = float(y_fit_valid.max() - y_fit_valid.min() + 1e-6)

    # Optional noise/outlier injection
    if corrupt_frac > 0.0:
        c_start = t_cal_end if corrupt_mode == "test_only" else 0
        X_std = inject_sensor_corruption(
            X_std,
            start_idx=c_start,
            corrupt_frac=corrupt_frac,
            noise_std=noise_std,
            spike_prob=spike_prob,
            spike_mag=spike_mag,
            seed=seed,
        )

    # Strict horizon-gapped origin indices for each split:
    # For fit: last label <= t_fit_end - 1  => last origin <= t_fit_end - 1 - horizon
    fit_origins = np.arange(lookback - 1, t_fit_end - horizon, subsample_stride)
    val_origins = np.arange(t_fit_end, t_val_end - horizon, subsample_stride)
    cal_origins = np.arange(t_val_end, t_cal_end - horizon, subsample_stride)
    test_origins = np.arange(t_cal_end, T - horizon, subsample_stride)

    return PreparedDataset(
        name=name,
        feature_names=feat_names,
        target_name=target_name,
        lookback=lookback,
        horizon=horizon,
        y_mean=y_mean,
        y_std=y_std,
        y_range_fit=y_range_fit,
        fit=_build_windows(X_std, y_std_vec, y_raw, y_valid_mask, fit_origins, lookback, horizon),
        val=_build_windows(X_std, y_std_vec, y_raw, y_valid_mask, val_origins, lookback, horizon),
        cal=_build_windows(X_std, y_std_vec, y_raw, y_valid_mask, cal_origins, lookback, horizon),
        test=_build_windows(X_std, y_std_vec, y_raw, y_valid_mask, test_origins, lookback, horizon),
    )
