# FGB-IPN

Reference implementation and benchmark datasets for **FGB-IPN** (Dynamic Fuzzy Granular-Ball Spectral-TSK Network with Adaptive Conformal Calibration for High-Dimensional Time-Series Interval Prediction).

## Repository Structure

```text
FGB-IPN/
├── data_loader.py          # Chronological 4-way splits (fit/val/cal/test) and sensor corruption
├── sfgb.py                 # Justifiable Spectral Granularity (JSG) & Dynamic Fuzzy Granular-Ball layer
├── fgb_tsk_net.py          # Complex rFFT spectral mixing backbone & ellipsoidal TSK interval head
├── conformal.py            # Rule-weighted delayed-feedback ACI calibration & evaluation metrics
├── baselines.py            # Interval prediction baselines (ECQR, MQR-NN, ACHF, CAFHT, DAQD-NN, MSDIPN)
├── trainer.py              # Training loop, baseline compression bases, and variant evaluation
├── run_experiments.py      # Main benchmark, ablation, corruption-stress, and sensitivity runner
├── generate_figures.py     # Generates Figures 1--6 (PDF and EPS) from results/
├── datasets/               # Eight multivariate benchmark datasets (7--321 dims)
├── results/                # Recorded JSON outputs across seeds (17, 31, 43)
└── figures/                # Output directory for generated figures
```

## Datasets

All eight benchmark datasets used in the paper are placed under `datasets/`:

| Dataset | File | Dims | Granularity |
| :--- | :--- | :---: | :---: |
| Electricity | `electricity.txt.gz` | 321 | 1 h |
| Solar-Energy | `solar_AL.txt.gz` | 137 | 10 min |
| Beijing-Air | `beijing_multi_site_uci501.zip` | 132 | 1 h |
| EnergyData | `EnergyDataAuthor.csv` | 28 | 10 min |
| SML2010 | `SML2010UCI.zip` | 22 | 15 min |
| AirQuality | `AirQualityUCI.zip` | 13 | 1 h |
| ETTh1 | `ETTh1.csv` | 7 | 1 h |
| ETTh2 | `ETTh2.csv` | 7 | 1 h |

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Experiments and Figures

Run main comparisons across the eight benchmarks:

```bash
python run_experiments.py --mode main \
  --datasets electricity solar uci501 energy sml airquality etth1 etth2 \
  --seeds 17 31 43 --epochs 20 --output-dir results
```

Run clean and 30%-corrupted ablation studies:

```bash
python run_experiments.py --mode ablation \
  --datasets electricity solar uci501 energy \
  --seeds 17 31 43 --epochs 20 --output-dir results

python run_experiments.py --mode ablation \
  --datasets electricity solar uci501 energy \
  --seeds 17 31 43 --epochs 20 \
  --corrupt-frac 0.3 --noise-std 1.5 --spike-prob 0.05 --output-dir results
```

Run sensor-corruption stress curves (0%--40%) and hyperparameter sensitivity:

```bash
python run_experiments.py --mode stress --datasets electricity uci501 --seeds 17 31 43 --epochs 20
python run_experiments.py --mode sensitivity --datasets electricity uci501 --seeds 17 31 43 --epochs 20
```

Regenerate all paper figures (PDF and EPS) in `figures/` from `results/`:

```bash
python generate_figures.py
```
