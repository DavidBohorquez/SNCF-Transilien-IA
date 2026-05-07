# SNCF-Transilien-IA

This repository contains an advanced machine learning pipeline developed for the **ENS Challenge 166**. The objective is to provide short-term predictions of train arrival delays across the SNCF Transilien network.

## Project Overview

The challenge focuses on predicting the waiting time difference (**p0q0**) for a train located two stations upstream. The target variable represents the deviation between theoretical and actual arrival times:
* **Negative values**: Indicate delays (longer waiting times than scheduled).
* **Positive values**: Indicate arrivals ahead of schedule.

**Current Performance (I-19)**: Website leaderboard MAE = **0.6719** (achieved with `gare_dow_enc` LOO encoding). Local metrics via 5-fold GroupKFold and time-holdout validation confirm stability across train/test distributions.

## Technical Architecture

The pipeline implements a sophisticated multi-model approach designed to minimize Mean Absolute Error (MAE) while ensuring robustness against distribution shifts.

### Feature Engineering
The current feature set consists of **111+ total variables** structured into specialized groups:
* **Flow Features (8)**: Real-time traffic density and network load (`flowavg1` – `flowavg8`).
* **Graph-based Features (88)**: Topological relationships and connectivity between stations (cycle-aware graph metrics).
* **Categorical Encodings**:
  - **Gare (Station) Embedding**: Categorical station identifier (84 unique gares) encoded as learned Embedding(84, 16) in the NN (replaces one-hot encoding). LightGBM receives ordinal encoding.
  - **Leave-One-Out (LOO) Target Encodings**: High-cardinality features capture historical signal without leakage. Currently active:
    - `gare_delay_enc`: Mean delay per station, grouped by 5-fold CV
    - `train_delay_enc`: Mean delay per train number
    - `gare_dow_enc`: Mean delay per station-day-of-week pair (low-cardinality, 420 combinations; accepted in I-19)
    - All computed within GroupKFold to prevent train/test contamination.
* **Temporal & Contextual**: Day of week, time of day, and other cyclical features.

### Model Strategy
The system utilizes a **weighted ensemble** optimized for MAE minimization:
1. **LightGBM (90% weight)**: Tuned boosting model with 5-fold GroupKFold OOF for training rows and bagged test predictions. Hyperparameters: `num_leaves=127`, `min_child_samples=40`, `reg_lambda=3.0`, `learning_rate=0.02`. Achieves LGBM OOF MAE ≈ 0.640.
2. **Neural Network (10% weight)**: Keras residual MLP with dual inputs: (a) learned `gare` Embedding(N_GARE=84, 16) for categorical station context, and (b) numeric features. NN validation MAE ≈ 0.673. Designed for residual correction and capturing smooth, non-linear temporal patterns.
3. **Ensemble Blend**: Simple weighted average at inference. CatBoost was previously included but dropped after I-15 (weight optimisation converged to 0.000, indicating LGBM captures all signal on the given feature set).

## Feature Selection & Importance

An `audit_feature_importance` function runs at every iteration — LGBM gain + `mutual_info_regression` on a 30% sample — and prints a combined KEEP/DEAD verdict table before training. All 31 current features pass the audit.

### Ranked by LGBM Gain (31 features, all KEEP)

| Rank | Feature | LGBM Gain | Mutual Info | Group |
|------|---------|-----------|-------------|-------|
| 1 | `gare_month_enc` | 512 277 | 0.277 | LOO Encoding |
| 2 | `gare_delay_enc` | 322 334 | 0.243 | LOO Encoding |
| 3 | `p2q0` | 274 516 | 0.059 | Upstream delay |
| 4 | `arret` | 247 936 | 0.034 | Stop position |
| 5 | `gare_in_count` | 190 729 | 0.243 | Graph |
| 6 | `p3q0` | 110 944 | 0.011 | Upstream delay |
| 7 | `p4q0` | 110 845 | 0.023 | Upstream delay |
| 8 | `doy_cos` | 78 592 | 0.009 | Temporal |
| 9 | `gare_in_mean_delay` | 54 438 | 0.244 | Graph |
| 10 | `gare_out_mean_delay` | 38 968 | 0.237 | Graph |
| 11 | `gare_cat` | 38 535 | 0.241 | Categorical |
| 12–19 | `flowavg2–5`, `gare_out_count`, `p0q2`, `p0q4`, `day_of_year` | 7k–36k | 0.09–0.24 | Flow / Upstream |
| 20 | `gare_dow_enc` | 14 623 | 0.240 | LOO Encoding |
| 21–31 | `p0q3`, `doy_sin`, `flowavg6–8`, `flowavg1`, `day_of_week`, `month`, `dow_sin`, `dow_cos`, `train_delay_enc` | 0–7k | 0.00–0.07 | Mixed |

**Key observations:**
- LOO encodings (`gare_month_enc`, `gare_delay_enc`) dominate LGBM gain, confirming station-level historical signal is the strongest predictor.
- Graph features (`gare_in_count`, `gare_in/out_mean_delay`) show high mutual information (~0.24), suggesting they capture structural delay propagation well.
- `flowavg1`, `month`, `dow_sin/cos`, `train_delay_enc` have zero LGBM gain but are retained — mutual information confirms they carry latent signal not captured by gain alone.
- No features were marked DEAD in the current configuration.

## Data Consistency and Validation

The pipeline enforces strict alignment between local validation and leaderboard performance through a **multi-level validation strategy**:
* **GroupKFold Cross-Validation**: 5-fold stratification by train/route groups ensures temporal and operational consistency. LGBM OOF MAE ≈ 0.640; reported as single source-of-truth for feature engineering decisions.
* **Time-Holdout Validation**: Chronological split to detect covariate shift and verify that features generalize across unseen time periods.
* **Distribution Audit**: Adversarial validation identifies features with significant train/test drift. Features flagged for removal if drift exceeds threshold (e.g., `is_weekend` was a constant — removed in I-14).
* **Leaderboard Alignment**: Website MAE 0.6719 vs. local GroupKFold OOF ~0.643 = 0.028 gap, indicating healthy generalization with no serious distribution mismatch.

## Recent Improvements (Iteration Log)

See `SKILL.md` for detailed iteration history and ablation studies. Key recent wins:

* **I-19 (2026-05-07) ★ CURRENT BEST**: Added `gare_dow_enc` — LOO encoding for (gare, day-of-week) pairs. Cardinality: 84×5=420 well-covered combinations. Website **0.6719** (improvement −0.0005 vs I-14 baseline 0.6724). ✅ ACCEPTED.
* **I-20 (2026-05-07)**: Attempted `gare_month_enc + train_dow_enc`. Website **0.7108** — catastrophic regression. ❌ REJECTED. Confirmed: `train`-based LOO encodings fail due to structurally different train ID distribution in test set.
* **I-14 (2026-05-05)**: Removed `is_weekend` constant feature and audited feature importance. Achieved **0.6724 MAE** (non-regression baseline).
* **I-13 (2026-05-05)**: OOF weight optimisation via `scipy.optimize.minimize_scalar` revealed CatBoost weight=0.000. Introduced `train_delay_enc` LOO encoding. Effective ensemble: **LGBM×0.90 + NN×0.10**.

## Requirements

The project is developed using a virtual environment (`mon_env`) with the following core dependencies:
* Python 3.12+
* TensorFlow 2.21.0
* Scikit-Learn 1.8.0
* Pandas 3.0.1
* LightGBM / CatBoost
* Category Encoders

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/SNCF-Transilien-IA.git](https://github.com/YourUsername/SNCF-Transilien-IA.git)