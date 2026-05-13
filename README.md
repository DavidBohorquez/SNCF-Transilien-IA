# SNCF-Transilien-IA

This repository contains an advanced machine learning pipeline developed for the **ENS Challenge 166**. The objective is to provide short-term predictions of train arrival delays across the SNCF Transilien network.

## Project Overview

The challenge focuses on predicting the waiting time difference (**p0q0**) for a train located two stations upstream. The target variable represents the deviation between theoretical and actual arrival times:
* **Negative values**: Indicate delays (longer waiting times than scheduled).
* **Positive values**: Indicate arrivals ahead of schedule.

**Current Performance (I-36)**: Website leaderboard MAE = **0.6700** — best so far. Ensemble LGBM×0.85 + NN×0.15. Local metrics: GroupKFold OOF 0.6397, temporal OOF 0.6318.

## Technical Architecture

The pipeline implements a sophisticated multi-model approach designed to minimize Mean Absolute Error (MAE) while ensuring robustness against distribution shifts.

### Feature Engineering
The current feature set contains **43 active variables** across 7 groups:
* **Upstream delay (6)**: `p2q0`, `p3q0`, `p4q0`, `p0q2`, `p0q3`, `p0q4` — raw delay signals from neighboring stops.
* **Flow Features (8)**: `flowavg1`–`flowavg8` — real-time traffic density and network load.
* **Graph features (5)**: `gare_in_count`, `gare_out_count`, `gare_in_mean_delay`, `gare_out_mean_delay`, `gare_cat` — topological relationships and connectivity (cycle-aware DiGraph).
* **Per-station statistics (12)**: Mean and std of each upstream/downstream delay signal grouped by station — added in I-30→I-34:
  - Upstream: `gare_p2q0_mean/std`, `gare_p3q0_mean/std`, `gare_p4q0_mean/std`
  - Downstream: `gare_p0q2_mean/std`, `gare_p0q3_mean/std`, `gare_p0q4_mean/std`
* **LOO Target Encodings (3)**: Computed within GroupKFold to prevent leakage:
  - `gare_delay_enc`: Mean target delay per station
  - `train_delay_enc`: Mean target delay per train number
  - `gare_dow_enc`: Mean target delay per (station, day-of-week) pair — 420 low-cardinality combinations (I-19)
* **Stop position (1)**: `arret`
* **Temporal (8)**: `day_of_week`, `day_of_year`, `month`, `doy_sin`, `doy_cos`, `dow_sin`, `dow_cos` + cyclical encoding.

### Model Strategy
The system utilizes a **weighted ensemble** optimized for MAE minimization:
1. **LightGBM (85% weight)**: 5-fold GroupKFold OOF for training rows, bagged test predictions. Hyperparameters: `num_leaves=127`, `min_child_samples=40`, `reg_lambda=3.0`, `learning_rate=0.02`, `colsample_bytree=0.90`, `num_boost_round=5000`. OOF MAE ≈ 0.640.
2. **Neural Network (15% weight)**: Keras residual MLP, dual-input — `gare` Embedding(84, 16) + numeric features. NN val MAE ≈ 0.677. Weight increased from 10% to 15% in I-36 after ensemble tuning confirmed added value.
3. **Ensemble Blend**: Simple weighted average. CatBoost was dropped in I-15 (OOF weight collapsed to 0.000 for two consecutive iterations).

## Feature Selection & Importance

An `audit_feature_importance` function runs at every iteration — LGBM gain + `mutual_info_regression` on a 30% sample — and prints a KEEP/DEAD verdict before training. **All 43 current features pass the audit.**

### Active Features (43 total, all KEEP — state at I-36)

| Group | Features | Count | Notes |
|-------|----------|-------|-------|
| Per-station stats (upstream) | `gare_p0q2_mean/std`, `gare_p0q3_mean/std`, `gare_p0q4_mean/std`, `gare_p2q0_mean/std`, `gare_p3q0_mean/std`, `gare_p4q0_mean/std` | 12 | Top LGBM gain features; `gare_p0q4_mean` ranked #1 (I-34) |
| Upstream/downstream delay (raw) | `p2q0`, `p3q0`, `p4q0`, `p0q2`, `p0q3`, `p0q4` | 6 | Strong LGBM signal |
| Flow | `flowavg1`–`flowavg8` | 8 | `flowavg1` has 0 LGBM gain but retained (MI signal via NN) |
| Graph | `gare_in_count`, `gare_out_count`, `gare_in_mean_delay`, `gare_out_mean_delay` | 4 | High mutual info (~0.24) |
| LOO encodings | `gare_delay_enc`, `train_delay_enc`, `gare_dow_enc` | 3 | GroupKFold-safe; time-stable keys only |
| Categorical | `gare_cat` | 1 | Ordinal for LGBM; Embedding(84,16) for NN |
| Stop position | `arret` | 1 | |
| Temporal | `day_of_week`, `day_of_year`, `month`, `doy_sin`, `doy_cos`, `dow_sin`, `dow_cos` | 7 | `month`/`dow_sin/cos` have 0 LGBM gain but retained |
| Stop position | `arret` | 1 | |

### Rejected Features (do not re-add)

| Feature | Iteration | Reason |
|---------|-----------|--------|
| `is_weekend` | I-14 | Constant feature — dataset contains no weekend days |
| `arret_delay_enc` | I-15, I-25 | High cardinality; arret distribution shifts between train/test |
| `train_gare_enc` | I-15 | Train ID distribution structurally different in test set |
| `gare_month_enc` | I-21 | Month is time-period key — LOO values don't transfer to test period |
| `train_dow_enc` | I-20 | Train-based LOO — train IDs shift between train/test |
| `arret_p2q0_mean/std` | I-35 | Heavy LGBM adoption locally but hurt OOD generalization (timetable change) |

**Rules derived from rejections:**
- Never use `train` as a component in LOO encodings (train ID distribution shifts)
- Never use time-period keys (`month`, `day_of_year`) in LOO encodings (temporal shift)
- Features with 0 LGBM gain are NOT safe to prune — they carry signal via the NN (confirmed in I-29)

## Data Consistency and Validation

The pipeline enforces strict alignment between local validation and leaderboard performance through a **multi-level validation strategy**:
* **GroupKFold Cross-Validation**: 5-fold stratification by train/route groups ensures temporal and operational consistency. LGBM OOF MAE ≈ 0.640; reported as single source-of-truth for feature engineering decisions.
* **Time-Holdout Validation**: Chronological split to detect covariate shift and verify that features generalize across unseen time periods.
* **Distribution Audit**: Adversarial validation identifies features with significant train/test drift. Features flagged for removal if drift exceeds threshold (e.g., `is_weekend` was a constant — removed in I-14).
* **Leaderboard Alignment**: Website MAE 0.6700 vs. GroupKFold OOF 0.6397 / temporal OOF 0.6318 — gap ≈ 0.030, stable across iterations.

## Recent Improvements (Iteration Log)

See `SKILL.md` for detailed iteration history and ablation studies. Key recent wins:

* **I-36 (2026-05-13) ★ CURRENT BEST**: `colsample_bytree` 0.85→0.90; ensemble rebalanced LGBM×0.85 + NN×0.15. Website **0.6700** (−0.0002). ✅ ACCEPTED.
* **I-34 (2026-05-12)**: Added `gare_p0q3_mean/std`, `gare_p0q4_mean/std` downstream per-station stats. Website **0.6702** (−0.0004). ✅ ACCEPTED.
* **I-33 (2026-05-12)**: Added `gare_p0q2_mean/std`. `gare_p0q2_mean` became #1 LGBM gain feature (586k). Website **0.6706**. ✅ ACCEPTED.
* **I-32 (2026-05-10)**: Added `gare_p3q0/p4q0_mean/std` per-station upstream stats. Website **0.6703**. ✅ ACCEPTED.
* **I-31 (2026-05-10)**: Increased `num_boost_round` 3000→5000 (folds were hitting ceiling). Website **0.6706**. ✅ ACCEPTED.
* **I-30 (2026-05-10)**: Added `gare_p2q0_mean/std`. Website **0.6707**. ✅ ACCEPTED.
* **I-19 (2026-05-07)**: Added `gare_dow_enc` LOO for (gare, day-of-week). Website **0.6719**. ✅ ACCEPTED.

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