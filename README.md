# SNCF-Transilien-IA

This repository contains an advanced machine learning pipeline developed for the **ENS Challenge 166**. The objective is to provide short-term predictions of train arrival delays across the SNCF Transilien network.

## Project Overview

The challenge focuses on predicting the waiting time difference (**p0q0**) for a train located two stations upstream. The target variable represents the deviation between theoretical and actual arrival times:
* **Negative values**: Indicate delays (longer waiting times than scheduled).
* **Positive values**: Indicate arrivals ahead of schedule.

**Current Performance (I-14→I-15)**: Website leaderboard MAE = **0.6724** (non-regression baseline). Local metrics via 5-fold GroupKFold and time-holdout validation confirm stability across train/test distributions.

## Technical Architecture

The pipeline implements a sophisticated multi-model approach designed to minimize Mean Absolute Error (MAE) while ensuring robustness against distribution shifts.

### Feature Engineering
The current feature set consists of **111+ total variables** structured into specialized groups:
* **Flow Features (8)**: Real-time traffic density and network load (`flowavg1` – `flowavg8`).
* **Graph-based Features (88)**: Topological relationships and connectivity between stations (cycle-aware graph metrics).
* **Categorical Encodings**:
  - **Gare (Station) Embedding**: Categorical station identifier (84 unique gares) encoded as learned Embedding(84, 16) in the NN (replaces one-hot encoding). LightGBM receives ordinal encoding.
  - **Leave-One-Out (LOO) Target Encodings**: High-cardinality features capture historical signal without leakage. Currently active: `gare_delay_enc` (mean delay per station, grouped by 5-fold CV), `train_delay_enc` (mean delay per train number). Computed within GroupKFold to prevent train/test contamination.
* **Temporal & Contextual**: Day of week, time of day, and other cyclical features.

### Model Strategy
The system utilizes a **weighted ensemble** optimized for MAE minimization:
1. **LightGBM (90% weight)**: Tuned boosting model with 5-fold GroupKFold OOF for training rows and bagged test predictions. Hyperparameters: `num_leaves=127`, `min_child_samples=40`, `reg_lambda=3.0`, `learning_rate=0.02`. Achieves LGBM OOF MAE ≈ 0.640.
2. **Neural Network (10% weight)**: Keras residual MLP with dual inputs: (a) learned `gare` Embedding(N_GARE=84, 16) for categorical station context, and (b) numeric features. NN validation MAE ≈ 0.673. Designed for residual correction and capturing smooth, non-linear temporal patterns.
3. **Ensemble Blend**: Simple weighted average at inference. CatBoost was previously included but dropped after I-15 (weight optimisation converged to 0.000, indicating LGBM captures all signal on the given feature set).

## Data Consistency and Validation

The pipeline enforces strict alignment between local validation and leaderboard performance through a **multi-level validation strategy**:
* **GroupKFold Cross-Validation**: 5-fold stratification by train/route groups ensures temporal and operational consistency. LGBM OOF MAE ≈ 0.640; reported as single source-of-truth for feature engineering decisions.
* **Time-Holdout Validation**: Chronological split to detect covariate shift and verify that features generalize across unseen time periods.
* **Distribution Audit**: Adversarial validation identifies features with significant train/test drift. Features flagged for removal if drift exceeds threshold (e.g., `is_weekend` was a constant — removed in I-14).
* **Leaderboard Alignment**: Website MAE 0.6724 vs. local GroupKFold OOF 0.640 = 0.032 gap, indicating healthy generalization with no serious distribution mismatch.

## Recent Improvements (Iteration Log)

See `SKILL.md` for detailed iteration history and ablation studies. Key recent wins:

* **I-14 (2026-05-05)**: Removed `is_weekend` constant feature and audited feature importance. Achieved **0.6724 MAE** (baseline).
* **I-15 (2026-05-05)**: Dropped CatBoost (zero weight in ensemble), added `arret_delay_enc` + `train_gare_enc` LOO encodings. Reverted LOO additions due to test-set cardinality mismatch; kept CatBoost off. Effective ensemble: **LGBM×0.90 + NN×0.10**.
* **I-13 (2026-05-05)**: OOF weight optimisation via `scipy.optimize.minimize_scalar` revealed CatBoost contributes zero weight—LGBM dominates on this feature set. Introduced `train_delay_enc` LOO encoding.

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