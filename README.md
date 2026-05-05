# SNCF-Transilien-IA

This repository contains an advanced machine learning pipeline developed for the **ENS Challenge 166**. The objective is to provide short-term predictions of train arrival delays across the SNCF Transilien network.

## Project Overview

The challenge focuses on predicting the waiting time difference (**p0q0**) for a train located two stations upstream. The target variable represents the deviation between theoretical and actual arrival times:
* **Negative values**: Indicate delays (longer waiting times than scheduled).
* **Positive values**: Indicate arrivals ahead of schedule.

## Technical Architecture

The pipeline implements a sophisticated multi-model approach designed to minimize Mean Absolute Error (MAE) while ensuring robustness against distribution shifts.

### Feature Engineering
The current feature set consists of **111 total variables** structured into specialized groups:
* **Flow Features (8)**: Capturing real-time traffic density and network load.
* **Graph-based Features (88)**: Representing the topological relationships and connectivity between stations.
* **One-Hot Encoded Categories**: Geographic context through station identifiers (Gare).
* **Leave-One-Out (LOO) Encoding**: High-cardinality features such as `train_number` are encoded using LOO techniques to capture historical performance without data leakage.

### Model Strategy
To reach the target performance of **0.64 MAE**, the system utilizes a hybrid architecture:
1. **Gradient Boosting (LightGBM/CatBoost)**: Leveraging sequential error correction to handle tabular data and non-linear feature interactions.
2. **Neural Networks (TensorFlow)**: Employed as feature extractors to identify complex temporal patterns.
3. **Stacking Ensemble**: The outputs of initial models serve as inputs for a meta-regressor, moving beyond simple blending to achieve higher predictive precision.

## Data Consistency and Validation

A core component of this pipeline is the **Distribution Audit**. Given that the test dataset contains randomly deleted rows, the model implements:
* **Adversarial Validation**: Identifying features that differ significantly between training and testing environments.
* **Covariate Shift Detection**: Filtering out high-drift variables (e.g., specific train counts) that may introduce noise.
* **K-Fold Cross-Validation**: Ensuring local validation scores (currently ~0.73) align with leaderboard performance.

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