# SNCF Delay Prediction — Project Rules

## Goal
Predict `p0q0` (SNCF train delay, minutes). **Target MAE < 0.64** on the website leaderboard.

## Source of truth
- `test.py` is the **stable pipeline** and the only model file in the repo. All accepted improvements land here.
- Old reference files (`full_code.py`, `challenge_sncf.py`, `filtre.py`, `EDA.ipynb`, `analyse.ipynb`, etc.) were removed during the I-03 cleanup. Do not recreate them — extend `test.py` instead.

## Active model (I-12, website 0.6737 — non-regression baseline)
- **Ensemble**: `LGBM×0.5 + CatBoost×0.3 + NN×0.2` (weights in execution block).
- **LGBM**: `train_lgbm_oof_predict` — 5-fold GroupKFold OOF for training rows, bagged test predictions. Params in `_LGBM_PARAMS` dict (num_leaves=255, lr=0.02, rounds=3000).
- **CatBoost**: `train_catboost_oof_predict` — same GroupKFold + bagging pattern. Params in `_CATBOOST_PARAMS` dict.
- **NN**: Keras residual MLP (`train_nn_full_predict`), dual-input: gare Embedding(N_GARE,16) + numeric. Uses `nn_features` (no OHE gare), gare_cat passed separately. Params in `NN_CONFIG` dict.
- RF = legacy baseline (defined, not executed).
- **Never** hardcode hyperparameters inside `_build_nn`, `_fit_nn`, or `lgb.train` — use `NN_CONFIG`, `_LGBM_PARAMS`, `_CATBOOST_PARAMS`.

## Non-regression baseline
- **Website MAE ≤ 0.6737** (I-12). Any change that regresses past this must be reverted via `git` before continuing.
- Safety checks already wired into `test.py`: `assert len(submission) == len(x_test)` and `assert set(row_ids) == set(x_test_row_ids)`.
- The row-ID alignment bug (I-06–I-08) was caused by an unconditional `sort_values` reordering test rows. The sort is now **conditional** on `FEATURE_FLAGS['lag'] or FEATURE_FLAGS['trip_stats']` — never make it unconditional again.

## Code rules
- **Minimal comments.** Only document the *why* when non-obvious (e.g. validation strategy, data quirks). Never comment what well-named code already says.
- **Clean and organized.** Group related logic in clearly labeled sections (`# --- Section ---`). No dead code; delete unused imports/blocks before committing.
- **No redundant files.** One submission CSV at a time → `submission_test_rf.csv`. Old submissions get deleted, not duplicated. No `*_v2.py`, `*_backup.py`, `*_copy.py`. Use `git` for history.
- **No new files unless required.** Prefer extending `test.py` over spawning modules. Only extract to a module when the same logic is needed in two places.
- **Reproducibility.** Every randomized step takes `random_state=42` (or a documented seed).

## Validation rules
- The website score is the truth. Any local metric must be reported alongside `GroupKFold` and the time-based holdout — random KFold alone is not allowed to drive decisions.
- A change is "accepted" only when **both** GroupKFold OOF MAE **and** time-holdout MAE improve (or one improves while the other holds within ±0.005). Random-KFold gains alone do not count.
- After accepting a change, log it in `SKILL.md` with the three numbers (random / group / time) and the resulting website score.

## Submission rules
- `submission_test_rf.csv` must have one row per test row, columns `row_id, p0q0`. Test rows are **never** dropped during cleaning.
- Row IDs must come from `x_test['x_test_row_id']`. The two `assert` statements in the execution block enforce this automatically — do not remove them.
- `x_train_row_id`, `x_test_row_id`, `y_train_row_id` are in `non_features` and must stay there — ID columns must never enter model training.

## Workflow
1. Make the change in `test.py`.
2. Run `python test.py`, capture the three local metrics + the produced submission.
3. Submit, record the website score.
4. Append an entry to `SKILL.md`. Commit with a short, factual message naming the change and the deltas.
5. If the change regresses the website, revert via `git` and note the regression in `SKILL.md` so we don't try the same path again.
