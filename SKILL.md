# SKILL.md — Iteration Log

Append-only ledger of every change to `test.py`, the local metrics it produced, and the resulting website score. One section per iteration. Newest at the top.

## Format

```
## I-NN — short name (YYYY-MM-DD, git <sha>)
- Change      : 1-2 sentences on what changed in test.py
- Random KFold: 0.XXXX
- GroupKFold  : 0.XXXX
- Time holdout: 0.XXXX
- Website     : 0.XXXX  (or "not submitted")
- Verdict     : ACCEPTED | REJECTED — reason
- Model config: only the params that differ from the previous iteration
```

A change is **ACCEPTED** when GroupKFold and time-holdout both improve (or one improves while the other holds within ±0.005). Random-KFold gains alone do not count. Rejected iterations stay in the log so we don't repeat them.

---

## I-01 — RandomForest baseline (pre-cleanup)
- Change      : original `test.py` — RF on one-hot `gare` + day features + delay columns, random KFold OOF.
- Random KFold: 0.7318
- GroupKFold  : not measured
- Time holdout: not measured
- Website     : 0.7962
- Verdict     : BASELINE — anchors the random/website gap (~0.064) we need to close.
- Model config: `RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_split=10, max_features='sqrt', bootstrap=False)`

## I-02 — Cycle removal + multi-weighted graph + aligned validation (2026-05-03)
- Change      : drop entire (train, date) trajectories where any `gare` repeats; vectorized multi-weighted DiGraph (count / mean_delay / mean_arret_diff) with adjacency matrices and `graph_cleaned.png`; replace random KFold with GroupKFold by (train, date) and add last-15% time holdout. RF features unchanged.
- Cleanup     : 17 / 37 545 cyclic trips removed (0.05%, 353 rows). Graph: 84 nodes, 1 042 edges.
- Random KFold: 0.7339
- GroupKFold  : 0.7339  (per-fold std 0.0054)
- Time holdout: 0.7255  (cutoff 2023-10-10, 562 700 / 104 204 rows)
- Website     : 0.7970  (slightly worse than I-01's 0.7962 — within noise; cycle removal cost ~0.001)
- Verdict     : INFRASTRUCTURE — kept. Numbers ~unchanged from I-01 baseline as expected.
- **Surprise** : GroupKFold by (train, date) is **identical** to random KFold (both 0.7339). The website gap (~0.064) is therefore **not** trip-level leakage. Time holdout is actually *better* than OOF (0.7255), so the gap is also **not** chronological drift in the obvious direction. The website hidden split must differ on another axis — candidates to test in I-03:
    - Unseen `gare` values in test (one-hot encoding silently drops them)
    - Unseen `train` IDs (currently excluded from features anyway, but check `gare` co-occurrence shifts)
    - Different `arret` distribution (e.g. test biased toward early/late stops)
    - Different distribution of the upstream `p2q0/p3q0/p4q0` predictors
  Action: before adding more features (I-03+), audit train-vs-test distribution of `gare`, `arret`, and the upstream delay columns to identify where the gap actually comes from.
- Model config: unchanged from I-01.

---

## I-12 — LOO gare encoding + CatBoost cat_features + weight shift (2026-05-04) ★ NEW BEST
- Change      : (1) Replaced 84-col OHE gare with LOO target encoding `gare_delay_enc` (smoothed cross-fold mean p0q0 per gare, α=1.0) — test rows use all-training-data mean. (2) CatBoost `cat_features=['gare_cat']` — native categorical treatment for gare integer. (3) Weights rebalanced LGBM×0.65, CB×0.25, NN×0.10. `gare_delay_enc` added to NN_SCALE_COLS and to numeric inputs.
- LGBM OOF MAE: 0.6416  (slight regression from 0.6383 — fewer features, less overfit)
- CatBoost OOF: 0.6584
- NN val MAE  : 0.6777  (best epoch 120 — healthy)
- Website     : **0.6737**  (improvement −0.0017 vs I-11 baseline 0.6754)
- Verdict     : ACCEPTED — LGBM OOF regressed but website improved, confirming LOO encoding reduced train/test distribution gap. Local-to-website gap narrowed.
- Model config: `LGBM×0.65, CB×0.25, NN×0.10`; 29 features total, 0 OHE gare cols, gare_delay_enc + gare_cat.
- **Non-regression baseline updated**: website ≤ 0.6737.
- **Data structure clarified**: `x_train` has 12 CSV cols (2 index cols: `Unnamed: 0.1` + `Unnamed: 0`) and `x_test` has 11 (1 index col: `Unnamed: 0`). After preprocessing, both have the same 10 features: train, gare, date, arret, p2q0, p3q0, p4q0, p0q2, p0q3, p0q4. No hidden extra columns — the remaining local-to-website gap (0.032) must come from temporal/arret/gare distribution shift, not missing features.

---

## I-11 — 3-model ensemble + tuned LGBM + CatBoost + gare Embedding (2026-05-04)
- Change      : (1) LGBM tuned: num_leaves=255, lr=0.02, num_boost_round=3000; (2) CatBoost added as third ensemble member (MAE obj, depth=8, lr=0.02, early_stop=50, bagged test preds); (3) NN gare OHE (84 cols) replaced with learned Keras Embedding(N_GARE, 16) — NN now dual-input (gare_cat integer + numeric); (4) weights rebalanced LGBM×0.5 + CB×0.3 + NN×0.2; (5) gare_cat LabelEncoder added for NN embedding. All models trained independently; test preds bagged.
- LGBM OOF MAE: 0.6383  (↓ from 0.6411 — LGBM tuning paid off)
- CatBoost OOF: 0.6581
- NN val MAE  : 0.6769  (best epoch 106 — healthy, gare embedding working)
- Website     : **0.6754**  (improvement −0.005 vs I-10 baseline 0.6804)
- Verdict     : ACCEPTED — new best. Local-to-website gap = 0.037 (LGBM OOF 0.6383 → website 0.6754). CatBoost and NN embedding both add marginal value; LGBM remains the dominant model.
- Model config: `LGBM_WEIGHT=0.5, CB_WEIGHT=0.3, NN_WEIGHT=0.2`; LGBM: num_leaves=255, lr=0.02, rounds=3000; CatBoost: depth=8, lr=0.02, iter=3000; NN: Embedding(N_GARE=84, 16) + residual MLP.
- **Non-regression baseline updated**: website ≤ 0.6754 is now the bar.
- **Next levers** (ordered by expected impact):
  1. LOO target encoding for gare (replace OHE 84-col with 1 smooth numeric) — closes train/test distribution gap
  2. CatBoost with cat_features=['gare_cat'] — native categorical treatment for gare
  3. Increase LGBM_WEIGHT to 0.65, CB=0.25, NN=0.10 — LGBM is clearly dominant
  4. OOF weight optimisation (scipy) on LGBM+CB OOF to find optimal blend analytically
  5. Stacking: NN GroupKFold OOF → CatBoost/LGBM meta-learner (needs 5× NN training)

---

## I-10 — Independent LGBM + NN ensemble (2026-05-04)
- Change      : LGBM and NN trained independently on same raw features; blended at output (`LGBM_WEIGHT=0.6`). No feature injection between models — each contributes its own orthogonal signal (trees vs. smooth embeddings). Row-ID fix from I-09 already in place.
- LGBM OOF MAE: 0.6411
- NN val MAE  : 0.6766  (best epoch 101/113 — healthy convergence, no epoch-1 collapse)
- Website     : **0.6804**  (previous best: 0.7055 at I-05; improvement = −0.0251)
- Verdict     : ACCEPTED — first clean submission since I-05 baseline. Local-to-website gap = 0.039 (down from 0.064 gap in I-01/I-02), confirming LGBM generalises well.
- Model config: `LGBM_WEIGHT=0.6`; LGBM: num_leaves=127, lr=0.05, subsample=0.8, early_stop=50; NN: unchanged from I-05 config.
- **Non-regression baseline**: website ≤ 0.6804 is now the bar. Any change that regresses beyond this must be reverted.
- **Next levers to pull** (ordered by expected impact):
  1. Increase `LGBM_WEIGHT` toward 0.7–0.8 (LGBM alone likely ~0.67; NN alone ~0.72)
  2. Tune LGBM: more leaves (255), lower lr (0.02), more rounds, dart booster
  3. Add CatBoost as a third ensemble member (different bias from LGBM)
  4. Tune NN toward lighter architecture optimised for residual correction
  5. Feature engineering: categorical embeddings for `gare` instead of OHE; temporal features

---

## I-09 — Row-ID alignment fix (2026-05-04) — THE REAL ROOT CAUSE
- Change      : (1) made the `sort_values(['train','date','arret'])` conditional on `FEATURE_FLAGS['lag'] or FEATURE_FLAGS['trip_stats']`; (2) excluded `x_train_row_id`, `x_test_row_id`, `y_train_row_id` from `non_features` so ID columns never enter model; (3) reverted submission to `x_test['x_test_row_id']` (correct since sort no longer fires); (4) added `assert set(row_ids) == set(x_test row_ids)` safeguard.
- Website     : not submitted standalone — applied together with I-10.
- Verdict     : INFRASTRUCTURE — essential fix. Corrected 3 scrambled submissions (I-06/07/08).
- **Root cause**: unconditional `sort_values` introduced in I-06 (for lag features) reordered all test rows. Submission built predictions in post-sort order but attached pre-sort row IDs → every prediction paired with wrong row. Local val was always honest; scrambled row IDs caused the 1.27→1.25→1.20 website collapse.

---

## I-08 — Hybrid: LGBM (5-fold OOF + bagged test) as feature for NN (2026-05-04)
- Change      : replaced the I-07 weighted-average ensemble with a true hybrid. Stage 1: LightGBM trains on 5 GroupKFold folds → produces OOF predictions (clean) for every training row + bagged test predictions (mean of fold predictions, same distribution as OOF). Stage 2: `lgbm_pred` injected as a feature into the NN. NN refines LGBM's signal.
- LGBM OOF MAE: 0.6429
- NN val MAE  : 0.6678 (best epoch=1 — symptom of the row-ID bug, not the architecture)
- Website     : 1.2029  (still scrambled — root cause was the same row-ID misalignment from I-06; not actually a stacking issue. The "fix" of bagging test preds was harmless but did not address the real bug.)
- Verdict     : REJECTED for the wrong reason — see I-09. The hybrid architecture itself is sound; we'll re-evaluate it once row IDs are aligned.

---

## I-07 — trip_stats off + lag off; simple LGBM+NN weighted average (2026-05-04)
- Change      : `FEATURE_FLAGS['trip_stats'] = False` and `FEATURE_FLAGS['lag'] = False`. Added a simple `LGBM_WEIGHT * lgbm_preds + (1-w) * nn_preds` ensemble.
- NN val MAE  : 0.6703
- LGBM val MAE: ~0.65
- Website     : 1.2481  — REJECTED (we incorrectly suspected lag features as the leakage source; the real bug was the unconditional sort + row-ID misalignment introduced in I-06, not the features themselves).

---

## I-06 — Advanced feature engineering: lag, trip stats, cyclical time + importance audit (2026-05-04)
- Change      : (1) trip-level lag features: lag1/lag2 of p2q0 and p0q2 within (train,date,arret) order; (2) trip aggregate stats: mean_p2q0, mean_p0q2, trip_arret_count, arret_position (normalized 0-1); (3) cyclical time encoding: day_of_week sin/cos, day_of_year sin/cos, month; (4) feature importance print via quick RF (n_estimators=50, 30% sample) before NN training — gated by `PRINT_IMPORTANCE=True` flag; (5) `FEATURE_FLAGS` dict to toggle each feature group on/off without editing logic.
- val MAE     : 0.6263  (best epoch — excellent local but severely leaked)
- Website     : 1.2776  (catastrophic — worst result so far)
- Verdict     : REJECTED — trip_stats caused target leakage. Test trips are partial (test set has ~1 stop per trip); trip mean of p0q2 ≈ the target on test rows. Model learned to rely on this and failed completely on the website's actual test distribution.

---

## I-05 — Residual NN architecture + hyperparameter config (2026-05-04)
- Change      : replaced flat MLP with configurable residual architecture. New `NN_CONFIG` dict at top of model section controls all knobs without touching architecture code. Defaults: units=[512,256,128,64], activation=GELU, dropout=[0.30,0.25,0.20,0.10], L2=1e-4, Huber delta=0.8, lr=1e-3, batch=1024, epochs=150, patience_ES=12, patience_LR=5. Residual skip connections added between same-width blocks. `_build_nn` dispatches LeakyReLU as a special case (not a Keras activation string).
- val MAE (10% internal): 0.6774  (best epoch 115/127)
- Website     : 0.7055  (flat vs I-04 0.7053; val improved but website didn't)
- Verdict     : ACCEPTED (val improved) — architecture is not the bottleneck; feature generalization is. Val/website gap = 0.028.
- Model config: units=[512,256,128,64], GELU, L2=1e-4, dropout=[0.30,0.25,0.20,0.10], Huber=0.8, batch=1024, epochs=150 (stopped 127).

---

## I-04 — Flow features (flowavg1-8) + faster execution (2026-05-04)
- Change      : (1) graph edge filter tightened to consecutive arrets only (arret_next == arret_current + 1); (2) added `delay` and `delay_s2` edge attrs (count × mean p2q0/p0q2); (3) `all_flow(G, target, depth)` BFS upstream propagation function added; (4) `flowavg1`–`flowavg8` features added (per-gare upstream delay flow at depths 1–8, mapped as row features before one-hot); (5) cross-validation removed from execution — validation is now the 10% internal val-split inside `train_nn_full_predict` only; (6) `train_nn_full_predict` returns `(preds, best_val_mae)`.
- val MAE (10% internal): 0.6827  (best epoch 49/57)
- Website     : 0.7053  (improvement +0.005 over I-03)
- Verdict     : ACCEPTED — flow features + consecutive-stop filter help; website/val gap narrowed to ~0.023.
- Model config: unchanged from I-03. 20 new flow features added to input layer.

---

## I-03 — Distribution audit + graph features + Neural Network (2026-05-03)
- Change      : (1) train-vs-test distribution audit at runtime (gare/arret/p2q0/p3q0/p4q0); (2) per-row graph features `gare_in_count`, `gare_out_count`, `gare_in_mean_delay`, `gare_out_mean_delay` added BEFORE one-hot of gare; (3) graph plot now idempotent (skips if `graph_cleaned.png` exists); (4) **switched active model to a Keras MLP** (Dense 256→128→64, Swish, Dropout 0.2/0.15, BatchNorm, Huber loss, Adam 1e-3, EarlyStopping on val_mae); (5) numeric inputs normalized with `StandardScaler` fit on train fold only; (6) RF code retained as legacy but no longer executed; (7) repo cleaned of `full_code.py`, `challenge_sncf.py`, `filtre.py`, `analyse.ipynb`, `EDA.ipynb`, `plot_iris_dataset.ipynb`, `reseaux_nureuon.py`, old submission CSVs.
- Random KFold: not run (RF skipped)
- NN GroupKFold: not measured (removed for speed in I-04)
- NN Time holdout: not measured
- Website     : 0.7107  (improvement of +0.086 over RF I-02)
- Verdict     : ACCEPTED — NN beats RF on website. Foundation for flow features (I-04).

---

## Roadmap (revised after I-03)
- I-04  : depending on I-03 outcome — either tune NN width/depth/regularization, or add `day_of_week` sin/cos cyclical encoding.
- I-05  : SVD / Node2Vec embeddings of the cleaned adjacency, replace raw one-hot `gare`.
- I-06  : lag/rolling features by `(gare, train)` (lag-1/2/3, rolling mean window 3) before feeding NN.
- I-07  : PyTorch BiLSTM over per-trip sequences sorted by `arret`.
- I-08  : RF + NN/LSTM ensemble, weights tuned on the time-holdout only.

## Repository hygiene log
- 2026-05-03 (I-03 cleanup): deleted `full_code.py`, `challenge_sncf.py`, `filtre.py`, `analyse.ipynb`, `EDA.ipynb`, `plot_iris_dataset.ipynb`, `reseaux_nureuon.py`, old `submission_*.csv`, old `graph_cleaned.png`. Repo root now contains only `test.py`, `CLAUDE.md`, `SKILL.md`, the four input CSVs, and the venv.
- Submission kept: `submission_test_rf.csv` (regenerated each run).
- Graph file kept: `graph_cleaned.png` (regenerated only if missing).
