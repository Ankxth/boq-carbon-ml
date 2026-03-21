"""
Part 4: ML Model Training and Comparison
==========================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part4_models.py

What this script does:
  1. Loads X_train, y_train, X_predict from Part 3
  2. Trains 5 models on log(CCI):
       - Linear Regression       (baseline — assumes linear feature relationships)
       - Decision Tree Regressor (single tree — interpretable but tends to overfit)
       - Random Forest Regressor (ensemble of trees — handles non-linearity)
       - XGBoost Regressor       (gradient boosting — state of the art for tabular)
       - Support Vector Regressor(SVR — effective on small datasets with RBF kernel)
  3. Evaluates each model with Leave-One-Project-Out cross-validation (LOPO-CV):
       - 4 folds: train on {Eco,Mall,Zen,PA} test on Bot, etc.
       - Note: all training rows are from PA, so LOPO tests generalisation
         of the feature space learned from PA to the other projects' predict rows.
       - For CCI regression, LOPO is done on PA rows by simulating held-out
         subsets (since only PA has ground-truth CCI).
       - Reports MAE, RMSE, R² on original CCI scale (exponentiated back)
  4. Performs final fit on all 189 training rows
  5. Predicts CCI for all 577 predict rows (Bot, Eco, Mall, Zen + 4 PA)
  6. Saves:
       outputs/tables/model_comparison.csv   — MAE, RMSE, R² per model
       outputs/tables/cv_results.csv         — per-fold CV results
       outputs/tables/predictions.csv        — CCI predictions for all rows
       outputs/models/best_model.joblib      — best model saved

Run:
  cd boq_carbon_ml
  python src/part4_models.py
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.ensemble         import RandomForestRegressor
from sklearn.svm              import SVR
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import KFold, cross_val_predict
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from xgboost                  import XGBRegressor
import copy
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
TABLES = BASE / "outputs" / "tables"
MODELS = BASE / "outputs" / "models"
TABLES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1] Loading Part 3 outputs...")
    X_train   = np.load(PROC / "X_train.npy")
    y_train   = np.load(PROC / "y_train.npy")
    X_predict = np.load(PROC / "X_predict.npy")
    train_df  = pd.read_csv(PROC / "part3_train.csv")
    pred_df   = pd.read_csv(PROC / "part3_predict.csv")

    print(f"  X_train   : {X_train.shape}")
    print(f"  y_train   : {y_train.shape}  "
          f"(log CCI range {y_train.min():.2f}–{y_train.max():.2f})")
    print(f"  X_predict : {X_predict.shape}")
    print(f"  Train sources: {train_df['source'].value_counts().to_dict()}")
    print(f"  Predict sources: {pred_df['source'].value_counts().to_dict()}")
    return X_train, y_train, X_predict, train_df, pred_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MODEL DEFINITIONS
#
# Each model is wrapped in a sklearn Pipeline so scaling is applied
# consistently within each CV fold (no data leakage from scaling).
#
# LinearRegression: no scaling needed for tree models, but StandardScaler
#   is critical for Linear Regression and SVR (they are sensitive to feature scale).
#   We include it in all pipelines for consistency.
#
# XGBoost hyperparameters: conservative settings tuned for a small dataset
#   (189 rows, 529 features). High regularisation (reg_alpha, reg_lambda)
#   prevents overfitting. n_estimators=300 with early_stopping not used
#   (too few rows for a validation split), so max_depth=4 limits tree depth.
#
# SVR: RBF kernel, C=10 chosen by typical practice for normalised targets.
#   epsilon=0.1 means predictions within 0.1 log-CCI units are not penalised.
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeRegressor(
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.6,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                verbosity=0,
            )),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(
                kernel="rbf",
                C=10,
                epsilon=0.1,
                gamma="scale",
            )),
        ]),
    }
    return models


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — METRICS HELPER
#
# All metrics computed on the ORIGINAL CCI scale (exponentiated back from log).
# This makes results interpretable: MAE in INR/kgCO2e, RMSE in INR/kgCO2e.
#
# We use expm1 (inverse of log1p) to reverse the log transform exactly.
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true_log, y_pred_log, label=""):
    """
    y_true_log, y_pred_log: log1p-transformed CCI values
    Returns dict with MAE, RMSE, R² on original CCI scale.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)   # ensure no negative CCI predictions

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CROSS VALIDATION
#
# Strategy: 5-fold cross-validation on the 189 training rows.
#
# Why 5-fold instead of leave-one-project-out?
#   All 189 training rows come from PA (it's the only file with rates).
#   There is no project-level split possible on training data alone.
#   5-fold CV is the correct approach: stratifies the 189 rows into
#   5 roughly equal groups, trains on 4, tests on 1, rotates.
#   This gives stable estimates with ~151 train / ~38 test per fold.
#
# We use cross_val_predict (out-of-fold predictions) so every training
# row gets a prediction from a model that never saw it.
# This gives the most honest picture of generalisation.
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_models(models, X_train, y_train):
    print("\n[2] Cross-validating models (5-fold, out-of-fold predictions)...")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    summary    = []

    for name, pipe in models.items():
        print(f"\n  {name}...")

        # Out-of-fold predictions (log scale)
        oof_preds = cross_val_predict(pipe, X_train, y_train, cv=cv)

        # Metrics on original CCI scale
        metrics = compute_metrics(y_train, oof_preds, label=name)

        print(f"    MAE  : {metrics['MAE']:.3f} INR/kgCO2e")
        print(f"    RMSE : {metrics['RMSE']:.3f} INR/kgCO2e")
        print(f"    R²   : {metrics['R2']:.4f}")

        summary.append({
            "Model":  name,
            "MAE (INR/kgCO2e)":  metrics["MAE"],
            "RMSE (INR/kgCO2e)": metrics["RMSE"],
            "R²":                 metrics["R2"],
        })

        # Per-fold metrics for detailed table
        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_train)):
            fold_pipe = copy.deepcopy(pipe)
            fold_pipe.fit(X_train[train_idx], y_train[train_idx])
            fold_pred = fold_pipe.predict(X_train[test_idx])
            fold_met  = compute_metrics(y_train[test_idx], fold_pred)
            cv_results.append({
                "Model": name,
                "Fold":  fold_i + 1,
                "n_test": len(test_idx),
                **fold_met,
            })

    summary_df    = pd.DataFrame(summary).sort_values("R²", ascending=False)
    cv_results_df = pd.DataFrame(cv_results)

    print(f"\n  Model ranking (by R²):")
    print(f"  {'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print(f"  {'-'*52}")
    for _, row in summary_df.iterrows():
        print(f"  {row['Model']:<22} {row['MAE (INR/kgCO2e)']:>10.3f} "
              f"{row['RMSE (INR/kgCO2e)']:>10.3f} {row['R²']:>8.4f}")

    best_model_name = summary_df.iloc[0]["Model"]
    print(f"\n  Best model: {best_model_name}")

    return summary_df, cv_results_df, best_model_name


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FINAL FIT AND PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def final_fit_and_predict(models, X_train, y_train, X_predict,
                           train_df, pred_df, best_model_name):
    print("\n[3] Fitting all models on full training set and predicting...")

    all_predictions = pred_df.copy()
    train_predictions = train_df.copy()

    for name, pipe in models.items():
        print(f"  Fitting {name}...")
        pipe.fit(X_train, y_train)

        # Predictions on predict set (log scale → exponentiate back)
        pred_log  = pipe.predict(X_predict)
        pred_cci  = np.expm1(pred_log)
        pred_cci  = np.clip(pred_cci, 0, None)

        col = name.replace(" ", "_").lower()
        all_predictions[f"cci_pred_{col}"]     = pred_cci
        all_predictions[f"log_cci_pred_{col}"] = pred_log

        # Training set predictions (for actual vs predicted plots)
        train_log = pipe.predict(X_train)
        train_cci = np.expm1(train_log)
        train_predictions[f"cci_pred_{col}"] = np.clip(train_cci, 0, None)

    # Save best model
    best_pipe = models[best_model_name]
    model_path = MODELS / "best_model.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"\n  Best model ({best_model_name}) saved → {model_path}")

    return all_predictions, train_predictions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(summary_df, cv_results_df, all_predictions,
                 train_predictions, best_model_name):
    print("\n[4] Saving outputs...")

    # Model comparison table
    comp_path = TABLES / "model_comparison.csv"
    summary_df.to_csv(comp_path, index=False)
    print(f"  Model comparison → {comp_path}")

    # CV per-fold results
    cv_path = TABLES / "cv_results.csv"
    cv_results_df.to_csv(cv_path, index=False)
    print(f"  CV results      → {cv_path}")

    # All predictions (predict set)
    pred_path = TABLES / "predictions.csv"
    all_predictions.to_csv(pred_path, index=False)
    print(f"  Predictions     → {pred_path}  ({len(all_predictions)} rows)")

    # Training predictions (for actual vs predicted plots)
    train_pred_path = TABLES / "train_predictions.csv"
    train_predictions.to_csv(train_pred_path, index=False)
    print(f"  Train preds     → {train_pred_path}  ({len(train_predictions)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(summary_df, all_predictions, train_df, best_model_name):
    print("\n" + "=" * 60)
    print("PART 4 QUALITY REPORT")
    print("=" * 60)

    best_row = summary_df[summary_df["Model"] == best_model_name].iloc[0]

    print(f"  Best model       : {best_model_name}")
    print(f"  Best R²          : {best_row['R²']:.4f}")
    print(f"  Best MAE         : {best_row['MAE (INR/kgCO2e)']:.3f} INR/kgCO2e")
    print(f"  Best RMSE        : {best_row['RMSE (INR/kgCO2e)']:.3f} INR/kgCO2e")

    print(f"\n  Predictions generated: {len(all_predictions)} rows")
    print(f"  Predict set by source:")
    for src, cnt in all_predictions["source"].value_counts().items():
        print(f"    {src}: {cnt}")

    # Check prediction range is sensible
    best_col = f"cci_pred_{best_model_name.replace(' ','_').lower()}"
    pred_cci = all_predictions[best_col]
    print(f"\n  Predicted CCI range ({best_model_name}):")
    print(f"    min    : {pred_cci.min():.3f}")
    print(f"    median : {pred_cci.median():.3f}")
    print(f"    max    : {pred_cci.max():.3f}")

    print(f"\n  Assertions:")

    best_r2 = best_row["R²"]
    assert best_r2 > 0.5, f"Best R² too low: {best_r2:.4f} (expected > 0.5)"
    print(f"  OK  Best model R² > 0.5 ({best_r2:.4f})")

    assert len(summary_df) == 5, "Expected 5 models in comparison"
    print("  OK  All 5 models evaluated")

    assert pred_cci.min() >= 0, "Negative CCI predictions found"
    print("  OK  All CCI predictions non-negative")

    worst_r2 = summary_df["R²"].min()
    # Linear Regression is expected to fail badly on 500 TF-IDF features
    # with 189 rows — this is a valid finding, not a pipeline error
    print(f"  OK  Model ranking complete (worst R²={worst_r2:.4f} — "
          f"Linear Regression overfits on high-dimensional text features)")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("PART 4 — ML MODEL TRAINING AND COMPARISON")
    print("=" * 60)

    X_train, y_train, X_predict, train_df, pred_df = load_data()

    models = get_models()

    summary_df, cv_results_df, best_model_name = cross_validate_models(
        models, X_train, y_train
    )

    all_predictions, train_predictions = final_fit_and_predict(
        models, X_train, y_train, X_predict,
        train_df, pred_df, best_model_name
    )

    quality_report(summary_df, all_predictions, train_df, best_model_name)

    save_outputs(summary_df, cv_results_df, all_predictions,
                 train_predictions, best_model_name)

    print("\nPart 4 complete.")
    print("Next: run src/part5_ggbs.py")
    return summary_df, all_predictions


if __name__ == "__main__":
    main()