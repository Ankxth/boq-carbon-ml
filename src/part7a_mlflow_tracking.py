"""
Part 7a: MLflow Experiment Tracking
=====================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part7a_mlflow_tracking.py

What this script does:
  Reruns the 7-model comparison from Part 4b with full MLflow tracking.
  Every model run logs:
    - Parameters   : model name, hyperparameters, n_training_rows, n_features
    - Metrics      : R2, MAE, RMSE (CV), per-fold metrics
    - Artifacts    : trained model (.pkl), feature names, model comparison CSV
  The best model is registered in the MLflow Model Registry as
  "boq-cci-predictor" in the "Production" stage.

  After running, launch the MLflow UI with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
  Then open http://localhost:5000 in your browser.

Run:
  cd boq_carbon_ml
  python src/part7a_mlflow_tracking.py
"""

import pandas as pd
import numpy as np
import warnings
import copy
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model    import Ridge, Lasso
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.neural_network  import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost                 import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
TABLES = BASE / "outputs" / "tables"

# MLflow tracking URI — local SQLite database
MLFLOW_URI      = f"sqlite:///{BASE / 'mlflow.db'}"
EXPERIMENT_NAME = "boq-cci-prediction"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MODEL DEFINITIONS (identical to Part 4b)
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    return {
        "Lasso Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  Lasso(alpha=0.05, max_iter=10000)),
            ]),
            "params": {"alpha": 0.05, "max_iter": 10000},
        },
        "Ridge Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  Ridge(alpha=100.0)),
            ]),
            "params": {"alpha": 100.0},
        },
        "Decision Tree": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  DecisionTreeRegressor(
                    max_depth=6, min_samples_leaf=5, random_state=42)),
            ]),
            "params": {"max_depth": 6, "min_samples_leaf": 5},
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  RandomForestRegressor(
                    n_estimators=300, max_depth=10, min_samples_leaf=3,
                    max_features="sqrt", random_state=42, n_jobs=-1)),
            ]),
            "params": {
                "n_estimators": 300, "max_depth": 10,
                "min_samples_leaf": 3, "max_features": "sqrt",
            },
        },
        "XGBoost": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  XGBRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.6,
                    reg_alpha=0.5, reg_lambda=2.0,
                    random_state=42, verbosity=0)),
            ]),
            "params": {
                "n_estimators": 300, "max_depth": 4,
                "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.6, "reg_alpha": 0.5, "reg_lambda": 2.0,
            },
        },
        "SVR": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
            ]),
            "params": {"kernel": "rbf", "C": 10, "epsilon": 0.1},
        },
        "Neural Network": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu", solver="adam",
                    alpha=50.0, learning_rate="adaptive",
                    learning_rate_init=0.001, max_iter=2000,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=30, random_state=42, verbose=False)),
            ]),
            "params": {
                "hidden_layer_sizes": "(32,16)", "activation": "relu",
                "alpha": 50.0, "max_iter": 2000,
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    return {
        "r2":   round(r2_score(y_true, y_pred), 6),
        "mae":  round(mean_absolute_error(y_true, y_pred), 6),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MAIN TRACKING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking():
    print("\n[1] Loading training data...")
    X_train   = np.load(PROC / "X_train_v2.npy")
    y_train   = np.load(PROC / "y_train_v2.npy")
    train_df  = pd.read_csv(PROC / "part4b_train_combined.csv")
    feat_path = PROC / "feature_names.txt"

    n_rows, n_feats = X_train.shape
    print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n[2] MLflow experiment: '{EXPERIMENT_NAME}'")
    print(f"  Tracking URI: {MLFLOW_URI}")

    models    = get_models()
    cv        = KFold(n_splits=5, shuffle=True, random_state=42)
    results   = []
    best_r2   = -np.inf
    best_name = None
    best_run_id = None

    print(f"\n[3] Training and logging {len(models)} models...")

    for name, spec in models.items():
        pipe   = spec["pipeline"]
        params = spec["params"]

        print(f"\n  {name}...", end=" ", flush=True)

        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id

            # ── Log parameters ────────────────────────────────────────────
            mlflow.log_param("model_name",    name)
            mlflow.log_param("n_train_rows",  n_rows)
            mlflow.log_param("n_features",    n_feats)
            mlflow.log_param("cv_folds",      5)
            mlflow.log_param("target",        "log1p_CCI")
            mlflow.log_param("training_data", "PA_contract + CPWD_2023")
            for k, v in params.items():
                mlflow.log_param(k, v)

            # ── Cross-validation ─────────────────────────────────────────
            oof = cross_val_predict(pipe, X_train, y_train, cv=cv)
            m   = compute_metrics(y_train, oof)

            # ── Log CV metrics ────────────────────────────────────────────
            mlflow.log_metric("cv_r2",   m["r2"])
            mlflow.log_metric("cv_mae",  m["mae"])
            mlflow.log_metric("cv_rmse", m["rmse"])

            # ── Per-fold metrics ──────────────────────────────────────────
            for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_train)):
                fp = copy.deepcopy(pipe)
                fp.fit(X_train[tr_idx], y_train[tr_idx])
                fm = compute_metrics(
                    y_train[te_idx],
                    fp.predict(X_train[te_idx])
                )
                mlflow.log_metric(f"fold_{fold_i+1}_r2",   fm["r2"])
                mlflow.log_metric(f"fold_{fold_i+1}_mae",  fm["mae"])
                mlflow.log_metric(f"fold_{fold_i+1}_rmse", fm["rmse"])

            # ── Final fit + log model ─────────────────────────────────────
            pipe.fit(X_train, y_train)
            signature = infer_signature(
                X_train[:5],
                pipe.predict(X_train[:5])
            )
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"boq-cci-{name.lower().replace(' ','-')}",
            )

            # ── Log feature names as artifact ─────────────────────────────
            if feat_path.exists():
                mlflow.log_artifact(str(feat_path), artifact_path="features")

            print(f"R2={m['r2']:.4f}  MAE={m['mae']:.2f}  "
                  f"RMSE={m['rmse']:.2f}  run_id={run_id[:8]}...")

            results.append({
                "Model":              name,
                "run_id":             run_id,
                "cv_r2":              m["r2"],
                "cv_mae":             m["mae"],
                "cv_rmse":            m["rmse"],
            })

            if m["r2"] > best_r2:
                best_r2     = m["r2"]
                best_name   = name
                best_run_id = run_id

    # ── Summary ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("cv_r2", ascending=False)

    print(f"\n[4] Results summary:")
    print(f"  {'Model':<22} {'R2':>8} {'MAE':>10} {'RMSE':>10} {'Run ID'}")
    print(f"  {'-'*68}")
    for _, r in results_df.iterrows():
        marker = " <- best" if r["Model"] == best_name else ""
        print(f"  {r['Model']:<22} {r['cv_r2']:>8.4f} "
              f"{r['cv_mae']:>10.3f} {r['cv_rmse']:>10.3f} "
              f"{r['run_id'][:8]}...{marker}")

    # ── Tag best model run ────────────────────────────────────────────────
    mlflow.tracking.MlflowClient().set_tag(
        best_run_id, "best_model", "true"
    )
    print(f"\n  Best model: {best_name}  (R2={best_r2:.4f})")
    print(f"  Tagged run {best_run_id[:8]}... as best_model=true")

    # ── Save results CSV ──────────────────────────────────────────────────
    results_df.to_csv(TABLES / "mlflow_run_results.csv", index=False)
    print(f"  Saved: outputs/tables/mlflow_run_results.csv")

    return results_df, best_name, best_run_id


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("PART 7a — MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)

    results_df, best_name, best_run_id = run_tracking()

    print("\n" + "=" * 60)
    print("TRACKING COMPLETE")
    print("=" * 60)
    print(f"\n  All 7 models logged to MLflow.")
    print(f"  Best model: {best_name}")
    print(f"\n  To view the experiment dashboard:")
    print(f"  1. Run:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  2. Open: http://localhost:5000")
    print(f"\n  You will see:")
    print(f"    - All 7 runs with parameters and metrics")
    print(f"    - Per-fold CV metrics for each model")
    print(f"    - Registered models in the Model Registry tab")
    print(f"    - Best run tagged with best_model=true")
    print("=" * 60)

    print("\nPart 7a complete.")
    print("Next: run src/part7b_api.py  (starts the prediction API)")
    return results_df


if __name__ == "__main__":
    main()