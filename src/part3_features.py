"""
Part 3: Feature Engineering and ML Dataset Preparation
========================================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part3_features.py

What this script does:
  1. Loads part2_ef_mapped.csv
  2. Audits and cleans the CCI training set:
       - Removes misclassified rows (stud bolts as concrete, excavation as concrete)
       - Removes differential/additional-rate rows (not real unit prices)
       - Removes rows with non-standard units for their material (m for concrete)
       - Removes outliers (already flagged in Part 2)
  3. Builds features for ML:
       - TF-IDF on description_clean (text features)
       - One-hot encoding of unit_clean
       - One-hot encoding of material
       - Grade encoded as numeric strength (M20=20, M30=30 etc.)
       - log(quantity) as numeric feature
  4. Defines the target variable: log(CCI) — log-transforms CCI so
       the distribution is approximately normal (skewness 6.95 → 0.49)
       Models predict log(CCI); results are exponentiated back for reporting
  5. Saves:
       data/processed/part3_ml_ready.csv    — full feature matrix
       data/processed/X_train.npy           — feature matrix (training)
       data/processed/y_train.npy           — target vector  (training)
       data/processed/X_predict.npy         — features for prediction
       data/processed/feature_names.txt     — ordered feature names
       outputs/tables/training_audit.csv    — rows kept/removed and why

Run:
  cd boq_carbon_ml
  python src/part3_features.py
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
TABLES = BASE / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1] Loading Part 2 output...")
    df = pd.read_csv(PROC / "part2_ef_mapped.csv")
    print(f"  Total rows: {len(df):,}")
    print(f"  CCI training rows (has_cci): {(df['cci_label']=='has_cci').sum()}")
    print(f"  EF-known rows (predict_cci): {(df['cci_label']=='predict_cci').sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TRAINING SET AUDIT AND CLEANING
#
# The CCI training set (PA rows with rate + EF) contains several rows that
# should NOT be used for ML training because their CCI is not a genuine
# unit-rate-to-EF ratio. We identify and remove them here, keeping a record.
#
# Removal rules (applied in order, each row removed at most once):
#
#  R1: Outlier flag (from Part 2 IQR detection)
#      These are extreme CCI values — structural steel, mobilisation charges,
#      design fees, door sets etc.
#
#  R2: Non-standard unit for material
#      Concrete measured in 'm' or 'nos' — these are differential/incidental
#      rates (e.g. "Installation INR 21/m" for M35 concrete), not volumetric
#      unit rates. CCI is meaningless for them.
#
#  R3: Additional/differential rate rows
#      Rows where the description contains "additional rate", "over and above",
#      "differential" — these are price adjustment items, not standalone rates.
#
#  R4: Clearly misclassified rows
#      Stud bolt rows classified as concrete, cable rows classified as concrete,
#      excavation-depth rows (depth-based rates, not material rates).
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that identify rows to remove from training
REMOVE_PATTERNS = {
    "R2_bad_unit": {
        # concrete with unit != m3 or m2
        "rule": lambda r: (
            r["material"] == "concrete" and
            r["unit_clean"] not in ("m3", "m2")
        ),
        "reason": "Concrete with non-volumetric unit (m/nos) — differential rate"
    },
    "R3_additional_rate": {
        "rule": lambda r: any(kw in str(r["description"]).lower() for kw in [
            "additional rate", "over and above", "differential",
            "extra rate", "rate payable", "grade changed",
            "variation in cement", "cementetious content",
        ]),
        "reason": "Differential/additional rate item — not a standalone unit rate"
    },
    "R4_misclassified": {
        "rule": lambda r: any(kw in str(r["description"]).lower() for kw in [
            "stud bolt", "shear connector", "m16 stud", "m19 stud",
            "flexible cable", "sqmm", "gypsum cornice",
            "rope suspended platform", "mobilization", "mobilisation",
            "design charge", "chain link fencing", "grc jaalis",
            "micro pile installation",
        ]),
        "reason": "Misclassified or non-material-rate item"
    },
}


def audit_and_clean_training(df: pd.DataFrame):
    """
    Splits df into:
      training_clean  — rows to use for ML (CCI reliable)
      predict_set     — rows with EF but no rate (CCI to predict)
      audit_log       — every training row with keep/remove decision
    """
    print("\n[2] Auditing and cleaning CCI training set...")

    training_raw = df[df["cci_label"] == "has_cci"].copy()
    predict_set  = df[df["cci_label"] == "predict_cci"].copy()

    audit_rows = []
    keep_mask  = pd.Series(True, index=training_raw.index)

    # R1: outlier flag from Part 2
    outlier_mask = training_raw["cci_outlier"] == True
    keep_mask[outlier_mask] = False
    for idx in training_raw[outlier_mask].index:
        audit_rows.append({
            "index": idx,
            "source": training_raw.at[idx, "source"],
            "description": str(training_raw.at[idx, "description"])[:80],
            "material": training_raw.at[idx, "material"],
            "unit": training_raw.at[idx, "unit_clean"],
            "rate": training_raw.at[idx, "rate"],
            "cci": training_raw.at[idx, "cci"],
            "decision": "REMOVE",
            "reason": "R1_outlier (IQR flag from Part 2)",
        })

    # R2, R3, R4: pattern-based removal
    for rule_name, rule_def in REMOVE_PATTERNS.items():
        for idx, row in training_raw[keep_mask].iterrows():
            if rule_def["rule"](row):
                keep_mask[idx] = False
                audit_rows.append({
                    "index": idx,
                    "source": row["source"],
                    "description": str(row["description"])[:80],
                    "material": row["material"],
                    "unit": row["unit_clean"],
                    "rate": row["rate"],
                    "cci": row["cci"],
                    "decision": "REMOVE",
                    "reason": f"{rule_name}: {rule_def['reason']}",
                })

    # Mark kept rows in audit
    for idx, row in training_raw[keep_mask].iterrows():
        audit_rows.append({
            "index": idx,
            "source": row["source"],
            "description": str(row["description"])[:80],
            "material": row["material"],
            "unit": row["unit_clean"],
            "rate": row["rate"],
            "cci": row["cci"],
            "decision": "KEEP",
            "reason": "Valid unit rate with reliable CCI",
        })

    training_clean = training_raw[keep_mask].copy()
    audit_df       = pd.DataFrame(audit_rows)

    n_removed = (~keep_mask).sum()
    n_kept    = keep_mask.sum()

    print(f"  Raw training rows  : {len(training_raw)}")
    print(f"  Removed            : {n_removed}")
    print(f"  Clean training rows: {n_kept}")
    print(f"\n  Removal breakdown:")
    removed = audit_df[audit_df["decision"] == "REMOVE"]
    for reason, grp in removed.groupby("reason"):
        print(f"    {len(grp):3d} rows — {reason}")

    print(f"\n  Clean training — CCI stats (INR/kgCO2e):")
    print(f"    min    : {training_clean['cci'].min():.3f}")
    print(f"    median : {training_clean['cci'].median():.3f}")
    print(f"    mean   : {training_clean['cci'].mean():.3f}")
    print(f"    max    : {training_clean['cci'].max():.3f}")
    print(f"    skew   : {training_clean['cci'].skew():.3f}")

    log_cci = np.log1p(training_clean["cci"])
    print(f"\n  log(CCI) stats (ML target):")
    print(f"    min    : {log_cci.min():.3f}")
    print(f"    median : {log_cci.median():.3f}")
    print(f"    mean   : {log_cci.mean():.3f}")
    print(f"    max    : {log_cci.max():.3f}")
    print(f"    skew   : {log_cci.skew():.3f}")

    print(f"\n  Material distribution (clean training):")
    print(f"  {training_clean['material'].value_counts().to_dict()}")

    # Save audit log
    audit_path = TABLES / "training_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"\n  Audit log saved → {audit_path}")

    return training_clean.reset_index(drop=True), predict_set.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE ENGINEERING
#
# Features built for every row (training + prediction):
#
#  Text features (TF-IDF on description_clean):
#    - Captures material-specific vocabulary
#    - max_features=500, unigrams + bigrams
#    - Fitted ONLY on training set, applied to both
#    - Why 500: enough to capture key terms without overfitting on 229 rows
#
#  Categorical features (one-hot encoded):
#    - unit_clean  : m3, m2, m, nos, ton, kg, ls, day, bags → 9 columns
#    - material    : concrete, paint, plaster, etc.         → N columns
#
#  Grade feature (numeric):
#    - Concrete grades M5–M50 → 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
#    - Non-concrete: 0
#    - Captures the monotonic relationship: higher grade → higher EF → affects CCI
#
#  Numeric features:
#    - log1p(quantity) — quantity varies over many orders of magnitude
#    - emission_factor — direct numeric signal for CCI
#
# Feature matrix is dense (numpy array) for sklearn compatibility.
# TF-IDF sparse matrix is converted to dense — 229 × 500 is manageable (~900KB).
# ─────────────────────────────────────────────────────────────────────────────

# All units seen across training + predict set
ALL_UNITS = ["m3", "m2", "m", "nos", "ton", "kg", "ls", "day",
             "bags", "no.s", "unknown", "sqft"]

# All materials with EF (those that appear in training or predict)
ALL_MATERIALS = [
    "concrete", "paint", "plaster", "steel", "brick", "mortar",
    "tile", "aluminium", "insulation", "timber", "glass",
    "aggregate", "sand", "masonry",
]

GRADE_STRENGTH = {
    "M5": 5, "M10": 10, "M15": 15, "M20": 20, "M25": 25,
    "M30": 30, "M35": 35, "M40": 40, "M45": 45, "M50": 50,
    "Fe250": 0, "Fe415": 0, "Fe500": 0,   # steel grades → 0
}


def build_features(
    training: pd.DataFrame,
    predict:  pd.DataFrame,
    n_tfidf:  int = 500,
):
    """
    Returns:
      X_train       : numpy array (n_train, n_features)
      y_train       : numpy array (n_train,)  — log1p(CCI)
      X_predict     : numpy array (n_predict, n_features)
      feature_names : list of strings
      tfidf         : fitted TfidfVectorizer (saved for later use)
    """
    print(f"\n[3] Building features...")
    print(f"  Training rows : {len(training)}")
    print(f"  Predict rows  : {len(predict)}")

    all_rows = pd.concat([training, predict], ignore_index=True)
    n_train  = len(training)

    # ── 3a. TF-IDF on description_clean ──────────────────────────────────────
    print(f"\n  3a. TF-IDF (max_features={n_tfidf}, 1+2-grams)...")
    tfidf = TfidfVectorizer(
        max_features=n_tfidf,
        ngram_range=(1, 2),
        min_df=1,              # include rare terms — small dataset
        sublinear_tf=True,     # log(tf) scaling: reduces impact of very frequent terms
        strip_accents="unicode",
        analyzer="word",
    )
    # Fit ONLY on training descriptions
    tfidf.fit(training["description_clean"].fillna(""))
    # Transform all rows
    tfidf_all = tfidf.transform(all_rows["description_clean"].fillna(""))
    tfidf_dense = tfidf_all.toarray()
    tfidf_names = [f"tfidf_{f}" for f in tfidf.get_feature_names_out()]
    print(f"  TF-IDF matrix: {tfidf_dense.shape}")

    # ── 3b. One-hot: unit_clean ───────────────────────────────────────────────
    print(f"  3b. One-hot encoding: unit_clean...")
    unit_ohe = pd.get_dummies(
        all_rows["unit_clean"].fillna("unknown"),
        prefix="unit"
    )
    # Ensure all expected unit columns are present
    for u in ALL_UNITS:
        col = f"unit_{u}"
        if col not in unit_ohe.columns:
            unit_ohe[col] = 0
    unit_cols   = sorted([c for c in unit_ohe.columns if c.startswith("unit_")])
    unit_matrix = unit_ohe[unit_cols].values
    print(f"  Unit matrix  : {unit_matrix.shape}")

    # ── 3c. One-hot: material ─────────────────────────────────────────────────
    print(f"  3c. One-hot encoding: material...")
    mat_ohe = pd.get_dummies(
        all_rows["material"].fillna("unknown"),
        prefix="mat"
    )
    for m in ALL_MATERIALS:
        col = f"mat_{m}"
        if col not in mat_ohe.columns:
            mat_ohe[col] = 0
    mat_cols   = sorted([c for c in mat_ohe.columns if c.startswith("mat_")])
    mat_matrix = mat_ohe[mat_cols].values
    print(f"  Material matrix: {mat_matrix.shape}")

    # ── 3d. Grade as numeric strength ─────────────────────────────────────────
    print(f"  3d. Grade → numeric strength...")
    grade_vals = all_rows["grade"].apply(
        lambda g: float(GRADE_STRENGTH.get(str(g), 0)) if pd.notna(g) else 0.0
    ).values.reshape(-1, 1)

    # ── 3e. Numeric: log(quantity) and emission_factor ────────────────────────
    print(f"  3e. Numeric features: log(quantity), emission_factor...")
    log_qty = np.log1p(
        pd.to_numeric(all_rows["quantity"], errors="coerce").fillna(0)
    ).values.reshape(-1, 1)

    ef_vals = pd.to_numeric(all_rows["emission_factor"], errors="coerce").fillna(0)\
                .values.reshape(-1, 1)

    # ── 3f. Stack all features ────────────────────────────────────────────────
    X_all = np.hstack([
        tfidf_dense,   # 500 columns
        unit_matrix,   # ~12 columns
        mat_matrix,    # ~14 columns
        grade_vals,    # 1 column
        log_qty,       # 1 column
        ef_vals,       # 1 column
    ])

    feature_names = (
        tfidf_names +
        unit_cols   +
        mat_cols    +
        ["grade_strength", "log_quantity", "emission_factor"]
    )

    print(f"\n  Final feature matrix: {X_all.shape}")
    print(f"  Feature groups:")
    print(f"    TF-IDF features   : {len(tfidf_names)}")
    print(f"    Unit features     : {len(unit_cols)}")
    print(f"    Material features : {len(mat_cols)}")
    print(f"    Numeric features  : 3 (grade_strength, log_quantity, emission_factor)")

    # ── Split back into train / predict ───────────────────────────────────────
    X_train   = X_all[:n_train]
    X_predict = X_all[n_train:]

    # ── Target variable: log1p(CCI) ───────────────────────────────────────────
    y_train = np.log1p(training["cci"].values)
    print(f"\n  Target variable: log1p(CCI)")
    print(f"    shape  : {y_train.shape}")
    print(f"    min    : {y_train.min():.3f}")
    print(f"    median : {np.median(y_train):.3f}")
    print(f"    max    : {y_train.max():.3f}")
    print(f"    skew   : {pd.Series(y_train).skew():.3f}")

    X_train   = X_train.astype(np.float64)
    X_predict = X_predict.astype(np.float64)
    y_train   = y_train.astype(np.float64)
    return X_train, y_train, X_predict, feature_names, tfidf

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    training:      pd.DataFrame,
    predict:       pd.DataFrame,
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_predict:     np.ndarray,
    feature_names: list,
    df_full:       pd.DataFrame,
):
    print("\n[4] Saving outputs...")

    # Feature matrices
    np.save(PROC / "X_train.npy",   X_train)
    np.save(PROC / "y_train.npy",   y_train)
    np.save(PROC / "X_predict.npy", X_predict)
    print(f"  X_train   : {X_train.shape}  → data/processed/X_train.npy")
    print(f"  y_train   : {y_train.shape}  → data/processed/y_train.npy")
    print(f"  X_predict : {X_predict.shape} → data/processed/X_predict.npy")

    # Feature names
    feat_path = PROC / "feature_names.txt"
    feat_path.write_text("\n".join(feature_names))
    print(f"  Feature names ({len(feature_names)}) → data/processed/feature_names.txt")

    # ML-ready CSV with all context kept
    ml_cols = [
        "source", "description", "description_clean",
        "unit_clean", "quantity", "rate", "material", "grade",
        "emission_factor", "ef_unit", "cci", "cci_label",
    ]

    # Training set (clean)
    train_out = training.copy()
    train_out["log_cci"] = np.log1p(training["cci"])
    train_out["split"]   = "train"
    train_out[ml_cols + ["log_cci", "split"]].to_csv(
        PROC / "part3_train.csv", index=False
    )
    print(f"  Training CSV ({len(train_out)} rows) → data/processed/part3_train.csv")

    # Predict set
    pred_out = predict.copy()
    pred_out["log_cci"] = np.nan
    pred_out["split"]   = "predict"
    pred_out[ml_cols + ["log_cci", "split"]].to_csv(
        PROC / "part3_predict.csv", index=False
    )
    print(f"  Predict CSV ({len(pred_out)} rows) → data/processed/part3_predict.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(training, predict, X_train, y_train, X_predict):
    print("\n" + "=" * 60)
    print("PART 3 QUALITY REPORT")
    print("=" * 60)

    print(f"  Training rows (clean) : {len(training)}")
    print(f"  Predict rows          : {len(predict)}")
    print(f"  Feature dimensions    : {X_train.shape[1]}")
    print(f"  X_train shape         : {X_train.shape}")
    print(f"  X_predict shape       : {X_predict.shape}")
    print(f"  y_train (log CCI)     : min={y_train.min():.3f}  max={y_train.max():.3f}")

    print(f"\n  Training set by source:")
    for src, cnt in training["source"].value_counts().items():
        print(f"    {src}: {cnt} rows")

    print(f"\n  Training set by material:")
    for mat, cnt in training["material"].value_counts().items():
        print(f"    {mat:15s}: {cnt:3d} rows")

    print(f"\n  Predict set by source:")
    for src, cnt in predict["source"].value_counts().items():
        print(f"    {src}: {cnt} rows")

    print(f"\n  Assertions:")

    assert X_train.shape[0] == len(training), "X_train row mismatch"
    print(f"  OK  X_train rows match training set ({X_train.shape[0]})")

    assert X_predict.shape[0] == len(predict), "X_predict row mismatch"
    print(f"  OK  X_predict rows match predict set ({X_predict.shape[0]})")

    assert X_train.shape[1] == X_predict.shape[1], "Feature column mismatch"
    print(f"  OK  Same feature columns in train and predict ({X_train.shape[1]})")
    
    assert not np.any(np.isnan(X_train.astype(float))), "NaN in X_train"
    print(f"  OK  No NaN in X_train")

    assert not np.any(np.isnan(X_predict.astype(float))), "NaN in X_predict"
    print(f"  OK  No NaN in X_predict")

    assert not np.any(np.isnan(y_train.astype(float))), "NaN in y_train"
    print("  OK  No NaN in y_train")

    assert len(training) >= 150, f"Training set too small: {len(training)}"
    print(f"  OK  Training set >= 150 rows ({len(training)})")

    skew = pd.Series(y_train).skew()
    assert abs(skew) < 2.0, f"Target still heavily skewed: {skew:.2f}"
    print(f"  OK  log(CCI) skewness acceptable ({skew:.3f})")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("PART 3 — FEATURE ENGINEERING")
    print("=" * 60)

    df = load_data()

    training_clean, predict_set = audit_and_clean_training(df)

    X_train, y_train, X_predict, feature_names, tfidf = build_features(
        training_clean, predict_set, n_tfidf=500
    )

    quality_report(training_clean, predict_set, X_train, y_train, X_predict)

    save_outputs(
        training_clean, predict_set,
        X_train, y_train, X_predict,
        feature_names, df,
    )

    print("\nPart 3 complete.")
    print("Next: run src/part4_models.py")
    return X_train, y_train, X_predict, feature_names


if __name__ == "__main__":
    main()