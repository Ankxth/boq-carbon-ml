"""
Part 2: ICE DB Emission Factor Mapping + Carbon Cost Intensity
================================================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part2_emission_factors.py

What this script does:
  1. Loads master_cleaned.csv from Part 1
  2. Assigns ICE DB V4.1 emission factors to every identifiable material
  3. For concrete m2 rows: resolves thickness and converts EF to kgCO2e/m2
  4. For PA rows: computes Carbon Cost Intensity (CCI) = rate / EF
  5. Labels every row with its status for ML:
       'has_cci'     → PA row with rate + EF → CCI computed  (ML training set)
       'known_ef'    → has EF but no rate    → EF known, CCI to predict
       'no_ice_ef'   → material known, no ICE EF entry (formwork, waterproofing…)
       'unknown_mat' → material=UNKNOWN (7 rows only)
  6. Saves:
       data/processed/part2_ef_mapped.csv   — full dataset with EF + CCI
       outputs/tables/cci_training_set.csv  — PA rows with CCI (ML training)
       outputs/tables/ef_summary.csv        — per-material EF stats

Run:
  cd boq_carbon_ml
  python src/part2_emission_factors.py
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
TABLES = BASE / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — ICE DB V4.1 EMISSION FACTORS
#
# Concrete: kgCO2e / m3  (CEM I, density 2400 kg/m3, ICE Cement Model V1.3)
# Others  : kgCO2e / kg  (ICE DB Advanced V4.1, Oct 2025)
#
# Materials with NO ICE EF entry:
#   waterproofing — product-specific (proprietary membranes, no generic ICE value)
#   formwork      — process/activity item, not a bulk material
#   flooring      — composite installed system (tile + mortar + levelling)
#   pipe          — varies by material (UPVC, GI, HDPE have very different EFs)
#   sealant       — proprietary product
#   grouting      — varies by type
#   excavation    — activity, not material
#   anti_termite  — chemical treatment, negligible material carbon
# ─────────────────────────────────────────────────────────────────────────────

CONCRETE_EF = {
    "M5":  150.6,
    "M10": 232.8,
    "M15": 268.1,
    "M20": 284.9,
    "M25": 301.8,
    "M30": 331.2,
    "M35": 356.3,
    "M40": 381.5,
    "M45": 400.0,
    "M50": 420.0,
}

# (EF value, EF unit, ICE DB source reference)
MATERIAL_EF = {
    "steel":      (1.720,    "kgCO2e/kg", "ICE V4.1 — Steel, Rebar (Virgin)"),
    "brick":      (0.213,    "kgCO2e/kg", "ICE V4.1 — Clay Bricks, General"),
    "masonry":    (0.213,    "kgCO2e/kg", "ICE V4.1 — Clay Bricks (proxy for block masonry)"),
    "aggregate":  (0.00747,  "kgCO2e/kg", "ICE V4.1 — Aggregates, General UK Mix"),
    "sand":       (0.00493,  "kgCO2e/kg", "ICE V4.1 — Aggregates & Sand, Virgin Mix"),
    "glass":      (1.437,    "kgCO2e/kg", "ICE V4.1 — Glass, General"),
    "timber":     (0.493,    "kgCO2e/kg", "ICE V4.1 — Timber, Average All Data"),
    "paint":      (2.152,    "kgCO2e/kg", "ICE V4.1 — Paint, Waterborne"),
    "tile":       (0.796,    "kgCO2e/kg", "ICE V4.1 — Ceramic Tile"),
    "plaster":    (0.238,    "kgCO2e/kg", "ICE V4.1 — Plaster, Plasterboard"),
    "aluminium":  (6.669,    "kgCO2e/kg", "ICE V4.1 — Aluminium, European Mix"),
    "mortar":     (0.208,    "kgCO2e/kg", "ICE V4.1 — Mortar, General"),
    "insulation": (1.860,    "kgCO2e/kg", "ICE V4.1 — Mineral Wool, General"),
}

# Materials that have no ICE EF and will be labelled 'no_ice_ef'
NO_ICE_EF_MATERIALS = {
    "waterproofing", "formwork", "flooring", "pipe",
    "sealant", "grouting", "excavation", "anti_termite",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — THICKNESS DEFAULTS FOR m2 CONCRETE ROWS
#
# When a concrete row is measured in m2 (area) rather than m3 (volume),
# we need thickness to convert EF from kgCO2e/m3 → kgCO2e/m2.
#
# Thickness hierarchy:
#   1. Extracted from description text (done in Part 1, stored in thickness_mm)
#   2. Inferred from description keywords (below)
#   3. Default fallback (75mm — typical screed/protective layer)
#
# All assumed thicknesses are flagged with thickness_assumed=True
# so uncertainty is transparent in the paper.
# ─────────────────────────────────────────────────────────────────────────────

THICKNESS_KEYWORDS = [
    # (keyword_in_description, thickness_mm)
    # Order: more specific first
    (r'raft\s+foundation|raft\s+slab',          300),
    (r'grade\s+slab|ground\s+slab',             150),
    (r'flat\s+slab|transfer\s+slab',            200),
    (r'roof\s+slab|terrace\s+slab',             150),
    (r'floor\s+slab|\bslab\b(?!.*screed)',       125),
    (r'screed.*75|75.*screed',                   75),
    (r'screed.*50|50.*screed',                   50),
    (r'screed.*100|100.*screed',                100),
    (r'\bscreed\b',                              75),  # default screed
    (r'protective.*layer|protective.*concrete',  50),
    (r'dpc\b',                                   50),
    (r'retaining\s+wall',                       200),
    (r'\bwall\b',                               150),
    (r'below\s+grade\s+slab',                   150),
]

DEFAULT_THICKNESS_MM = 75.0   # absolute fallback


def resolve_thickness(row) -> tuple:
    """
    Returns (thickness_mm, was_assumed).
    was_assumed = False if thickness came from Part 1 text extraction.
    was_assumed = True if we are inferring or using a default.
    """
    # Already extracted in Part 1
    if pd.notna(row["thickness_mm"]):
        return float(row["thickness_mm"]), False

    # Try to infer from keywords in description
    desc = str(row["description"]).lower()
    for pat, thk in THICKNESS_KEYWORDS:
        if re.search(pat, desc):
            return float(thk), True

    # Absolute default
    return DEFAULT_THICKNESS_MM, True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EF ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_emission_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      emission_factor   — numeric EF value (in EF_unit per unit_clean)
      ef_unit           — unit of the EF (kgCO2e/m3 or kgCO2e/kg or kgCO2e/m2)
      ef_source         — ICE DB reference string
      thickness_mm      — resolved thickness (may override Part 1 value)
      thickness_assumed — True if thickness was inferred/defaulted, not extracted
      ef_label          — 'has_ef' / 'no_ice_ef' / 'unknown_mat'
    """
    print("\n[2] Assigning ICE DB emission factors...")

    efs, ef_units, ef_sources = [], [], []
    thick_resolved, thick_assumed = [], []
    labels = []

    for _, row in df.iterrows():
        mat   = str(row["material"]).lower()
        grade = str(row["grade"]) if pd.notna(row.get("grade")) else None
        unit  = str(row["unit_clean"])

        ef = ef_u = ef_s = None
        thk = row.get("thickness_mm")
        thk_assumed = False

        # ── Concrete ──────────────────────────────────────────────────────────
        if mat == "concrete":
            ef_m3 = CONCRETE_EF.get(grade) if grade else None

            if ef_m3 is not None:
                if unit == "m3":
                    ef   = ef_m3
                    ef_u = "kgCO2e/m3"
                    ef_s = f"ICE V4.1 Concrete CEM I {grade}"

                elif unit == "m2":
                    # Need thickness to convert m3 EF → m2 EF
                    thk, thk_assumed = resolve_thickness(row)
                    ef   = round(ef_m3 * (thk / 1000.0), 4)   # mm → m
                    ef_u = "kgCO2e/m2"
                    ef_s = f"ICE V4.1 Concrete CEM I {grade} × {thk:.0f}mm"

                else:
                    # nos, kg, m — concrete measured in non-standard units
                    # Assign m3 EF and note the unit mismatch; flag for review
                    ef   = ef_m3
                    ef_u = "kgCO2e/m3"
                    ef_s = f"ICE V4.1 Concrete CEM I {grade} [unit={unit}, review]"

        # ── Non-concrete with ICE EF ──────────────────────────────────────────
        # ── Non-concrete with ICE EF ──────────────────────────────────────────
        elif mat in MATERIAL_EF:
            ef_kg, ef_u_kg, ef_s = MATERIAL_EF[mat]
            # Convert EF from per-kg to per-installed-unit using unit context
            DENSITY  = {'brick':1800, 'masonry':1800, 'mortar':2000}
            COVERAGE = {'plaster':20.0, 'paint':0.3, 'tile':22.0,
                        'insulation':1.5, 'glass':25.0, 'aluminium':0.5}
            u = str(row["unit_clean"])
            if u == 'kg':
                ef, ef_u = ef_kg, ef_u_kg
            elif u == 'ton':
                ef, ef_u = ef_kg * 1000, "kgCO2e/ton"
            elif u == 'm3' and mat in DENSITY:
                ef, ef_u = ef_kg * DENSITY[mat], "kgCO2e/m3"
            elif u == 'm2' and mat in COVERAGE:
                ef, ef_u = ef_kg * COVERAGE[mat], "kgCO2e/m2"
            elif u == 'm2':
                # fallback for m2 materials without coverage — use per-kg and flag
                ef, ef_u = ef_kg, ef_u_kg + "_per_kg_review"
            elif u in ('m', 'rmt'):
                # linear metre items — use per-kg, flag for review
                ef, ef_u = ef_kg, ef_u_kg + "_per_kg_review"
            else:
                ef, ef_u = ef_kg, ef_u_kg

        # No EF for this material (process items, composite systems)
        # ef, ef_u, ef_s remain None

        efs.append(ef)
        ef_units.append(ef_u)
        ef_sources.append(ef_s)
        thick_resolved.append(thk if pd.notna(thk) else np.nan)
        thick_assumed.append(thk_assumed)

        # Label
        if ef is not None:
            labels.append("has_ef")
        elif mat in NO_ICE_EF_MATERIALS:
            labels.append("no_ice_ef")
        elif mat == "unknown":
            labels.append("unknown_mat")
        else:
            labels.append("no_ice_ef")

    df["emission_factor"]   = efs
    df["ef_unit"]           = ef_units
    df["ef_source"]         = ef_sources
    df["thickness_mm"]      = thick_resolved
    df["thickness_assumed"] = thick_assumed
    df["ef_label"]          = labels

    has_ef  = (df["ef_label"] == "has_ef").sum()
    no_ice  = (df["ef_label"] == "no_ice_ef").sum()
    unk_mat = (df["ef_label"] == "unknown_mat").sum()

    print(f"  has_ef      : {has_ef:4d} rows  ({has_ef/len(df)*100:.1f}%)")
    print(f"  no_ice_ef   : {no_ice:4d} rows  ({no_ice/len(df)*100:.1f}%)")
    print(f"  unknown_mat : {unk_mat:4d} rows")

    print(f"\n  m2 concrete rows    : {((df['material']=='concrete') & (df['unit_clean']=='m2')).sum()}")
    print(f"  thickness extracted : {((df['material']=='concrete') & (df['unit_clean']=='m2') & (~df['thickness_assumed'])).sum()}")
    print(f"  thickness assumed   : {((df['material']=='concrete') & (df['unit_clean']=='m2') & (df['thickness_assumed'])).sum()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CARBON COST INTENSITY (CCI)
#
# CCI = Rate (INR / unit) ÷ EF (kgCO2e / unit)
#      = INR per kgCO2e
#
# Only computable for PA rows where BOTH rate and EF are known.
# This is the ML regression target — a novel metric for Indian construction.
#
# Interpretation:
#   Low CCI  → cheap per unit of carbon (carbon-intensive, cost-efficient)
#   High CCI → expensive per unit of carbon (lower carbon intensity)
#
# For the paper: we report CCI in INR/kgCO2e and discuss which materials
# offer the most carbon reduction per rupee of cost premium.
# ─────────────────────────────────────────────────────────────────────────────

def compute_cci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      cci               — Carbon Cost Intensity (INR / kgCO2e), PA rows only
      cci_label         — 'has_cci' / 'predict_cci'
    """
    print("\n[3] Computing Carbon Cost Intensity (CCI)...")

    ccis       = []
    cci_labels = []

    for _, row in df.iterrows():
        rate = row.get("rate")
        ef   = row.get("emission_factor")
        src  = row["source"]

        cci = np.nan
        cci_lbl = "predict_cci"

        if (src == "PA"
                and pd.notna(rate) and rate > 0
                and pd.notna(ef)   and ef   > 0):
            cci     = round(float(rate) / float(ef), 4)
            cci_lbl = "has_cci"

        elif pd.notna(ef) and ef > 0:
            cci_lbl = "predict_cci"

        else:
            cci_lbl = "no_ef_no_cci"

        ccis.append(cci)
        cci_labels.append(cci_lbl)

    df["cci"]       = ccis
    df["cci_label"] = cci_labels

    has_cci  = df["cci"].notna().sum()
    pred_cci = (df["cci_label"] == "predict_cci").sum()

    print(f"  has_cci (ML training) : {has_cci:4d} rows")
    print(f"  predict_cci (ML target): {pred_cci:4d} rows")

    if has_cci > 0:
        cci_data = df[df["cci"].notna()]
        print(f"\n  CCI stats (INR/kgCO2e):")
        print(f"    min    : {cci_data['cci'].min():.3f}")
        print(f"    median : {cci_data['cci'].median():.3f}")
        print(f"    mean   : {cci_data['cci'].mean():.3f}")
        print(f"    max    : {cci_data['cci'].max():.3f}")
        print(f"    std    : {cci_data['cci'].std():.3f}")

        print(f"\n  CCI by material (PA rows):")
        pa_cci = cci_data.groupby("material")["cci"].agg(
            count="count", mean="mean", median="median",
            min="min", max="max"
        ).round(3)
        print(pa_cci.to_string())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — OUTLIER DETECTION ON CCI
#
# PA has some rate anomalies: very high rates for specialist items
# (thermo couplers = INR 290,030/nos; structural steel = INR 132,793/ton).
# These are real rates but outliers in the CCI distribution.
# We flag but do not remove them — the paper should discuss them.
# ─────────────────────────────────────────────────────────────────────────────

def flag_cci_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses IQR method to flag CCI outliers.
    Adds column: cci_outlier (True/False)
    """
    cci_data = df[df["cci"].notna()]["cci"]
    q1  = cci_data.quantile(0.25)
    q3  = cci_data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr   # 3×IQR — conservative, keeps more data
    upper = q3 + 3.0 * iqr

    df["cci_outlier"] = False
    mask = df["cci"].notna()
    df.loc[mask, "cci_outlier"] = (
        (df.loc[mask, "cci"] < lower) | (df.loc[mask, "cci"] > upper)
    )

    n_out = df["cci_outlier"].sum()
    print(f"\n[4] CCI outlier detection (3×IQR method):")
    print(f"  IQR range : {lower:.2f} – {upper:.2f} INR/kgCO2e")
    print(f"  Outliers  : {n_out} rows  (flagged, not removed)")
    if n_out > 0:
        out_rows = df[df["cci_outlier"]][["source","description","material","rate","cci"]]
        for _, r in out_rows.iterrows():
            print(f"    [{r['source']}] {str(r['description'])[:60]} "
                  f"rate={r['rate']:.0f} CCI={r['cci']:.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

FINAL_COLS = [
    "source",
    "description",
    "description_clean",
    "unit_clean",
    "quantity",
    "rate",
    "material",
    "grade",
    "thickness_mm",
    "thickness_assumed",
    "emission_factor",
    "ef_unit",
    "ef_source",
    "ef_label",
    "cci",
    "cci_label",
    "cci_outlier",
    "extr_method",
    "prop_grade",
]

CCI_TRAINING_COLS = [
    "source",
    "description",
    "description_clean",
    "unit_clean",
    "quantity",
    "rate",
    "material",
    "grade",
    "emission_factor",
    "ef_unit",
    "cci",
    "cci_outlier",
    "extr_method",
]


def save_outputs(df: pd.DataFrame):
    # Full dataset
    full_path = PROC / "part2_ef_mapped.csv"
    df[FINAL_COLS].to_csv(full_path, index=False)
    print(f"\n[5] Saved full dataset → {full_path}")

    # ML training set: PA rows with CCI
    training = df[df["cci_label"] == "has_cci"][CCI_TRAINING_COLS].copy()
    train_path = TABLES / "cci_training_set.csv"
    training.to_csv(train_path, index=False)
    print(f"  Saved CCI training set → {train_path}  ({len(training)} rows)")

    # EF summary table
    ef_summary = (
        df[df["ef_label"] == "has_ef"]
        .groupby(["material", "grade", "ef_unit"])["emission_factor"]
        .agg(count="count", ef_value="first")
        .reset_index()
        .sort_values(["material", "grade"])
    )
    summary_path = TABLES / "ef_summary.csv"
    ef_summary.to_csv(summary_path, index=False)
    print(f"  Saved EF summary → {summary_path}  ({len(ef_summary)} unique material-grade combos)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PART 2 QUALITY REPORT")
    print("=" * 60)

    total    = len(df)
    has_ef   = (df["ef_label"] == "has_ef").sum()
    has_cci  = df["cci"].notna().sum()
    outliers = df["cci_outlier"].sum()

    print(f"  Total rows            : {total:,}")
    print(f"  Rows with ICE EF      : {has_ef:,}  ({has_ef/total*100:.1f}%)")
    print(f"  CCI training rows     : {has_cci:,}")
    print(f"  CCI outliers flagged  : {outliers}")

    print(f"\n  EF label breakdown:")
    for lbl, cnt in df["ef_label"].value_counts().items():
        print(f"    {lbl:20s}: {cnt:4d}  ({cnt/total*100:.1f}%)")

    print(f"\n  CCI label breakdown:")
    for lbl, cnt in df["cci_label"].value_counts().items():
        print(f"    {lbl:20s}: {cnt:4d}")

    print(f"\n  Concrete EF check:")
    conc = df[df["material"] == "concrete"]
    for grade, grp in conc.groupby("grade"):
        ef_vals = grp["emission_factor"].dropna().unique()
        expected = {
            "M5":150.6,"M10":232.8,"M15":268.1,"M20":284.9,"M25":301.8,
            "M30":331.2,"M35":356.3,"M40":381.5,"M45":400.0,"M50":420.0,
        }.get(str(grade), "?")
        # For m2 rows EF will differ (includes thickness) — check m3 only
        m3_ef = grp[grp["unit_clean"] == "m3"]["emission_factor"].dropna().unique()
        if len(m3_ef) > 0:
            ok = "OK" if abs(m3_ef[0] - expected) < 0.01 else "MISMATCH"
            print(f"    {grade:5s}: m3 EF={m3_ef[0]:.1f} expected={expected} [{ok}]")

    print(f"\n  Assertions:")
    assert has_ef > 700, f"Expected >700 rows with EF, got {has_ef}"
    print(f"  OK  Rows with ICE EF > 700 ({has_ef})")
    assert has_cci >= 150, f"Expected >=150 CCI training rows, got {has_cci}"
    print(f"  OK  CCI training rows >= 150 ({has_cci})")
    assert df[df["cci"].notna()]["cci"].min() > 0, "Negative CCI found"
    print("  OK  All CCI values positive")
    assert df["emission_factor"].isna().sum() < total * 0.35, \
        "Too many missing EF values"
    print(f"  OK  EF coverage > 65% ({has_ef/total*100:.1f}%)")

    # Verify no conduit in steel, no tile in sand (carried from Part 1)
    steel_rows = df[df["material"] == "steel"]
    assert not steel_rows["description"].str.contains(
        r'conduit|frls', case=False, na=False).any(), \
        "Conduit still in steel rows"
    print("  OK  No conduit in steel")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("PART 2 — EMISSION FACTOR MAPPING + CCI COMPUTATION")
    print("=" * 60)

    # Load Part 1 output
    print("\n[1] Loading Part 1 output...")
    df = pd.read_csv(PROC / "master_cleaned.csv")
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")

    df = assign_emission_factors(df)
    df = compute_cci(df)
    df = flag_cci_outliers(df)

    quality_report(df)
    save_outputs(df)

    print("\nPart 2 complete.")
    print("Next: run src/part3_features.py")
    return df


if __name__ == "__main__":
    main()