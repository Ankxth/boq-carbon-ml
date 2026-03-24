"""
Part 5: GGBS Substitution Scenario Analysis
=============================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part5_ggbs.py

What this script does:
  1. Loads the full cleaned dataset (Part 1 output)
  2. Isolates all structural concrete rows across all 5 projects
  3. Computes baseline embodied carbon (0% GGBS, standard CEM I)
  4. Models three GGBS substitution scenarios:
       Scenario A: 0%  GGBS (baseline — current standard practice)
       Scenario B: 30% GGBS substitution
       Scenario C: 50% GGBS substitution
  5. For each scenario, computes:
       Total embodied carbon (tCO2e) per project
       Carbon reduction vs baseline (tCO2e and %)
       Cost premium vs baseline (INR) using PA rates + CPWD rates
  6. Uses PA's own GGBS differential rate data (two real PA items):
       PCC grade change M7.5->M10 with 50% GGBS: +INR 49/m3
       M30 raft with 40% GGBS (350 kg cementitious): +INR 85/m3
  7. Saves:
       outputs/tables/ggbs_scenario_results.csv
       outputs/tables/ggbs_by_project.csv
       outputs/tables/ggbs_by_grade.csv

GGBS carbon reduction methodology:
  Standard concrete EF uses 100% CEM I (Portland cement)
  CEM I EF = ~0.83 kgCO2e/kg (ICE V4.1)
  GGBS EF  = ~0.07 kgCO2e/kg (ICE V4.1, slag cement)
  Typical mix: ~350 kg cementitious per m3 of structural concrete

  Carbon saved per m3 = cement_content * (1 - GGBS_fraction) * GGBS_EF_diff
  Where GGBS_EF_diff = CEM_I_EF - GGBS_EF = 0.83 - 0.07 = 0.76 kgCO2e/kg

  At 30% GGBS: carbon reduction = 350 * 0.30 * 0.76 = 79.8 kgCO2e/m3
  At 50% GGBS: carbon reduction = 350 * 0.50 * 0.76 = 133.0 kgCO2e/m3

Cost premium methodology:
  Based on PA's actual GGBS differential rates:
    ~INR 49-85/m3 for GGBS substitution (PA contract data)
  CPWD published premium: ~INR 120-180/m3 for 30-50% GGBS
  We use PA rates as primary (actual contract), CPWD as validation

Run:
  cd boq_carbon_ml
  python src/part5_ggbs.py
"""

import pandas as pd
import numpy as np
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
# SECTION 1 -- CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# ICE DB V4.1 concrete EF (kgCO2e/m3) — 100% CEM I baseline
CONCRETE_EF_M3 = {
    'M5':  150.6, 'M10': 232.8, 'M15': 268.1,
    'M20': 284.9, 'M25': 301.8, 'M30': 331.2,
    'M35': 356.3, 'M40': 381.5, 'M45': 400.0, 'M50': 420.0,
}

# ICE DB V4.1 binder EF values (kgCO2e/kg)
CEM_I_EF  = 0.830   # Portland cement
GGBS_EF   = 0.070   # Ground Granulated Blast-furnace Slag
EF_DIFF   = CEM_I_EF - GGBS_EF   # = 0.760 kgCO2e/kg saved per kg cement replaced

# Typical cementitious content in Indian structural concrete (kg/m3)
# Source: IS 456:2000 + PA contract specifications
CEMENT_CONTENT_KG_M3 = {
    'M5':  200, 'M10': 250, 'M15': 280,
    'M20': 300, 'M25': 320, 'M30': 350,
    'M35': 380, 'M40': 400, 'M45': 420, 'M50': 450,
}

# GGBS substitution scenarios
SCENARIOS = {
    'A_baseline': 0.00,
    'B_30pct':    0.30,
    'C_50pct':    0.50,
}

# Cost premium per m3 for GGBS substitution
# Derived from PA contract GGBS differential rates:
#   M10 with 50% GGBS: +INR 49/m3
#   M30 with 40% GGBS: +INR 85/m3
# Extrapolated linearly for other fractions
# Premium = base_premium * (ggbs_fraction / 0.40)
PA_GGBS_BASE_PREMIUM_PER_M3 = 85.0    # INR/m3 at 40% GGBS (from PA contract)
PA_GGBS_BASE_FRACTION       = 0.40    # the reference fraction

def cost_premium_per_m3(ggbs_fraction: float) -> float:
    """INR per m3 cost premium for GGBS substitution, scaled from PA contract data."""
    if ggbs_fraction == 0:
        return 0.0
    return PA_GGBS_BASE_PREMIUM_PER_M3 * (ggbs_fraction / PA_GGBS_BASE_FRACTION)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1] Loading cleaned dataset...")
    df = pd.read_csv(PROC / "master_cleaned.csv")
    print(f"  Total rows: {len(df):,}")

    # Filter to structural concrete rows with m3 unit
    # m2 rows excluded — volume cannot be computed without confirmed thickness
    concrete = df[
        (df['material'] == 'concrete') &
        (df['unit_clean'] == 'm3') &
        (df['grade'].notna()) &
        (df['quantity'] > 0)
    ].copy()

    print(f"  Concrete m3 rows: {len(concrete)}")
    print(f"  By source:")
    for src, grp in concrete.groupby('source'):
        total_vol = grp['quantity'].sum()
        print(f"    {src:5s}: {len(grp):3d} rows | "
              f"total volume = {total_vol:,.1f} m3")

    return concrete

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- COMPUTE BASELINE CARBON
# ─────────────────────────────────────────────────────────────────────────────

def compute_baseline(concrete: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] Computing baseline embodied carbon (0% GGBS, CEM I)...")

    df = concrete.copy()

    # EF for each row (kgCO2e/m3)
    df['ef_m3'] = df['grade'].map(CONCRETE_EF_M3)

    # Rows with unrecognised grade — assign M20 as default
    missing_ef = df['ef_m3'].isna().sum()
    if missing_ef > 0:
        print(f"  {missing_ef} rows with unrecognised grade — defaulting to M20 EF")
        df['ef_m3'].fillna(CONCRETE_EF_M3['M20'], inplace=True)

    # Baseline embodied carbon (kgCO2e)
    df['carbon_baseline_kgCO2e'] = df['quantity'] * df['ef_m3']

    # Cement content per row (kg)
    df['cement_content_kg_m3'] = df['grade'].map(CEMENT_CONTENT_KG_M3).fillna(350)

    total_vol    = df['quantity'].sum()
    total_carbon = df['carbon_baseline_kgCO2e'].sum() / 1000  # convert to tCO2e

    print(f"  Total concrete volume: {total_vol:,.1f} m3")
    print(f"  Total baseline carbon: {total_carbon:,.1f} tCO2e")
    print(f"\n  By project:")
    for src, grp in df.groupby('source'):
        vol = grp['quantity'].sum()
        co2 = grp['carbon_baseline_kgCO2e'].sum() / 1000
        avg_ef = grp['ef_m3'].mean()
        print(f"    {src:5s}: {vol:8,.1f} m3 | "
              f"{co2:8,.1f} tCO2e | avg EF={avg_ef:.1f} kgCO2e/m3")

    print(f"\n  By grade:")
    for grade, grp in df.groupby('grade'):
        vol = grp['quantity'].sum()
        co2 = grp['carbon_baseline_kgCO2e'].sum() / 1000
        print(f"    {grade:4s}: {vol:8,.1f} m3 | {co2:8,.1f} tCO2e")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- COMPUTE GGBS SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ggbs_carbon(ef_m3: float, grade: str,
                         quantity_m3: float, ggbs_fraction: float) -> dict:
    """
    Compute carbon and cost for one concrete row under a given GGBS fraction.

    Carbon reduction mechanism:
      Each m3 of concrete contains ~cement_content kg of cementitious material.
      Replacing fraction f of CEM I with GGBS saves:
        carbon_saved_per_m3 = cement_content * f * (CEM_I_EF - GGBS_EF)

    The adjusted EF per m3:
        ef_adjusted = ef_m3 - cement_content * f * EF_DIFF
    """
    cement_kg   = CEMENT_CONTENT_KG_M3.get(str(grade), 350)
    carbon_saved_per_m3 = cement_kg * ggbs_fraction * EF_DIFF
    ef_adjusted = max(ef_m3 - carbon_saved_per_m3, 0)

    carbon_kgCO2e = quantity_m3 * ef_adjusted
    premium_inr   = quantity_m3 * cost_premium_per_m3(ggbs_fraction)

    return {
        'ef_adjusted':    round(ef_adjusted, 3),
        'carbon_kgCO2e':  round(carbon_kgCO2e, 3),
        'premium_inr':    round(premium_inr, 2),
        'saved_per_m3':   round(carbon_saved_per_m3, 3),
    }


def run_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] Running GGBS substitution scenarios...")

    results = []
    for _, row in df.iterrows():
        for scenario_name, ggbs_frac in SCENARIOS.items():
            r = compute_ggbs_carbon(
                ef_m3       = row['ef_m3'],
                grade       = row['grade'],
                quantity_m3 = row['quantity'],
                ggbs_fraction = ggbs_frac,
            )
            results.append({
                'source':         row['source'],
                'description':    str(row['description'])[:80],
                'grade':          row['grade'],
                'unit_clean':     row['unit_clean'],
                'quantity_m3':    row['quantity'],
                'ef_baseline':    row['ef_m3'],
                'scenario':       scenario_name,
                'ggbs_fraction':  ggbs_frac,
                **r,
            })

    results_df = pd.DataFrame(results)

    # Summary by scenario and project
    print(f"\n  Scenario summary across all 5 projects:")
    print(f"  {'Scenario':<20} {'GGBS%':>6} {'Carbon (tCO2e)':>16} "
          f"{'Reduction (tCO2e)':>18} {'Reduction %':>12} {'Cost Premium (INR M)':>20}")
    print(f"  {'-'*94}")

    baseline_carbon = results_df[
        results_df['scenario'] == 'A_baseline'
    ]['carbon_kgCO2e'].sum() / 1000

    scenario_summary = []
    for scenario_name, ggbs_frac in SCENARIOS.items():
        sub = results_df[results_df['scenario'] == scenario_name]
        carbon_tCO2e  = sub['carbon_kgCO2e'].sum() / 1000
        reduction     = baseline_carbon - carbon_tCO2e
        pct_reduction = (reduction / baseline_carbon * 100) if baseline_carbon > 0 else 0
        premium_inr_m = sub['premium_inr'].sum() / 1_000_000  # convert to INR million

        label = f"{int(ggbs_frac*100)}% GGBS"
        print(f"  {label:<20} {ggbs_frac*100:>5.0f}% "
              f"{carbon_tCO2e:>16,.1f} "
              f"{reduction:>18,.1f} "
              f"{pct_reduction:>11.1f}% "
              f"{premium_inr_m:>19.2f}M")

        scenario_summary.append({
            'scenario':           scenario_name,
            'ggbs_fraction_pct':  int(ggbs_frac * 100),
            'total_carbon_tCO2e': round(carbon_tCO2e, 2),
            'reduction_tCO2e':    round(reduction, 2),
            'reduction_pct':      round(pct_reduction, 2),
            'cost_premium_INR_M': round(premium_inr_m, 4),
        })

    return results_df, pd.DataFrame(scenario_summary)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 -- BREAKDOWN TABLES
# ─────────────────────────────────────────────────────────────────────────────

def build_breakdown_tables(results_df: pd.DataFrame):
    print("\n[4] Building breakdown tables...")

    # By project
    by_project_rows = []
    for src in ['Bot', 'Eco', 'Mall', 'Zen', 'PA']:
        sub = results_df[results_df['source'] == src]
        if sub.empty:
            continue
        for scenario_name, ggbs_frac in SCENARIOS.items():
            s = sub[sub['scenario'] == scenario_name]
            baseline_s = sub[sub['scenario'] == 'A_baseline']
            carbon    = s['carbon_kgCO2e'].sum() / 1000
            baseline  = baseline_s['carbon_kgCO2e'].sum() / 1000
            reduction = baseline - carbon
            pct       = (reduction / baseline * 100) if baseline > 0 else 0
            premium   = s['premium_inr'].sum() / 1_000_000
            vol       = s['quantity_m3'].sum()
            by_project_rows.append({
                'project':            src,
                'scenario':           scenario_name,
                'ggbs_pct':           int(ggbs_frac * 100),
                'concrete_volume_m3': round(vol, 1),
                'carbon_tCO2e':       round(carbon, 2),
                'baseline_tCO2e':     round(baseline, 2),
                'reduction_tCO2e':    round(reduction, 2),
                'reduction_pct':      round(pct, 2),
                'cost_premium_INR_M': round(premium, 4),
            })

    by_project_df = pd.DataFrame(by_project_rows)

    print(f"\n  Carbon reduction by project (50% GGBS scenario):")
    c50 = by_project_df[by_project_df['ggbs_pct'] == 50]
    print(f"  {'Project':<8} {'Volume m3':>12} {'Baseline tCO2e':>16} "
          f"{'Saved tCO2e':>12} {'Saved %':>8} {'Premium INR M':>14}")
    print(f"  {'-'*72}")
    for _, r in c50.iterrows():
        print(f"  {r['project']:<8} {r['concrete_volume_m3']:>12,.1f} "
              f"{r['baseline_tCO2e']:>16,.1f} "
              f"{r['reduction_tCO2e']:>12,.1f} "
              f"{r['reduction_pct']:>7.1f}% "
              f"{r['cost_premium_INR_M']:>13.3f}M")

    # By grade
    by_grade_rows = []
    for grade in sorted(CONCRETE_EF_M3.keys(),
                        key=lambda g: int(g.replace('M',''))):
        sub = results_df[results_df['grade'] == grade]
        if sub.empty:
            continue
        for scenario_name, ggbs_frac in SCENARIOS.items():
            s = sub[sub['scenario'] == scenario_name]
            baseline_s = sub[sub['scenario'] == 'A_baseline']
            carbon   = s['carbon_kgCO2e'].sum() / 1000
            baseline = baseline_s['carbon_kgCO2e'].sum() / 1000
            vol      = s['quantity_m3'].sum()
            saved_pm3 = s['saved_per_m3'].mean() if len(s) > 0 else 0
            by_grade_rows.append({
                'grade':                  grade,
                'scenario':               scenario_name,
                'ggbs_pct':               int(ggbs_frac * 100),
                'total_volume_m3':        round(vol, 1),
                'baseline_ef_kgCO2e_m3':  CONCRETE_EF_M3[grade],
                'adjusted_ef_kgCO2e_m3':  round(CONCRETE_EF_M3[grade] - saved_pm3, 2),
                'carbon_saved_tCO2e':     round(baseline - carbon, 2),
                'carbon_saved_pct':       round((baseline - carbon) / baseline * 100, 2)
                                          if baseline > 0 else 0,
            })

    by_grade_df = pd.DataFrame(by_grade_rows)

    print(f"\n  EF reduction by grade (50% GGBS):")
    g50 = by_grade_df[by_grade_df['ggbs_pct'] == 50]
    print(f"  {'Grade':<6} {'Volume m3':>10} {'Baseline EF':>12} "
          f"{'Adjusted EF':>12} {'Saved %':>8}")
    print(f"  {'-'*52}")
    for _, r in g50.iterrows():
        print(f"  {r['grade']:<6} {r['total_volume_m3']:>10,.1f} "
              f"{r['baseline_ef_kgCO2e_m3']:>12.1f} "
              f"{r['adjusted_ef_kgCO2e_m3']:>12.1f} "
              f"{r['carbon_saved_pct']:>7.1f}%")

    return by_project_df, by_grade_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 -- KEY FINDINGS
# ─────────────────────────────────────────────────────────────────────────────

def print_key_findings(scenario_summary: pd.DataFrame,
                       by_project_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    base = scenario_summary[scenario_summary['ggbs_fraction_pct'] == 0].iloc[0]
    s30  = scenario_summary[scenario_summary['ggbs_fraction_pct'] == 30].iloc[0]
    s50  = scenario_summary[scenario_summary['ggbs_fraction_pct'] == 50].iloc[0]

    print(f"\n  Baseline (0% GGBS):")
    print(f"    Total embodied carbon: {base['total_carbon_tCO2e']:,.1f} tCO2e")
    print(f"    Cost premium:          INR 0")

    print(f"\n  Scenario B — 30% GGBS substitution:")
    print(f"    Total carbon:          {s30['total_carbon_tCO2e']:,.1f} tCO2e")
    print(f"    Carbon saved:          {s30['reduction_tCO2e']:,.1f} tCO2e "
          f"({s30['reduction_pct']:.1f}% reduction)")
    print(f"    Cost premium:          INR {s30['cost_premium_INR_M']:.2f}M")
    if s30['cost_premium_INR_M'] > 0:
        cost_per_tonne = (s30['cost_premium_INR_M'] * 1_000_000) / \
                         (s30['reduction_tCO2e'] * 1000) if s30['reduction_tCO2e'] > 0 else 0
        print(f"    Cost per tCO2e saved:  INR {cost_per_tonne:,.0f}")

    print(f"\n  Scenario C — 50% GGBS substitution:")
    print(f"    Total carbon:          {s50['total_carbon_tCO2e']:,.1f} tCO2e")
    print(f"    Carbon saved:          {s50['reduction_tCO2e']:,.1f} tCO2e "
          f"({s50['reduction_pct']:.1f}% reduction)")
    print(f"    Cost premium:          INR {s50['cost_premium_INR_M']:.2f}M")
    if s50['cost_premium_INR_M'] > 0:
        cost_per_tonne = (s50['cost_premium_INR_M'] * 1_000_000) / \
                         (s50['reduction_tCO2e'] * 1000) if s50['reduction_tCO2e'] > 0 else 0
        print(f"    Cost per tCO2e saved:  INR {cost_per_tonne:,.0f}")

    print(f"\n  Interpretation:")
    print(f"    50% GGBS substitution reduces embodied carbon in structural")
    print(f"    concrete by {s50['reduction_pct']:.1f}% — the single most impactful")
    print(f"    design-stage intervention available to Indian construction")
    print(f"    practitioners, at a cost premium of INR {s50['cost_premium_INR_M']:.2f}M")
    print(f"    across the 5 projects studied.")
    if s50['reduction_tCO2e'] > 0 and s50['cost_premium_INR_M'] > 0:
        eff = s50['reduction_tCO2e'] / s50['cost_premium_INR_M']
        print(f"    Efficiency: {eff:.1f} tCO2e saved per INR million premium.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 -- SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(results_df, scenario_summary, by_project_df, by_grade_df):
    print("\n[5] Saving outputs...")

    results_df.to_csv(TABLES / "ggbs_scenario_results.csv", index=False)
    scenario_summary.to_csv(TABLES / "ggbs_summary.csv", index=False)
    by_project_df.to_csv(TABLES / "ggbs_by_project.csv", index=False)
    by_grade_df.to_csv(TABLES / "ggbs_by_grade.csv", index=False)

    print(f"  outputs/tables/ggbs_scenario_results.csv  ({len(results_df)} rows)")
    print(f"  outputs/tables/ggbs_summary.csv           ({len(scenario_summary)} rows)")
    print(f"  outputs/tables/ggbs_by_project.csv        ({len(by_project_df)} rows)")
    print(f"  outputs/tables/ggbs_by_grade.csv          ({len(by_grade_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 -- QUALITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def quality_checks(scenario_summary, by_project_df):
    print("\n  Assertions:")

    s50 = scenario_summary[scenario_summary['ggbs_fraction_pct'] == 50].iloc[0]
    s30 = scenario_summary[scenario_summary['ggbs_fraction_pct'] == 30].iloc[0]
    s0  = scenario_summary[scenario_summary['ggbs_fraction_pct'] == 0].iloc[0]

    assert s50['total_carbon_tCO2e'] < s0['total_carbon_tCO2e'], \
        "50% GGBS should reduce carbon vs baseline"
    print(f"  OK  50% GGBS reduces carbon "
          f"({s50['total_carbon_tCO2e']:.1f} < {s0['total_carbon_tCO2e']:.1f} tCO2e)")

    assert s30['total_carbon_tCO2e'] < s0['total_carbon_tCO2e'], \
        "30% GGBS should reduce carbon vs baseline"
    print(f"  OK  30% GGBS reduces carbon "
          f"({s30['total_carbon_tCO2e']:.1f} < {s0['total_carbon_tCO2e']:.1f} tCO2e)")

    assert s50['total_carbon_tCO2e'] < s30['total_carbon_tCO2e'], \
        "50% GGBS should reduce more than 30%"
    print(f"  OK  50% GGBS reduces more than 30% GGBS")

    assert 15 < s50['reduction_pct'] < 50, \
        f"50% GGBS reduction out of expected range: {s50['reduction_pct']:.1f}%"
    print(f"  OK  50% GGBS reduction in expected range ({s50['reduction_pct']:.1f}%)")

    assert s50['cost_premium_INR_M'] >= 0, "Cost premium cannot be negative"
    print(f"  OK  Cost premium non-negative")

    # All 5 projects should appear in by_project results
    projects_present = set(by_project_df['project'].unique())
    assert len(projects_present) == 5, \
        f"Expected 5 projects, got: {projects_present}"
    print(f"  OK  All 5 projects present: {sorted(projects_present)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("PART 5 -- GGBS SUBSTITUTION SCENARIO ANALYSIS")
    print("=" * 60)

    concrete_df = load_data()
    concrete_df = compute_baseline(concrete_df)

    results_df, scenario_summary = run_scenarios(concrete_df)
    by_project_df, by_grade_df   = build_breakdown_tables(results_df)

    print_key_findings(scenario_summary, by_project_df)
    quality_checks(scenario_summary, by_project_df)
    save_outputs(results_df, scenario_summary, by_project_df, by_grade_df)

    print("\nPart 5 complete.")
    print("Next: run src/part6_visualisations.py")
    return scenario_summary, by_project_df


if __name__ == "__main__":
    main()