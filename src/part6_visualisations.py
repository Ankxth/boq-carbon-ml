"""
Part 6: Visualisations and Paper Figures
==========================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part6_visualisations.py

Generates all figures needed for the paper:
  Fig 1 — Model comparison bar chart (R2, MAE, RMSE for 7 models)
  Fig 2 — Actual vs Predicted scatter (XGBoost, best model)
  Fig 3 — Feature importance (top 20 features, XGBoost)
  Fig 4 — CCI distribution by material (box plot)
  Fig 5 — GGBS scenario grouped bar chart (carbon + cost per project)
  Fig 6 — Embodied carbon by grade (baseline vs 50% GGBS)
  Fig 7 — Material coverage across 5 projects (stacked bar)

All figures saved as:
  PNG at 300 DPI  (for paper submission)
  PDF             (for vector editing)

Run:
  cd boq_carbon_ml
  python src/part6_visualisations.py
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE    = Path(__file__).resolve().parent.parent
PROC    = BASE / "data" / "processed"
TABLES  = BASE / "outputs" / "tables"
FIGURES = BASE / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "XGBoost":          "#1f4e79",
    "SVR":              "#2e75b6",
    "Lasso Regression": "#70ad47",
    "Decision Tree":    "#ed7d31",
    "Ridge Regression": "#ffc000",
    "Random Forest":    "#a9d18e",
    "Neural Network":   "#c00000",
    "baseline":         "#2e75b6",
    "30pct":            "#70ad47",
    "50pct":            "#c00000",
}

def apply_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        "legend.fontsize":    9,
        "legend.framealpha":  0.8,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })

def save_fig(fig, name: str):
    png_path = FIGURES / f"{name}.png"
    pdf_path = FIGURES / f"{name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {png_path.name}  +  {pdf_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def fig1_model_comparison():
    print("\n  Fig 1 — Model comparison...")
    df = pd.read_csv(TABLES / "model_comparison_v2.csv")
    df = df.sort_values("R2", ascending=True)   # ascending for horizontal bar

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Fig 1: Comparison of 7 ML Models for CCI Prediction\n"
        "(5-fold Cross-Validation, PA + CPWD Training Set, n=233)",
        fontsize=11, fontweight="bold", y=1.01
    )

    metrics = [
        ("R2",                "R² (higher is better)",  True),
        ("MAE (INR/kgCO2e)",  "MAE  INR/kgCO2e (lower is better)", False),
        ("RMSE (INR/kgCO2e)", "RMSE INR/kgCO2e (lower is better)", False),
    ]

    for ax, (col, label, higher_better) in zip(axes, metrics):
        colours = [PALETTE.get(m, "#888888") for m in df["Model"]]
        bars = ax.barh(df["Model"], df[col], color=colours,
                       edgecolor="white", linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, df[col]):
            x_pos = bar.get_width()
            ax.text(x_pos + abs(x_pos) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)

        ax.set_xlabel(label, fontsize=9)
        ax.set_title(col.replace(" (INR/kgCO2e)", ""), fontweight="bold")

        # Highlight best
        best_idx = df[col].idxmax() if higher_better else df[col].idxmin()
        best_model = df.loc[best_idx, "Model"]
        ax.axhline(
            y=list(df["Model"]).index(best_model),
            color="gold", linewidth=1.5, alpha=0.6, linestyle="--"
        )

    plt.tight_layout()
    save_fig(fig, "fig1_model_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — ACTUAL VS PREDICTED (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_actual_vs_predicted():
    print("  Fig 2 — Actual vs predicted (XGBoost)...")
    train_df = pd.read_csv(TABLES / "train_predictions_v2.csv")

    actual    = train_df["cci"].values
    predicted = train_df["cci_pred_xgboost"].values

    # Remove extreme outliers for visual clarity (> 99th percentile)
    p99   = np.percentile(actual, 99)
    mask  = actual <= p99
    actual_plot    = actual[mask]
    predicted_plot = predicted[mask]

    from sklearn.metrics import r2_score, mean_absolute_error
    r2  = r2_score(actual_plot, predicted_plot)
    mae = mean_absolute_error(actual_plot, predicted_plot)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter
    ax.scatter(actual_plot, predicted_plot,
               alpha=0.5, s=20, color="#1f4e79", edgecolors="none",
               label=f"Training predictions (n={mask.sum()})")

    # Perfect prediction line
    lim = max(actual_plot.max(), predicted_plot.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5,
            label="Perfect prediction (y=x)")

    # Annotations
    ax.text(0.05, 0.92,
            f"R² = {r2:.4f}\nMAE = {mae:.2f} INR/kgCO2e",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Actual CCI (INR/kgCO2e)", fontsize=10)
    ax.set_ylabel("Predicted CCI (INR/kgCO2e)", fontsize=10)
    ax.set_title(
        "Fig 2: XGBoost — Actual vs Predicted CCI\n"
        "(Training set, values ≤ 99th percentile shown)",
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    plt.tight_layout()
    save_fig(fig, "fig2_actual_vs_predicted")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — FEATURE IMPORTANCE (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_feature_importance():
    print("  Fig 3 — Feature importance...")

    model_pipe = joblib.load(BASE / "outputs" / "models" / "best_model_v2.joblib")
    feat_names = Path(PROC / "feature_names.txt").read_text().splitlines()

    # XGBoost is the last step in the pipeline
    xgb_model = model_pipe.named_steps["model"]
    importances = xgb_model.feature_importances_

    # Align feature names with importances
    # Note: feature matrix is X_train_v2 which has 64 features
    # feat_names from part3 may have different count — rebuild if needed
    if len(feat_names) != len(importances):
        feat_names = [f"feature_{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(20)

    # Clean up feature names for display
    def clean_name(name):
        if name.startswith("tfidf_"):
            return f'"{name[6:]}"'
        if name.startswith("unit_"):
            return f"unit={name[5:]}"
        if name.startswith("mat_"):
            return f"mat={name[4:]}"
        return name

    imp_df["label"] = imp_df["feature"].apply(clean_name)
    imp_df = imp_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    colours = ["#1f4e79" if i >= len(imp_df) - 5 else "#9dc3e6"
               for i in range(len(imp_df))]
    ax.barh(imp_df["label"], imp_df["importance"],
            color=colours, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=10)
    ax.set_title(
        "Fig 3: Top 20 Feature Importances — XGBoost Model\n"
        "(Darker bars = top 5 most important)",
        fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig3_feature_importance")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — CCI DISTRIBUTION BY MATERIAL
# ─────────────────────────────────────────────────────────────────────────────

def fig4_cci_by_material():
    print("  Fig 4 — CCI distribution by material...")
    train_df = pd.read_csv(TABLES / "train_predictions_v2.csv")

    # Keep materials with enough rows
    mat_counts = train_df["material"].value_counts()
    keep_mats  = mat_counts[mat_counts >= 3].index.tolist()
    df_plot    = train_df[train_df["material"].isin(keep_mats)].copy()

    # Cap at 95th percentile per material for visual clarity
    df_plot = df_plot[df_plot["cci"] <= df_plot["cci"].quantile(0.95)]

    order = df_plot.groupby("material")["cci"].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_plot, x="material", y="cci", order=order,
        palette="Blues_d", width=0.5, fliersize=3,
        linewidth=0.8, ax=ax
    )
    ax.set_xlabel("Material Category", fontsize=10)
    ax.set_ylabel("Carbon Cost Intensity (INR/kgCO2e)", fontsize=10)
    ax.set_title(
        "Fig 4: CCI Distribution by Material (PA + CPWD Training Set)\n"
        "(Values ≤ 95th percentile; n≥3 materials shown)",
        fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=30)

    # Annotate median values
    for i, mat in enumerate(order):
        med = df_plot[df_plot["material"] == mat]["cci"].median()
        ax.text(i, med + 2, f"{med:.0f}", ha="center", va="bottom",
                fontsize=7, color="navy")

    plt.tight_layout()
    save_fig(fig, "fig4_cci_by_material")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — GGBS SCENARIO: CARBON + COST BY PROJECT
# ─────────────────────────────────────────────────────────────────────────────

def fig5_ggbs_scenarios():
    print("  Fig 5 — GGBS scenario analysis...")
    by_proj = pd.read_csv(TABLES / "ggbs_by_project.csv")

    projects = ["Bot", "Eco", "Mall", "Zen", "PA"]
    scenarios = [
        ("A_baseline", "0% GGBS (Baseline)",  PALETTE["baseline"]),
        ("B_30pct",    "30% GGBS",            PALETTE["30pct"]),
        ("C_50pct",    "50% GGBS",            PALETTE["50pct"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Fig 5: GGBS Substitution Scenarios — Carbon Reduction and Cost Premium\n"
        "Across 5 Indian Construction Projects",
        fontsize=11, fontweight="bold"
    )

    x      = np.arange(len(projects))
    width  = 0.25

    # Left: Carbon
    for i, (scen_id, label, colour) in enumerate(scenarios):
        vals = [
            by_proj[(by_proj["project"] == p) &
                    (by_proj["scenario"] == scen_id)]["carbon_tCO2e"].values[0]
            for p in projects
        ]
        bars = ax1.bar(x + i * width, vals, width,
                       label=label, color=colour, alpha=0.85,
                       edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("Project", fontsize=10)
    ax1.set_ylabel("Embodied Carbon (tCO2e)", fontsize=10)
    ax1.set_title("Embodied Carbon by Scenario", fontweight="bold")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(projects)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

    # Right: Cost premium (0% = 0, so only 30% and 50%)
    cost_scenarios = [
        ("B_30pct", "30% GGBS", PALETTE["30pct"]),
        ("C_50pct", "50% GGBS", PALETTE["50pct"]),
    ]
    x2 = np.arange(len(projects))
    w2 = 0.35
    for i, (scen_id, label, colour) in enumerate(cost_scenarios):
        vals = [
            by_proj[(by_proj["project"] == p) &
                    (by_proj["scenario"] == scen_id)]["cost_premium_INR_M"].values[0]
            for p in projects
        ]
        ax2.bar(x2 + i * w2, vals, w2,
                label=label, color=colour, alpha=0.85,
                edgecolor="white", linewidth=0.5)

    ax2.set_xlabel("Project", fontsize=10)
    ax2.set_ylabel("Cost Premium (INR Million)", fontsize=10)
    ax2.set_title("Cost Premium for GGBS Substitution", fontweight="bold")
    ax2.set_xticks(x2 + w2 / 2)
    ax2.set_xticklabels(projects)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, "fig5_ggbs_scenarios")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — EMBODIED CARBON BY GRADE (baseline vs 50% GGBS)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_carbon_by_grade():
    print("  Fig 6 — Carbon by concrete grade...")
    by_grade = pd.read_csv(TABLES / "ggbs_by_grade.csv")

    grades     = ["M5","M10","M15","M20","M25","M30","M35","M40","M45","M50"]
    baseline   = []
    adjusted   = []
    for g in grades:
        row_b = by_grade[(by_grade["grade"] == g) &
                         (by_grade["ggbs_pct"] == 0)]
        row_a = by_grade[(by_grade["grade"] == g) &
                         (by_grade["ggbs_pct"] == 50)]
        if row_b.empty or row_a.empty:
            baseline.append(0)
            adjusted.append(0)
        else:
            baseline.append(row_b["baseline_ef_kgCO2e_m3"].values[0])
            adjusted.append(row_a["adjusted_ef_kgCO2e_m3"].values[0])

    x     = np.arange(len(grades))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width/2, baseline, width,
                label="Baseline (0% GGBS, CEM I)",
                color=PALETTE["baseline"], alpha=0.85,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width/2, adjusted, width,
                label="50% GGBS substitution",
                color=PALETTE["50pct"], alpha=0.85,
                edgecolor="white", linewidth=0.5)

    # Reduction % labels above pairs
    for i, (b, a) in enumerate(zip(baseline, adjusted)):
        if b > 0:
            pct = (b - a) / b * 100
            ax.text(i, max(b, a) + 5, f"-{pct:.0f}%",
                    ha="center", fontsize=8, color="darkred", fontweight="bold")

    ax.set_xlabel("Concrete Grade", fontsize=10)
    ax.set_ylabel("Emission Factor (kgCO2e/m³)", fontsize=10)
    ax.set_title(
        "Fig 6: Concrete EF by Grade — Baseline vs 50% GGBS Substitution\n"
        "(Percentage shows carbon reduction per m³)",
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig6_carbon_by_grade")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — MATERIAL COVERAGE ACROSS PROJECTS
# ─────────────────────────────────────────────────────────────────────────────

def fig7_material_coverage():
    print("  Fig 7 — Material coverage...")
    df = pd.read_csv(PROC / "master_cleaned.csv")

    top_mats = (df["material"].value_counts()
                  .head(8).index.tolist())

    coverage = []
    for src in ["Bot", "Eco", "Mall", "Zen", "PA"]:
        sub = df[df["source"] == src]
        total = len(sub)
        row = {"project": src}
        for mat in top_mats:
            row[mat] = (sub["material"] == mat).sum() / total * 100
        row["other"] = (
            (~sub["material"].isin(top_mats)).sum() / total * 100
        )
        coverage.append(row)

    cov_df = pd.DataFrame(coverage).set_index("project")

    mat_colours = sns.color_palette("tab10", len(cov_df.columns))

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(cov_df))
    for j, col in enumerate(cov_df.columns):
        vals = cov_df[col].values
        bars = ax.bar(cov_df.index, vals, bottom=bottom,
                      label=col, color=mat_colours[j],
                      edgecolor="white", linewidth=0.4)
        # Label segments > 5%
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(i, b + v/2, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals

    ax.set_xlabel("Project", fontsize=10)
    ax.set_ylabel("Proportion of BOQ Rows (%)", fontsize=10)
    ax.set_title(
        "Fig 7: Material Category Distribution Across 5 Projects\n"
        "(Proportion of all identified BOQ line items)",
        fontweight="bold"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1),
              fontsize=8, title="Material")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    save_fig(fig, "fig7_material_coverage")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — CCI HEATMAP (grade vs project, concrete only)
# ─────────────────────────────────────────────────────────────────────────────

def fig8_cci_heatmap():
    print("  Fig 8 — CCI heatmap (concrete grades by project)...")
    train_df = pd.read_csv(TABLES / "train_predictions_v2.csv")

    conc = train_df[train_df["material"] == "concrete"].copy()
    conc = conc[conc["cci"] <= conc["cci"].quantile(0.95)]

    grades   = ["M10","M15","M20","M25","M30","M35","M40","M45","M50"]
    projects = ["Bot","Eco","Mall","Zen","PA"]

    matrix = pd.DataFrame(index=grades, columns=projects, dtype=float)
    for g in grades:
        for p in projects:
            sub = conc[(conc["grade"] == g) & (conc["source"] == p)]
            matrix.loc[g, p] = sub["cci"].median() if len(sub) >= 2 else np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".1f",
        cmap="YlOrRd", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Median CCI (INR/kgCO2e)"},
        ax=ax, mask=matrix.astype(float).isna()
    )
    ax.set_xlabel("Project", fontsize=10)
    ax.set_ylabel("Concrete Grade", fontsize=10)
    ax.set_title(
        "Fig 8: Median CCI (INR/kgCO2e) by Concrete Grade and Project\n"
        "(Blank = fewer than 2 observations)",
        fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig8_cci_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("PART 6 -- VISUALISATIONS AND PAPER FIGURES")
    print("=" * 60)

    apply_style()

    figures = [
        ("Fig 1 — Model comparison",          fig1_model_comparison),
        ("Fig 2 — Actual vs predicted",        fig2_actual_vs_predicted),
        ("Fig 3 — Feature importance",         fig3_feature_importance),
        ("Fig 4 — CCI by material",            fig4_cci_by_material),
        ("Fig 5 — GGBS scenarios",             fig5_ggbs_scenarios),
        ("Fig 6 — Carbon by grade",            fig6_carbon_by_grade),
        ("Fig 7 — Material coverage",          fig7_material_coverage),
        ("Fig 8 — CCI heatmap",                fig8_cci_heatmap),
    ]

    failed = []
    for name, fn in figures:
        try:
            fn()
        except Exception as e:
            print(f"  WARNING: {name} failed — {e}")
            failed.append(name)

    print(f"\n[Summary]")
    print(f"  Figures generated : {len(figures) - len(failed)} / {len(figures)}")
    if failed:
        print(f"  Failed            : {failed}")

    generated = list(FIGURES.glob("*.png"))
    print(f"\n  Files in outputs/figures/:")
    for f in sorted(generated):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size_kb:6.1f} KB")

    print(f"\nPart 6 complete.")
    print("All figures saved to outputs/figures/ as PNG (300 DPI) and PDF.")


if __name__ == "__main__":
    main()