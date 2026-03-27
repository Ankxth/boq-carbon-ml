"""
Part 7b: FastAPI Prediction API
=================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part7b_api.py

What this script does:
  Serves the best trained model (XGBoost) as a REST API using FastAPI.
  Accepts a BOQ description and returns:
    - Predicted CCI (INR/kgCO2e)
    - Predicted embodied carbon (kgCO2e) given quantity
    - GGBS scenario projections (0%, 30%, 50%)
    - Confidence indicator

  Endpoints:
    GET  /              — health check
    GET  /model-info    — model metadata (R2, MAE, training data)
    POST /predict       — predict CCI for one BOQ item
    POST /predict-batch — predict CCI for multiple BOQ items
    GET  /docs          — auto-generated Swagger UI (FastAPI built-in)

Run:
  cd boq_carbon_ml
  python src/part7b_api.py

  Or with uvicorn directly:
  uvicorn src.part7b_api:app --reload --port 8000

  Then open: http://localhost:8000/docs
  for the interactive Swagger UI.

Example request (POST /predict):
  {
    "description": "Providing and laying RCC M30 in columns beams and slabs",
    "unit": "m3",
    "quantity": 150.0,
    "grade": "M30"
  }

Example response:
  {
    "description": "Providing and laying RCC M30...",
    "material_detected": "concrete",
    "grade": "M30",
    "predicted_cci": 23.5,
    "emission_factor": 331.2,
    "ef_unit": "kgCO2e/m3",
    "total_carbon_kgCO2e": 49680.0,
    "ggbs_scenarios": {
      "baseline_0pct":  {"carbon_kgCO2e": 49680.0, "cost_premium_inr": 0},
      "scenario_30pct": {"carbon_kgCO2e": 37693.5, "cost_premium_inr": 9562.5},
      "scenario_50pct": {"carbon_kgCO2e": 29872.5, "cost_premium_inr": 15937.5}
    }
  }
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

# ── FastAPI imports ───────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# ── sklearn for TF-IDF rebuild ────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
MODELS = BASE / "outputs" / "models"
TABLES = BASE / "outputs" / "tables"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (same as Part 4b / Part 5)
# ─────────────────────────────────────────────────────────────────────────────

CONCRETE_EF_M3 = {
    'M5':150.6,'M10':232.8,'M15':268.1,'M20':284.9,'M25':301.8,
    'M30':331.2,'M35':356.3,'M40':381.5,'M45':400.0,'M50':420.0,
}
ICE_EF_KG = {
    'steel':1.720,'brick':0.213,'masonry':0.213,'glass':1.437,
    'timber':0.493,'paint':2.152,'tile':0.796,'plaster':0.238,
    'aluminium':6.669,'mortar':0.208,'insulation':1.860,
}
DENSITY_KG_M3  = {'brick':1800,'masonry':1800,'mortar':2000}
COVERAGE_KG_M2 = {
    'plaster':20.0,'paint':0.3,'tile':22.0,'insulation':1.5,'glass':25.0,
}
CEMENT_CONTENT = {
    'M5':200,'M10':250,'M15':280,'M20':300,'M25':320,
    'M30':350,'M35':380,'M40':400,'M45':420,'M50':450,
}
CEM_I_EF = 0.830
GGBS_EF  = 0.070
EF_DIFF  = CEM_I_EF - GGBS_EF
PA_GGBS_PREMIUM_BASE     = 85.0   # INR/m3 at 40% GGBS
PA_GGBS_PREMIUM_FRACTION = 0.40

ALL_UNITS = ["m3","m2","m","nos","ton","kg","ls","day","bags","no.s","unknown","sqft"]
GRADE_STRENGTH = {
    "M5":5,"M10":10,"M15":15,"M20":20,"M25":25,"M30":30,
    "M35":35,"M40":40,"M45":45,"M50":50,
    "Fe250":0,"Fe415":0,"Fe500":0,
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (done once at startup)
# ─────────────────────────────────────────────────────────────────────────────

class ModelState:
    """Holds the loaded model and feature pipeline. Loaded once at startup."""
    model_pipe     = None
    tfidf          = None
    unit_cols      = None
    feat_names     = None
    model_metadata = {}


STATE = ModelState()


def load_model_and_pipeline():
    """Load the best model and rebuild the TF-IDF pipeline from training data."""
    print("Loading model and feature pipeline...")

    # Load best model
    model_path = MODELS / "best_model_v2.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run part4b_extended_models.py first."
        )
    STATE.model_pipe = joblib.load(model_path)

    # Rebuild TF-IDF from training data (same params as Part 4b)
    train_df = pd.read_csv(PROC / "part4b_train_combined.csv")
    tfidf = TfidfVectorizer(
        max_features=50, ngram_range=(1, 2), min_df=1,
        sublinear_tf=True, strip_accents="unicode", analyzer="word",
    )
    tfidf.fit(train_df["description_clean"].fillna(""))
    STATE.tfidf = tfidf

    # Build unit column list (must match training exactly)
    unit_ohe = pd.get_dummies(
        train_df["unit_clean"].fillna("unknown"), prefix="unit"
    )
    for u in ALL_UNITS:
        if f"unit_{u}" not in unit_ohe.columns:
            unit_ohe[f"unit_{u}"] = 0
    STATE.unit_cols = sorted(
        [c for c in unit_ohe.columns if c.startswith("unit_")]
    )

    # Load model metadata
    comparison_path = TABLES / "model_comparison_v2.csv"
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        best = comp.iloc[0]
        STATE.model_metadata = {
            "model_name":     best["Model"],
            "cv_r2":          float(best["R2"]),
            "cv_mae":         float(best["MAE (INR/kgCO2e)"]),
            "cv_rmse":        float(best["RMSE (INR/kgCO2e)"]),
            "training_rows":  233,
            "training_data":  "PA contract (INR rates) + CPWD Schedule of Rates 2023",
            "target_variable":"log1p(CCI) = log1p(rate / emission_factor)",
            "features":       "TF-IDF(50) + unit one-hot + grade_strength + log_quantity",
        }

    print(f"  Model loaded: {STATE.model_metadata.get('model_name','XGBoost')}")
    print(f"  TF-IDF vocabulary: {len(STATE.tfidf.vocabulary_)} terms")
    print(f"  Ready.")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDING (for a single prediction)
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def detect_material(desc_clean: str) -> str:
    patterns = [
        ("concrete",   [r'\brcc\b',r'\bpcc\b',r'reinforced.*concrete',
                        r'plain.*concrete',r'ready.*mix',r'grade.*m\d{2}',
                        r'm\d{2}.*grade',r'concrete.*m\d{2}']),
        ("steel",      [r'tmt.*bar',r'reinforcement.*fe',r'steel.*bar',
                        r'hsd.*bar',r'tor.*steel']),
        ("brick",      [r'brick.*masonry',r'masonry.*brick',r'\bbrick\b']),
        ("masonry",    [r'block.*masonry',r'masonry.*block']),
        ("plaster",    [r'\bplaster',r'cement.*plaster',r'gypsum.*plaster']),
        ("paint",      [r'\bpaint',r'emulsion',r'distemper',r'whitewash']),
        ("tile",       [r'\btile\b',r'ceramic.*tile',r'vitrified.*tile',
                        r'granite.*floor',r'marble.*floor']),
        ("glass",      [r'toughened.*glass',r'float.*glass',r'\bglazing\b']),
        ("aluminium",  [r'\baluminium\b',r'\baluminum\b',r'curtain.*wall']),
        ("insulation", [r'thermal.*insul',r'rock.*wool',r'mineral.*wool']),
        ("timber",     [r'\btimber\b',r'\bplywood\b',r'wooden.*door']),
        ("mortar",     [r'cement.*mortar',r'bedding.*mortar']),
    ]
    for mat, pats in patterns:
        for p in pats:
            if re.search(p, desc_clean):
                return mat
    return "unknown"


def extract_grade(desc: str) -> Optional[str]:
    m = re.search(r'\b(m\s?5|m\s?10|m\s?15|m\s?20|m\s?25|m\s?30|m\s?35|m\s?40|m\s?45|m\s?50)\b',
                  desc, re.IGNORECASE)
    return re.sub(r'\s', '', m.group(1)).upper() if m else None


def get_ef(material: str, grade: Optional[str], unit: str) -> Optional[float]:
    if material == "concrete":
        ef_m3 = CONCRETE_EF_M3.get(grade or "M20")
        if unit == "m3": return ef_m3
        if unit == "m2": return ef_m3 * 0.075 if ef_m3 else None
        return ef_m3
    if unit == "m3" and material in DENSITY_KG_M3:
        return ICE_EF_KG[material] * DENSITY_KG_M3[material]
    if unit == "m2" and material in COVERAGE_KG_M2:
        return ICE_EF_KG.get(material, 0) * COVERAGE_KG_M2[material]
    if unit == "ton" and material == "steel":
        return ICE_EF_KG["steel"] * 1000
    if unit == "kg":
        return ICE_EF_KG.get(material)
    return ICE_EF_KG.get(material)


def build_feature_vector(
    description: str,
    unit: str,
    quantity: float,
    grade: Optional[str],
) -> np.ndarray:
    """Build a (1, n_features) feature vector for a single BOQ item."""
    desc_clean = clean_text(description)

    # TF-IDF
    tfidf_vec = STATE.tfidf.transform([desc_clean]).toarray()   # (1, 50)

    # Unit one-hot
    unit_lower = str(unit).lower().strip()
    unit_vec   = np.zeros((1, len(STATE.unit_cols)))
    col_name   = f"unit_{unit_lower}"
    if col_name in STATE.unit_cols:
        unit_vec[0, STATE.unit_cols.index(col_name)] = 1
    else:
        # fallback to unknown
        unk = "unit_unknown"
        if unk in STATE.unit_cols:
            unit_vec[0, STATE.unit_cols.index(unk)] = 1

    # Grade strength
    grade_strength = float(GRADE_STRENGTH.get(grade or "", 0))

    # log(quantity)
    log_qty = np.log1p(max(float(quantity), 0))

    X = np.hstack([
        tfidf_vec,
        unit_vec,
        [[grade_strength]],
        [[log_qty]],
    ]).astype(np.float64)

    return X


def ggbs_scenarios(
    ef: float, grade: Optional[str], quantity: float, unit: str
) -> dict:
    """Compute carbon and cost under 0%, 30%, 50% GGBS for concrete items."""
    if ef is None or quantity <= 0:
        return {}

    cement_kg = CEMENT_CONTENT.get(grade or "M30", 350)
    scenarios = {}
    for pct, frac in [(0, 0.0), (30, 0.30), (50, 0.50)]:
        saved_per_unit = cement_kg * frac * EF_DIFF if unit == "m3" else 0
        ef_adj = max(ef - saved_per_unit, 0)
        carbon = quantity * ef_adj
        premium = (
            quantity * PA_GGBS_PREMIUM_BASE * (frac / PA_GGBS_PREMIUM_FRACTION)
            if unit == "m3" else 0.0
        )
        scenarios[f"scenario_{pct}pct"] = {
            "ggbs_fraction_pct":  pct,
            "ef_adjusted":        round(ef_adj, 3),
            "carbon_kgCO2e":      round(carbon, 2),
            "cost_premium_inr":   round(premium, 2),
        }
    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BOQ Carbon Cost Intensity Predictor",
    description=(
        "Predicts Carbon Cost Intensity (CCI = INR/kgCO2e) for Indian "
        "construction BOQ items using XGBoost trained on PA contract rates "
        "and CPWD Schedule of Rates 2023. Also provides GGBS substitution "
        "scenario analysis for concrete items."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_model_and_pipeline()


# ─── Request / Response schemas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    description: str = Field(
        ...,
        example="Providing and laying RCC M30 in columns beams and slabs",
        description="BOQ item description text"
    )
    unit: str = Field(
        default="m3",
        example="m3",
        description="Unit of measurement (m3, m2, m, nos, ton, kg)"
    )
    quantity: float = Field(
        default=1.0,
        ge=0,
        example=150.0,
        description="Quantity of the BOQ item"
    )
    grade: Optional[str] = Field(
        default=None,
        example="M30",
        description="Concrete grade if known (M5–M50). Auto-detected if not provided."
    )


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]


class GGBSScenario(BaseModel):
    ggbs_fraction_pct: int
    ef_adjusted: float
    carbon_kgCO2e: float
    cost_premium_inr: float


class PredictResponse(BaseModel):
    description:           str
    material_detected:     str
    grade:                 Optional[str]
    unit:                  str
    quantity:              float
    emission_factor:       Optional[float]
    ef_unit:               Optional[str]
    predicted_log_cci:     float
    predicted_cci:         float
    total_carbon_kgCO2e:   Optional[float]
    ggbs_scenarios:        Optional[dict]
    note:                  str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status":  "ok",
        "service": "BOQ CCI Predictor",
        "model":   STATE.model_metadata.get("model_name", "XGBoost"),
        "version": "1.0.0",
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_metadata": STATE.model_metadata,
        "supported_units":     ["m3","m2","m","nos","ton","kg"],
        "supported_grades":    list(CONCRETE_EF_M3.keys()),
        "ggbs_scenarios":      "available for concrete m3 items",
        "api_docs":            "http://localhost:8000/docs",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_single(req: PredictRequest):
    if STATE.model_pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    desc_clean = clean_text(req.description)
    material   = detect_material(desc_clean)
    grade      = req.grade or extract_grade(desc_clean)
    unit       = req.unit.lower().strip()
    quantity   = float(req.quantity)

    # Build features and predict
    try:
        X = build_feature_vector(req.description, unit, quantity, grade)
        log_cci = float(STATE.model_pipe.predict(X)[0])
        cci     = float(np.expm1(log_cci))
        cci     = max(cci, 0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # EF and total carbon
    ef      = get_ef(material, grade, unit)
    ef_unit = None
    total_c = None
    if ef and ef > 0:
        ef_unit = f"kgCO2e/{unit}"
        total_c = round(quantity * ef, 2)

    # GGBS scenarios (concrete only)
    scenarios = None
    if material == "concrete" and ef:
        scenarios = ggbs_scenarios(ef, grade, quantity, unit)

    note = (
        f"CCI predicted from BOQ text using XGBoost "
        f"(R²={STATE.model_metadata.get('cv_r2', 'N/A')}, "
        f"MAE={STATE.model_metadata.get('cv_mae', 'N/A'):.1f} INR/kgCO2e). "
        f"Material auto-detected as '{material}'."
    )

    return PredictResponse(
        description=req.description,
        material_detected=material,
        grade=grade,
        unit=unit,
        quantity=quantity,
        emission_factor=round(ef, 4) if ef else None,
        ef_unit=ef_unit,
        predicted_log_cci=round(log_cci, 4),
        predicted_cci=round(cci, 4),
        total_carbon_kgCO2e=total_c,
        ggbs_scenarios=scenarios,
        note=note,
    )


@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided")
    if len(req.items) > 500:
        raise HTTPException(status_code=400,
                            detail="Max 500 items per batch request")

    results = []
    for item in req.items:
        try:
            result = predict_single(item)
            results.append(result.dict())
        except Exception as e:
            results.append({
                "description": item.description,
                "error": str(e),
            })

    return {
        "n_items":    len(req.items),
        "n_success":  sum(1 for r in results if "error" not in r),
        "n_failed":   sum(1 for r in results if "error" in r),
        "predictions": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — starts the server
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PART 7b — BOQ CCI PREDICTION API")
    print("=" * 60)
    print("\n  Starting FastAPI server...")
    print("  API docs : http://localhost:8000/docs")
    print("  Health   : http://localhost:8000/")
    print("  Model    : http://localhost:8000/model-info")
    print("\n  Press Ctrl+C to stop the server.")
    print("=" * 60 + "\n")

    uvicorn.run(
        "part7b_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
    