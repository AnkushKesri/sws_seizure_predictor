"""
main.py -- FastAPI backend for Sturge-Weber Seizure Onset Prediction
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------
# APP SETUP
# ----------------------------------------------------------------

app = FastAPI(
    title="Sturge-Weber Seizure Onset Predictor",
    description="Predicts 2-year seizure onset risk using a 15-model ensemble.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# FEATURE METADATA
# ----------------------------------------------------------------

FEATURE_METADATA = {
    "ethnicity": {
        "label": "Ethnicity (Hispanic or Latino)",
        "options": {0: "No", 1: "Yes"},
        "depends_on": None
    },
    "treatment": {
        "label": "Presymptomatic Treatment done for SWS",
        "options": {0: "No", 1: "Yes"},
        "depends_on": None
    },
    "forehead": {
        "label": "Forehead involved in Port Wine Birthmark",
        "options": {0: "No", 1: "Yes"},
        "depends_on": None
    },
    "treatments___1": {
        "label": "Treatment with Low-dose Aspirin",
        "options": {0: "No", 1: "Yes"},
        "depends_on": "treatment"
    },
    "sex": {
        "label": "Sex",
        "options": {0: "Female", 1: "Male"},
        "depends_on": None
    },
}

# ----------------------------------------------------------------
# LOAD MODELS AND CONFIG ON STARTUP
# ----------------------------------------------------------------

MODELS_DIR = os.getenv("MODELS_DIR", "saved_models")

MODEL_TYPES = {
    "Random Forest": "random_forest",
    "Gaussian SVC":  "gaussian_svc",
    "Linear SVC":    "linear_svc",
}

def load_assets():
    config_path = os.path.join(MODELS_DIR, "feature_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"feature_config.json not found in '{MODELS_DIR}'."
        )

    with open(config_path) as f:
        feature_config = json.load(f)

    n_folds = feature_config["nFold"]
    models  = {}

    for fold in range(1, n_folds + 1):
        for display_name, file_key in MODEL_TYPES.items():
            filename = f"fold{fold}_{file_key}.joblib"
            path     = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            models[(fold, file_key)] = joblib.load(path)

    print(f"Loaded {len(models)} models from '{MODELS_DIR}'")
    print(f"Features: {feature_config['final_features']}")
    return models, feature_config


MODELS, FEATURE_CONFIG = load_assets()

# ----------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ----------------------------------------------------------------

class PredictRequest(BaseModel):
    ethnicity:      int
    treatment:      int
    forehead:       int
    treatments___1: int
    sex:            int

class ModelTypeResult(BaseModel):
    fold_probabilities:  list
    average_probability: float
    prediction:          int

class PredictResponse(BaseModel):
    prediction:       int
    prediction_label: str
    probability:      float
    risk_level:       str
    model_breakdown:  dict
    disclaimer:       str

# ----------------------------------------------------------------
# INFERENCE LOGIC
# ----------------------------------------------------------------

def run_inference(feature_values: dict) -> PredictResponse:
    final_features = FEATURE_CONFIG["final_features"]
    n_folds        = FEATURE_CONFIG["nFold"]

    # Enforce dependency: treatments___1 must be 0 if treatment is 0
    if feature_values.get("treatment", 0) == 0:
        feature_values["treatments___1"] = 0

    # Validate all values are 0 or 1
    for feat, val in feature_values.items():
        if val not in (0, 1):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{feat}' must be 0 or 1, got {val}."
            )

    # Build input DataFrame in the exact feature order used during training
    try:
        input_df = pd.DataFrame([{feat: feature_values[feat] for feat in final_features}])
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    # Method 2 aggregation:
    # Step 1 - collect per-fold probs grouped by model type
    # Step 2 - average within each type  -> 3 type-level probs
    # Step 3 - average across types      -> final probability
    # Step 4 - majority vote on type-level probs >= 0.5 -> final prediction

    type_probs = {file_key: [] for file_key in MODEL_TYPES.values()}

    for fold in range(1, n_folds + 1):
        for file_key in MODEL_TYPES.values():
            prob = float(MODELS[(fold, file_key)].predict_proba(input_df)[0][1])
            type_probs[file_key].append(prob)

    type_avg = {
        file_key: float(np.mean(probs))
        for file_key, probs in type_probs.items()
    }

    final_prob       = round(float(np.mean(list(type_avg.values()))), 3)
    type_predictions = [1 if p >= 0.5 else 0 for p in type_avg.values()]
    final_prediction = 1 if sum(type_predictions) >= 2 else 0

    if final_prob >= 0.65:
        risk_level = "High"
    elif final_prob >= 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    file_key_to_display = {v: k for k, v in MODEL_TYPES.items()}
    model_breakdown = {
        file_key_to_display[file_key]: ModelTypeResult(
            fold_probabilities=[round(p, 3) for p in type_probs[file_key]],
            average_probability=round(type_avg[file_key], 3),
            prediction=1 if type_avg[file_key] >= 0.5 else 0
        )
        for file_key in MODEL_TYPES.values()
    }

    return PredictResponse(
        prediction=final_prediction,
        prediction_label="Seizure Onset Likely" if final_prediction == 1 else "Seizure Onset Unlikely",
        probability=final_prob,
        risk_level=risk_level,
        model_breakdown=model_breakdown,
        disclaimer=(
            "This tool is intended for research and clinical decision support only. "
            "It does not replace clinical judgment. Always consult a qualified physician."
        )
    )

# ----------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "running", "tool": "Sturge-Weber Seizure Onset Predictor"}


@app.get("/features")
def get_features():
    return {
        "features":      FEATURE_METADATA,
        "feature_order": FEATURE_CONFIG.get("final_features", []),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    feature_values = {
        "ethnicity":      request.ethnicity,
        "treatment":      request.treatment,
        "forehead":       request.forehead,
        "treatments___1": request.treatments___1,
        "sex":            request.sex,
    }
    return run_inference(feature_values)


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": len(MODELS),
        "n_folds":       FEATURE_CONFIG.get("nFold", 0),
        "features":      FEATURE_CONFIG.get("final_features", []),
    }
