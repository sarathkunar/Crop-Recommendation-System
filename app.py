import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory for resolving artifact paths safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifact(filename):
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, "rb") as f:
        return pickle.load(f)

model = load_artifact("crop_model.pkl")
scaler = load_artifact("scaler.pkl")
crop_encoder = load_artifact("crop_encoder.pkl")
season_encoder = load_artifact("season_encoder.pkl")

# Cached valid season list and feature columns
VALID_SEASONS = list(season_encoder.classes_)
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season"]

# Input validation rules
VALIDATION_RULES = {
    "N": {"min": 0.0, "max": 300.0, "name": "Nitrogen (N)"},
    "P": {"min": 0.0, "max": 300.0, "name": "Phosphorus (P)"},
    "K": {"min": 0.0, "max": 300.0, "name": "Potassium (K)"},
    "temperature": {"min": -10.0, "max": 65.0, "name": "Temperature"},
    "humidity": {"min": 0.0, "max": 100.0, "name": "Humidity"},
    "ph": {"min": 0.0, "max": 14.0, "name": "pH"},
    "rainfall": {"min": 0.0, "max": 1000.0, "name": "Rainfall"},
}

def validate_and_extract_inputs(data):
    """Validate numerical and categorical inputs."""
    cleaned = {}
    for field, rule in VALIDATION_RULES.items():
        val = data.get(field)
        if val is None or str(val).strip() == "":
            return None, f"Field '{rule['name']}' is required."
        try:
            num = float(val)
        except (ValueError, TypeError):
            return None, f"Field '{rule['name']}' must be a valid number."
        if num < rule["min"] or num > rule["max"]:
            return None, f"{rule['name']} must be between {rule['min']} and {rule['max']}."
        cleaned[field] = num

    season = data.get("season")
    if not season or str(season).strip() == "":
        return None, "Field 'Season' is required."
    
    season_clean = str(season).strip().capitalize()
    if season_clean not in VALID_SEASONS:
        return None, f"Invalid season '{season}'. Allowed: {', '.join(VALID_SEASONS)}."
    
    cleaned["season"] = season_clean
    return cleaned, None

def run_prediction(features):
    """Run preprocessing, scaling, and inference with probabilistic ranking."""
    encoded_season = season_encoder.transform([features["season"]])[0]
    df_features = pd.DataFrame([{
        "N": features["N"],
        "P": features["P"],
        "K": features["K"],
        "temperature": features["temperature"],
        "humidity": features["humidity"],
        "ph": features["ph"],
        "rainfall": features["rainfall"],
        "season": encoded_season
    }], columns=FEATURE_COLUMNS)

    scaled_data = scaler.transform(df_features)

    # If the model supports probabilities, extract top-3 candidates
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_data)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        
        top_recommendations = []
        for idx in top_indices:
            crop_name = crop_encoder.classes_[idx]
            prob = float(probabilities[idx])
            top_recommendations.append({
                "crop": str(crop_name).capitalize(),
                "confidence_percent": round(prob * 100, 1)
            })

        best_crop = top_recommendations[0]["crop"]
        best_confidence = top_recommendations[0]["confidence_percent"]
        return best_crop, best_confidence, top_recommendations
    else:
        pred_code = model.predict(scaled_data)
        recommended_crop = crop_encoder.inverse_transform(pred_code)[0]
        return str(recommended_crop).capitalize(), 100.0, [{"crop": str(recommended_crop).capitalize(), "confidence_percent": 100.0}]

@app.route("/")
def home():
    return render_template("index.html", seasons=VALID_SEASONS, form_data={})

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for container orchestrators and cloud load balancers."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "features": FEATURE_COLUMNS,
        "available_seasons": VALID_SEASONS
    }), 200

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Returns feature importance ranking and algorithm benchmark results."""
    info = {"model": type(model).__name__, "feature_columns": FEATURE_COLUMNS}
    
    report_path = os.path.join(BASE_DIR, "feature_analysis_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            info["feature_analysis"] = json.load(f)

    comparison_path = os.path.join(BASE_DIR, "model_comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, "r") as f:
            info["algorithm_benchmarks"] = json.load(f)

    return jsonify(info), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Web form submission route."""
    form_dict = request.form.to_dict()
    cleaned_inputs, error = validate_and_extract_inputs(form_dict)

    if error:
        return render_template(
            "index.html",
            error_text=error,
            form_data=form_dict,
            seasons=VALID_SEASONS
        ), 400

    try:
        best_crop, confidence, top_recs = run_prediction(cleaned_inputs)
        return render_template(
            "index.html",
            prediction_text=f"Recommended Crop: {best_crop}",
            confidence=confidence,
            top_recommendations=top_recs,
            form_data=form_dict,
            seasons=VALID_SEASONS
        )
    except Exception as e:
        return render_template(
            "index.html",
            error_text=f"Prediction error: {str(e)}",
            form_data=form_dict,
            seasons=VALID_SEASONS
        ), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic access."""
    if request.is_json:
        payload = request.get_json() or {}
    else:
        payload = request.form.to_dict()

    cleaned_inputs, error = validate_and_extract_inputs(payload)
    if error:
        return jsonify({"status": "error", "message": error}), 400

    try:
        best_crop, confidence, top_recs = run_prediction(cleaned_inputs)
        return jsonify({
            "status": "success",
            "recommended_crop": best_crop,
            "confidence_percent": confidence,
            "top_recommendations": top_recs,
            "inputs": cleaned_inputs
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host=host, port=port, debug=debug)
