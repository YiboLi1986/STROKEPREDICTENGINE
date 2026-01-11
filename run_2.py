import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import joblib
import torch
import pandas as pd
from flask import Flask, request, jsonify
from xgboost import XGBClassifier

app = Flask(__name__)

FEATURE_ORDER = [
    "age", "race", "sex", "HTN", "DM", "HLD", "Smoking", "HxOfStroke",
    "HxOfAfib", "HxOfPsychIllness", "HxOfESRD", "HxOfSeizure", 
    "SBP", "DBP", "BloodSugar", "NIHSS", "FacialDroop (weakness)"
]

MODELS_PATH = "app/services/featureset_multimodels/trained_models"

def clean_key_names(keys):
    return {key.replace(" ", "") for key in keys}

def find_matching_model_folder(keys):
    cleaned_keys = clean_key_names(keys)
    for folder in os.listdir(MODELS_PATH):
        folder_path = os.path.join(MODELS_PATH, folder)
        if os.path.isdir(folder_path):
            folder_keys = set(folder.split("_"))
            if folder_keys == cleaned_keys:
                return folder_path
    return None

def load_models(model_path):
    models = {}
    for file in os.listdir(model_path):
        file_path = os.path.join(model_path, file)
        if file.endswith(".json"):
            xgb_model = XGBClassifier()
            xgb_model.load_model(file_path)
            models["xgboost"] = xgb_model
        elif file.endswith(".pkl"):
            models["random_forest"] = joblib.load(file_path)
        elif file.endswith(".pth"):
            models["neural_network"] = torch.load(file_path, map_location=torch.device("cpu"))
    return models

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        model_folder = find_matching_model_folder(data.keys())
        if not model_folder:
            return jsonify({"error": "No matching model found"}), 400

        models = load_models(model_folder)

        sorted_values = [data[key] for key in FEATURE_ORDER if key in data]
        input_data = pd.DataFrame([sorted_values])

        predictions = {}

        if "xgboost" in models:
            y_pred_xgb = models["xgboost"].predict_proba(input_data)
            predictions["XGB_model_for_stroke_prediction"] = {
                "non-stroke_prob": float(y_pred_xgb[0][0]),
                "stroke_prob": float(y_pred_xgb[0][1]),
                "classification": "stroke" if y_pred_xgb[0][1] > 0.5 else "non-stroke"
            }

        if "random_forest" in models:
            y_pred_rf = models["random_forest"].predict_proba(input_data)
            predictions["RF_model_for_stroke_prediction"] = {
                "non-stroke_prob": float(y_pred_rf[0][0]),
                "stroke_prob": float(y_pred_rf[0][1]),
                "classification": "stroke" if y_pred_rf[0][1] > 0.5 else "non-stroke"
            }

        if "neural_network" in models:
            nn_model = models["neural_network"]
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
            y_pred_nn = nn_model(input_tensor).detach().numpy()
            predictions["NN_model_for_stroke_prediction"] = {
                "non-stroke_prob": float(y_pred_nn[0][0]),
                "stroke_prob": float(1 - y_pred_nn[0][0]),
                "classification": "stroke" if (1 - y_pred_nn[0][0]) > 0.5 else "non-stroke"
            }

        response = {
            "message": "Data received successfully",
            "received_data": data,
            "predictions": predictions
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
