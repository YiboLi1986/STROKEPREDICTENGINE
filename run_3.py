import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import joblib
import torch
import pandas as pd
from flask import Flask, request, jsonify
from xgboost import XGBClassifier

from app.services.featureset_multimodels.model_trainer_with_reports_3 import NeuralNetwork

app = Flask(__name__)

FEATURE_ORDER = [
    "age", "race", "sex", "HTN", "DM", "HLD", "Smoking", "HxOfStroke", "HxOfAfib", "HxOfSeizure", "SBP", "DBP", "BloodSugar"
]

MODELS_PATH = "app/services/featureset_multimodels/trained_models"

FEATURE_CONSTRAINTS = {
    "age": {"type": (int, float), "range": (0, 120)}, #integer or float
    "race": {"type": str, "range": ("white", "black", "other")},
    "sex": {"type": str, "range": ("M", "F")},
    "HTN": {"type": str, "range": ("True", "False")},
    "DM": {"type": str, "range": ("True", "False")},
    "HLD": {"type": str, "range": ("True", "False")},
    "Smoking": {"type": str, "range": ("True", "False")},
    "HxOfStroke": {"type": str, "range": ("True", "False")},
    "HxOfAfib": {"type": str, "range": ("True", "False")},
    "HxOfPsychIllness": {}, # accept if missing
    "HxOfESRD": {}, # accept if missing
    "HxOfSeizure": {"type": str, "range": ("True", "False")},
    "SBP": {"type": (int, float), "range": (0.0, 600.0)}, #integer or float
    "DBP": {"type": (int, float), "range": (0.0, 600.0)}, #integer or float
    "BloodSugar": {"type": (int, float), "range": (0.0, 1000.0)}, #integer or float
    "NIHSS": {}, # accept if missing
    "FacialDroop (weakness)": {} # accept if missing
}

THRESHOLD = 0.7

def validate_input_features(data, feature_constraints):
    """
    Validate input features based on type and value constraints.

    This function supports two types of validation:
    1. Numerical range validation: For features with int/float type,
       the 'range' should be a tuple of two numbers (min, max), and
       the feature value must fall into the interval (min, max], i.e., min < value <= max.

    2. Categorical value validation: For features with string type,
       the 'range' should be a tuple/list of allowed values.
       The input will be considered valid only if it exactly matches one of these values.

    If a feature is missing from input or has no constraints defined, it is skipped.

    Parameters
    ----------
    data : dict
        The input data containing user-provided feature values.

    feature_constraints : dict
        A dictionary defining validation rules for each feature. Each entry may include:
            - "type": expected Python type (e.g., int, float, str)
            - "range": either a tuple (min, max) for numeric features,
                        or a list/tuple of allowed values for categorical features

    Returns
    -------
    Tuple[bool, str]
        - True and None if validation passes
        - False and an error message string if any validation fails
    """

    for key, constraints in feature_constraints.items():
        if key not in data:
            continue  # Missing feature is acceptable

        if not constraints:
            continue  # No validation rules; skip

        value = data[key]
        expected_type = constraints.get("type")
        value_range = constraints.get("range")

        # Type check
        if expected_type and not isinstance(value, expected_type):
            return False, (
                f"Feature '{key}' should be of type {expected_type.__name__}, "
                f"but got {type(value).__name__}."
            )

        # Range check
        if value_range:
            # Numeric range: (min, max)
            if isinstance(value_range, tuple) and len(value_range) == 2 and all(isinstance(x, (int, float)) for x in value_range):
                min_val, max_val = value_range
                if not (min_val < value <= max_val):  # 左开右闭区间
                    return False, (
                        f"Feature '{key}' value {value} is out of range. "
                        f"Expected in the interval ({min_val}, {max_val}]."
                    )
            elif isinstance(value, str) and all(isinstance(x, str) for x in value_range):
                # Categorical value set: e.g. ("M", "F"), ("True", "False")
                if value.lower() not in {x.lower() for x in value_range}:
                    return False, (
                        f"Feature '{key}' has invalid value '{value}'. "
                        f"Expected one of {value_range} (case-insensitive)."
                    )
            else:
                if value not in value_range:
                    return False, (
                        f"Feature '{key}' has invalid value '{value}'. "
                        f"Expected one of {value_range}."
                    )

    return True, None

def transform_input_data(data: dict) -> dict:
    """
    Transform validated user input data into model-ready numeric format.

    Includes:
    - Mapping categorical values (race, sex)
    - Converting binary features ("True"/"False") to 1/0
    - Scaling numerical features according to predefined rules

    Parameters
    ----------
    data : dict
        Validated input dictionary

    Returns
    -------
    dict
        Transformed and scaled feature dictionary
    """

    # Fixed mappings
    race_map = {"white": 1, "black": 2, "other": 3}
    sex_map = {"m": 1, "f": 2}
    binary_map = {"true": 1, "false": 0}

    # Scaling factors from screenshot
    scaling_factors = {
        "age": 100,
        "race": 10,
        "sex": 10,
        "DM": 10,
        "HTN": 10,
        "HLD": 10,
        "Smoking": 10,
        "HxOfStroke": 10,
        "HxOfAfib": 10,
        "HxOfPsychIllness": 10,
        "HxOfESRD": 10,
        "HxOfSeizure": 10,
        "SBP": 200,
        "DBP": 109,
        "BloodSugar": 109,
        "NIHSS": 10,
        "FacialDroop": 10,
    }

    transformed = {}

    for key, value in data.items():
        # Handle race
        if key == "race":
            value = race_map.get(str(value).lower(), 1)  # default to "white" → 1

        # Handle sex
        elif key == "sex":
            value = sex_map.get(str(value).lower(), 1)  # default to "m" → 1

        # Handle binary features
        elif key in {
            "HTN", "DM", "HLD", "Smoking", "HxOfStroke", "HxOfAfib", "HxOfSeizure"
        }:
            value = binary_map.get(str(value).lower(), 0)

        # Apply scaling
        if key in scaling_factors:
            scale = scaling_factors[key]
            value = float(value) / scale

        transformed[key] = value

    return transformed

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
            input_dim = len(model_path.split(os.sep)[-1].split("_"))
            model = NeuralNetwork(input_dim=input_dim)
            model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
            model.eval()
            models["neural_network"] = model
    return models

def classify_with_threshold(non_p, stroke_p, thr=THRESHOLD):
    """Return classification result for a single model with explanation."""
    if stroke_p >= thr:
        return "stroke", f"stroke_prob={stroke_p:.2f} ≥ {thr}"
    elif non_p >= thr:
        return "non-stroke", f"non-stroke_prob={non_p:.2f} ≥ {thr}"
    else:
        return "uncertainty", f"both probs < {thr} (non={non_p:.2f}, stroke={stroke_p:.2f})"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        is_valid, error_message = validate_input_features(data, FEATURE_CONSTRAINTS)
        if not is_valid:
            return jsonify({"error": error_message}), 400
    
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        valid_data = {k: v for k, v in data.items() if v not in [None, "", "None"]}
        if not valid_data:
            return jsonify({"error": "No valid features with values provided"}), 400

        transformed_data = transform_input_data(valid_data)

        model_folder = find_matching_model_folder(valid_data.keys())
        if not model_folder:
            return jsonify({"error": "No matching model found"}), 400

        models = load_models(model_folder)

        feature_order = model_folder.split(os.sep)[-1].split("_")

        try:
            sorted_values = [transformed_data[key] for key in feature_order]
        except KeyError as e:
            return jsonify({"error": f"Missing required feature: {str(e)}"}), 400

        input_data = pd.DataFrame([sorted_values])

        predictions = {}

        explanations = []

        # For each individual model: only output a class ("stroke" or "non-stroke") if its probability ≥ 0.7; otherwise output "uncertainty".
        if "xgboost" in models:
            y_pred_xgb = models["xgboost"].predict_proba(input_data)
            non_p, stroke_p = float(y_pred_xgb[0][0]), float(y_pred_xgb[0][1])
            cls, reason = classify_with_threshold(non_p, stroke_p)
            predictions["XGB_model_for_stroke_prediction"] = {
                "non-stroke_prob": non_p,
                "stroke_prob": stroke_p,
                "classification": cls
            }
            explanations.append(f"XGBoost → {cls} ({reason})")

        if "random_forest" in models:
            y_pred_rf = models["random_forest"].predict_proba(input_data)
            non_p, stroke_p = float(y_pred_rf[0][0]), float(y_pred_rf[0][1])
            cls, reason = classify_with_threshold(non_p, stroke_p)
            predictions["RF_model_for_stroke_prediction"] = {
                "non-stroke_prob": non_p,
                "stroke_prob": stroke_p,
                "classification": cls
            }
            explanations.append(f"RandomForest → {cls} ({reason})")

        if "neural_network" in models:
            nn_model = models["neural_network"]
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
            y_pred_nn = nn_model(input_tensor).detach().numpy()
            non_p, stroke_p = float(y_pred_nn[0][0]), float(1 - y_pred_nn[0][0])
            cls, reason = classify_with_threshold(non_p, stroke_p)
            predictions["NN_model_for_stroke_prediction"] = {
                "non-stroke_prob": non_p,
                "stroke_prob": stroke_p,
                "classification": cls
            }
            explanations.append(f"NeuralNet → {cls} ({reason})")

        # For the final_classification: only output a class if at least two models agree; otherwise output "uncertainty".
        votes = [pred["classification"] for pred in predictions.values() if pred["classification"] in ("stroke", "non-stroke")]
        if votes.count("stroke") >= 2:
            final_cls = "stroke"
            explanations.append("Final consensus → stroke (at least two models predicted stroke)")
        elif votes.count("non-stroke") >= 2:
            final_cls = "non-stroke"
            explanations.append("Final consensus → non-stroke (at least two models predicted non-stroke)")
        else:
            final_cls = "uncertainty"
            explanations.append("Final consensus → uncertainty (no agreement from at least two models)")

        response = {
            "message": "Data received successfully",
            "received_data": data,
            "predictions": predictions,
            "final_classification": final_cls,
            "explanations": explanations
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
