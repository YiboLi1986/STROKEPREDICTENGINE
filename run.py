import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import tensorflow as tf
from xgboost import XGBClassifier

app = Flask(__name__)

COLUMNS = [
    "age", "race", "sex", "HTN", "DM", "HLD", "Smoking", "HxOfStroke",
    "HxOfAfib", "HxOfPsychIllness", "HxOfESRD", "HxOfSeizure", 
    "SBP", "DBP", "BloodSugar", "NIHSS", "FacialDroop (weakness)"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        print(data)
        
        try:
            input_data = pd.DataFrame(data, index=[0])
        except KeyError as e:
            missing_cols = set(COLUMNS) - set(data.keys())
            return jsonify({"error": f"Missing required data columns: {', '.join(missing_cols)}"}), 400
        
        ml_model = tf.keras.models.load_model('app/services/models/ML_model_for_stroke_prediction.h5')
        rf_model = joblib.load('app/services/models/RF_model_for_stroke_prediction.pkl')
        xgb_model = XGBClassifier()
        xgb_model.load_model('app/services/models/XGB_model_for_stroke_prediction.json')

        y_pred_ml = ml_model.predict(input_data)
        y_pred_rf = rf_model.predict_proba(input_data)
        y_pred_xgb = xgb_model.predict_proba(input_data)

        threshold = 0.5
        res_ml = "non-stroke" if y_pred_ml[0][0] > threshold else "stroke" if (1 - y_pred_ml[0][0]) > threshold else "uncertainty"
        res_rf = "non-stroke" if y_pred_rf[0][0] > threshold else "stroke" if y_pred_rf[0][1] > threshold else "uncertainty"
        res_xgb = "non-stroke" if y_pred_xgb[0][0] > threshold else "stroke" if y_pred_xgb[0][1] > threshold else "uncertainty"

        response = {
            "message": "Data received successfully",
            "received_data": data,
            "predictions": {
                "ML_model_for_stroke_prediction": {
                    "non-stroke_prob": float(y_pred_ml[0][0]),
                    "stroke_prob": float(1 - y_pred_ml[0][0]),
                    "classification": res_ml
                },
                "RF_model_for_stroke_prediction": {
                    "non-stroke_prob": float(y_pred_rf[0][0]),
                    "stroke_prob": float(y_pred_rf[0][1]),
                    "classification": res_rf
                },
                "XGB_model_for_stroke_prediction": {
                    "non-stroke_prob": float(y_pred_xgb[0][0]),
                    "stroke_prob": float(y_pred_xgb[0][1]),
                    "classification": res_xgb
                }
            }
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)