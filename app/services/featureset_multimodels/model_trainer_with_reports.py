import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.optimizers import adam_v2


class ModelTrainerWithReports:
    """
    A class to train and evaluate Random Forest, XGBoost, and Neural Network models 
    on different feature subsets of a dataset. It stores the models and generates 
    detailed performance reports.
    """

    def __init__(self, file_path: str, label_col: str, feature_sets: list, base_dir="app/services/trained_models"):
        """
        Initializes the ModelTrainerWithReports by loading an Excel file.

        Args:
            file_path (str): Path to the Excel file containing the dataset.
            label_col (str): The name of the label column.
            feature_sets (list): A list of lists, where each inner list contains a subset of features to be used for training.
            base_dir (str): The base directory where all models and reports will be stored.
        """
        self.dataset = pd.read_excel(file_path)
        self.dataset = self.dataset.dropna()

        self.label_col = label_col
        self.feature_sets = feature_sets
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def train_models(self):
        """ Trains Random Forest, XGBoost, and Neural Network on different feature sets and stores results. """
        results = {}

        for i, feature_subset in enumerate(self.feature_sets):
            feature_str = "_".join(feature_subset)
            feature_str = feature_str.replace(" ", "")
            model_dir = os.path.join(self.base_dir, feature_str)
            os.makedirs(model_dir, exist_ok=True)

            print(f"\nTraining models on feature subset {i + 1}/{len(self.feature_sets)}: {feature_subset}")

            # Extract features and labels
            X = self.dataset[feature_subset]
            y = self.dataset[self.label_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize for Neural Network
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            rf_report = self.evaluate_performance(y_test, y_pred_rf)
            joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

            # Train XGBoost
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            xgb_report = self.evaluate_performance(y_test, y_pred_xgb)
            xgb.save_model(os.path.join(model_dir, "xgboost.json"))

            # Train Neural Network
            nn = Sequential([
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            nn.compile(optimizer=adam_v2.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            nn.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=0)
            y_pred_nn = (nn.predict(X_test_scaled) > 0.5).astype(int)
            nn_report = self.evaluate_performance(y_test, y_pred_nn)
            nn.save(os.path.join(model_dir, "neural_network.h5"))

            # Store performance reports
            performance_report = {
                "Random Forest": rf_report,
                "XGBoost": xgb_report,
                "Neural Network": nn_report
            }
            results[feature_str] = performance_report

            # Save performance report as CSV
            self.save_performance_report(performance_report, model_dir)

            print(f"Models and performance report saved to {model_dir}")

        return results

    @staticmethod
    def evaluate_performance(y_true, y_pred):
        """
        Computes evaluation metrics for model performance.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.

        Returns:
            dict: Dictionary containing accuracy, precision, recall, F1-score, and ROC AUC.
        """
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_true, y_pred)
        }

    @staticmethod
    def save_performance_report(report, directory):
        """
        Saves the performance report as a CSV file, including explanations.

        Args:
            report (dict): The performance report.
            directory (str): The directory where the report should be saved.
        """
        # Convert to DataFrame
        df = pd.DataFrame(report).T
        df.loc["Explanation"] = [
            "Overall correctness of the model (correct predictions / total predictions). Higher is better. / 模型整体正确率（正确预测数 / 总预测数）。值越高越好。",
            "Proportion of true positive predictions among all positive predictions (TP / (TP + FP)). Measures the model’s ability to avoid false positives. / 预测为正例的样本中，真正的正例比例（TP / (TP + FP)）。用于衡量模型减少误报的能力。",
            "Proportion of actual positive cases correctly predicted (TP / (TP + FN)). Measures the model’s ability to capture positive instances. / 实际正例中，被正确预测的比例（TP / (TP + FN)）。用于衡量模型减少漏报的能力。",
            "Harmonic mean of Precision and Recall (2 * (Precision * Recall) / (Precision + Recall)). A balanced metric when Precision and Recall are both important. / Precision 和 Recall 的调和平均数（2 * (Precision * Recall) / (Precision + Recall)）。用于平衡 Precision 和 Recall。",
            "Area under the ROC curve, measuring how well the model distinguishes between positive and negative classes. 1.0 means perfect distinction, 0.5 means random guessing. / ROC 曲线下的面积，衡量模型区分正负类的能力。1.0 表示完美，0.5 表示随机猜测。"
        ]
        df.to_csv(os.path.join(directory, "performance_report.csv"))


if __name__ == "__main__":
    file_path = "app/services/data/data_clean.xlsx"
    label_col = "Output"
    feature_sets = [
        ["age", "race", "sex", "HTN", "DM", "HLD", "Smoking", "HxOfStroke", "HxOfAfib", "HxOfPsychIllness", "HxOfESRD", "HxOfSeizure", "SBP", "DBP", "BloodSugar", "NIHSS", "FacialDroop (weakness)"],
    ]

    trainer = ModelTrainerWithReports(file_path, label_col=label_col, feature_sets=feature_sets)
    trainer.train_models()
