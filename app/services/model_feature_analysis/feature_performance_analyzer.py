import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

class FeaturePerformanceAnalyzer:
    """
    A class to evaluate the performance of Decision Tree, Random Forest, and LightGBM models.

    Attributes:
        data (pd.DataFrame): The dataset loaded from the specified Excel file.
        target_column (str): The name of the target column in the dataset.
        features (pd.DataFrame): Feature columns of the dataset.
        target (pd.Series): Target variable of the dataset.

    Methods:
        evaluate_decision_tree_performance: Evaluate and print the performance of a Decision Tree.
        evaluate_random_forest_performance: Evaluate and print the performance of a Random Forest.
        evaluate_lightgbm_performance: Evaluate and print the performance of a LightGBM model.
    """

    def __init__(self, file_path, target_column):
        """
        Initialize the class by loading the dataset and splitting it into features and target.

        :param file_path: Path to the Excel file containing the dataset.
        :param target_column: The name of the target column in the dataset.
        """
        # Read the Excel file
        self.data = pd.read_excel(file_path)
        self.target_column = target_column

        # Replace whitespaces in feature names with underscores
        self.data.columns = self.data.columns.str.replace(" ", "_")

        #self.features = self.data.drop(columns=[target_column])
        #self.target = self.data[target_column]

        # Drop the specified features: NIHSS and FacialDroop_(weakness)
        features_to_drop = ["NIHSS", "FacialDroop_(weakness)"]
        self.features = self.data.drop(columns=[target_column] + features_to_drop, errors='ignore')
        self.target = self.data[target_column]

        # Encode categorical target if necessary
        if self.target.dtype == 'object':
            self.target = LabelEncoder().fit_transform(self.target)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )

        # Unified hyperparameter settings
        self.max_depth = 6  # Maximum tree depth
        self.min_samples_leaf = 20  # Minimum number of samples per leaf
        self.n_estimators = 100  # Number of trees for Random Forest and LightGBM

    def evaluate_decision_tree_performance(self):
        """
        Evaluate the performance of a Decision Tree model.
        """
        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        model.fit(self.X_train, self.y_train)
        self._evaluate_model_performance(model, "Decision Tree")

    def evaluate_random_forest_performance(self):
        """
        Evaluate the performance of a Random Forest model.
        """
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        model.fit(self.X_train, self.y_train)
        self._evaluate_model_performance(model, "Random Forest")

    def evaluate_lightgbm_performance(self):
        """
        Evaluate the performance of a LightGBM model.
        """
        model = lgb.LGBMClassifier(
            random_state=42,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_samples=self.min_samples_leaf,
            verbose=-1
        )
        model.fit(self.X_train, self.y_train)
        self._evaluate_model_performance(model, "LightGBM")

    def _evaluate_model_performance(self, model, model_name):
        """
        Helper function to evaluate and print performance metrics for a given model.

        :param model: The trained model.
        :param model_name: The name of the model being evaluated.
        """
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Compute performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="binary")
        recall = recall_score(self.y_test, y_pred, average="binary")
        f1 = f1_score(self.y_test, y_pred, average="binary")
        auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

        # Print results
        print(f"\n{model_name} Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}" if auc != "N/A" else "AUC: Not applicable")

        # Optional: Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

if __name__ == "__main__":
    file_path = "app/services/data/data_clean.xlsx"
    target_column = "Output"

    analyzer = FeaturePerformanceAnalyzer(file_path, target_column)

    print("Evaluating Decision Tree Performance:")
    analyzer.evaluate_decision_tree_performance()

    print("Evaluating Random Forest Performance:")
    analyzer.evaluate_random_forest_performance()

    print("Evaluating LightGBM Performance:")
    analyzer.evaluate_lightgbm_performance()
