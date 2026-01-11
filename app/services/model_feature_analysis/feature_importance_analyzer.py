import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
import shap 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FeatureImportanceAnalyzer:
    """
    A class to analyze feature importance using Decision Tree, Random Forest, and LightGBM.

    Attributes:
        data (pd.DataFrame): The dataset loaded from the specified Excel file.
        target_column (str): The name of the target column in the dataset.
        features (pd.DataFrame): Feature columns of the dataset.
        target (pd.Series): Target variable of the dataset.

    Methods:
        decision_tree_importance: Calculate and plot feature importance using a Decision Tree.
        random_forest_importance: Calculate and plot feature importance using Random Forest.
        lightgbm_importance: Calculate and plot feature importance using LightGBM.
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

        self.features = self.data.drop(columns=[target_column])
        self.target = self.data[target_column]

        ''''
        # Drop the specified features: NIHSS and FacialDroop_(weakness)
        features_to_drop = ["NIHSS", "FacialDroop_(weakness)"]
        self.features = self.data.drop(columns=[target_column] + features_to_drop, errors='ignore')
        self.target = self.data[target_column]
        '''
        
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

    def decision_tree_importance(self):
        """
        Calculate and plot feature importance using a Decision Tree.
        """
        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        model.fit(self.X_train, self.y_train)
        self._plot_feature_importance(
            model.feature_importances_,
            title="Decision Tree Feature Importance"
        )

    def random_forest_importance(self):
        """
        Calculate and plot feature importance using Random Forest.
        """
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        model.fit(self.X_train, self.y_train)
        self._plot_feature_importance(
            model.feature_importances_,
            title="Random Forest Feature Importance"
        )

    def lightgbm_importance(self):
        """
        Calculate and plot feature importance using LightGBM.
        """
        model = lgb.LGBMClassifier(
            random_state=42,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_samples=self.min_samples_leaf,  # Corresponds to min_samples_leaf
            verbose=-1  # Suppress additional logs
        )
        model.fit(self.X_train, self.y_train)
        self._plot_feature_importance(
            model.feature_importances_,
            title="LightGBM Feature Importance"
        )

    def _plot_feature_importance(self, importances, title):
        """
        Plot feature importance as a bar chart.

        :param importances: Array of feature importance scores.
        :param title: Title of the plot.
        """
        feature_names = self.features.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], align='center')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()


    def shap_analysis(self, model_name="decision_tree"):
        """
        Perform SHAP analysis for the specified model and visualize the SHAP values.

        :param model_name: The name of the model to use for SHAP analysis. 
                        Options: "decision_tree", "random_forest", "lightgbm".
        """
        # Select the model based on the provided model name
        if model_name == "decision_tree":
            model = DecisionTreeClassifier(
                random_state=42,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                random_state=42,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
        elif model_name == "lightgbm":
            model = lgb.LGBMClassifier(
                random_state=42,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_samples=self.min_samples_leaf,
                verbose=-1
            )
        else:
            raise ValueError("Invalid model_name. Choose from 'decision_tree', 'random_forest', 'lightgbm'.")

        # Train the selected model
        model.fit(self.X_train, self.y_train)

        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values for the test set
        shap_values = explainer.shap_values(self.X_test)

        # Visualize the SHAP summary plot
        # For classification, shap_values may return a list (one for each class); take the class of interest (e.g., 1)
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], self.X_test, feature_names=self.features.columns)
        else:
            shap.summary_plot(shap_values, self.X_test, feature_names=self.features.columns)


    def analyze_feature_correlation(self):
        """
        Analyze the correlation between features and visualize it as a heatmap.
        """
        # Compute the correlation matrix
        correlation_matrix = self.features.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Feature Correlation Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


    def cross_validate_feature_importance(self, model_name="decision_tree", n_splits=5):
        """
        Perform cross-validation to calculate feature importance for stability analysis.

        :param model_name: The name of the model to use for feature importance analysis.
                        Options: "decision_tree", "random_forest", "lightgbm".
        :param n_splits: Number of splits for K-Fold cross-validation.
        """
        # Select the model based on the provided model name
        if model_name == "decision_tree":
            model = DecisionTreeClassifier(
                random_state=42,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                random_state=42,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
        elif model_name == "lightgbm":
            model = lgb.LGBMClassifier(
                random_state=42,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_samples=self.min_samples_leaf,
                verbose=-1
            )
        else:
            raise ValueError("Invalid model_name. Choose from 'decision_tree', 'random_forest', 'lightgbm'.")

        # K-Fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        feature_importances = []

        for train_index, test_index in kf.split(self.features):
            X_train, X_test = self.features.iloc[train_index], self.features.iloc[test_index]
            y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Record feature importances
            if model_name == "lightgbm":
                feature_importances.append(model.feature_importances_)
            else:
                feature_importances.append(model.feature_importances_)

        # Compute the mean and standard deviation of feature importances
        feature_importances = np.array(feature_importances)
        mean_importance = feature_importances.mean(axis=0)
        std_importance = feature_importances.std(axis=0)

        # Display the results
        importance_df = pd.DataFrame({
            'Feature': self.features.columns,
            'Mean Importance': mean_importance,
            'Std Importance': std_importance
        }).sort_values(by='Mean Importance', ascending=False)

        print(importance_df)

        # Plot the mean and standard deviation of feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Mean Importance'], xerr=importance_df['Std Importance'], align='center')
        plt.xlabel("Mean Importance ± Std Dev")
        plt.ylabel("Features")
        plt.title(f"Cross-Validated Feature Importance ({model_name})")
        plt.gca().invert_yaxis()
        plt.show()

if __name__ == "__main__":
    # Replace with your Excel file path
    file_path = "app/services/data/data_clean.xlsx"
    target_column = "Output"

    analyzer = FeatureImportanceAnalyzer(file_path, target_column)

    print("Decision Tree Feature Importance:")
    analyzer.decision_tree_importance()

    print("Random Forest Feature Importance:")
    analyzer.random_forest_importance()

    print("LightGBM Feature Importance:")
    analyzer.lightgbm_importance()


    print("SHAP Analysis for Decision Tree:")
    analyzer.shap_analysis(model_name="decision_tree")

    print("SHAP Analysis for Random Forest:")
    analyzer.shap_analysis(model_name="random_forest")

    print("SHAP Analysis for LightGBM:")
    analyzer.shap_analysis(model_name="lightgbm")


    print("Analyzing Feature Correlation:")
    analyzer.analyze_feature_correlation()


    print("Cross-Validated Feature Importance for Random Forest:")
    analyzer.cross_validate_feature_importance(model_name="random_forest", n_splits=5)

    print("Cross-Validated Feature Importance for LightGBM:")
    analyzer.cross_validate_feature_importance(model_name="lightgbm", n_splits=5)