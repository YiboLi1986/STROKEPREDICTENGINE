import os
import sys
import pandas as pd
import joblib
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 🔹 PyTorch 神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ModelTrainerWithReports:
    def __init__(self, file_path: str, label_col: str, fixed_features: list, base_dir="app/services/featureset_multimodels/trained_models"):
        self.dataset = pd.read_excel(file_path).dropna()
        self.label_col = label_col
        self.fixed_features = fixed_features
        self.base_dir = base_dir

        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        # 🔧 MODIFIED: 使用新的组合逻辑，允许最多 3 个变量特征缺失
        self.feature_sets = self.generate_feature_sets(max_missing_features=3)

    # 🔧 MODIFIED: 替换为缺失特征组合逻辑
    def generate_feature_sets(self, max_missing_features=3):
        variable_features = ["HTN", "DM", "HLD", "HxOfStroke", "HxOfAfib", "HxOfSeizure"]
        feature_sets = []

        for m in range(0, max_missing_features + 1):
            keep_size = len(variable_features) - m
            for subset in combinations(variable_features, keep_size):
                full_feature_set = self.fixed_features + list(subset)
                feature_sets.append(full_feature_set)

        return feature_sets

    def train_models(self):
        results = {}
        for i, feature_subset in enumerate(self.feature_sets):
            feature_str = "_".join(feature_subset).replace(" ", "")
            model_dir = os.path.join(self.base_dir, feature_str)
            os.makedirs(model_dir, exist_ok=True)

            print(f"\nTraining models on feature subset {i + 1}/{len(self.feature_sets)}: {feature_subset}")
            X = self.dataset[feature_subset].values
            y = self.dataset[self.label_col].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            rf_report = self.evaluate_performance(y_test, y_pred_rf)
            joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

            xgb = XGBClassifier(eval_metric='logloss')
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            xgb_report = self.evaluate_performance(y_test, y_pred_xgb)
            xgb.save_model(os.path.join(model_dir, "xgboost.json"))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NeuralNetwork(input_dim=X_train.shape[1]).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            model.train()
            for epoch in range(15):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred_nn = (model(X_test_tensor) > 0.5).cpu().numpy().astype(int)
            nn_report = self.evaluate_performance(y_test, y_pred_nn)
            torch.save(model.state_dict(), os.path.join(model_dir, "neural_network.pth"))

            performance_report = {
                "Random Forest": rf_report,
                "XGBoost": xgb_report,
                "Neural Network": nn_report
            }
            results[feature_str] = performance_report
            self.save_performance_report(performance_report, model_dir)
            print(f"Models and performance report saved to {model_dir}")

        return results

    @staticmethod
    def evaluate_performance(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_true, y_pred)
        }

    @staticmethod
    def save_performance_report(report, directory):
        df = pd.DataFrame(report).T
        explanation_row = {
            "Accuracy": "Overall correctness of the model (correct predictions / total predictions). Higher is better.",
            "Precision": "Proportion of true positive predictions among all positive predictions.",
            "Recall": "Proportion of actual positive cases correctly predicted.",
            "F1-Score": "Harmonic mean of Precision and Recall.",
            "ROC AUC": "Area under the ROC curve, measuring class separation ability."
        }
        explanation_df = pd.DataFrame(explanation_row, index=["Explanation"])
        df = pd.concat([df, explanation_df])
        os.makedirs(directory, exist_ok=True)
        df.to_csv(os.path.join(directory, "performance_report.csv"), encoding="utf-8-sig")
        print(f"Performance report saved to {directory}/performance_report.csv")


if __name__ == "__main__":
    trainer = ModelTrainerWithReports(
        file_path="app/services/data/data_clean.xlsx",
        label_col="Output",
        fixed_features=["age", "race", "sex", "SBP", "DBP", "BloodSugar", "Smoking"]  # 🔧 MODIFIED: updated fixed features
    )
    trainer.train_models()
