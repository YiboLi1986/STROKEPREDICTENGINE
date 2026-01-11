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
    """ PyTorch 版神经网络模型 """
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

# 🔹 训练管理类
class ModelTrainerWithReports:
    """ 训练 Random Forest, XGBoost 和 PyTorch 神经网络 """

    def __init__(self, file_path: str, label_col: str, fixed_features: list, max_combination_size=3, base_dir="app/services/featureset_multimodels/trained_models"):
        self.dataset = pd.read_excel(file_path).dropna()
        self.label_col = label_col
        self.fixed_features = fixed_features
        self.base_dir = base_dir

        # **删除 `base_dir`，确保每次都是干净的训练环境**
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)  # 递归删除整个目录

        os.makedirs(self.base_dir, exist_ok=True)

        # **自动生成 feature_sets**
        self.feature_sets = self.generate_feature_sets(max_combination_size)

    def generate_feature_sets(self, max_combination_size):
        """
        自动生成 feature_sets（特征组合）。
        """
        all_features = list(self.dataset.columns)
        variable_features = [f for f in all_features if f not in self.fixed_features + [self.label_col]]

        feature_sets = []
        max_size = len(variable_features) if max_combination_size is None else max_combination_size
        for r in range(1, max_size + 1):
            for subset in combinations(variable_features, r):
                feature_sets.append(self.fixed_features + list(subset))

        return feature_sets

    def train_models(self):
        """ 训练所有模型并保存 """
        results = {}

        for i, feature_subset in enumerate(self.feature_sets):
            feature_str = "_".join(feature_subset).replace(" ", "")
            model_dir = os.path.join(self.base_dir, feature_str)
            os.makedirs(model_dir, exist_ok=True)

            print(f"\nTraining models on feature subset {i + 1}/{len(self.feature_sets)}: {feature_subset}")

            # 数据处理
            X = self.dataset[feature_subset].values
            y = self.dataset[self.label_col].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 仅对神经网络标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # **训练 Random Forest**
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            rf_report = self.evaluate_performance(y_test, y_pred_rf)
            joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

            # **训练 XGBoost**
            xgb = XGBClassifier(eval_metric='logloss')
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            xgb_report = self.evaluate_performance(y_test, y_pred_xgb)
            xgb.save_model(os.path.join(model_dir, "xgboost.json"))

            # **训练 PyTorch 神经网络**
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NeuralNetwork(input_dim=X_train.shape[1]).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 转换数据为 PyTorch 张量
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # 训练循环
            model.train()
            for epoch in range(15):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

            # 预测
            model.eval()
            with torch.no_grad():
                y_pred_nn = (model(X_test_tensor) > 0.5).cpu().numpy().astype(int)

            nn_report = self.evaluate_performance(y_test, y_pred_nn)

            # **保存 PyTorch 模型**
            torch.save(model.state_dict(), os.path.join(model_dir, "neural_network.pth"))

            # **保存性能报告**
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
        """
        Saves the performance report as a CSV file, including explanations.
        
        Args:
            report (dict): The performance report.
            directory (str): The directory where the report should be saved.
        """
        # 转换性能报告为 DataFrame
        df = pd.DataFrame(report).T

        # 直接添加一行 Explanation
        explanation_row = {
            "Accuracy": "Overall correctness of the model (correct predictions / total predictions). Higher is better. / 模型整体正确率（正确预测数 / 总预测数）。值越高越好。",
            "Precision": "Proportion of true positive predictions among all positive predictions (TP / (TP + FP)). Measures the model’s ability to avoid false positives. / 预测为正例的样本中，真正的正例比例（TP / (TP + FP)）。用于衡量模型减少误报的能力。",
            "Recall": "Proportion of actual positive cases correctly predicted (TP / (TP + FN)). Measures the model’s ability to capture positive instances. / 实际正例中，被正确预测的比例（TP / (TP + FN)）。用于衡量模型减少漏报的能力。",
            "F1-Score": "Harmonic mean of Precision and Recall (2 * (Precision * Recall) / (Precision + Recall)). A balanced metric when Precision and Recall are both important. / Precision 和 Recall 的调和平均数（2 * (Precision * Recall) / (Precision + Recall)）。用于平衡 Precision 和 Recall。",
            "ROC AUC": "Area under the ROC curve, measuring how well the model distinguishes between positive and negative classes. 1.0 means perfect distinction, 0.5 means random guessing. / ROC 曲线下的面积，衡量模型区分正负类的能力。1.0 表示完美，0.5 表示随机猜测。"
        }

        # 将解释作为新的一行添加
        explanation_df = pd.DataFrame(explanation_row, index=["Explanation"])
        df = pd.concat([df, explanation_df])

        # 确保目录存在
        os.makedirs(directory, exist_ok=True)

        # 保存 CSV，避免乱码问题
        df.to_csv(os.path.join(directory, "performance_report.csv"), encoding="utf-8-sig")

        print(f"Performance report saved to {directory}/performance_report.csv")


if __name__ == "__main__":
    trainer = ModelTrainerWithReports(
        file_path="app/services/data/data_clean.xlsx",
        label_col="Output",
        fixed_features=["age", "sex"],  # 固定特征
        max_combination_size=1 # 限制最大组合特征数
    )
    trainer.train_models()
