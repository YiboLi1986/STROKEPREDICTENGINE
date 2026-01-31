import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


@dataclass
class RocEvalConfig:
    file_path: str
    label_col: str
    feature_cols: List[str]
    test_size: float = 0.2
    random_state: int = 42
    out_dir: str = "outputs/roc_eval"
    stratify: bool = True


class BaselineRocEvaluator:
    """
    Baseline ROC/AUC evaluator (investor-ready):
    - Fixed held-out test split (saved indices)
    - Per-sample continuous scores (probabilities) saved for test set
    - ROC curves (PNG) + ROC-AUC (JSON)
    - Test size breakdown (N total / N stroke / N non-stroke)

    Outputs under cfg.out_dir:
      - split_indices.npz
      - scores_baseline_{rf|xgb|nn}.csv  (y_true, y_score)
      - roc_baseline_{rf|xgb|nn}.png
      - summary_baseline_roc_auc.json
    """

    def __init__(self, cfg: RocEvalConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        # Load & clean
        self.df = pd.read_excel(self.cfg.file_path).dropna()

        # Basic validation
        missing = [c for c in ([self.cfg.label_col] + self.cfg.feature_cols) if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in dataset: {missing}\n"
                f"Please verify file_path={self.cfg.file_path} and column names."
            )

        self.X = self.df[self.cfg.feature_cols].to_numpy()
        self.y = self.df[self.cfg.label_col].to_numpy()

        # Ensure binary int labels 0/1
        # (If your label is already 0/1, this is fine.)
        try:
            self.y = self.y.astype(int)
        except Exception as e:
            raise ValueError(
                f"Label column '{self.cfg.label_col}' cannot be cast to int. "
                f"Please map labels to 0/1 first. Original error: {e}"
            )

        unique_labels = set(np.unique(self.y).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Label values must be 0/1 for ROC/AUC. Found labels: {sorted(list(unique_labels))}. "
                f"Please map stroke=1, non-stroke=0."
            )

        # Build reproducible split
        idx = np.arange(len(self.df))
        stratify_y = self.y if self.cfg.stratify else None
        self.train_idx, self.test_idx = train_test_split(
            idx,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=stratify_y,
        )

        # Persist split indices
        np.savez(
            os.path.join(self.cfg.out_dir, "split_indices.npz"),
            train_idx=self.train_idx,
            test_idx=self.test_idx,
        )

        self.X_train, self.y_train = self.X[self.train_idx], self.y[self.train_idx]
        self.X_test, self.y_test = self.X[self.test_idx], self.y[self.test_idx]

    def _counts(self) -> Dict[str, int]:
        n_total = int(len(self.y_test))
        n_pos = int((self.y_test == 1).sum())
        n_neg = int((self.y_test == 0).sum())
        return {
            "N_total": n_total,
            "N_stroke(positive)": n_pos,
            "N_nonstroke(negative)": n_neg,
        }

    @staticmethod
    def _build_nn(input_dim: int) -> Sequential:
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _save_scores(self, name: str, y_true: np.ndarray, y_score: np.ndarray) -> str:
        path = os.path.join(self.cfg.out_dir, f"scores_{name}.csv")
        pd.DataFrame({
            "y_true": y_true.astype(int),
            "y_score": y_score.astype(float),
        }).to_csv(path, index=False)
        return path

    def _plot_roc(self, name: str, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name} (stroke=positive)")
        plt.legend(loc="lower right")

        fig_path = os.path.join(self.cfg.out_dir, f"roc_{name}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()

        return {"auc": auc, "roc_png": fig_path}

    def run_all(self, epochs: int = 15, batch_size: int = 32) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "file_path": self.cfg.file_path,
            "label_col": self.cfg.label_col,
            "feature_cols": self.cfg.feature_cols,
            "test_counts": self._counts(),
            "test_split": {
                "test_size": self.cfg.test_size,
                "random_state": self.cfg.random_state,
                "stratify": self.cfg.stratify,
            }
        }

        # -----------------------
        # Random Forest
        # -----------------------
        rf = RandomForestClassifier(n_estimators=100, random_state=self.cfg.random_state)
        rf.fit(self.X_train, self.y_train)
        rf_score = rf.predict_proba(self.X_test)[:, 1]
        self._save_scores("baseline_rf", self.y_test, rf_score)
        report["baseline_rf"] = self._plot_roc("baseline_rf", self.y_test, rf_score)

        # -----------------------
        # XGBoost
        # -----------------------
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.cfg.random_state
        )
        xgb.fit(self.X_train, self.y_train)
        xgb_score = xgb.predict_proba(self.X_test)[:, 1]
        self._save_scores("baseline_xgb", self.y_test, xgb_score)
        report["baseline_xgb"] = self._plot_roc("baseline_xgb", self.y_test, xgb_score)

        # -----------------------
        # Neural Network
        # -----------------------
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(self.X_train)
        X_test_s = scaler.transform(self.X_test)

        nn = self._build_nn(X_train_s.shape[1])
        nn.fit(
            X_train_s, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        nn_score = nn.predict(X_test_s, verbose=0).ravel()
        self._save_scores("baseline_nn", self.y_test, nn_score)
        report["baseline_nn"] = self._plot_roc("baseline_nn", self.y_test, nn_score)

        # Save summary JSON
        summary_path = os.path.join(self.cfg.out_dir, "summary_baseline_roc_auc.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        report["summary_json"] = summary_path
        report["out_dir"] = self.cfg.out_dir
        return report


if __name__ == "__main__":
    # Standalone entry point: run baseline ROC/AUC evaluation and write artifacts.
    cfg = RocEvalConfig(
        file_path="app/services/data/data_clean.xlsx",
        label_col="Output",
        feature_cols=[
            "age", "race", "sex", "HTN", "DM", "HLD", "Smoking",
            "HxOfStroke", "HxOfAfib", "HxOfPsychIllness", "HxOfESRD", "HxOfSeizure",
            "SBP", "DBP", "BloodSugar", "NIHSS", "FacialDroop (weakness)"
        ],
        test_size=0.2,
        random_state=42,
        out_dir="outputs/roc_eval",
        stratify=True
    )

    evaluator = BaselineRocEvaluator(cfg)
    report = evaluator.run_all(epochs=15, batch_size=32)

    print("\n" + "=" * 90)
    print("[DONE] Baseline ROC/AUC evaluation completed.")
    print(f"[OUTPUT DIR] {report['out_dir']}")
    print(f"[SUMMARY JSON] {report['summary_json']}")
    print("[TEST SET COUNTS]", report["test_counts"])
    print("[AUC] baseline_rf :", report["baseline_rf"]["auc"])
    print("[AUC] baseline_xgb:", report["baseline_xgb"]["auc"])
    print("[AUC] baseline_nn :", report["baseline_nn"]["auc"])
    print("=" * 90)