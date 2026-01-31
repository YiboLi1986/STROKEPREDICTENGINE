import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from app.services.processing.interval_divider import IntervalDivider
from app.services.processing.grid_merger import GridMerger
from app.services.processing.grid_processor import GridProcessor
from app.services.processing.gravitational_kmeans import GravitationalKMeans


@dataclass
class RoutedEvalConfig:
    """
    Config for "best/rest + combined routing" evaluation.
    Designed to be investor-ready (scores + ROC/AUC + standard classification metrics).
    """

    # Data
    file_path: str = "app/services/data/data_clean.xlsx"
    label_col: str = "Output"

    # Global split (same idea as baseline evaluator)
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # GKM pipeline params (defaults aligned with your existing script)
    divider: int = 2
    merge_threshold: float = 0.3

    len_threshold: int = 2
    len_threshold_0: int = 2
    len_threshold_1: int = 3
    num_threshold_0: float = 0.7
    num_threshold_1: float = 0.7

    enhancement_factor: float = 0.0
    eval_every: int = 3
    eval_stop_at: Optional[int] = 200

    # Training params
    nn_epochs: int = 15
    nn_batch_size: int = 32
    decision_threshold: float = 0.5

    # Output
    out_dir: str = "outputs/cluster_routed_eval"

    # If baseline split exists, prefer reusing it for fair comparison
    baseline_split_path: str = "outputs/roc_eval/split_indices.npz"


class ClusterRoutedEvaluator:
    """
    Cluster-routed evaluation (no change to your existing GKM code):

    Step A: Use your existing GKM pipeline to find best_clusters.
            best_set = union(best_clusters)

    Step B: Use a fixed held-out split (prefer baseline split if present),
            then form cohort splits:
              - best cohort: indices in best_set ∩ {train/test}
              - rest cohort: indices in rest_set ∩ {train/test}

    Step C: Train+evaluate RF / XGB / NN separately on:
              - best cohort
              - rest cohort
            Each produces:
              - per-sample continuous scores (CSV)
              - ROC curve (PNG) + ROC-AUC (JSON)
              - standard metrics (Accuracy/Precision/Recall/F1/Sensitivity/Specificity/Confusion)

    Step D: Combined routing evaluation on the FULL test set:
              - if sample is in best cohort -> use best model score
              - else -> use rest model score
            Then compute the same metrics/ROC/AUC.

    Outputs under cfg.out_dir:
      - best_set_indices.json
      - split_indices.npz (if baseline split absent)
      - best/  (scores_*.csv, roc_*.png, summary_best.json)
      - rest/  (scores_*.csv, roc_*.png, summary_rest.json)
      - combined/ (scores_*.csv, roc_*.png, summary_combined_*.json)
      - summary_all.json
    """

    def __init__(self, cfg: RoutedEvalConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        # Load & validate data
        self.df = pd.read_excel(self.cfg.file_path).dropna()
        if self.cfg.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.cfg.label_col}' not found in: {self.cfg.file_path}")

        # Features: all columns except label (aligned with your GKM usage)
        self.feature_cols = [c for c in self.df.columns if c != self.cfg.label_col]
        self.X = self.df[self.feature_cols].to_numpy()

        # Labels must be 0/1 for ROC/AUC (stroke=1 is positive)
        self.y = self.df[self.cfg.label_col].to_numpy()
        try:
            self.y = self.y.astype(int)
        except Exception as e:
            raise ValueError(f"Label column '{self.cfg.label_col}' must be castable to int. Error: {e}")

        unique_labels = set(np.unique(self.y).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Label must be 0/1 (stroke=1, nonstroke=0). Found labels: {sorted(list(unique_labels))}"
            )

        # Fixed split indices (reuse baseline split if available)
        self.train_idx, self.test_idx = self._load_or_create_split()

        # Compute best_set indices via existing GKM pipeline
        self.best_set = self._compute_best_set_indices()
        self.rest_set = set(range(len(self.df))) - self.best_set

        # Cohort-specific splits = intersection with global split
        self.best_train_idx = self._intersect(self.train_idx, self.best_set)
        self.best_test_idx = self._intersect(self.test_idx, self.best_set)

        self.rest_train_idx = self._intersect(self.train_idx, self.rest_set)
        self.rest_test_idx = self._intersect(self.test_idx, self.rest_set)

        # Basic sanity
        if len(self.best_train_idx) == 0 or len(self.best_test_idx) == 0:
            raise ValueError(
                f"Best cohort has empty train/test split: "
                f"best_train={len(self.best_train_idx)}, best_test={len(self.best_test_idx)}"
            )
        if len(self.rest_train_idx) == 0 or len(self.rest_test_idx) == 0:
            raise ValueError(
                f"Rest cohort has empty train/test split: "
                f"rest_train={len(self.rest_train_idx)}, rest_test={len(self.rest_test_idx)}"
            )

    # ---------------------------------------------------------------------
    # Split helpers
    # ---------------------------------------------------------------------
    def _load_or_create_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prefer baseline split indices for apples-to-apples comparison.
        If not found, create a new split and save into cfg.out_dir/split_indices.npz.
        """
        if os.path.exists(self.cfg.baseline_split_path):
            data = np.load(self.cfg.baseline_split_path)
            train_idx = data["train_idx"]
            test_idx = data["test_idx"]
            return train_idx, test_idx

        idx = np.arange(len(self.df))
        stratify_y = self.y if self.cfg.stratify else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=stratify_y,
        )
        np.savez(os.path.join(self.cfg.out_dir, "split_indices.npz"), train_idx=train_idx, test_idx=test_idx)
        return train_idx, test_idx

    @staticmethod
    def _intersect(split_idx: np.ndarray, cohort_set: set) -> np.ndarray:
        return np.array([int(i) for i in split_idx if int(i) in cohort_set], dtype=int)

    # ---------------------------------------------------------------------
    # GKM: best cluster set
    # ---------------------------------------------------------------------
    def _compute_best_set_indices(self) -> set:
        """
        Call your existing GKM pipeline (no code changes) to get best_clusters.
        Return union(best_clusters) as a set of row indices in the full dataset.
        """
        # (1) interval divider
        interval_divider = IntervalDivider(self.cfg.file_path, self.cfg.divider)
        grid_map = interval_divider.grid_map

        # (2) merge grids until threshold
        merger = GridMerger(grid_map, self.cfg.merge_threshold)
        final_grid_map, histograms = merger.merge_until_threshold(interval_divider.data)

        # (3) majority grid selection
        processor = GridProcessor(
            final_grid_map,
            histograms,
            self.cfg.file_path,
            self.cfg.len_threshold,
            self.cfg.len_threshold_0,
            self.cfg.len_threshold_1,
            self.cfg.num_threshold_0,
            self.cfg.num_threshold_1,
        )
        new_final_grid_map = processor.select_majority_grids()

        initial_clusters: List[List[int]] = [samples for _, samples in new_final_grid_map.items()]
        if len(initial_clusters) == 0:
            raise ValueError("initial_clusters is empty. Check GridProcessor.select_majority_grids() output.")

        # (4) run GKM incremental + track best_clusters internally
        gkm = GravitationalKMeans(
            pd.read_excel(self.cfg.file_path).dropna(),
            initial_clusters,
            enhancement_factor=self.cfg.enhancement_factor,
            eval_every=self.cfg.eval_every,
            eval_stop_at=self.cfg.eval_stop_at,
        )
        gkm.fit_incremental()

        best_clusters = getattr(gkm, "best_clusters", None)
        if not best_clusters or not isinstance(best_clusters, list):
            raise ValueError("best_clusters not found / invalid from GravitationalKMeans.")

        best_set: set = set()
        for cl in best_clusters:
            for i in cl:
                best_set.add(int(i))

        # Persist for traceability
        path = os.path.join(self.cfg.out_dir, "best_set_indices.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_set_size": int(len(best_set)),
                    "best_set_indices": sorted(list(best_set)),
                    "gkm_params": {
                        "divider": self.cfg.divider,
                        "merge_threshold": self.cfg.merge_threshold,
                        "len_threshold": self.cfg.len_threshold,
                        "len_threshold_0": self.cfg.len_threshold_0,
                        "len_threshold_1": self.cfg.len_threshold_1,
                        "num_threshold_0": self.cfg.num_threshold_0,
                        "num_threshold_1": self.cfg.num_threshold_1,
                        "enhancement_factor": self.cfg.enhancement_factor,
                        "eval_every": self.cfg.eval_every,
                        "eval_stop_at": self.cfg.eval_stop_at,
                    },
                },
                f,
                indent=2,
            )

        return best_set

    # ---------------------------------------------------------------------
    # Model builders
    # ---------------------------------------------------------------------
    @staticmethod
    def _build_nn(input_dim: int) -> Sequential:
        model = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(64, activation="relu"),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    # ---------------------------------------------------------------------
    # Metrics / ROC helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _counts(y_true: np.ndarray) -> Dict[str, int]:
        return {
            "N_total": int(len(y_true)),
            "N_stroke(positive)": int((y_true == 1).sum()),
            "N_nonstroke(negative)": int((y_true == 0).sum()),
        }

    def _compute_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
        thr = float(self.cfg.decision_threshold)
        y_pred = (y_score >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total = tp + tn + fp + fn

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        sen = (tp / (tp + fn)) if (tp + fn) else 0.0  # Sensitivity / TPR
        spe = (tn / (tn + fp)) if (tn + fp) else 0.0  # Specificity / TNR

        auc = None
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_score))

        return {
            "threshold": thr,
            "counts": self._counts(y_true),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "sensitivity": float(sen),
            "specificity": float(spe),
            "roc_auc": auc,
        }

    @staticmethod
    def _save_scores(out_dir: str, name: str, row_idx: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"scores_{name}.csv")
        pd.DataFrame(
            {"row_index": row_idx.astype(int), "y_true": y_true.astype(int), "y_score": y_score.astype(float)}
        ).to_csv(path, index=False)
        return path

    @staticmethod
    def _plot_roc(out_dir: str, name: str, y_true: np.ndarray, y_score: np.ndarray) -> Optional[str]:
        if len(np.unique(y_true)) < 2:
            return None

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name} (stroke=positive)")
        plt.legend(loc="lower right")

        fig_path = os.path.join(out_dir, f"roc_{name}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        return fig_path

    # ---------------------------------------------------------------------
    # Training functions (return continuous scores on given test set)
    # ---------------------------------------------------------------------
    def _rf_scores(self, X_train, y_train, X_test) -> np.ndarray:
        model = RandomForestClassifier(n_estimators=100, random_state=self.cfg.random_state)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    def _xgb_scores(self, X_train, y_train, X_test) -> np.ndarray:
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=self.cfg.random_state)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    def _nn_scores(self, X_train, y_train, X_test) -> np.ndarray:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        nn = self._build_nn(X_train_s.shape[1])
        nn.fit(
            X_train_s,
            y_train,
            epochs=self.cfg.nn_epochs,
            batch_size=self.cfg.nn_batch_size,
            validation_split=0.2,
            verbose=0,
        )
        return nn.predict(X_test_s, verbose=0).ravel()

    # ---------------------------------------------------------------------
    # Cohort evaluation
    # ---------------------------------------------------------------------
    def _eval_cohort(self, cohort_name: str, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
        out_dir = os.path.join(self.cfg.out_dir, cohort_name)
        os.makedirs(out_dir, exist_ok=True)

        X_train, y_train = self.X[train_idx], self.y[train_idx]
        X_test, y_test = self.X[test_idx], self.y[test_idx]

        result: Dict[str, Any] = {
            "cohort": cohort_name,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "test_counts": self._counts(y_test),
            "decision_threshold": float(self.cfg.decision_threshold),
        }

        # RF
        rf_score = self._rf_scores(X_train, y_train, X_test)
        rf_scores_path = self._save_scores(out_dir, f"{cohort_name}_rf", test_idx, y_test, rf_score)
        rf_roc_path = self._plot_roc(out_dir, f"{cohort_name}_rf", y_test, rf_score)
        result["rf"] = {"scores_csv": rf_scores_path, "roc_png": rf_roc_path, "metrics": self._compute_metrics(y_test, rf_score)}

        # XGB
        xgb_score = self._xgb_scores(X_train, y_train, X_test)
        xgb_scores_path = self._save_scores(out_dir, f"{cohort_name}_xgb", test_idx, y_test, xgb_score)
        xgb_roc_path = self._plot_roc(out_dir, f"{cohort_name}_xgb", y_test, xgb_score)
        result["xgb"] = {"scores_csv": xgb_scores_path, "roc_png": xgb_roc_path, "metrics": self._compute_metrics(y_test, xgb_score)}

        # NN
        nn_score = self._nn_scores(X_train, y_train, X_test)
        nn_scores_path = self._save_scores(out_dir, f"{cohort_name}_nn", test_idx, y_test, nn_score)
        nn_roc_path = self._plot_roc(out_dir, f"{cohort_name}_nn", y_test, nn_score)
        result["nn"] = {"scores_csv": nn_scores_path, "roc_png": nn_roc_path, "metrics": self._compute_metrics(y_test, nn_score)}

        # Persist cohort summary
        with open(os.path.join(out_dir, f"summary_{cohort_name}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

    # ---------------------------------------------------------------------
    # Combined routing evaluation on global test set
    # ---------------------------------------------------------------------
    def _combined_eval(
        self,
        model_key: str,
        best_scores: np.ndarray,
        rest_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Create a combined score vector aligned with global test_idx ordering:
          - if row_index in best cohort -> use best_scores at that row
          - else -> use rest_scores at that row
        """
        out_dir = os.path.join(self.cfg.out_dir, "combined")
        os.makedirs(out_dir, exist_ok=True)

        # Build mapping row_index -> score for each cohort
        best_map = {int(i): float(s) for i, s in zip(self.best_test_idx, best_scores)}
        rest_map = {int(i): float(s) for i, s in zip(self.rest_test_idx, rest_scores)}

        test_idx = self.test_idx
        y_true = self.y[test_idx]
        y_score = np.zeros(len(test_idx), dtype=float)

        for k, row_i in enumerate(test_idx):
            ri = int(row_i)
            if ri in best_map:
                y_score[k] = best_map[ri]
            else:
                y_score[k] = rest_map[ri]

        name = f"combined_{model_key}"
        scores_path = self._save_scores(out_dir, name, test_idx, y_true, y_score)
        roc_path = self._plot_roc(out_dir, name, y_true, y_score)

        out = {
            "name": name,
            "scores_csv": scores_path,
            "roc_png": roc_path,
            "metrics": self._compute_metrics(y_true, y_score),
        }

        with open(os.path.join(out_dir, f"summary_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        return out

    # ---------------------------------------------------------------------
    # Public runner
    # ---------------------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        """
        Run best/rest training + combined routing evaluation, then write summary_all.json.
        """
        # (1) evaluate cohorts (this trains models and produces cohort outputs)
        best_res = self._eval_cohort("best", self.best_train_idx, self.best_test_idx)
        rest_res = self._eval_cohort("rest", self.rest_train_idx, self.rest_test_idx)

        # (2) for combined evaluation, compute scores for each cohort/model again (simple, consistent)
        #     (We intentionally re-train here to keep this module standalone and robust.)
        Xb_tr, yb_tr = self.X[self.best_train_idx], self.y[self.best_train_idx]
        Xb_te, yb_te = self.X[self.best_test_idx], self.y[self.best_test_idx]
        Xr_tr, yr_tr = self.X[self.rest_train_idx], self.y[self.rest_train_idx]
        Xr_te, yr_te = self.X[self.rest_test_idx], self.y[self.rest_test_idx]

        # Best cohort scores
        best_rf = self._rf_scores(Xb_tr, yb_tr, Xb_te)
        best_xgb = self._xgb_scores(Xb_tr, yb_tr, Xb_te)
        best_nn = self._nn_scores(Xb_tr, yb_tr, Xb_te)

        # Rest cohort scores
        rest_rf = self._rf_scores(Xr_tr, yr_tr, Xr_te)
        rest_xgb = self._xgb_scores(Xr_tr, yr_tr, Xr_te)
        rest_nn = self._nn_scores(Xr_tr, yr_tr, Xr_te)

        combined_rf = self._combined_eval("rf", best_rf, rest_rf)
        combined_xgb = self._combined_eval("xgb", best_xgb, rest_xgb)
        combined_nn = self._combined_eval("nn", best_nn, rest_nn)

        summary: Dict[str, Any] = {
            "file_path": self.cfg.file_path,
            "label_col": self.cfg.label_col,
            "feature_cols": self.feature_cols,
            "global_test_counts": self._counts(self.y[self.test_idx]),
            "test_split": {
                "test_size": self.cfg.test_size,
                "random_state": self.cfg.random_state,
                "stratify": self.cfg.stratify,
                "baseline_split_reused": os.path.exists(self.cfg.baseline_split_path),
            },
            "best_set_size": int(len(self.best_set)),
            "rest_set_size": int(len(self.rest_set)),
            "cohorts": {
                "best": {
                    "train_size": int(len(self.best_train_idx)),
                    "test_size": int(len(self.best_test_idx)),
                    "test_counts": self._counts(self.y[self.best_test_idx]),
                },
                "rest": {
                    "train_size": int(len(self.rest_train_idx)),
                    "test_size": int(len(self.rest_test_idx)),
                    "test_counts": self._counts(self.y[self.rest_test_idx]),
                },
            },
            "best": best_res,
            "rest": rest_res,
            "combined": {
                "rf": combined_rf,
                "xgb": combined_xgb,
                "nn": combined_nn,
            },
        }

        with open(os.path.join(self.cfg.out_dir, "summary_all.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


if __name__ == "__main__":
    cfg = RoutedEvalConfig(
        file_path="app/services/data/data_clean.xlsx",
        label_col="Output",
        out_dir="outputs/cluster_routed_eval",
        random_state=42,
        test_size=0.2,
        stratify=True,
        eval_every=3,
        eval_stop_at=200,
        decision_threshold=0.5,
        nn_epochs=15,
        nn_batch_size=32,
    )

    evaluator = ClusterRoutedEvaluator(cfg)
    report = evaluator.run_all()

    print("\n" + "=" * 95)
    print("[DONE] Cluster-routed evaluation completed.")
    print(f"[OUTPUT DIR] {cfg.out_dir}")
    print("[GLOBAL TEST COUNTS]", report["global_test_counts"])
    print("[BEST SET SIZE]", report["best_set_size"], " | [REST SET SIZE]", report["rest_set_size"])
    print("[COMBINED AUC] RF :", report["combined"]["rf"]["metrics"]["roc_auc"])
    print("[COMBINED AUC] XGB:", report["combined"]["xgb"]["metrics"]["roc_auc"])
    print("[COMBINED AUC] NN :", report["combined"]["nn"]["metrics"]["roc_auc"])
    print("=" * 95)
