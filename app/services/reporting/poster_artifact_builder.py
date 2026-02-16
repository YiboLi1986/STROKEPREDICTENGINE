import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


@dataclass
class PosterReportConfig:
    """
    Build poster-ready artifacts (Figure 2 ROC + Figure 3 Confusion Matrices)
    by reading existing evaluation outputs (scores CSVs).

    This module is intentionally "read-only": it does NOT train models.
    """

    # Inputs
    baseline_scores_csv: str = "outputs/roc_eval/scores_baseline_rf.csv"
    routed_scores_csv: str = "outputs/cluster_routed_eval/combined/scores_combined_rf.csv"

    # Output base
    report_base_dir: str = "outputs/reports"
    report_name: str = "aan_2026"

    # Confusion matrix threshold
    decision_threshold: float = 0.5

    # Positive label definition (stroke=1)
    pos_label: int = 1

    # Clinical labels
    negative_label_name: str = "No Stroke"
    positive_label_name: str = "Stroke"

    # ROC plot style
    roc_linewidth: float = 2.5
    show_diagonal: bool = True
    legend_loc: str = "lower right"

    # ROC annotation placement (ALL at lower-right, separated)
    # Axes coordinates (0~1)
    roc_legend_loc: str = "lower right"
    roc_note_box_alpha: float = 0.85

    # Three blocks position (lower-right area)
    # 1) legend already in lower-right
    # 2) delta box near legend, slightly above it
    delta_auc_pos: tuple = (0.62, 0.20)      # right-bottom, above legend
    # 3) test counts near legend, above delta box
    test_counts_pos: tuple = (0.62, 0.30)    # right-bottom, above delta box

    # CM figure style
    cm_dpi: int = 300
    roc_dpi: int = 300


class PosterArtifactBuilder:
    """
    Outputs (under outputs/reports/<report_name>/):
      figures/
        fig2_roc_rf_baseline_vs_routed.png
        fig3_cm_rf_baseline_vs_routed.png
      tables/
        table_auc_comparison.csv
        table_cm_metrics.csv
      summary/
        poster_metrics.json
    """

    def __init__(self, cfg: PosterReportConfig):
        self.cfg = cfg

        # Output directories
        self.out_dir = os.path.join(self.cfg.report_base_dir, self.cfg.report_name)
        self.fig_dir = os.path.join(self.out_dir, "figures")
        self.tbl_dir = os.path.join(self.out_dir, "tables")
        self.sum_dir = os.path.join(self.out_dir, "summary")

        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.tbl_dir, exist_ok=True)
        os.makedirs(self.sum_dir, exist_ok=True)

        # Load score files
        self.baseline = self._load_scores(self.cfg.baseline_scores_csv)
        self.routed = self._load_scores(self.cfg.routed_scores_csv)

        # Validate
        self._validate(self.baseline, self.routed)

        # Cache
        self._test_counts_cache: Optional[Dict[str, int]] = None

    # -----------------------
    # Load / validate
    # -----------------------
    @staticmethod
    def _load_scores(path: str) -> Dict[str, np.ndarray]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scores CSV not found: {path}")

        df = pd.read_csv(path)
        required = {"y_true", "y_score"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Scores CSV must contain columns {sorted(list(required))}. Got: {list(df.columns)}"
            )

        out = {
            "y_true": df["y_true"].astype(int).to_numpy(),
            "y_score": df["y_score"].astype(float).to_numpy(),
        }
        if "row_index" in df.columns:
            out["row_index"] = df["row_index"].astype(int).to_numpy()
        return out

    @staticmethod
    def _validate(b: Dict[str, np.ndarray], r: Dict[str, np.ndarray]) -> None:
        yb, yr = b["y_true"], r["y_true"]

        if len(yb) != len(yr):
            raise ValueError(f"y_true lengths differ: baseline={len(yb)} routed={len(yr)}")

        if not set(np.unique(yb)).issubset({0, 1}) or not set(np.unique(yr)).issubset({0, 1}):
            raise ValueError("Labels must be binary {0,1}.")

        # Loose check: same class counts (covers reordering differences)
        if (yb == 1).sum() != (yr == 1).sum() or (yb == 0).sum() != (yr == 0).sum():
            raise ValueError("Baseline and routed label counts differ — likely not the same test set.")

    # -----------------------
    # Metrics
    # -----------------------
    def _roc(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=self.cfg.pos_label)
        auc = float(roc_auc_score(y_true, y_score))
        return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}

    def _confusion(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
        thr = float(self.cfg.decision_threshold)
        y_pred = (y_score >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = (tp / (tp + fn)) if (tp + fn) else 0.0
        specificity = (tn / (tn + fp)) if (tn + fp) else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        return {
            "threshold": thr,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "accuracy": float(accuracy),
        }

    @staticmethod
    def _counts_from_y(y_true: np.ndarray) -> Dict[str, int]:
        return {
            "N_total": int(len(y_true)),
            "N_stroke": int((y_true == 1).sum()),
            "N_no_stroke": int((y_true == 0).sum()),
        }

    # -----------------------
    # Plotting
    # -----------------------
    def _plot_roc_comparison(self, roc_b: Dict[str, Any], roc_r: Dict[str, Any], out_path: str) -> None:
        """
        Put 3 blocks all in the lower-right corner area WITHOUT overlap:
          - Legend (lower-right)
          - ΔAUC box (just above legend)
          - Test counts box (above ΔAUC)
        """
        auc_b, auc_r = roc_b["auc"], roc_r["auc"]
        delta = auc_r - auc_b

        # Always compute counts from baseline y_true (source of truth for test set)
        y_true = self.baseline["y_true"]
        counts = self._counts_from_y(y_true)

        test_text = f"Test N={counts['N_total']} (Stroke={counts['N_stroke']}, No Stroke={counts['N_no_stroke']})"
        delta_text = f"ΔAUC = {delta:+.3f}"

        plt.figure(figsize=(8, 6))

        # ROC curves
        plt.plot(roc_b["fpr"], roc_b["tpr"], linewidth=self.cfg.roc_linewidth, label=f"Baseline (AUC={auc_b:.3f})")
        plt.plot(roc_r["fpr"], roc_r["tpr"], linewidth=self.cfg.roc_linewidth, label=f"Cohort-Routed (AUC={auc_r:.3f})")

        # Diagonal
        if self.cfg.show_diagonal:
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray")

        plt.xlabel("1 − Specificity (False Positive Rate)", fontsize=12)
        plt.ylabel("Sensitivity (True Positive Rate)", fontsize=12)
        plt.title("ROC Curve – Random Forest (Baseline vs Cohort-Routed)", fontsize=16)

        ax = plt.gca()

        # Legend in lower-right
        leg = ax.legend(
            loc=self.cfg.roc_legend_loc,
            frameon=True,
            fontsize=12,
            borderpad=0.8,
            labelspacing=0.6,
            handlelength=2.5,
            handletextpad=0.8,
        )
        leg.get_frame().set_alpha(0.90)

        # Two separate text boxes near lower-right, but ABOVE legend so they don't cover curves
        ax.text(
            0.98, 0.20,
            delta_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=self.cfg.roc_note_box_alpha),
        )

        ax.text(
            0.98, 0.30,
            test_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=self.cfg.roc_note_box_alpha),
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=self.cfg.roc_dpi)
        plt.close()

    def _plot_cm_side_by_side(self, cm_b: Dict[str, Any], cm_r: Dict[str, Any], out_path: str) -> None:
        """
        Clinical convention:
          - columns = Actual (No Stroke, Stroke)
          - rows    = Predicted (No Stroke, Stroke)

        arr = [[TN, FP],
               [FN, TP]]
        """
        arr_b = np.array([[cm_b["tn"], cm_b["fp"]], [cm_b["fn"], cm_b["tp"]]], dtype=int)
        arr_r = np.array([[cm_r["tn"], cm_r["fp"]], [cm_r["fn"], cm_r["tp"]]], dtype=int)

        thr = float(self.cfg.decision_threshold)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Confusion Matrix – Random Forest (Threshold = {thr:.2f})", fontsize=16)

        neg = self.cfg.negative_label_name
        pos = self.cfg.positive_label_name

        x_tick = [f"Actual {neg}", f"Actual {pos}"]
        y_tick = [f"Predicted {neg}", f"Predicted {pos}"]

        for ax, arr, title, cm in [
            (axes[0], arr_b, "Baseline", cm_b),
            (axes[1], arr_r, "Cohort-Routed", cm_r),
        ]:
            ax.imshow(arr, interpolation="nearest")
            ax.set_title(title, fontsize=14)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(x_tick, fontsize=11)
            ax.set_yticklabels(y_tick, fontsize=11)

            # Cell values
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=14)

            # Metrics box
            text = (
                f"Sensitivity={cm['sensitivity']:.3f}\n"
                f"Specificity={cm['specificity']:.3f}\n"
                f"Accuracy={cm['accuracy']:.3f}\n"
                f"Precision={cm['precision']:.3f}"
            )
            ax.text(
                1.03, 0.5, text,
                transform=ax.transAxes,
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(out_path, dpi=self.cfg.cm_dpi, bbox_inches="tight")
        plt.close()

    # -----------------------
    # Build all artifacts
    # -----------------------
    def build(self) -> Dict[str, Any]:
        y_true = self.baseline["y_true"]
        self._test_counts_cache = self._counts_from_y(y_true)

        roc_b = self._roc(self.baseline["y_true"], self.baseline["y_score"])
        roc_r = self._roc(self.routed["y_true"], self.routed["y_score"])

        cm_b = self._confusion(self.baseline["y_true"], self.baseline["y_score"])
        cm_r = self._confusion(self.routed["y_true"], self.routed["y_score"])

        auc_b, auc_r = roc_b["auc"], roc_r["auc"]
        delta_auc = auc_r - auc_b

        # Tradeoff deltas (interpretation-friendly)
        delta_cm = {
            "delta_tn": int(cm_r["tn"] - cm_b["tn"]),
            "delta_fp": int(cm_r["fp"] - cm_b["fp"]),
            "delta_fn": int(cm_r["fn"] - cm_b["fn"]),
            "delta_tp": int(cm_r["tp"] - cm_b["tp"]),
            "note": "Negative delta_fn means fewer missed strokes (false negatives).",
        }

        # Figures
        fig2 = os.path.join(self.fig_dir, "fig2_roc_rf_baseline_vs_routed.png")
        self._plot_roc_comparison(roc_b, roc_r, fig2)

        fig3 = os.path.join(self.fig_dir, "fig3_cm_rf_baseline_vs_routed.png")
        self._plot_cm_side_by_side(cm_b, cm_r, fig3)

        # Tables
        auc_table = pd.DataFrame(
            [
                {"Model": "Random Forest", "Strategy": "Baseline", "AUC": auc_b},
                {"Model": "Random Forest", "Strategy": "Cohort-Routed", "AUC": auc_r},
                {"Model": "Random Forest", "Strategy": "Δ (Routed - Baseline)", "AUC": delta_auc},
            ]
        )
        auc_csv = os.path.join(self.tbl_dir, "table_auc_comparison.csv")
        auc_table.to_csv(auc_csv, index=False)

        cm_table = pd.DataFrame(
            [
                {"Strategy": "Baseline", **cm_b},
                {"Strategy": "Cohort-Routed", **cm_r},
                {"Strategy": "Δ (Routed - Baseline)", **delta_cm},
            ]
        )
        cm_csv = os.path.join(self.tbl_dir, "table_cm_metrics.csv")
        cm_table.to_csv(cm_csv, index=False)

        # Summary JSON
        summary = {
            "inputs": {
                "baseline_scores_csv": self.cfg.baseline_scores_csv,
                "routed_scores_csv": self.cfg.routed_scores_csv,
            },
            "test_counts": self._test_counts_cache,
            "roc_auc": {
                "baseline": auc_b,
                "routed": auc_r,
                "delta": delta_auc,
            },
            "decision_threshold": self.cfg.decision_threshold,
            "confusion_matrix": {
                "baseline": cm_b,
                "routed": cm_r,
            },
            "clinical_tradeoff": delta_cm,
            "outputs": {
                "report_dir": self.out_dir,
                "figure_roc": fig2,
                "figure_confusion": fig3,
                "table_auc": auc_csv,
                "table_confusion_metrics": cm_csv,
            },
        }

        summary_json = os.path.join(self.sum_dir, "poster_metrics.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        summary["outputs"]["summary_json"] = summary_json
        return summary


if __name__ == "__main__":
    cfg = PosterReportConfig(
        baseline_scores_csv="outputs/roc_eval/scores_baseline_rf.csv",
        routed_scores_csv="outputs/cluster_routed_eval/combined/scores_combined_rf.csv",
        report_base_dir="outputs/reports",
        report_name="aan_2026",
        decision_threshold=0.5,
    )

    builder = PosterArtifactBuilder(cfg)
    out = builder.build()

    print("\n" + "=" * 95)
    print("[DONE] Poster artifacts generated.")
    print("[REPORT DIR]", out["outputs"]["report_dir"])
    print("[FIGURE 2]", out["outputs"]["figure_roc"])
    print("[FIGURE 3]", out["outputs"]["figure_confusion"])
    print("[TABLE AUC]", out["outputs"]["table_auc"])
    print("[TABLE CM ]", out["outputs"]["table_confusion_metrics"])
    print("[SUMMARY  ]", out["outputs"]["summary_json"])
    print("[AUC] baseline:", out["roc_auc"]["baseline"])
    print("[AUC] routed  :", out["roc_auc"]["routed"])
    print("[ΔAUC]       :", out["roc_auc"]["delta"])
    print("[ΔCM]        :", out["clinical_tradeoff"])
    print("=" * 95)
