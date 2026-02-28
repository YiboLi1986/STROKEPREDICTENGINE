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
    Build poster-ready artifacts (ROC + Confusion Matrices)
    by reading existing evaluation outputs (scores CSVs).

    This module is intentionally "read-only": it does NOT train models.
    """

    # Inputs
    baseline_scores_csv: str = "outputs/roc_eval/scores_baseline_rf.csv"
    routed_scores_csv: str = "outputs/cluster_routed_eval/combined/scores_combined_rf.csv"

    # Output base
    report_base_dir: str = "outputs/reports"
    report_name: str = "aan_2026"

    # Decision threshold for confusion matrix operating point
    decision_threshold: float = 0.5

    # Positive label definition (stroke=1)
    pos_label: int = 1

    # Clinical labels
    negative_label_name: str = "No Stroke"
    positive_label_name: str = "Stroke"

    # ROC plot style
    roc_linewidth: float = 2.5
    show_diagonal: bool = True
    roc_legend_loc: str = "lower right"
    roc_note_box_alpha: float = 0.85

    # Output DPI (PNG only)
    cm_dpi: int = 300
    roc_dpi: int = 300

    # Export formats (to satisfy "editable ROC" request)
    export_png: bool = True
    export_svg: bool = True
    export_pdf: bool = True


class PosterArtifactBuilder:
    """
    Outputs (under outputs/reports/<report_name>/):
      figures/
        fig2_roc_rf_baseline_vs_routed.png (+ .svg + .pdf)
        fig3_cm_rf_baseline_vs_routed.png  (+ .svg + .pdf)
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

        # Clinical definitions
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
    # Save helper (multi-format)
    # -----------------------
    def _save_multi(self, fig: plt.Figure, out_png_path: str, dpi: int) -> None:
        base, _ = os.path.splitext(out_png_path)

        if self.cfg.export_png:
            fig.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
        if self.cfg.export_svg:
            fig.savefig(base + ".svg", bbox_inches="tight")
        if self.cfg.export_pdf:
            fig.savefig(base + ".pdf", bbox_inches="tight")

    # -----------------------
    # Plotting: ROC
    # -----------------------
    def _plot_roc_comparison(self, roc_b: Dict[str, Any], roc_r: Dict[str, Any], out_path: str) -> None:
        """
        ROC curve figure + editable exports (SVG/PDF).
        """
        auc_b, auc_r = roc_b["auc"], roc_r["auc"]
        delta = auc_r - auc_b

        # Counts from baseline (source of truth)
        counts = self._counts_from_y(self.baseline["y_true"])
        test_text = f"Test N={counts['N_total']} (Stroke={counts['N_stroke']}, No Stroke={counts['N_no_stroke']})"
        delta_text = f"ΔAUC = {delta:+.3f}"

        fig = plt.figure(figsize=(8, 6))

        plt.plot(roc_b["fpr"], roc_b["tpr"], linewidth=self.cfg.roc_linewidth,
                 label=f"Baseline (AUC={auc_b:.3f})")
        plt.plot(roc_r["fpr"], roc_r["tpr"], linewidth=self.cfg.roc_linewidth,
                 label=f"Cohort-Routed (AUC={auc_r:.3f})")

        if self.cfg.show_diagonal:
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray")

        plt.xlabel("1 − Specificity (False Positive Rate)", fontsize=12)
        plt.ylabel("Sensitivity (True Positive Rate)", fontsize=12)
        plt.title("ROC Curve – Random Forest (Baseline vs Cohort-Routed)", fontsize=16)

        ax = plt.gca()

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
        self._save_multi(fig, out_path, dpi=self.cfg.roc_dpi)
        plt.close(fig)

    # -----------------------
    # Plotting: Confusion Matrix (Clinical Table)
    # -----------------------
    def _plot_cm_side_by_side(self, cm_b: Dict[str, Any], cm_r: Dict[str, Any], out_path: str) -> None:
        """
        Clinical-table confusion matrices:
          columns = Actual (No Stroke, Stroke, Total)
          rows    = Predicted (No Stroke, Stroke, Total)

        This avoids heatmap colors and makes totals obvious (N, Stroke, No Stroke).
        """
        thr = float(self.cfg.decision_threshold)

        neg = self.cfg.negative_label_name
        pos = self.cfg.positive_label_name

        # build 3x3 table: rows=Pred, cols=Actual, with totals
        def build_table(cm: Dict[str, Any]) -> np.ndarray:
            tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
            return np.array([
                [tn, fn, tn + fn],                      # Pred NS
                [fp, tp, fp + tp],                      # Pred S
                [tn + fp, fn + tp, tn + fp + fn + tp],  # Totals
            ], dtype=int)

        tab_b = build_table(cm_b)
        tab_r = build_table(cm_r)

        # Template-like light green styling
        header_green = "#D9EAD3"   # light green
        edge_color = "#2E5E2E"     # dark green border

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        fig.suptitle(f"Confusion Matrix – Random Forest (Operating Threshold = {thr:.2f})", fontsize=16)

        col_labels = [f"Actual {neg}", f"Actual {pos}", "Total"]
        row_labels = [f"Predicted {neg}", f"Predicted {pos}", "Total"]

        def metrics_text(cm: Dict[str, Any]) -> str:
            # Keep it simple & clinically explicit
            return (
                f"Sensitivity (TP/(TP+FN)) = {cm['sensitivity']:.3f}\n"
                f"Specificity (TN/(TN+FP)) = {cm['specificity']:.3f}\n"
                f"Accuracy = {cm['accuracy']:.3f}"
            )

        for ax, table_data, title, cm in [
            (axes[0], tab_b, "Baseline", cm_b),
            (axes[1], tab_r, "Cohort-Routed", cm_r),
        ]:
            ax.axis("off")
            ax.set_title(title, fontsize=14, pad=10)

            tbl = ax.table(
                cellText=table_data.astype(str),
                rowLabels=row_labels,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
            )

            tbl.auto_set_font_size(False)
            tbl.set_fontsize(12)
            tbl.scale(1.15, 1.6)

            # Style: header row and row-label column in green; bold totals
            for (r, c), cell in tbl.get_celld().items():
                cell.set_linewidth(1.0)
                cell.set_edgecolor(edge_color)

                # column header row
                if r == 0:
                    cell.set_facecolor(header_green)
                    cell.set_text_props(weight="bold")

                # row labels column
                if c == -1:
                    cell.set_facecolor(header_green)
                    cell.set_text_props(weight="bold")

                # totals emphasis: last row (r==3) and last col (c==2)
                if r == 3 or c == 2:
                    cell.set_text_props(weight="bold")

            # Metrics box below table (readable)
            ax.text(
                0.5, 0.08,
                metrics_text(cm),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor=edge_color),
            )

        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        self._save_multi(fig, out_path, dpi=self.cfg.cm_dpi)
        plt.close(fig)

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
                "figure_roc_png": fig2,
                "figure_roc_svg": os.path.splitext(fig2)[0] + ".svg",
                "figure_roc_pdf": os.path.splitext(fig2)[0] + ".pdf",
                "figure_confusion_png": fig3,
                "figure_confusion_svg": os.path.splitext(fig3)[0] + ".svg",
                "figure_confusion_pdf": os.path.splitext(fig3)[0] + ".pdf",
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
    print("[ROC  PNG]", out["outputs"]["figure_roc_png"])
    print("[ROC  SVG]", out["outputs"]["figure_roc_svg"])
    print("[ROC  PDF]", out["outputs"]["figure_roc_pdf"])
    print("[CM   PNG]", out["outputs"]["figure_confusion_png"])
    print("[CM   SVG]", out["outputs"]["figure_confusion_svg"])
    print("[CM   PDF]", out["outputs"]["figure_confusion_pdf"])
    print("[TABLE AUC]", out["outputs"]["table_auc"])
    print("[TABLE CM ]", out["outputs"]["table_confusion_metrics"])
    print("[SUMMARY  ]", out["outputs"]["summary_json"])
    print("=" * 95)