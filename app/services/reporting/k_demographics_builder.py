import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
@dataclass
class KDemographicsConfig:
    data_clean_path: str
    scores_path: str                    # scores_combined_rf.csv

    row_index_col: str = "row_index"
    y_true_col: str = "y_true"

    # column names in data_clean.xlsx
    age_col: str = "age"
    sex_col: str = "sex"
    race_col: str = "race"
    facial_droop_col: str = "FacialDroop (weakness)"
    nihss_col: str = "NIHSS"

    # ---- IMPORTANT: encoding based on your screenshot ----
    female_value: float = 0.1           # sex == 0.1 → female
    black_value: float = 0.1            # race == 0.1 → black
    droop_value: float = 0.1            # FacialDroop == 0.1 → present

    # optional clinically relevant columns (Mark's question)
    other_cols: Optional[List[str]] = None

    out_dir: str = "outputs"
    out_xlsx: str = "k_demographics_table.xlsx"


# =========================
# Builder
# =========================
class KDemographicsBuilder:
    """
    Build Kristine request #2 demographics table
    using the SAME 159 evaluation cohort as combined_rf.
    """

    def __init__(self, cfg: KDemographicsConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        self.df_all = pd.read_excel(cfg.data_clean_path)
        self.scores = pd.read_csv(cfg.scores_path)

        self._recover_cohort()

    def _recover_cohort(self):
        idx = self.scores[self.cfg.row_index_col].astype(int).to_numpy()
        n = len(self.df_all)

        # auto-detect 0-based vs 1-based
        if idx.min() >= 1 and idx.max() <= n:
            idx0 = idx - 1
        else:
            idx0 = idx

        self.df_eval = self.df_all.iloc[idx0].copy()
        self.df_eval["_y_true"] = self.scores[self.cfg.y_true_col].astype(int).to_numpy()

        self.stroke = self.df_eval[self.df_eval["_y_true"] == 1]
        self.nonstroke = self.df_eval[self.df_eval["_y_true"] == 0]

    @staticmethod
    def _pct(count: int, total: int) -> str:
        return f"{count / total * 100:.1f}%" if total > 0 else "NA"

    @staticmethod
    def _median(series: pd.Series) -> str:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return f"{s.median():.3f}" if len(s) > 0 else "NA"

    def build(self) -> pd.DataFrame:
        rows = []

        rows.append({
            "Demographic": "Age (median)",
            "Stroke": self._median(self.stroke[self.cfg.age_col]),
            "Non-stroke": self._median(self.nonstroke[self.cfg.age_col]),
        })

        rows.append({
            "Demographic": "Sex (% female)",
            "Stroke": self._pct((self.stroke[self.cfg.sex_col] == self.cfg.female_value).sum(), len(self.stroke)),
            "Non-stroke": self._pct((self.nonstroke[self.cfg.sex_col] == self.cfg.female_value).sum(), len(self.nonstroke)),
        })

        rows.append({
            "Demographic": "Race (% black)",
            "Stroke": self._pct((self.stroke[self.cfg.race_col] == self.cfg.black_value).sum(), len(self.stroke)),
            "Non-stroke": self._pct((self.nonstroke[self.cfg.race_col] == self.cfg.black_value).sum(), len(self.nonstroke)),
        })

        rows.append({
            "Demographic": "Facial droop (%)",
            "Stroke": self._pct((self.stroke[self.cfg.facial_droop_col] == self.cfg.droop_value).sum(), len(self.stroke)),
            "Non-stroke": self._pct((self.nonstroke[self.cfg.facial_droop_col] == self.cfg.droop_value).sum(), len(self.nonstroke)),
        })

        # optional: NIHSS median (good clinical signal)
        if self.cfg.nihss_col in self.df_eval.columns:
            rows.append({
                "Demographic": "NIHSS (median)",
                "Stroke": self._median(self.stroke[self.cfg.nihss_col]),
                "Non-stroke": self._median(self.nonstroke[self.cfg.nihss_col]),
            })

        return pd.DataFrame(rows)

    def save(self) -> pd.DataFrame:
        table = self.build()
        out_path = os.path.join(self.cfg.out_dir, self.cfg.out_xlsx)
        table.to_excel(out_path, index=False)
        return table

if __name__ == "__main__":
    cfg = KDemographicsConfig(
        data_clean_path="app/services/data/data_clean.xlsx",
        scores_path="outputs/cluster_routed_eval/combined/scores_combined_rf.csv",
        out_dir="outputs",
        out_xlsx="k_demographics_table.xlsx",
    )

    builder = KDemographicsBuilder(cfg)
    df = builder.save()
    print(df)