"""IQR-based outlier detection and capping module for financial dataset preprocessing.

This module applies Interquartile Range (IQR) logic to detect and cap extreme financial outliers
(Winsorization) without deleting records. Target variables and categorical identifiers are preserved.

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


class IQROutlierHandler:
    """Outlier treatment handler utilizing IQR upper/lower capping bounds."""

    def __init__(self, iqr_multiplier: float = 1.5) -> None:
        """Initialize outlier handler with IQR multiplier factor.

        Args:
            iqr_multiplier: IQR threshold multiplier (default 1.5).
        """
        self.iqr_multiplier: float = iqr_multiplier
        self.eligible_columns: List[str] = [
            "Planned_Budget",
            "Revenue",
            "Employee_Count",
            "Project_Count",
            "Operational_Cost",
            "Marketing_Cost",
            "Travel_Cost",
            "Utility_Cost",
            "Vendor_Cost",
            "Miscellaneous_Cost",
            "Previous_Month_Expense",
            "Previous_Quarter_Expense",
        ]
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "IQROutlierHandler":
        """Calculate Q1, Q3, IQR, and lower/upper capping bounds for eligible features.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Fitted IQROutlierHandler instance.
        """
        for col in self.eligible_columns:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    q1 = float(series.quantile(0.25))
                    q3 = float(series.quantile(0.75))
                    iqr = q3 - q1
                    lower_bound = max(0.0, q1 - (self.iqr_multiplier * iqr))  # Non-negative lower bound
                    upper_bound = q3 + (self.iqr_multiplier * iqr)
                    self.bounds_[col] = (lower_bound, upper_bound)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cap extreme outliers to computed IQR lower and upper bounds.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Tuple of (capped_dataframe, outlier_stats_dict).

        Raises:
            RuntimeError: If handler has not been fitted prior to transform.
        """
        if not self.bounds_:
            raise RuntimeError("IQROutlierHandler must be fitted before calling transform().")

        df_capped = df.copy()

        affected_columns: List[str] = []
        total_outliers_capped = 0
        column_stats: Dict[str, Dict[str, Any]] = {}

        for col, (lower_b, upper_b) in self.bounds_.items():
            if col in df_capped.columns:
                series = df_capped[col].dropna()

                # Calculate stats before capping
                stats_before = {
                    "min": float(series.min()) if len(series) > 0 else 0.0,
                    "max": float(series.max()) if len(series) > 0 else 0.0,
                    "mean": round(float(series.mean()), 2) if len(series) > 0 else 0.0,
                    "std": round(float(series.std()), 2) if len(series) > 0 else 0.0,
                }

                # Identify outliers
                low_mask = df_capped[col] < lower_b
                high_mask = df_capped[col] > upper_b
                outlier_mask = low_mask | high_mask
                count_outliers = int(outlier_mask.sum())

                if count_outliers > 0:
                    affected_columns.append(col)
                    total_outliers_capped += count_outliers

                    # Perform Winsorization capping (clip values within bounds)
                    df_capped[col] = df_capped[col].clip(lower=lower_b, upper=upper_b)

                # Calculate stats after capping
                series_after = df_capped[col].dropna()
                stats_after = {
                    "min": float(series_after.min()) if len(series_after) > 0 else 0.0,
                    "max": float(series_after.max()) if len(series_after) > 0 else 0.0,
                    "mean": round(float(series_after.mean()), 2) if len(series_after) > 0 else 0.0,
                    "std": round(float(series_after.std()), 2) if len(series_after) > 0 else 0.0,
                }

                column_stats[col] = {
                    "outliers_capped": count_outliers,
                    "lower_bound": round(lower_b, 2),
                    "upper_bound": round(upper_b, 2),
                    "before_stats": stats_before,
                    "after_stats": stats_after,
                }

        summary: Dict[str, Any] = {
            "total_outliers_capped": total_outliers_capped,
            "columns_affected_count": len(affected_columns),
            "columns_affected": affected_columns,
            "column_details": column_stats,
        }

        return df_capped, summary

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit IQR bounds and cap extreme values in a single step.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (capped_dataframe, outlier_stats_dict).
        """
        self.fit(df)
        return self.transform(df)
