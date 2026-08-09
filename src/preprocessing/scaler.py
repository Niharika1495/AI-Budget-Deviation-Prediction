"""Numerical feature scaling module for financial dataset preprocessing.

This module applies StandardScaler to numerical model features while preserving target variables
and non-numeric identifiers (Record_ID, Cost_Center, Actual_Expense, Budget_Deviation).
Supports joblib serialization for production deployment.

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """StandardScaler wrapper for numerical financial model features."""

    def __init__(self) -> None:
        """Initialize FeatureScaler and identify scaling exclusion list."""
        self.scaler: StandardScaler = StandardScaler()
        self.excluded_columns: List[str] = [
            "Record_ID",
            "Cost_Center",
            "Actual_Expense",
            "Budget_Deviation",
        ]
        self.scalable_columns_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Identify numerical model features and compute mean and variance vectors.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Fitted FeatureScaler instance.
        """
        # Select numeric columns excluding non-feature targets and IDs
        candidate_cols = [
            col
            for col in df.columns
            if col not in self.excluded_columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        self.scalable_columns_ = candidate_cols

        if self.scalable_columns_:
            self.scaler.fit(df[self.scalable_columns_])
            self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features in DataFrame using fitted StandardScaler.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Tuple of (scaled_dataframe, scaling_summary_dict).

        Raises:
            RuntimeError: If scaler has not been fitted prior to transform.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling transform().")

        df_scaled = df.copy()

        if self.scalable_columns_:
            scaled_array = self.scaler.transform(df_scaled[self.scalable_columns_])
            df_scaled[self.scalable_columns_] = scaled_array

        means_dict = {
            col: round(float(m), 4)
            for col, m in zip(self.scalable_columns_, self.scaler.mean_)
        }
        vars_dict = {
            col: round(float(v), 4)
            for col, v in zip(self.scalable_columns_, self.scaler.var_)
        }

        summary: Dict[str, Any] = {
            "scaled_feature_count": len(self.scalable_columns_),
            "scaled_columns": self.scalable_columns_,
            "excluded_columns": self.excluded_columns,
            "feature_means": means_dict,
            "feature_variances": vars_dict,
        }

        return df_scaled, summary

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit scaler parameters and transform DataFrame in a single step.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (scaled_dataframe, scaling_summary_dict).
        """
        self.fit(df)
        return self.transform(df)

    def save(self, file_path: Path) -> None:
        """Serialize fitted scaler object to disk using joblib.

        Args:
            file_path: Destination file path (.joblib).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureScaler.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "FeatureScaler":
        """Load serialized FeatureScaler from disk.

        Args:
            file_path: Path to serialized joblib file.

        Returns:
            Restored FeatureScaler instance.
        """
        return joblib.load(file_path)
