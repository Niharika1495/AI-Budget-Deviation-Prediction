"""Missing value handler module for financial dataset preprocessing.

This module analyzes missing data across dataset features and applies domain-aligned median
imputation for eligible numeric columns, while strictly preserving key metadata identifiers and
target variables (Record_ID, Department, Year, Quarter, Month, Cost_Center, Actual_Expense, Budget_Deviation).

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class MissingValueHandler:
    """Intelligent missing value analysis and median imputation handler."""

    def __init__(self) -> None:
        """Initialize imputation target features and immutable preservation lists."""
        self.imputable_columns: List[str] = [
            "Employee_Count",
            "Project_Count",
            "Travel_Cost",
            "Utility_Cost",
            "Marketing_Cost",
            "Miscellaneous_Cost",
            "Previous_Month_Expense",
            "Previous_Quarter_Expense",
        ]
        self.protected_columns: List[str] = [
            "Record_ID",
            "Department",
            "Year",
            "Quarter",
            "Month",
            "Cost_Center",
            "Actual_Expense",
            "Budget_Deviation",
        ]
        self.medians_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        """Compute training medians for imputable feature columns.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Fitted MissingValueHandler instance.
        """
        for col in self.imputable_columns:
            if col in df.columns:
                median_val = float(df[col].median(skipna=True))
                self.medians_[col] = median_val
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute missing values in input DataFrame and generate comparison summary.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Tuple of (imputed_dataframe, before_after_summary_dict).

        Raises:
            RuntimeError: If handler has not been fitted prior to transform.
        """
        if not self.medians_:
            raise RuntimeError("MissingValueHandler must be fitted before calling transform().")

        df_imputed = df.copy()

        # Capture Before Statistics
        before_null_counts: Dict[str, int] = df_imputed.isnull().sum().to_dict()
        before_total_nulls = sum(before_null_counts.values())

        imputed_columns_detail: Dict[str, Dict[str, Any]] = {}

        for col in self.imputable_columns:
            if col in df_imputed.columns:
                missing_count = before_null_counts.get(col, 0)
                median_val = self.medians_[col]

                # Impute missing values with computed median
                if missing_count > 0:
                    df_imputed[col] = df_imputed[col].fillna(median_val)

                # Integer columns remain integer type
                if col in ("Employee_Count", "Project_Count"):
                    df_imputed[col] = df_imputed[col].round().astype(int)

                imputed_columns_detail[col] = {
                    "missing_before": missing_count,
                    "imputed_value": median_val,
                    "missing_after": int(df_imputed[col].isnull().sum()),
                }

        # Verify Protected Columns were untouched
        for pcol in self.protected_columns:
            if pcol in df_imputed.columns and pcol in before_null_counts:
                after_nulls = int(df_imputed[pcol].isnull().sum())
                if after_nulls != before_null_counts[pcol]:
                    raise ValueError(
                        f"Protected column '{pcol}' missing values altered during imputation!"
                    )

        # Capture After Statistics
        after_null_counts: Dict[str, int] = df_imputed.isnull().sum().to_dict()
        after_total_nulls = sum(after_null_counts.values())

        summary: Dict[str, Any] = {
            "total_nulls_before": before_total_nulls,
            "total_nulls_after": after_total_nulls,
            "nulls_imputed": before_total_nulls - after_total_nulls,
            "column_details": imputed_columns_detail,
        }

        return df_imputed, summary

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit median parameters and transform DataFrame in a single step.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (imputed_dataframe, before_after_summary_dict).
        """
        self.fit(df)
        return self.transform(df)
