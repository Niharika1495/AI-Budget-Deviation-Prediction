"""Post-preprocessing validation module for financial dataset pipeline.

This module validates that the preprocessed dataset meets model-ready standards:
- Zero duplicate Record_IDs
- Zero missing values across all features
- Proper numeric dtypes across encoded/scaled features
- Target column (Budget_Deviation) remains 100% identical to raw input

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


class PreprocessedDataValidator:
    """Audit suite for verifying dataset state after preprocessing pipeline execution."""

    def __init__(self) -> None:
        """Initialize post-preprocessing validator."""
        self.target_column: str = "Budget_Deviation"
        self.id_column: str = "Record_ID"

    def validate(
        self, df_raw: pd.DataFrame, df_processed: pd.DataFrame
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate preprocessed DataFrame against raw input baseline.

        Args:
            df_raw: Original raw Pandas DataFrame.
            df_processed: Preprocessed transformed Pandas DataFrame.

        Returns:
            Tuple of (is_valid_boolean, validation_report_dict).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Row Count Integrity
        if len(df_processed) != len(df_raw):
            errors.append(
                f"Row count altered during preprocessing! Raw: {len(df_raw)}, Processed: {len(df_processed)}"
            )

        # 2. Record_ID Uniqueness Check
        if self.id_column in df_processed.columns:
            duplicates = int(df_processed[self.id_column].duplicated().sum())
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate {self.id_column}s in preprocessed dataset.")

        # 3. Zero Missing Values in Model Features
        missing_counts = df_processed.isnull().sum()
        total_missing = int(missing_counts.sum())
        if total_missing > 0:
            cols_with_missing = missing_counts[missing_counts > 0].to_dict()
            errors.append(
                f"Preprocessed dataset contains {total_missing} missing values across features: {cols_with_missing}"
            )

        # 4. Target Variable Unchanged Integrity Check
        if self.target_column in df_raw.columns and self.target_column in df_processed.columns:
            raw_target = df_raw[self.target_column].values
            proc_target = df_processed[self.target_column].values
            
            # Check maximum absolute divergence
            target_diff = np.max(np.abs(raw_target - proc_target))
            if target_diff > 1e-4:
                errors.append(
                    f"CRITICAL ERROR: Target column '{self.target_column}' was altered during preprocessing! Max diff: {target_diff}"
                )

        # 5. Data Types Verification (All non-ID columns must be numeric)
        non_numeric_cols = [
            col
            for col in df_processed.columns
            if col != self.id_column and col != "Cost_Center" and not pd.api.types.is_numeric_dtype(df_processed[col])
        ]
        if non_numeric_cols:
            errors.append(
                f"Preprocessed feature columns contain non-numeric data types: {non_numeric_cols}"
            )

        passed = len(errors) == 0

        report: Dict[str, Any] = {
            "passed": bool(passed),
            "total_records": int(len(df_processed)),
            "feature_columns_count": int(len(df_processed.columns)),
            "duplicate_ids": int(duplicates) if self.id_column in df_processed.columns else 0,
            "total_missing_values": int(total_missing),
            "target_unchanged_verified": bool(passed and float(target_diff) <= 1e-4) if 'target_diff' in locals() else False,
            "errors": errors,
            "warnings": warnings,
        }

        return passed, report
