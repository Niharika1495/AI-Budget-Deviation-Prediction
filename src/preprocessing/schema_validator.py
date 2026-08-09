"""Raw schema validation module for financial dataset preprocessing.

This module validates that incoming raw dataset CSV files conform to expected column schemas,
data types, mandatory non-null key fields, categorical allowed sets, and presence of target variables.

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from src.config import DATASET_COLUMNS, DEPARTMENTS, MONTHS, QUARTER_MAPPING


class RawSchemaValidator:
    """Validator engine for verifying structural integrity of raw ERP financial datasets."""

    def __init__(self) -> None:
        """Initialize validator with expected schema constraints."""
        self.expected_columns: List[str] = DATASET_COLUMNS
        self.mandatory_key_columns: List[str] = [
            "Record_ID",
            "Year",
            "Quarter",
            "Month",
            "Department",
            "Cost_Center",
            "Actual_Expense",
            "Budget_Deviation",
        ]
        self.allowed_departments: Set[str] = set(DEPARTMENTS)
        self.allowed_months: Set[str] = set(MONTHS)
        self.allowed_quarters: Set[str] = set(QUARTER_MAPPING.values())
        self.target_column: str = "Budget_Deviation"

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate raw DataFrame schema and integrity.

        Args:
            df: Input raw Pandas DataFrame.

        Returns:
            Tuple of (is_valid_boolean, detailed_report_dict).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Required Columns Check
        current_cols = set(df.columns)
        missing_required = set(self.expected_columns) - current_cols
        extra_columns = current_cols - set(self.expected_columns)

        if missing_required:
            errors.append(f"Missing mandatory schema columns: {sorted(list(missing_required))}")

        if extra_columns:
            warnings.append(f"Found non-standard extra columns: {sorted(list(extra_columns))}")

        # 2. Target Column Check
        if self.target_column not in df.columns:
            errors.append(f"Target column '{self.target_column}' is missing from dataset.")

        # 3. Mandatory Key Null Checks
        for kcol in self.mandatory_key_columns:
            if kcol in df.columns:
                null_count = df[kcol].isnull().sum()
                if null_count > 0:
                    errors.append(
                        f"Mandatory key column '{kcol}' contains {null_count} null/missing values."
                    )

        # 4. Categorical Allowed Values Checks
        if "Department" in df.columns:
            invalid_depts = set(df["Department"].dropna().unique()) - self.allowed_departments
            if invalid_depts:
                errors.append(f"Invalid department values found: {invalid_depts}")

        if "Month" in df.columns:
            invalid_months = set(df["Month"].dropna().unique()) - self.allowed_months
            if invalid_months:
                errors.append(f"Invalid month values found: {invalid_months}")

        if "Quarter" in df.columns:
            invalid_qtrs = set(df["Quarter"].dropna().unique()) - self.allowed_quarters
            if invalid_qtrs:
                errors.append(f"Invalid quarter values found: {invalid_qtrs}")

        # 5. Data Type Verification
        numeric_cols = [
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
            "Inflation_Rate",
            "Actual_Expense",
            "Budget_Deviation",
        ]

        for ncol in numeric_cols:
            if ncol in df.columns:
                if not pd.api.types.is_numeric_dtype(df[ncol]):
                    errors.append(
                        f"Column '{ncol}' has non-numeric data type '{df[ncol].dtype}'."
                    )

        passed = len(errors) == 0

        report: Dict[str, Any] = {
            "passed": passed,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_required_columns": list(missing_required),
            "extra_columns": list(extra_columns),
            "target_column_present": self.target_column in df.columns,
            "errors": errors,
            "warnings": warnings,
        }

        return passed, report
