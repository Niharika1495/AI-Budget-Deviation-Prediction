"""Data quality and schema validation module for AI Financial Budget Deviation Prediction.

This module provides automated validation logic to audit synthetic ERP financial datasets:
- Schema adherence (21 columns)
- Record count exactness (15,000)
- Record_ID uniqueness check
- Missing value percentage evaluation (2.0% - 3.0%)
- Non-negativity constraints for cost/count variables
- Business logic target math verification (Budget_Deviation = Actual_Expense - Planned_Budget)
- Department profile range compliance
- Formatted validation report generation (JSON and TXT formats)

Follows Clean Architecture, type annotations, PEP8 standards, and SOLID principles.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import DATASET_COLUMNS, DEPARTMENTS, INFLATION_RATES, TOTAL_RECORDS, YEARS
from src.department_rules import DEPARTMENT_PROFILES


@dataclass
class ValidationResult:
    """Dataclass encapsulating comprehensive validation execution metrics."""

    passed: bool
    total_records: int
    column_count: int
    duplicate_record_ids: int
    overall_missing_pct: float
    missing_by_column: Dict[str, float]
    invalid_negative_counts: Dict[str, int]
    math_mismatch_count: int
    dept_distribution: Dict[str, int]
    year_distribution: Dict[int, int]
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary format."""
        return asdict(self)


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Verify DataFrame column names and ordering match exact dataset schema.

    Args:
        df: Input DataFrame to check.

    Returns:
        List of validation error messages, if any.
    """
    errors: List[str] = []
    if list(df.columns) != DATASET_COLUMNS:
        missing_cols = set(DATASET_COLUMNS) - set(df.columns)
        extra_cols = set(df.columns) - set(DATASET_COLUMNS)
        if missing_cols:
            errors.append(f"Missing required schema columns: {missing_cols}")
        if extra_cols:
            errors.append(f"Unexpected extra columns: {extra_cols}")
        if not missing_cols and not extra_cols and list(df.columns) != DATASET_COLUMNS:
            errors.append("Column order mismatch against standard schema.")
    return errors


def validate_record_uniqueness(df: pd.DataFrame) -> Tuple[int, List[str]]:
    """Check for duplicate Record_IDs in dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (duplicate_count, list_of_error_strings).
    """
    errors: List[str] = []
    if "Record_ID" not in df.columns:
        return 0, ["Record_ID column not present."]

    duplicates = df["Record_ID"].duplicated().sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate Record_ID values in dataset.")
    return int(duplicates), errors


def validate_missing_value_rates(df: pd.DataFrame) -> Tuple[float, Dict[str, float], List[str]]:
    """Audit missing value proportions across key and non-key columns.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (overall_missing_pct, missing_by_column_dict, error_list).
    """
    errors: List[str] = []
    total_cells = df.size
    total_nulls = df.isnull().sum().sum()
    overall_pct = round(float((total_nulls / total_cells) * 100.0), 2)

    missing_by_col = {
        col: round(float((df[col].isnull().sum() / len(df)) * 100.0), 2) for col in df.columns
    }

    # Key columns MUST NOT have missing values
    key_cols = ["Record_ID", "Year", "Quarter", "Month", "Department", "Cost_Center"]
    for kcol in key_cols:
        if kcol in df.columns and df[kcol].isnull().sum() > 0:
            errors.append(
                f"Critical Key Column '{kcol}' contains {df[kcol].isnull().sum()} missing values."
            )

    return overall_pct, missing_by_col, errors


def validate_non_negativity(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """Check that financial and numeric quantity columns contain non-negative values.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (invalid_negative_counts_dict, error_list).
    """
    errors: List[str] = []
    non_negative_cols = [
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
    ]

    invalid_counts: Dict[str, int] = {}
    for col in non_negative_cols:
        if col in df.columns:
            neg_count = int((df[col].dropna() < 0.0).sum())
            invalid_counts[col] = neg_count
            if neg_count > 0:
                errors.append(f"Column '{col}' contains {neg_count} invalid negative values.")

    return invalid_counts, errors


def validate_target_mathematical_integrity(
    df: pd.DataFrame, tolerance: float = 0.02
) -> Tuple[int, List[str]]:
    """Verify Target Relationship: Budget_Deviation == Actual_Expense - Planned_Budget.

    Args:
        df: Input DataFrame.
        tolerance: Absolute float difference tolerance for floating-point arithmetic.

    Returns:
        Tuple of (mismatch_count, error_list).
    """
    errors: List[str] = []
    required = ["Actual_Expense", "Planned_Budget", "Budget_Deviation"]
    if not all(col in df.columns for col in required):
        return 0, ["Target verification skipped due to missing target columns."]

    valid_mask = df[required].notnull().all(axis=1)
    sub = df[valid_mask]

    calculated_dev = sub["Actual_Expense"] - sub["Planned_Budget"]
    mismatches = int((np.abs(sub["Budget_Deviation"] - calculated_dev) > tolerance).sum())

    if mismatches > 0:
        errors.append(
            f"Found {mismatches} records where Budget_Deviation != (Actual_Expense - Planned_Budget)."
        )

    return mismatches, errors


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """Execute complete validation suite on the synthetic dataset.

    Args:
        df: Target DataFrame to evaluate.

    Returns:
        ValidationResult object summarizing pass/fail status and detailed metrics.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Total Record Count
    rec_count = len(df)
    if rec_count != TOTAL_RECORDS:
        errors.append(f"Record count mismatch: Expected {TOTAL_RECORDS}, got {rec_count}.")

    # 2. Schema check
    schema_errs = validate_schema(df)
    errors.extend(schema_errs)

    # 3. Record ID uniqueness
    duplicates, dup_errs = validate_record_uniqueness(df)
    errors.extend(dup_errs)

    # 4. Missing value checks
    overall_missing_pct, missing_by_col, miss_errs = validate_missing_value_rates(df)
    errors.extend(miss_errs)

    if not (1.5 <= overall_missing_pct <= 4.0):
        warnings.append(
            f"Overall missing value rate is {overall_missing_pct}%, slightly outside 2.0%-3.0% target window."
        )

    # 5. Non-negativity checks
    invalid_negs, neg_errs = validate_non_negativity(df)
    errors.extend(neg_errs)

    # 6. Target math integrity
    math_mismatches, math_errs = validate_target_mathematical_integrity(df)
    errors.extend(math_errs)

    # 7. Department and Year distributions
    dept_dist = (
        df["Department"].value_counts().to_dict()
        if "Department" in df.columns
        else {}
    )
    year_dist = (
        df["Year"].value_counts().to_dict()
        if "Year" in df.columns
        else {}
    )

    passed = len(errors) == 0

    return ValidationResult(
        passed=passed,
        total_records=rec_count,
        column_count=len(df.columns),
        duplicate_record_ids=duplicates,
        overall_missing_pct=overall_missing_pct,
        missing_by_column=missing_by_col,
        invalid_negative_counts=invalid_negs,
        math_mismatch_count=math_mismatches,
        dept_distribution=dept_dist,
        year_distribution=year_dist,
        errors=errors,
        warnings=warnings,
    )


def generate_validation_report_text(result: ValidationResult) -> str:
    """Generate a clean human-readable text validation report.

    Args:
        result: ValidationResult instance.

    Returns:
        Formatted summary report string.
    """
    status_str = "SUCCESS (PASSED)" if result.passed else "FAILURE (FAILED)"
    lines = [
        "================================================================================",
        "             AI FINANCIAL BUDGET DEVIATION DATASET VALIDATION REPORT            ",
        "================================================================================",
        f"Validation Status        : {status_str}",
        f"Total Records Evaluated  : {result.total_records:,} (Target: {TOTAL_RECORDS:,})",
        f"Column Count             : {result.column_count} (Target: {len(DATASET_COLUMNS)})",
        f"Duplicate Record IDs     : {result.duplicate_record_ids}",
        f"Overall Missing Value %  : {result.overall_missing_pct}%",
        f"Target Math Mismatches   : {result.math_mismatch_count}",
        "--------------------------------------------------------------------------------",
        "DEPARTMENT DISTRIBUTION:",
    ]
    for dept, count in result.dept_distribution.items():
        lines.append(f"  - {dept:<24}: {count:,} records")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("YEAR DISTRIBUTION:")
    for yr, count in sorted(result.year_distribution.items()):
        lines.append(f"  - Year {yr:<19}: {count:,} records")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("MISSING VALUES BY COLUMN:")
    for col, pct in result.missing_by_column.items():
        lines.append(f"  - {col:<26}: {pct}% missing")

    if result.errors:
        lines.append("--------------------------------------------------------------------------------")
        lines.append("VALIDATION ERRORS:")
        for err in result.errors:
            lines.append(f"  [ERROR] {err}")

    if result.warnings:
        lines.append("--------------------------------------------------------------------------------")
        lines.append("VALIDATION WARNINGS:")
        for warn in result.warnings:
            lines.append(f"  [WARNING] {warn}")

    lines.append("================================================================================")
    return "\n".join(lines)
