"""Unit tests for dataset preprocessing and validation pipeline."""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing.schema_validator import RawSchemaValidator
from src.preprocessing.missing_value_handler import MissingValueHandler
from src.preprocessing.outlier_handler import IQROutlierHandler


def test_raw_schema_validator():
    validator = RawSchemaValidator()
    # Create valid dummy row
    df_valid = pd.DataFrame(
        [
            {
                "Record_ID": "REC_00001",
                "Year": 2024,
                "Quarter": "Q1",
                "Month": "January",
                "Department": "Finance",
                "Cost_Center": "CC-FIN-001",
                "Planned_Budget": 100000.0,
                "Revenue": 150000.0,
                "Employee_Count": 20,
                "Project_Count": 5,
                "Operational_Cost": 40000.0,
                "Marketing_Cost": 20000.0,
                "Travel_Cost": 10000.0,
                "Utility_Cost": 5000.0,
                "Vendor_Cost": 15000.0,
                "Miscellaneous_Cost": 5000.0,
                "Inflation_Rate": 0.054,
                "Previous_Month_Expense": 35000.0,
                "Previous_Quarter_Expense": 110000.0,
                "Actual_Expense": 95000.0,
                "Budget_Deviation": -5000.0,
            }
        ]
    )

    is_valid, report = validator.validate(df_valid)
    assert is_valid is True
    assert len(report["errors"]) == 0


def test_missing_value_handler():
    handler = MissingValueHandler()
    df = pd.DataFrame(
        {
            "Travel_Cost": [10.0, 20.0, np.nan, 40.0],
            "Utility_Cost": [5.0, 10.0, np.nan, 20.0],
        }
    )

    df_imputed, report = handler.fit_transform(df)
    assert df_imputed.isnull().sum().sum() == 0
    assert report["total_nulls_before"] == 2
    assert report["total_nulls_after"] == 0


def test_iqr_outlier_handler():
    handler = IQROutlierHandler(iqr_multiplier=1.5)
    # Generate data with extreme outlier
    data = [10.0, 12.0, 11.0, 13.0, 12.0, 1000.0]
    df = pd.DataFrame({"Planned_Budget": data})

    df_capped, report = handler.fit_transform(df)
    assert df_capped["Planned_Budget"].max() < 1000.0
    assert report["total_outliers_capped"] > 0
