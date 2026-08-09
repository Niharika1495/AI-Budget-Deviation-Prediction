"""Unit tests for feature engineering module."""

import pytest
import pandas as pd
from src.features.feature_builder import FinancialFeatureBuilder
from src.features.feature_selector import FinancialFeatureSelector


def test_financial_feature_builder():
    builder = FinancialFeatureBuilder()
    df = pd.DataFrame(
        [
            {
                "Planned_Budget": 100000.0,
                "Employee_Count": 10,
                "Project_Count": 2,
                "Revenue": 200000.0,
                "Inflation_Rate": 0.05,
                "Previous_Month_Expense": 30000.0,
                "Previous_Quarter_Expense": 90000.0,
                "Budget_Deviation": 15000.0,
            }
        ]
    )

    df_feat = builder.transform(df, include_target_categories=True)

    assert "Planned_Budget_Per_Employee" in df_feat.columns
    assert "Projects_Per_Employee" in df_feat.columns
    assert "Inflation_Cost_Impact" in df_feat.columns
    assert "Expense_Momentum_Ratio" in df_feat.columns
    assert "Deviation_Risk_Category" in df_feat.columns

    assert df_feat["Planned_Budget_Per_Employee"].iloc[0] > 0
    assert df_feat["Deviation_Risk_Category"].iloc[0] in [
        "Underbudget",
        "Low Risk (On-Budget)",
        "Moderate Overbudget",
        "Severe Overbudget",
    ]


def test_financial_feature_selector():
    selector = FinancialFeatureSelector(top_k_features=2)
    X = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "f_const": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

    df_selected = selector.fit_transform(X, y)
    assert "f_const" not in df_selected.columns
    assert len(df_selected.columns) <= 2
