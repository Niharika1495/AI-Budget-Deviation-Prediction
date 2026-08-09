"""Unit tests for production predictor and inference engine."""

import pytest
from src.inference.predictor import BudgetDeviationPredictor


def test_budget_deviation_predictor_single():
    predictor = BudgetDeviationPredictor()

    sample_input = {
        "Record_ID": "TEST_001",
        "Year": 2025,
        "Quarter": "Q4",
        "Month": "December",
        "Department": "Finance",
        "Cost_Center": "CC-FIN-001",
        "Planned_Budget": 200000.0,
        "Revenue": 300000.0,
        "Employee_Count": 20,
        "Project_Count": 4,
        "Operational_Cost": 80000.0,
        "Marketing_Cost": 30000.0,
        "Travel_Cost": 15000.0,
        "Utility_Cost": 10000.0,
        "Vendor_Cost": 40000.0,
        "Miscellaneous_Cost": 5000.0,
        "Inflation_Rate": 0.048,
        "Previous_Month_Expense": 60000.0,
        "Previous_Quarter_Expense": 180000.0,
    }

    res = predictor.predict_single(sample_input)

    assert "predicted_budget_deviation" in res
    assert "predicted_actual_expense" in res
    assert "predicted_risk_level" in res
    assert "top_cost_drivers" in res
    assert "recommendations" in res

    assert isinstance(res["predicted_budget_deviation"], float)
    assert len(res["recommendations"]) > 0
