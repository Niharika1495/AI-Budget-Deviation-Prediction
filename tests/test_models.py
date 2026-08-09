"""Unit tests for model training and evaluation components."""

import pytest
import pandas as pd
import numpy as np

from src.models.model_evaluator import ModelEvaluator


def test_model_evaluator_regression():
    evaluator = ModelEvaluator()
    y_true = pd.Series([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    metrics = evaluator.evaluate_regression(y_true, y_pred, model_name="TestRegressor")
    assert metrics["model_name"] == "TestRegressor"
    assert metrics["mae"] == 10.0
    assert metrics["rmse"] == 10.0
    assert metrics["r2_score"] > 0.90


def test_model_evaluator_classification():
    evaluator = ModelEvaluator()
    y_true = pd.Series(["Low Risk", "Severe Overbudget", "Low Risk"])
    y_pred = np.array(["Low Risk", "Severe Overbudget", "Low Risk"])

    metrics = evaluator.evaluate_classification(
        y_true, y_pred, model_name="TestClassifier"
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_weighted"] == 1.0
