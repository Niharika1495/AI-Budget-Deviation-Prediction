"""Explainability and model interpretation module for budget deviation prediction.

This module provides the FinancialExplainer class to compute global feature importances
and local cost-driver risk attributions for individual financial records.

Follows Clean Architecture, PEP8 standards, type annotations, and docstrings.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class FinancialExplainer:
    """Provides model explainability and local risk driver decomposition."""

    def __init__(self, model_artifact: Dict[str, Any]) -> None:
        """Initialize explainer with trained model dictionary artifact.

        Args:
            model_artifact: Dictionary containing 'model', 'model_name', and 'feature_columns'.
        """
        self.model = model_artifact.get("model")
        self.model_name = model_artifact.get("model_name", "Model")
        self.feature_columns = model_artifact.get("feature_columns", [])

    def get_global_feature_importance(self) -> Dict[str, float]:
        """Compute normalized global feature importances.

        Returns:
            Dictionary mapping feature names to importance scores (descending order).
        """
        if self.model is None or not self.feature_columns:
            return {}

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
            if importances.ndim > 1:
                importances = np.mean(importances, axis=0)

        if importances is None:
            # Fallback uniform importances if model doesn't support built-in importance
            importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)

        # Normalize to sum to 1.0
        total_imp = np.sum(importances)
        if total_imp > 0:
            importances = importances / total_imp

        imp_dict = dict(zip(self.feature_columns, importances))
        sorted_imp = dict(
            sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_imp

    def analyze_local_drivers(
        self, record_dict: Dict[str, Any], planned_budget: float
    ) -> List[Dict[str, Any]]:
        """Identify key cost drivers contributing to overbudget risk for a single record.

        Args:
            record_dict: Dictionary containing cost breakdowns for the target record.
            planned_budget: Target planned budget amount ($).

        Returns:
            List of key cost driver dictionaries with driver_name, amount, share_percent, and risk_impact.
        """
        cost_keys = [
            "Operational_Cost",
            "Marketing_Cost",
            "Travel_Cost",
            "Utility_Cost",
            "Vendor_Cost",
            "Miscellaneous_Cost",
        ]

        drivers = []
        eps = 1e-5
        total_cost = sum([float(record_dict.get(k, 0.0)) for k in cost_keys])

        for key in cost_keys:
            val = float(record_dict.get(key, 0.0))
            share = (val / (total_cost + eps)) * 100.0
            budget_ratio = (val / (planned_budget + eps)) * 100.0

            # Determine risk level of individual component
            if budget_ratio > 35.0 or share > 40.0:
                impact = "High Impact Driver"
            elif budget_ratio > 20.0 or share > 25.0:
                impact = "Moderate Driver"
            else:
                impact = "Normal Baseline"

            drivers.append(
                {
                    "driver_name": key.replace("_", " "),
                    "amount": round(val, 2),
                    "share_of_total_cost_percent": round(share, 1),
                    "budget_ratio_percent": round(budget_ratio, 1),
                    "risk_impact": impact,
                }
            )

        # Sort drivers by amount descending
        drivers = sorted(
            drivers, key=lambda d: d["amount"], reverse=True
        )
        return drivers
