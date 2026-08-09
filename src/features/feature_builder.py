"""Financial feature engineering module for budget deviation prediction.

STRICT ML INTEGRITY COMPLIANCE:
This module extracts features ONLY from pre-prediction time inputs:
- Planned_Budget, Revenue, Employee_Count, Project_Count, Inflation_Rate,
  Previous_Month_Expense, Previous_Quarter_Expense.

Current-period actual expense breakdown components (Operational_Cost, Marketing_Cost,
Travel_Cost, Utility_Cost, Vendor_Cost, Miscellaneous_Cost, Actual_Expense) are
STRICTLY EXCLUDED to prevent target leakage.

Follows Clean Architecture, PEP8 standards, type annotations, and docstrings.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class FinancialFeatureBuilder:
    """Engineers valid pre-prediction time domain features for budget analysis."""

    def __init__(self) -> None:
        """Initialize feature builder."""
        self.engineered_feature_names: List[str] = []

    def transform(
        self, df: pd.DataFrame, include_target_categories: bool = True
    ) -> pd.DataFrame:
        """Transform financial DataFrame into feature-rich DataFrame using pre-prediction features only.

        Args:
            df: Input financial DataFrame.
            include_target_categories: Whether to compute target 'Deviation_Risk_Category'.

        Returns:
            DataFrame augmented with valid engineered domain features.
        """
        df_feat = df.copy()

        # Prevent zero-division using small epsilon
        eps = 1e-5

        # 1. Operational & Headcount Ratios (Pre-Prediction Inputs Only)
        if "Planned_Budget" in df_feat.columns and "Employee_Count" in df_feat.columns:
            df_feat["Planned_Budget_Per_Employee"] = df_feat["Planned_Budget"] / (
                df_feat["Employee_Count"] + eps
            )

        if "Project_Count" in df_feat.columns and "Employee_Count" in df_feat.columns:
            df_feat["Projects_Per_Employee"] = df_feat["Project_Count"] / (
                df_feat["Employee_Count"] + eps
            )

        if "Planned_Budget" in df_feat.columns and "Project_Count" in df_feat.columns:
            df_feat["Planned_Budget_Per_Project"] = df_feat["Planned_Budget"] / (
                df_feat["Project_Count"] + eps
            )

        # 2. Revenue & Budget Allocation Ratios
        if "Revenue" in df_feat.columns and "Planned_Budget" in df_feat.columns:
            df_feat["Revenue_to_Budget_Ratio"] = df_feat["Revenue"] / (
                df_feat["Planned_Budget"] + eps
            )

        # 3. Macroeconomic Inflation Impact Features
        if "Planned_Budget" in df_feat.columns and "Inflation_Rate" in df_feat.columns:
            df_feat["Inflation_Cost_Impact"] = (
                df_feat["Planned_Budget"] * df_feat["Inflation_Rate"]
            )
            df_feat["Inflation_Adjusted_Budget"] = (
                df_feat["Planned_Budget"] * (1 + df_feat["Inflation_Rate"])
            )

        # 4. Trailing Historical Expense Dynamics
        if "Previous_Month_Expense" in df_feat.columns and "Planned_Budget" in df_feat.columns:
            df_feat["Prev_Month_Expense_Ratio"] = df_feat["Previous_Month_Expense"] / (
                df_feat["Planned_Budget"] + eps
            )

        if "Previous_Quarter_Expense" in df_feat.columns and "Planned_Budget" in df_feat.columns:
            df_feat["Prev_Quarter_Expense_Ratio"] = df_feat["Previous_Quarter_Expense"] / (
                df_feat["Planned_Budget"] + eps
            )

        if (
            "Previous_Month_Expense" in df_feat.columns
            and "Previous_Quarter_Expense" in df_feat.columns
        ):
            df_feat["Expense_Momentum_Ratio"] = df_feat["Previous_Month_Expense"] / (
                (df_feat["Previous_Quarter_Expense"] / 3.0) + eps
            )

        # 5. Risk Category Target Generation (if requested and Budget_Deviation present)
        if include_target_categories and "Budget_Deviation" in df_feat.columns:
            df_feat["Deviation_Risk_Category"] = self._create_risk_categories(
                df_feat["Budget_Deviation"],
                df_feat["Planned_Budget"] if "Planned_Budget" in df_feat.columns else None,
            )

        # Track new engineered features
        new_cols = [c for c in df_feat.columns if c not in df.columns]
        self.engineered_feature_names = new_cols

        return df_feat

    @staticmethod
    def _create_risk_categories(
        deviation_series: pd.Series, budget_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """Categorize continuous budget deviation into risk tiers.

        Args:
            deviation_series: Continuous Budget_Deviation values ($).
            budget_series: Planned_Budget values for relative percentage thresholds.

        Returns:
            Categorical Series with risk tiers.
        """
        if budget_series is not None:
            rel_dev = deviation_series / (budget_series + 1e-5)
            conditions = [
                rel_dev < -0.05,
                (rel_dev >= -0.05) & (rel_dev <= 0.05),
                (rel_dev > 0.05) & (rel_dev <= 0.15),
                rel_dev > 0.15,
            ]
        else:
            conditions = [
                deviation_series < -10000,
                (deviation_series >= -10000) & (deviation_series <= 10000),
                (deviation_series > 10000) & (deviation_series <= 50000),
                deviation_series > 50000,
            ]

        choices = [
            "Underbudget",
            "Low Risk (On-Budget)",
            "Moderate Overbudget",
            "Severe Overbudget",
        ]

        return pd.Series(
            np.select(conditions, choices, default="Low Risk (On-Budget)"),
            index=deviation_series.index,
            name="Deviation_Risk_Category",
        )
