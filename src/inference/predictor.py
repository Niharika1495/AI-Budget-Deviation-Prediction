"""Production inference engine for AI Budget Deviation Prediction system.

This module provides the BudgetDeviationPredictor class to execute batch and real-time
predictions, computing predicted deviation ($), expected actual expense ($),
risk categorization, cost-driver attribution, and mitigation recommendations.

Follows Clean Architecture, PEP8 standards, type annotations, and docstrings.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.features.feature_builder import FinancialFeatureBuilder
from src.explainability.explainability import FinancialExplainer


class BudgetDeviationPredictor:
    """Production predictor for financial budget deviation estimation and risk assessment."""

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        """Initialize predictor by loading trained model artifacts."""
        self.models_dir: Path = models_dir or MODELS_DIR

        self.regressor_artifact = self._load_joblib(
            self.models_dir / "budget_deviation_regressor.joblib"
        )
        self.classifier_artifact = self._load_joblib(
            self.models_dir / "budget_deviation_classifier.joblib"
        )
        self.encoder_artifact = self._load_joblib(
            self.models_dir / "encoder.joblib"
        )
        self.scaler_artifact = self._load_joblib(
            self.models_dir / "scaler.joblib"
        )

        self.feature_builder = FinancialFeatureBuilder()
        self.explainer = (
            FinancialExplainer(self.regressor_artifact)
            if self.regressor_artifact
            else None
        )

    def _load_joblib(self, path: Path) -> Optional[Any]:
        """Safely load joblib artifact if exists."""
        if path.exists():
            return joblib.load(path)
        return None

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict budget deviation and risk metrics for a single input record dictionary.

        Args:
            input_dict: Dictionary containing raw or semi-processed financial metrics.

        Returns:
            Dictionary containing predictions, risk badges, drivers, and recommendations.
        """
        df_input = pd.DataFrame([input_dict])
        df_pred, pred_results = self.predict_batch(df_input)
        return pred_results[0]

    def predict_batch(
        self, df_raw: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Process batch of financial records and generate predictions and recommendations.

        Args:
            df_raw: DataFrame containing input financial records.

        Returns:
            Tuple of (DataFrame with appended predictions, List of prediction result dicts).
        """
        df_feat = self.feature_builder.transform(df_raw, include_target_categories=False)

        # Encode categorical features if encoder exists
        if self.encoder_artifact is not None and hasattr(self.encoder_artifact, "transform"):
            df_encoded, _ = self.encoder_artifact.transform(df_feat)
        else:
            df_encoded = pd.get_dummies(df_feat, drop_first=True)

        # Scale numerical features if scaler exists
        if self.scaler_artifact is not None and hasattr(self.scaler_artifact, "transform"):
            df_ml, _ = self.scaler_artifact.transform(df_encoded)
        else:
            df_ml = df_encoded.copy()

        # Extract model feature columns
        reg_model = self.regressor_artifact.get("model") if self.regressor_artifact else None
        clf_model = (
            self.classifier_artifact.get("model") if self.classifier_artifact else None
        )
        feature_cols = (
            self.regressor_artifact.get("feature_columns", [])
            if self.regressor_artifact
            else []
        )

        if not feature_cols:
            forbidden_cols = [
                "Record_ID",
                "Actual_Expense",
                "Budget_Deviation",
                "Deviation_Risk_Category",
                "Operational_Cost",
                "Marketing_Cost",
                "Travel_Cost",
                "Utility_Cost",
                "Vendor_Cost",
                "Miscellaneous_Cost",
                "Calculated_Total_Cost",
            ]
            feature_cols = [c for c in df_ml.columns if c not in forbidden_cols]

        # Prepare X matrix with exact columns expected by trained model
        X = pd.DataFrame(index=df_ml.index)
        for col in feature_cols:
            if col in df_ml.columns:
                X[col] = df_ml[col]
            else:
                X[col] = 0.0

        # Fill NaNs with 0
        X = X.fillna(0.0)

        # Predict continuous deviation ($)
        if reg_model is not None:
            pred_deviations = reg_model.predict(X)
        else:
            # Fallback zero deviation if model not loaded
            pred_deviations = np.zeros(len(df_raw))

        # Predict risk category
        if clf_model is not None:
            pred_risk_levels = clf_model.predict(X)
        else:
            pred_risk_levels = np.full(shape=len(df_raw), fill_value="Low Risk (On-Budget)")

        results_list = []
        df_out = df_raw.copy()

        df_out["Predicted_Budget_Deviation"] = np.round(pred_deviations, 2)
        df_out["Predicted_Actual_Expense"] = np.round(
            df_out["Planned_Budget"] + pred_deviations, 2
        )
        df_out["Predicted_Risk_Level"] = pred_risk_levels

        for idx, row in df_raw.iterrows():
            dev = float(pred_deviations[idx])
            planned = float(row.get("Planned_Budget", 1.0))
            expected_actual = planned + dev
            dev_pct = (dev / (planned + 1e-5)) * 100.0
            risk_level = str(pred_risk_levels[idx])

            record_dict = row.to_dict()
            drivers = (
                self.explainer.analyze_local_drivers(record_dict, planned)
                if self.explainer
                else []
            )

            recs = self._generate_recommendations(record_dict, dev, dev_pct, drivers)

            res = {
                "record_id": record_dict.get("Record_ID", f"REC_{idx+1}"),
                "department": record_dict.get("Department", "N/A"),
                "planned_budget": round(planned, 2),
                "predicted_actual_expense": round(expected_actual, 2),
                "predicted_budget_deviation": round(dev, 2),
                "deviation_percentage": round(dev_pct, 2),
                "predicted_risk_level": risk_level,
                "top_cost_drivers": drivers,
                "recommendations": recs,
            }
            results_list.append(res)

        return df_out, results_list

    @staticmethod
    def _generate_recommendations(
        record: Dict[str, Any], dev: float, dev_pct: float, drivers: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate targeted cost optimization recommendations based on prediction metrics.

        Args:
            record: Input financial metrics dictionary.
            dev: Predicted deviation amount ($).
            dev_pct: Percentage deviation (%).
            drivers: Top cost drivers list.

        Returns:
            List of actionable recommendation strings.
        """
        recs = []

        if dev > 0:
            recs.append(
                f"⚠️ Projecting a budget overrun of ${dev:,.2f} ({dev_pct:+.1f}% above planned baseline)."
            )
            if drivers:
                top_driver = drivers[0]
                recs.append(
                    f"🔍 Primary spending surge identified in **{top_driver['driver_name']}** (${top_driver['amount']:,.2f}, accounting for {top_driver['share_of_total_cost_percent']}% of expenditures)."
                )

            vendor_cost = float(record.get("Vendor_Cost", 0.0))
            planned = float(record.get("Planned_Budget", 1.0))
            if vendor_cost / (planned + 1e-5) > 0.25:
                recs.append(
                    "🤝 High vendor expenditure detected: Initiate vendor contract renegotiation and enforce SLA procurement caps."
                )

            travel_cost = float(record.get("Travel_Cost", 0.0))
            if travel_cost / (planned + 1e-5) > 0.15:
                recs.append(
                    "✈️ Travel cost ratio exceeds benchmark: Implement strict pre-approval policies for executive travel."
                )

            mktg_cost = float(record.get("Marketing_Cost", 0.0))
            if mktg_cost / (planned + 1e-5) > 0.30:
                recs.append(
                    "📢 Marketing spend ratio is elevated: Perform campaign ROI audit before approving supplementary Q4 budget allocations."
                )

            recs.append(
                "📉 Implement weekly budget milestone tracking to curb non-essential discretionary operational expenses."
            )
        else:
            recs.append(
                f"✅ Projecting a favorable budget performance of ${abs(dev):,.2f} under planned allocation ({dev_pct:.1f}%)."
            )
            recs.append(
                "💡 Department operations are running efficiently within financial constraints. Reallocate surplus reserves to high-ROI growth initiatives."
            )

        return recs
