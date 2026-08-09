"""Model training orchestrator module for budget deviation prediction.

STRICT ML INTEGRITY COMPLIANCE:
1. Temporal Train/Test Split (2021-2024 TRAIN, 2025 UNSEEN TEST).
2. Fit all preprocessing, scaling, encoding, imputation, and feature selection ONLY on TRAIN.
3. Strict Target Leakage Prevention: Excludes current-period expense breakdown components.
4. Baseline Comparisons: Mean Regression & Majority Class Classification.

Follows Clean Architecture, PEP8 standards, type annotations, and docstrings.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from src.config import MODELS_DIR, RANDOM_SEED
from src.preprocessing.missing_value_handler import MissingValueHandler
from src.preprocessing.outlier_handler import IQROutlierHandler
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.scaler import FeatureScaler
from src.features.feature_builder import FinancialFeatureBuilder
from src.features.feature_selector import FinancialFeatureSelector
from src.models.model_evaluator import ModelEvaluator


class BudgetDeviationModelTrainer:
    """Orchestrates leak-free, temporally valid model training, evaluation, and persistence."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """Initialize trainer parameters and output directory."""
        self.models_dir: Path = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.evaluator = ModelEvaluator(reports_dir=self.models_dir.parent / "reports")

        # Fitted Preprocessors
        self.missing_handler = MissingValueHandler()
        self.outlier_handler = IQROutlierHandler(iqr_multiplier=1.5)
        self.feature_builder = FinancialFeatureBuilder()
        self.encoder = CategoricalEncoder()
        self.scaler = FeatureScaler()
        self.selector = FinancialFeatureSelector(top_k_features=25)

        self.best_regressor: Optional[Any] = None
        self.best_regressor_name: str = ""
        self.best_classifier: Optional[Any] = None
        self.best_classifier_name: str = ""
        self.valid_feature_columns: List[str] = []

    def train_and_audit(
        self, df_raw: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute leak-free pipeline fitting, temporal train/test validation, and evaluation.

        Args:
            df_raw: Original or clean raw DataFrame.

        Returns:
            Audit summary dictionary containing performance, baselines, and lineage metrics.
        """
        # ==============================================================================
        # 1. TEMPORAL TRAIN / TEST SPLIT (TRAIN: 2021-2024, TEST: 2025)
        # ==============================================================================
        train_mask = df_raw["Year"] <= 2024
        test_mask = df_raw["Year"] == 2025

        df_train_raw = df_raw[train_mask].copy()
        df_test_raw = df_raw[test_mask].copy()

        # ==============================================================================
        # 2. FIT PREPROCESSING ONLY ON TRAIN DATA
        # ==============================================================================
        # Imputation
        df_train_imp, _ = self.missing_handler.fit_transform(df_train_raw)
        df_test_imp, _ = self.missing_handler.transform(df_test_raw)

        # Outlier Treatment
        df_train_clean, _ = self.outlier_handler.fit_transform(df_train_imp)
        df_test_clean, _ = self.outlier_handler.transform(df_test_imp)

        # Feature Building (Engineers valid features; target risk tier generated)
        df_train_feat = self.feature_builder.transform(df_train_clean, include_target_categories=True)
        df_test_feat = self.feature_builder.transform(df_test_clean, include_target_categories=True)

        # Categorical One-Hot Encoding (Fit ONLY on Train)
        df_train_encoded, _ = self.encoder.fit_transform(df_train_feat)
        df_test_encoded, _ = self.encoder.transform(df_test_feat)

        # Feature Scaling (Fit ONLY on Train)
        df_train_scaled, _ = self.scaler.fit_transform(df_train_encoded)
        df_test_scaled, _ = self.scaler.transform(df_test_encoded)

        # Save fitted preprocessing objects
        self.encoder.save(self.models_dir / "encoder.joblib")
        self.scaler.save(self.models_dir / "scaler.joblib")

        # ==============================================================================
        # 3. STRICT FEATURE PURGE & SEPARATION (LEAKAGE REMOVAL)
        # ==============================================================================
        # FORBIDDEN COLUMNS: Metadata, targets, and current-period cost breakdown
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
            "Cost_to_Budget_Ratio",
            "Operational_Cost_Ratio",
            "Marketing_Cost_Ratio",
            "Travel_Cost_Ratio",
            "Utility_Cost_Ratio",
            "Vendor_Cost_Ratio",
            "Miscellaneous_Cost_Ratio",
            "Revenue_to_OpCost_Ratio",
        ]

        feature_cols = [c for c in df_train_scaled.columns if c not in forbidden_cols]
        # Keep strictly numeric columns
        X_train_num = df_train_scaled[feature_cols].select_dtypes(include=[np.number]).copy()
        X_test_num = df_test_scaled[feature_cols].select_dtypes(include=[np.number]).copy()

        # Align columns
        common_cols = [c for c in X_train_num.columns if c in X_test_num.columns]
        X_train_raw_feats = X_train_num[common_cols].copy()
        X_test_raw_feats = X_test_num[common_cols].copy()

        # Target Series
        y_reg_train = df_train_feat["Budget_Deviation"]
        y_reg_test = df_test_feat["Budget_Deviation"]

        y_clf_train = df_train_feat["Deviation_Risk_Category"]
        y_clf_test = df_test_feat["Deviation_Risk_Category"]

        # Feature Selection (Fit ONLY on Train)
        X_train = self.selector.fit_transform(X_train_raw_feats, y_reg_train)
        X_test = self.selector.transform(X_test_raw_feats)

        self.valid_feature_columns = list(X_train.columns)

        # ==============================================================================
        # 4. BASELINE EVALUATION
        # ==============================================================================
        reg_baseline = self.evaluator.compute_regression_baseline(y_reg_train, y_reg_test)
        clf_baseline = self.evaluator.compute_classification_baseline(y_clf_train, y_clf_test)

        # ==============================================================================
        # 5. REGRESSOR MODEL SELECTION & TRAINING (5-FOLD CV ON TRAIN 2021-2024)
        # ==============================================================================
        regressors = {
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                random_state=self.random_seed, max_iter=150
            ),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100, random_state=self.random_seed, n_jobs=-1
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_seed
            ),
            "RidgeRegression": Ridge(alpha=1.0),
        }

        best_rmse_cv = float("inf")
        best_reg_model = None
        best_reg_name = ""
        reg_cv_scores_dict: Dict[str, float] = {}

        for name, model in regressors.items():
            cv_scores = cross_val_score(
                model, X_train, y_reg_train, cv=5, scoring="neg_root_mean_squared_error"
            )
            mean_rmse = float(-np.mean(cv_scores))
            reg_cv_scores_dict[name] = round(mean_rmse, 2)
            if mean_rmse < best_rmse_cv:
                best_rmse_cv = mean_rmse
                best_reg_model = model
                best_reg_name = name

        # Fit best regressor on full TRAIN (2021-2024)
        best_reg_model.fit(X_train, y_reg_train)
        self.best_regressor = best_reg_model
        self.best_regressor_name = best_reg_name

        # Evaluate Regressor
        y_reg_train_pred = self.best_regressor.predict(X_train)
        y_reg_test_pred = self.best_regressor.predict(X_test)

        reg_train_metrics = self.evaluator.evaluate_regression(y_reg_train, y_reg_train_pred, model_name=best_reg_name)
        reg_test_metrics = self.evaluator.evaluate_regression(y_reg_test, y_reg_test_pred, model_name=best_reg_name)
        reg_test_metrics["cv_rmse"] = round(best_rmse_cv, 2)

        # ==============================================================================
        # 6. CLASSIFIER MODEL SELECTION & TRAINING (5-FOLD CV ON TRAIN 2021-2024)
        # ==============================================================================
        classifiers = {
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
                random_state=self.random_seed, max_iter=150
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=100, random_state=self.random_seed, n_jobs=-1
            ),
        }

        best_acc_cv = 0.0
        best_clf_model = None
        best_clf_name = ""
        clf_cv_scores_dict: Dict[str, float] = {}

        for name, model in classifiers.items():
            cv_scores = cross_val_score(
                model, X_train, y_clf_train, cv=5, scoring="accuracy"
            )
            mean_acc = float(np.mean(cv_scores))
            clf_cv_scores_dict[name] = round(mean_acc, 4)
            if mean_acc > best_acc_cv:
                best_acc_cv = mean_acc
                best_clf_model = model
                best_clf_name = name

        # Fit best classifier on full TRAIN (2021-2024)
        best_clf_model.fit(X_train, y_clf_train)
        self.best_classifier = best_clf_model
        self.best_classifier_name = best_clf_name

        # Evaluate Classifier
        y_clf_test_pred = self.best_classifier.predict(X_test)
        clf_test_metrics = self.evaluator.evaluate_classification(y_clf_test, y_clf_test_pred, model_name=best_clf_name)
        clf_test_metrics["cv_accuracy"] = round(best_acc_cv, 4)

        # Save Metrics Report
        self.evaluator.save_evaluation_report(reg_test_metrics, clf_test_metrics)

        # Save Trained Model Artifacts
        reg_path = self.models_dir / "budget_deviation_regressor.joblib"
        clf_path = self.models_dir / "budget_deviation_classifier.joblib"

        joblib.dump(
            {
                "model": self.best_regressor,
                "model_name": self.best_regressor_name,
                "feature_columns": self.valid_feature_columns,
            },
            reg_path,
        )

        joblib.dump(
            {
                "model": self.best_classifier,
                "model_name": self.best_classifier_name,
                "feature_columns": self.valid_feature_columns,
            },
            clf_path,
        )

        # Calculate Feature Importances on legitimate predictors
        feature_importances: Dict[str, float] = {}
        if hasattr(self.best_regressor, "feature_importances_"):
            imps = self.best_regressor.feature_importances_
            feature_importances = dict(
                sorted(
                    zip(self.valid_feature_columns, [round(float(x), 4) for x in imps]),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        audit_summary = {
            "train_period": "2021-2024",
            "test_period": "2025 (Unseen)",
            "train_rows": len(df_train_raw),
            "test_rows": len(df_test_raw),
            "forbidden_columns_removed": forbidden_cols,
            "valid_feature_columns": self.valid_feature_columns,
            "regression_baseline": reg_baseline,
            "classification_baseline": clf_baseline,
            "best_regressor": reg_test_metrics,
            "best_regressor_train_metrics": reg_train_metrics,
            "regressor_cv_scores": reg_cv_scores_dict,
            "best_classifier": clf_test_metrics,
            "classifier_cv_scores": clf_cv_scores_dict,
            "feature_importances": feature_importances,
            "regressor_artifact_path": str(reg_path),
            "classifier_artifact_path": str(clf_path),
        }

        return audit_summary
