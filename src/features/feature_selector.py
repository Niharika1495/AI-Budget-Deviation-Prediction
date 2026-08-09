"""Feature selection module for financial budget deviation prediction.

This module provides the FinancialFeatureSelector class to filter non-informative,
constant, highly correlated, or low-importance features before model training.

Follows Clean Architecture, PEP8 standards, type annotations, and docstrings.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class FinancialFeatureSelector:
    """Selects high-impact numerical and categorical features for machine learning."""

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 1e-4,
        top_k_features: Optional[int] = None,
    ) -> None:
        """Initialize feature selector with thresholds.

        Args:
            correlation_threshold: Cutoff for removing collinear numerical features.
            variance_threshold: Minimum variance required for a feature to be retained.
            top_k_features: Optional limit on number of features to retain by importance.
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.top_k_features = top_k_features

        self.selected_features_: List[str] = []
        self.dropped_features_: Dict[str, List[str]] = {
            "low_variance": [],
            "high_correlation": [],
        }
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FinancialFeatureSelector":
        """Fit selector to identify non-redundant, informative features.

        Args:
            X: Feature matrix DataFrame (numerical/encoded).
            y: Target Series (continuous Budget_Deviation).

        Returns:
            Fitted FinancialFeatureSelector instance.
        """
        X_num = X.select_dtypes(include=[np.number]).copy()
        initial_features = list(X_num.columns)

        # 1. Low Variance Filter
        variances = X_num.var(numeric_only=True)
        low_var_cols = list(variances[variances < self.variance_threshold].index)
        self.dropped_features_["low_variance"] = low_var_cols
        retained_features = [f for f in initial_features if f not in low_var_cols]

        # 2. High Correlation Filter
        df_corr = X_num[retained_features].corr().abs()
        upper_tri = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
        high_corr_cols = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] > self.correlation_threshold)
        ]
        self.dropped_features_["high_correlation"] = high_corr_cols
        candidate_features = [f for f in retained_features if f not in high_corr_cols]

        # 3. Tree Importance Ranking (if top_k_features specified)
        if candidate_features and self.top_k_features is not None:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_num[candidate_features].fillna(0), y)
            importances = dict(zip(candidate_features, rf.feature_importances_))
            self.feature_importances_ = dict(
                sorted(importances.items(), key=lambda item: item[1], reverse=True)
            )

            sorted_cols = list(self.feature_importances_.keys())
            self.selected_features_ = sorted_cols[: self.top_k_features]
        else:
            self.selected_features_ = candidate_features

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X to retain only selected features.

        Args:
            X: Input feature DataFrame.

        Returns:
            DataFrame containing only selected features.
        """
        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]
        return X[cols_to_keep]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform feature DataFrame.

        Args:
            X: Feature matrix DataFrame.
            y: Target Series.

        Returns:
            Filtered DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)
