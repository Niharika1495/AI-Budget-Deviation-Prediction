"""Correlation analysis module for identifying primary budget deviation drivers.

This module evaluates:
- Pearson correlation matrix across all numeric features
- Top positive and negative feature correlations
- Target correlation ranking with Budget_Deviation
- Primary financial levers influencing cost overruns

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


class CorrelationAnalyzer:
    """Target correlation driver analysis engine."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize CorrelationAnalyzer with cleaned Pandas DataFrame.

        Args:
            df: Cleaned financial DataFrame.
        """
        self.df: pd.DataFrame = df
        self.target_col: str = "Budget_Deviation"

    def analyze_correlations(self) -> Dict[str, Any]:
        """Perform correlation analysis and rank target drivers.

        Returns:
            Dictionary containing target correlation rankings, top positive/negative correlations.
        """
        numeric_df = self.df.select_dtypes(include=[np.number]).dropna()
        corr_matrix = numeric_df.corr()

        if self.target_col not in corr_matrix.columns:
            return {}

        target_corrs = corr_matrix[self.target_col].drop(self.target_col).sort_values(ascending=False)

        top_positive = target_corrs[target_corrs > 0].head(5).to_dict()
        top_negative = target_corrs[target_corrs < 0].tail(5).to_dict()

        # Unstack pairwise correlations to find overall strongest pairs
        unstacked = corr_matrix.unstack()
        # Filter out self-correlations
        unstacked = unstacked[unstacked.index.get_level_values(0) != unstacked.index.get_level_values(1)]
        # Remove duplicate pairs
        pair_corrs = {}
        for (f1, f2), val in unstacked.items():
            pair_key = tuple(sorted([f1, f2]))
            if pair_key not in pair_corrs:
                pair_corrs[pair_key] = float(val)

        sorted_pairs = sorted(pair_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        top_pairwise_correlations = [
            {"feature_1": k[0], "feature_2": k[1], "correlation": round(v, 4)}
            for k, v in sorted_pairs[:10]
        ]

        return {
            "target_variable": self.target_col,
            "target_correlation_ranking": {k: round(float(v), 4) for k, v in target_corrs.items()},
            "top_positive_target_drivers": {k: round(float(v), 4) for k, v in top_positive.items()},
            "top_negative_target_drivers": {k: round(float(v), 4) for k, v in top_negative.items()},
            "top_pairwise_correlations": top_pairwise_correlations,
        }
