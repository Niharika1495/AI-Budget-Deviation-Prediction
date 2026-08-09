"""Data profiling engine for financial dataset analysis.

This module executes automated profiling on cleaned ERP financial data:
- Structural shape & dtypes
- Missing value audit
- Numerical summary statistics (Mean, Median, Std, Skewness, Kurtosis)
- Categorical cardinality & frequency analysis
- Memory footprint calculation
- Composite Data Quality Score evaluation (0–100%)

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


class DataProfiler:
    """Automated data profiling engine for enterprise financial datasets."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize DataProfiler with input Pandas DataFrame.

        Args:
            df: Target DataFrame to profile.
        """
        self.df: pd.DataFrame = df

    def profile(self) -> Dict[str, Any]:
        """Execute complete data profiling suite and return structured stats dictionary.

        Returns:
            Dictionary containing shape, types, nulls, statistical summaries, memory, and quality score.
        """
        total_rows, total_cols = self.df.shape
        memory_bytes = self.df.memory_usage(deep=True).sum()
        memory_mb = round(float(memory_bytes / (1024 * 1024)), 2)

        # 1. Column Dtypes
        dtypes_dict = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

        # 2. Missing Value Analysis
        null_counts = self.df.isnull().sum()
        missing_summary = {
            col: {
                "null_count": int(count),
                "null_percentage": round(float((count / total_rows) * 100.0), 2),
            }
            for col, count in null_counts.items()
        }
        total_nulls = int(null_counts.sum())

        # 3. Duplicate Summary
        duplicates_count = int(self.df.duplicated().sum())

        # 4. Numerical Summary Statistics
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        num_summary: Dict[str, Dict[str, float]] = {}
        for col in num_cols:
            series = self.df[col].dropna()
            num_summary[col] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "q25": round(float(series.quantile(0.25)), 2),
                "median": round(float(series.median()), 2),
                "q75": round(float(series.quantile(0.75)), 2),
                "max": round(float(series.max()), 2),
                "skewness": round(float(series.skew()), 2),
                "kurtosis": round(float(series.kurtosis()), 2),
            }

        # 5. Categorical Summary Statistics
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_summary: Dict[str, Dict[str, Any]] = {}
        for col in cat_cols:
            series = self.df[col].dropna()
            val_counts = series.value_counts()
            top_val = str(val_counts.index[0]) if len(val_counts) > 0 else ""
            top_freq = int(val_counts.iloc[0]) if len(val_counts) > 0 else 0
            cat_summary[col] = {
                "unique_count": int(series.nunique()),
                "most_frequent": top_val,
                "frequency": top_freq,
                "top_categories": val_counts.head(5).to_dict(),
            }

        # 6. Calculate Composite Data Quality Score (0-100%)
        # Deduct penalties for nulls, duplicates, and type issues
        null_penalty = min(30.0, (total_nulls / (total_rows * total_cols)) * 100.0 * 5.0)
        dup_penalty = min(20.0, (duplicates_count / total_rows) * 100.0 * 10.0)
        quality_score = round(max(0.0, 100.0 - null_penalty - dup_penalty), 2)

        return {
            "dataset_shape": {"rows": total_rows, "columns": total_cols},
            "memory_usage_mb": memory_mb,
            "duplicate_rows": duplicates_count,
            "total_nulls": total_nulls,
            "data_quality_score": quality_score,
            "dtypes": dtypes_dict,
            "missing_summary": missing_summary,
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary,
        }
