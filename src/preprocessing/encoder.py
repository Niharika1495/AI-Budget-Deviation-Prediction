"""Categorical encoding module for financial dataset preprocessing.

This module applies One-Hot Encoding (OHE) to categorical dimensions (Department, Quarter, Month)
using scikit-learn's OneHotEncoder. Retains feature mappings and supports joblib model persistence.

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class CategoricalEncoder:
    """One-Hot Encoder wrapper for categorical financial features with joblib persistence."""

    def __init__(self) -> None:
        """Initialize encoder with target categorical columns and scikit-learn OneHotEncoder."""
        self.categorical_columns: List[str] = ["Department", "Quarter", "Month"]
        self.encoder: OneHotEncoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=float
        )
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """Fit OneHotEncoder on categorical columns.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Fitted CategoricalEncoder instance.
        """
        existing_cats = [col for col in self.categorical_columns if col in df.columns]
        if existing_cats:
            self.encoder.fit(df[existing_cats])
            self.feature_names_ = list(self.encoder.get_feature_names_out(existing_cats))
            self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transform categorical columns into One-Hot Encoded features.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Tuple of (encoded_dataframe, encoding_summary_dict).

        Raises:
            RuntimeError: If encoder has not been fitted prior to transform.
        """
        if not self.is_fitted:
            raise RuntimeError("CategoricalEncoder must be fitted before calling transform().")

        df_result = df.copy()

        existing_cats = [col for col in self.categorical_columns if col in df_result.columns]

        if existing_cats:
            # Perform OneHot Encoding transformation
            encoded_array = self.encoder.transform(df_result[existing_cats])
            encoded_df = pd.DataFrame(
                encoded_array, columns=self.feature_names_, index=df_result.index
            )

            # Drop original categorical columns and concatenate encoded binary features
            df_result = df_result.drop(columns=existing_cats)

            # Insert encoded columns before target columns
            df_result = pd.concat([df_result, encoded_df], axis=1)

        summary: Dict[str, Any] = {
            "original_categorical_columns": existing_cats,
            "encoded_feature_count": len(self.feature_names_),
            "new_column_names": self.feature_names_,
            "categories_per_feature": {
                col: [str(c) for c in cats]
                for col, cats in zip(existing_cats, self.encoder.categories_)
            },
        }

        return df_result, summary

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit encoder and transform DataFrame in a single step.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (encoded_dataframe, encoding_summary_dict).
        """
        self.fit(df)
        return self.transform(df)

    def save(self, file_path: Path) -> None:
        """Serialize fitted encoder object to disk using joblib.

        Args:
            file_path: Destination file path (.joblib).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted CategoricalEncoder.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "CategoricalEncoder":
        """Load serialized CategoricalEncoder from disk.

        Args:
            file_path: Path to serialized joblib file.

        Returns:
            Restored CategoricalEncoder instance.
        """
        return joblib.load(file_path)
