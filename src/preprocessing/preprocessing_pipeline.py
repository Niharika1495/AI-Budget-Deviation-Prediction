"""Preprocessing pipeline orchestrator module.

This module encapsulates the end-to-end preprocessing pipeline execution:
1. Raw Dataset Loading & Schema Validation
2. Intelligent Missing Value Median Imputation
3. IQR Outlier Detection & Capping (Produces Clean Dataset for EDA)
4. One-Hot Categorical Encoding
5. StandardScaler Numerical Feature Scaling (Produces ML Dataset for Training)
6. Post-Preprocessing Integrity Validation
7. Artifact Model Persistence (encoder, scaler, and full pipeline joblib files)

Follows Clean Architecture, PEP8 standards, SOLID design, type annotations, and docstrings.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.missing_value_handler import MissingValueHandler
from src.preprocessing.outlier_handler import IQROutlierHandler
from src.preprocessing.preprocessing_validator import PreprocessedDataValidator
from src.preprocessing.scaler import FeatureScaler
from src.preprocessing.schema_validator import RawSchemaValidator


class PreprocessingPipeline:
    """Orchestrator for financial dataset transformation, scaling, and quality validation."""

    def __init__(
        self,
        encoder_save_path: Optional[Path] = None,
        scaler_save_path: Optional[Path] = None,
        pipeline_save_path: Optional[Path] = None,
    ) -> None:
        """Initialize pipeline sub-components and persistence paths."""
        self.schema_validator = RawSchemaValidator()
        self.missing_handler = MissingValueHandler()
        self.outlier_handler = IQROutlierHandler(iqr_multiplier=1.5)
        self.encoder = CategoricalEncoder()
        self.scaler = FeatureScaler()
        self.post_validator = PreprocessedDataValidator()

        self.encoder_save_path: Path = encoder_save_path or (MODELS_DIR / "encoder.joblib")
        self.scaler_save_path: Path = scaler_save_path or (MODELS_DIR / "scaler.joblib")
        self.pipeline_save_path: Path = pipeline_save_path or (MODELS_DIR / "preprocessing_pipeline.joblib")

    def run(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Run complete preprocessing pipeline on raw financial DataFrame.

        Args:
            df_raw: Original raw DataFrame loaded from data/raw/financial_budget_deviation_raw.csv.

        Returns:
            Tuple of (df_clean, df_ml, execution_summary_report_dict).
                - df_clean: Imputed & outlier-capped unscaled/unencoded dataset for EDA.
                - df_ml: Encoded & standard-scaled dataset ready for ML training.

        Raises:
            ValueError: If raw schema validation fails.
            RuntimeError: If post-preprocessing validation fails.
        """
        summary: Dict[str, Any] = {
            "initial_rows": len(df_raw),
            "initial_cols": len(df_raw.columns),
            "steps_completed": [],
        }

        # Step 1: Raw Schema Validation
        is_schema_valid, schema_report = self.schema_validator.validate(df_raw)
        summary["schema_validation"] = schema_report
        if not is_schema_valid:
            raise ValueError(f"Raw dataset schema validation failed: {schema_report['errors']}")
        summary["steps_completed"].append("Schema Validation")

        # Step 2: Missing Value Median Imputation
        df_imputed, missing_report = self.missing_handler.fit_transform(df_raw)
        summary["missing_value_handling"] = missing_report
        summary["steps_completed"].append("Missing Value Handling")

        # Step 3: IQR Outlier Detection and Capping -> CLEAN DATASET FOR EDA
        df_clean, outlier_report = self.outlier_handler.fit_transform(df_imputed)
        summary["outlier_treatment"] = outlier_report
        summary["steps_completed"].append("Outlier Treatment (Clean Dataset Produced)")

        # Step 4: Categorical One-Hot Encoding
        df_encoded, encoding_report = self.encoder.fit_transform(df_clean)
        summary["encoding"] = encoding_report
        summary["steps_completed"].append("Categorical Encoding")

        # Save Fitted Encoder Object
        self.encoder.save(self.encoder_save_path)
        summary["encoder_model_saved_path"] = str(self.encoder_save_path)

        # Step 5: Numerical Feature Scaling (StandardScaler) -> ML DATASET FOR TRAINING
        df_ml, scaling_report = self.scaler.fit_transform(df_encoded)
        summary["scaling"] = scaling_report
        summary["steps_completed"].append("Numerical Scaling (ML Dataset Produced)")

        # Save Fitted Scaler Object
        self.scaler.save(self.scaler_save_path)
        summary["scaler_model_saved_path"] = str(self.scaler_save_path)

        # Step 6: Post-Preprocessing Integrity Validation
        is_post_valid, post_report = self.post_validator.validate(
            df_raw=df_raw, df_processed=df_ml
        )
        summary["post_validation"] = post_report
        if not is_post_valid:
            raise RuntimeError(f"Post-preprocessing validation failed: {post_report['errors']}")
        summary["steps_completed"].append("Post Validation")

        # Save Complete Pipeline Object for Production Inference
        self.save(self.pipeline_save_path)
        summary["pipeline_model_saved_path"] = str(self.pipeline_save_path)

        summary["clean_rows"] = len(df_clean)
        summary["clean_cols"] = len(df_clean.columns)
        summary["ml_rows"] = len(df_ml)
        summary["ml_cols"] = len(df_ml.columns)

        return df_clean, df_ml, summary

    def save(self, file_path: Path) -> None:
        """Serialize complete fitted pipeline object to disk using joblib.

        Args:
            file_path: Target .joblib file path.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "PreprocessingPipeline":
        """Load serialized PreprocessingPipeline instance from disk.

        Args:
            file_path: Path to serialized joblib file.

        Returns:
            Restored PreprocessingPipeline instance.
        """
        return joblib.load(file_path)
