"""Model training pipeline execution script for AI Budget Deviation Prediction system.

This script loads the preprocessed ML dataset, executes model selection,
training, cross-validation, evaluation, and saves serialized model artifacts.

Follows Clean Architecture, PEP8 standards, type annotations, and docstrings.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATASET_PATH
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.model_trainer import BudgetDeviationModelTrainer


def run_training_pipeline() -> None:
    """Run model training and evaluation end-to-end."""
    print("=" * 70)
    print("      AI BUDGET DEVIATION PREDICTION - MODEL TRAINING PIPELINE     ")
    print("=" * 70)

    # 1. Load ML dataset
    ml_path_parquet = PROCESSED_DATA_DIR / "financial_budget_deviation_ml.parquet"
    ml_path_csv = PROCESSED_DATA_DIR / "financial_budget_deviation_ml.csv"

    if ml_path_parquet.exists():
        print(f"[1/4] Loading preprocessed ML dataset from {ml_path_parquet.name}...")
        df_ml = pd.read_parquet(ml_path_parquet)
    elif ml_path_csv.exists():
        print(f"[1/4] Loading preprocessed ML dataset from {ml_path_csv.name}...")
        df_ml = pd.read_csv(ml_path_csv)
    else:
        print("[1/4] Preprocessed dataset not found. Running preprocessing pipeline...")
        df_raw = pd.read_csv(RAW_DATASET_PATH)
        pipeline = PreprocessingPipeline()
        _, df_ml, _ = pipeline.run(df_raw)

    print(f"      Dataset loaded: {len(df_ml):,} rows x {len(df_ml.columns)} columns.")

    # 2. Initialize and execute trainer
    print("[2/4] Initializing Model Trainer & Feature Generator...")
    trainer = BudgetDeviationModelTrainer()

    print("[3/4] Training candidate Regressors and Classifiers with 5-Fold Cross-Validation...")
    reg_metrics, clf_metrics, reg_path, clf_path = trainer.train_all(df_ml)

    # 3. Print Results Summary
    print("[4/4] Model Training & Persistence Completed Successfully!")
    print("-" * 70)
    print("SUMMARY OF BEST MODELS & PERFORMANCE METRICS:")
    print("-" * 70)
    print(f" [*] Best Regressor Model      : {reg_metrics['model_name']}")
    print(f"     - Root Mean Squared Error : ${reg_metrics['rmse']:,.2f}")
    print(f"     - Mean Absolute Error     : ${reg_metrics['mae']:,.2f}")
    print(f"     - R-Squared (R2) Score    : {reg_metrics['r2_score']:.4f}")
    print(f"     - MAPE (%)                : {reg_metrics['mape_percent']:.2f}%")
    print(f"     - Saved Regressor Path    : {reg_path}")
    print()
    print(f" [*] Best Classifier Model     : {clf_metrics['model_name']}")
    print(f"     - Accuracy                : {clf_metrics['accuracy']*100:.2f}%")
    print(f"     - Weighted F1-Score       : {clf_metrics['f1_weighted']:.4f}")
    print(f"     - Saved Classifier Path   : {clf_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
