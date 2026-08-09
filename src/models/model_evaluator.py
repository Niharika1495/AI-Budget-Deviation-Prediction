"""Model evaluation module for financial budget deviation prediction.

This module provides the ModelEvaluator class to compute comprehensive evaluation metrics,
cross-validation summaries, confusion matrices, per-class metrics, baselines, and reports.

Follows Clean Architecture, PEP8 standards, type annotations, and docstrings.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.config import REPORTS_DIR


class ModelEvaluator:
    """Evaluates regression and classification models on financial budget prediction tasks."""

    def __init__(self, reports_dir: Optional[Path] = None) -> None:
        """Initialize evaluator and output reporting directory."""
        self.reports_dir: Path = reports_dir or REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_regression(
        self, y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Regressor"
    ) -> Dict[str, Any]:
        """Compute regression metrics (MAE, RMSE, R2, MAPE, Max Error).

        Args:
            y_true: True continuous target values ($ Budget Deviation).
            y_pred: Predicted target values ($ Budget Deviation).
            model_name: Name of evaluated model.

        Returns:
            Dictionary of regression metrics.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        # Mean Absolute Percentage Error with epsilon
        non_zero_mask = np.abs(y_true) > 1e-3
        if np.any(non_zero_mask):
            mape = float(
                np.mean(
                    np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
                )
                * 100
            )
        else:
            mape = 0.0

        max_err = float(np.max(np.abs(y_true - y_pred)))

        metrics = {
            "model_name": model_name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 4),
            "mape_percent": round(mape, 2),
            "max_error": round(max_err, 2),
        }
        return metrics

    def compute_regression_baseline(
        self, y_train: pd.Series, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Compute mean prediction baseline for regression.

        Args:
            y_train: Training continuous target values.
            y_test: Test continuous target values.

        Returns:
            Regression metrics dictionary for Mean Baseline.
        """
        mean_val = float(y_train.mean())
        y_pred_baseline = np.full(shape=len(y_test), fill_value=mean_val)
        return self.evaluate_regression(
            y_test, y_pred_baseline, model_name="Mean_Baseline"
        )

    def evaluate_classification(
        self, y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Classifier"
    ) -> Dict[str, Any]:
        """Compute classification metrics for risk tiers.

        Args:
            y_true: True categorical risk labels.
            y_pred: Predicted categorical risk labels.
            model_name: Name of evaluated model.

        Returns:
            Dictionary containing accuracy, weighted/macro precision/recall/f1, per-class breakdown, and confusion matrix.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        labels = sorted(list(set(y_true).union(set(y_pred))))
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        cls_report = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )

        # Per-class metrics
        per_class_metrics = {}
        for lbl in labels:
            if lbl in cls_report:
                per_class_metrics[lbl] = {
                    "precision": round(float(cls_report[lbl]["precision"]), 4),
                    "recall": round(float(cls_report[lbl]["recall"]), 4),
                    "f1_score": round(float(cls_report[lbl]["f1-score"]), 4),
                    "support": int(cls_report[lbl]["support"]),
                }

        metrics = {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision_weighted": round(float(prec_w), 4),
            "recall_weighted": round(float(rec_w), 4),
            "f1_weighted": round(float(f1_w), 4),
            "precision_macro": round(float(prec_m), 4),
            "recall_macro": round(float(rec_m), 4),
            "f1_macro": round(float(f1_m), 4),
            "labels": labels,
            "confusion_matrix": cm,
            "per_class_metrics": per_class_metrics,
        }
        return metrics

    def compute_classification_baseline(
        self, y_train: pd.Series, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Compute majority class prediction baseline for classification.

        Args:
            y_train: Training risk tier labels.
            y_test: Test risk tier labels.

        Returns:
            Classification metrics dictionary for Majority Class Baseline.
        """
        majority_class = str(y_train.mode()[0])
        y_pred_baseline = np.full(shape=len(y_test), fill_value=majority_class)
        return self.evaluate_classification(
            y_test, y_pred_baseline, model_name="Majority_Class_Baseline"
        )

    def save_evaluation_report(
        self,
        regression_metrics: Dict[str, Any],
        classification_metrics: Dict[str, Any],
        filename_prefix: str = "model_evaluation_report",
    ) -> Tuple[Path, Path]:
        """Save comprehensive evaluation metrics to JSON and TXT report files.

        Args:
            regression_metrics: Evaluated regression results.
            classification_metrics: Evaluated classification results.
            filename_prefix: Base filename for output artifacts.

        Returns:
            Tuple of (json_file_path, txt_file_path).
        """
        report_data = {
            "regression_evaluation": regression_metrics,
            "classification_evaluation": classification_metrics,
        }

        json_path = self.reports_dir / f"{filename_prefix}.json"
        txt_path = self.reports_dir / f"{filename_prefix}.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("======================================================================\n")
            f.write("         AI BUDGET DEVIATION PREDICTION - MODEL EVALUATION REPORT     \n")
            f.write("======================================================================\n\n")

            f.write("--- 1. REGRESSION MODEL PERFORMANCE (BUDGET DEVIATION $) ---\n")
            reg = regression_metrics
            f.write(f"Model Architecture     : {reg.get('model_name', 'N/A')}\n")
            f.write(f"Root Mean Squared Error: ${reg.get('rmse', 0.0):,.2f}\n")
            f.write(f"Mean Absolute Error    : ${reg.get('mae', 0.0):,.2f}\n")
            f.write(f"R-Squared (R²) Score   : {reg.get('r2_score', 0.0):.4f}\n")
            f.write(f"MAPE (%)               : {reg.get('mape_percent', 0.0):.2f}%\n\n")

            f.write("--- 2. RISK CLASSIFICATION MODEL PERFORMANCE ---\n")
            clf = classification_metrics
            f.write(f"Model Architecture     : {clf.get('model_name', 'N/A')}\n")
            f.write(f"Overall Accuracy       : {clf.get('accuracy', 0.0)*100:.2f}%\n")
            f.write(f"Weighted F1-Score      : {clf.get('f1_weighted', 0.0):.4f}\n")
            f.write(f"Macro F1-Score         : {clf.get('f1_macro', 0.0):.4f}\n\n")

            f.write("----------------------------------------------------------------------\n")
            f.write("End of Report\n")

        return json_path, txt_path
