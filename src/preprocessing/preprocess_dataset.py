"""CLI runner script for Phase 2 Financial Dataset Preprocessing.

This script coordinates end-to-end preprocessing:
1. Load raw dataset (data/raw/financial_budget_deviation_raw.csv)
2. Instantiate and run PreprocessingPipeline
3. Export two processed datasets:
   - Clean Dataset (Unencoded, Unscaled for EDA):
     data/processed/financial_budget_deviation_clean.csv
     data/processed/financial_budget_deviation_clean.parquet
   - ML Dataset (One-Hot Encoded, StandardScaled for Training):
     data/processed/financial_budget_deviation_ml.csv
     data/processed/financial_budget_deviation_ml.parquet
4. Save serialized model artifacts:
   - models/encoder.joblib
   - models/scaler.joblib
   - models/preprocessing_pipeline.joblib
5. Generate JSON and styled HTML preprocessing reports

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict
import pandas as pd

# Add project root to sys.path to allow direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATASET_PATH,
    REPORTS_DIR,
)
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

# Output Dataset Paths
CLEAN_CSV_PATH: Path = PROCESSED_DATA_DIR / "financial_budget_deviation_clean.csv"
CLEAN_PARQUET_PATH: Path = PROCESSED_DATA_DIR / "financial_budget_deviation_clean.parquet"
ML_CSV_PATH: Path = PROCESSED_DATA_DIR / "financial_budget_deviation_ml.csv"
ML_PARQUET_PATH: Path = PROCESSED_DATA_DIR / "financial_budget_deviation_ml.parquet"

# Output Report Paths
PREPROCESSING_REPORT_JSON_PATH: Path = REPORTS_DIR / "preprocessing_report.json"
PREPROCESSING_REPORT_HTML_PATH: Path = REPORTS_DIR / "preprocessing_report.html"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess_dataset")


def generate_html_report(summary: Dict[str, Any], execution_time: float) -> str:
    """Generate a responsive HTML preprocessing report covering Clean & ML datasets.

    Args:
        summary: Structured summary dictionary from pipeline execution.
        execution_time: Pipeline execution duration in seconds.

    Returns:
        String containing complete HTML document markup.
    """
    missing_before = summary.get("missing_value_handling", {}).get("total_nulls_before", 0)
    missing_after = summary.get("missing_value_handling", {}).get("total_nulls_after", 0)
    outliers_capped = summary.get("outlier_treatment", {}).get("total_outliers_capped", 0)
    encoded_count = summary.get("encoding", {}).get("encoded_feature_count", 0)
    scaled_count = summary.get("scaling", {}).get("scaled_feature_count", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Preprocessing Pipeline Report - AI Budget Deviation</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-green: #4ade80;
            --accent-purple: #c084fc;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            color: var(--accent);
        }}
        .badge {{
            background-color: rgba(74, 222, 128, 0.15);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 35px;
        }}
        .kpi-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--accent);
            margin-top: 5px;
        }}
        .kpi-label {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        .section h2 {{
            margin-top: 0;
            font-size: 20px;
            color: var(--accent);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            font-size: 13px;
            text-transform: uppercase;
        }}
        tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .steps-list {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        .step-pill {{
            background-color: rgba(56, 189, 248, 0.1);
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 6px 14px;
            border-radius: 15px;
            font-size: 13px;
        }}
        .dataset-box {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .box {{
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px dashed var(--border-color);
            border-radius: 8px;
            padding: 15px;
        }}
        .box h3 {{
            margin-top: 0;
            color: var(--accent-purple);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>AI Financial Budget Deviation - Preprocessing Report</h1>
                <div style="color: var(--text-muted); font-size: 14px; margin-top: 5px;">
                    Phase 2 Enterprise Dual-Dataset & Pipeline Model Report
                </div>
            </div>
            <div class="badge">STATUS: PASSED</div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Records</div>
                <div class="kpi-value">{summary.get('clean_rows', 0):,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Clean Features (EDA)</div>
                <div class="kpi-value" style="color: var(--accent-purple);">{summary.get('clean_cols', 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">ML Features (Model)</div>
                <div class="kpi-value" style="color: var(--accent);">{summary.get('ml_cols', 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Missing Imputed</div>
                <div class="kpi-value" style="color: var(--accent-green);">{missing_before:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Outliers Capped</div>
                <div class="kpi-value">{outliers_capped:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Execution Time</div>
                <div class="kpi-value">{execution_time:.2f}s</div>
            </div>
        </div>

        <div class="section">
            <h2>Dual-Dataset Architectural Separation</h2>
            <div class="dataset-box">
                <div class="box">
                    <h3>1. Clean Dataset (For EDA & Analysis)</h3>
                    <p><strong>Paths:</strong><br><code>data/processed/financial_budget_deviation_clean.csv</code><br><code>data/processed/financial_budget_deviation_clean.parquet</code></p>
                    <p><strong>Description:</strong> Imputed missing values & IQR capped outliers without One-Hot Encoding or StandardScaler. Retains original text categories for interpretable charts and domain visualization.</p>
                </div>
                <div class="box">
                    <h3>2. ML Dataset (For Model Training)</h3>
                    <p><strong>Paths:</strong><br><code>data/processed/financial_budget_deviation_ml.csv</code><br><code>data/processed/financial_budget_deviation_ml.parquet</code></p>
                    <p><strong>Description:</strong> Full One-Hot Encoding ({encoded_count} binary columns) and StandardScaler normalization ({scaled_count} features). Zero nulls, ready for scikit-learn training.</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Pipeline Execution Steps</h2>
            <div class="steps-list">
                {"".join([f'<div class="step-pill">✔ {step}</div>' for step in summary.get("steps_completed", [])])}
            </div>
        </div>

        <div class="section">
            <h2>Missing Value Imputation Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature Name</th>
                        <th>Missing Values Before</th>
                        <th>Imputation Strategy</th>
                        <th>Imputed Median Value</th>
                        <th>Missing Values After</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([
                        f"<tr><td>{col}</td><td>{details['missing_before']:,}</td><td>Median Imputation</td><td>{details['imputed_value']:,.2f}</td><td style='color:var(--accent-green); font-weight:600;'>{details['missing_after']}</td></tr>"
                        for col, details in summary.get("missing_value_handling", {}).get("column_details", {}).items()
                    ])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>IQR Outlier Capping Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature Name</th>
                        <th>Outliers Capped</th>
                        <th>Lower Bound</th>
                        <th>Upper Bound</th>
                        <th>Min (Before / After)</th>
                        <th>Max (Before / After)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([
                        f"<tr><td>{col}</td><td>{details['outliers_capped']:,}</td><td>{details['lower_bound']:,.2f}</td><td>{details['upper_bound']:,.2f}</td><td>{details['before_stats']['min']:,.2f} / {details['after_stats']['min']:,.2f}</td><td>{details['before_stats']['max']:,.2f} / {details['after_stats']['max']:,.2f}</td></tr>"
                        for col, details in summary.get("outlier_treatment", {}).get("column_details", {}).items()
                        if details['outliers_capped'] > 0
                    ])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Serialized Inference Artifacts</h2>
            <p>✔ <code>models/encoder.joblib</code> (OneHotEncoder mapping)</p>
            <p>✔ <code>models/scaler.joblib</code> (StandardScaler feature parameters)</p>
            <p>✔ <code>models/preprocessing_pipeline.joblib</code> (Full End-to-End Inference Pipeline Object)</p>
        </div>
    </div>
</body>
</html>
"""
    return html


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to convert NumPy data types to native Python types."""

    def default(self, obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main() -> None:
    """Execute main Phase 2 preprocessing workflow."""
    start_time = time.time()
    logger.info("Starting Phase 2 Dual-Dataset Preprocessing Engine...")

    # 1. Load Raw Dataset
    logger.info("Loading raw dataset from %s ...", RAW_DATASET_PATH)
    if not RAW_DATASET_PATH.exists():
        logger.error("Raw dataset not found at %s. Please run Phase 1 dataset generator first.", RAW_DATASET_PATH)
        sys.exit(1)

    df_raw = pd.read_csv(RAW_DATASET_PATH)
    logger.info("Raw dataset successfully loaded. Shape: %s", df_raw.shape)

    # 2. Run Preprocessing Pipeline
    logger.info("Instantiating PreprocessingPipeline...")
    pipeline = PreprocessingPipeline()

    logger.info("Running preprocessing pipeline steps (generating Clean and ML datasets)...")
    df_clean, df_ml, summary = pipeline.run(df_raw)
    
    elapsed_time = time.time() - start_time
    summary["execution_time_seconds"] = round(elapsed_time, 4)

    logger.info("Pipeline executed successfully in %.2f seconds.", elapsed_time)
    logger.info("Clean Dataset Shape (EDA): %s", df_clean.shape)
    logger.info("ML Dataset Shape (Training): %s", df_ml.shape)

    # 3. Export Processed Datasets (Clean & ML versions in CSV and Parquet)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Export Clean Datasets for EDA
    logger.info("Exporting Clean CSV to %s ...", CLEAN_CSV_PATH)
    df_clean.to_csv(CLEAN_CSV_PATH, index=False)

    logger.info("Exporting Clean Parquet to %s ...", CLEAN_PARQUET_PATH)
    df_clean.to_parquet(CLEAN_PARQUET_PATH, index=False, engine="pyarrow")

    # Export ML Datasets for Model Training
    logger.info("Exporting ML CSV to %s ...", ML_CSV_PATH)
    df_ml.to_csv(ML_CSV_PATH, index=False)

    logger.info("Exporting ML Parquet to %s ...", ML_PARQUET_PATH)
    df_ml.to_parquet(ML_PARQUET_PATH, index=False, engine="pyarrow")

    # 4. Export Execution Reports (JSON & HTML)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Saving preprocessing JSON report to %s ...", PREPROCESSING_REPORT_JSON_PATH)
    with open(PREPROCESSING_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)

    logger.info("Saving preprocessing HTML report to %s ...", PREPROCESSING_REPORT_HTML_PATH)
    html_report = generate_html_report(summary, elapsed_time)
    with open(PREPROCESSING_REPORT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_report)

    logger.info("Phase 2 Preprocessing Pipeline completed with 100%% SUCCESS.")


if __name__ == "__main__":
    main()
