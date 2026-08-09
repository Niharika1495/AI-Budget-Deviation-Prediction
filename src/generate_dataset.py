"""Main dataset synthesis orchestration script for AI Budget Deviation Prediction.

This script coordinates the entire Phase 1 data architecture:
1. System configuration and random seed initialization
2. Department business profile loading
3. ERP Financial record generation (15,000 records)
4. Comprehensive data quality and mathematical integrity validation
5. Exporting raw dataset to data/raw/financial_budget_deviation_raw.csv
6. Writing JSON and TXT validation reports to reports/

Follows SOLID design principles, Clean Architecture, PEP8, type hints, and robust logging.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path to allow execution directly via `python src/generate_dataset.py`
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    RANDOM_SEED,
    RAW_DATASET_PATH,
    TOTAL_RECORDS,
    VALIDATION_REPORT_JSON_PATH,
    VALIDATION_REPORT_TXT_PATH,
)
from src.data_generator import generate_synthetic_dataset
from src.validator import generate_validation_report_text, validate_dataset

# Setup Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_dataset")


def main() -> None:
    """Orchestrate dataset generation, validation, and file exports."""
    start_time = time.time()
    logger.info("Starting AI Financial Budget Deviation Dataset Synthesis Engine...")
    logger.info("Target Dataset Size: %d records", TOTAL_RECORDS)
    logger.info("Random Seed: %d", RANDOM_SEED)

    # 1. Synthesize Dataset
    logger.info("Synthesizing ERP financial records across 8 departments (2021-2025)...")
    df = generate_synthetic_dataset(num_records=TOTAL_RECORDS, seed=RANDOM_SEED)
    logger.info("Successfully generated %d records with shape %s", len(df), df.shape)

    # 2. Validate Dataset
    logger.info("Executing comprehensive data quality and mathematical validation...")
    validation_result = validate_dataset(df)

    # 3. Print Validation Report Summary
    report_text = generate_validation_report_text(validation_result)
    print("\n" + report_text + "\n")

    # 4. Export Raw Dataset to CSV
    logger.info("Exporting raw CSV dataset to %s ...", RAW_DATASET_PATH)
    df.to_csv(RAW_DATASET_PATH, index=False)
    logger.info("Dataset successfully saved to %s", RAW_DATASET_PATH)

    # 5. Export Validation Reports (TXT & JSON)
    logger.info("Saving validation reports to %s ...", VALIDATION_REPORT_TXT_PATH)
    with open(VALIDATION_REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(VALIDATION_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(validation_result.to_dict(), f, indent=4)

    elapsed = time.time() - start_time
    logger.info("Phase 1 Generation Pipeline completed successfully in %.2f seconds.", elapsed)


if __name__ == "__main__":
    main()
