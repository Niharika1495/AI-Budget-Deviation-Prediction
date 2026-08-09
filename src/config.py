"""System configuration and constants module for AI Budget Deviation Prediction.

This module serves as the single source of truth for time dimensions, inflation rates,
quarter mappings, department names, random seeds, schema definitions, and file paths.
Follows PEP8, type annotations, and Clean Architecture standards.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# PROJECT DIRECTORY PATHS
# ==============================================================================
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Ensure output directories exist at runtime
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATASET_PATH: Path = RAW_DATA_DIR / "financial_budget_deviation_raw.csv"
VALIDATION_REPORT_JSON_PATH: Path = REPORTS_DIR / "validation_report.json"
VALIDATION_REPORT_TXT_PATH: Path = REPORTS_DIR / "validation_report.txt"

# ==============================================================================
# GENERAL DATASET PARAMETERS
# ==============================================================================
TOTAL_RECORDS: int = 15000
RANDOM_SEED: int = 42

# Missing Value & Outlier Simulation Constraints
MISSING_VALUE_RATE: float = 0.025  # 2.5% missing value target (between 2% and 3%)
OUTLIER_RATE: float = 0.008  # 0.8% realistic financial anomaly rate

# ==============================================================================
# TIME DIMENSIONS
# ==============================================================================
YEARS: Tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

MONTHS: Tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

QUARTER_MAPPING: Dict[str, str] = {
    "January": "Q1",
    "February": "Q1",
    "March": "Q1",
    "April": "Q2",
    "May": "Q2",
    "June": "Q2",
    "July": "Q3",
    "August": "Q3",
    "September": "Q3",
    "October": "Q4",
    "November": "Q4",
    "December": "Q4",
}

# ==============================================================================
# INFLATION RATES (2021 - 2025)
# ==============================================================================
INFLATION_RATES: Dict[int, float] = {
    2021: 0.042,  # 4.2%
    2022: 0.058,  # 5.8%
    2023: 0.061,  # 6.1%
    2024: 0.054,  # 5.4%
    2025: 0.048,  # 4.8%
}

# ==============================================================================
# SEASONAL BEHAVIOR MULTIPLIERS
# ==============================================================================
SEASONAL_MULTIPLIERS: Dict[str, float] = {
    "Q1": 1.00,  # Stable baseline spending
    "Q2": 1.05,  # Moderate growth / project rollout
    "Q3": 1.10,  # Procurement growth & operational scale
    "Q4": 1.22,  # Highest spending, budget utilization, vendor spikes
}

# ==============================================================================
# DEPARTMENTS
# ==============================================================================
DEPARTMENTS: Tuple[str, ...] = (
    "Finance",
    "HR",
    "IT",
    "Marketing",
    "Sales",
    "Operations",
    "Procurement",
    "Research & Development",
)

# ==============================================================================
# DATASET SCHEMA DEFINITION (EXACTLY 21 COLUMNS)
# ==============================================================================
DATASET_COLUMNS: List[str] = [
    "Record_ID",
    "Year",
    "Quarter",
    "Month",
    "Department",
    "Cost_Center",
    "Planned_Budget",
    "Revenue",
    "Employee_Count",
    "Project_Count",
    "Operational_Cost",
    "Marketing_Cost",
    "Travel_Cost",
    "Utility_Cost",
    "Vendor_Cost",
    "Miscellaneous_Cost",
    "Inflation_Rate",
    "Previous_Month_Expense",
    "Previous_Quarter_Expense",
    "Actual_Expense",
    "Budget_Deviation",
]

# Non-key features eligible for missing value simulation (Key IDs/dimensions preserved)
INJECTABLE_MISSING_COLUMNS: List[str] = [
    "Previous_Month_Expense",
    "Previous_Quarter_Expense",
    "Travel_Cost",
    "Miscellaneous_Cost",
    "Utility_Cost",
    "Project_Count",
]
