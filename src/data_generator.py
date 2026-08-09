"""Data generator engine for synthesizing realistic financial ERP records.

This module provides functional primitives and orchestration to generate 15,000 production-grade
synthetic financial records. It models department profiles, time-series dynamics, employee scaling,
project workload, seasonal multipliers, inflation adjustments, and target calculations:
    Actual Expense = Sum(Expense Components) + Business Variation
    Budget Deviation = Actual Expense - Planned Budget
Follows SOLID principles, type annotations, PEP8, and Clean Architecture standards.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DATASET_COLUMNS,
    DEPARTMENTS,
    INFLATION_RATES,
    INJECTABLE_MISSING_COLUMNS,
    MISSING_VALUE_RATE,
    MONTHS,
    OUTLIER_RATE,
    QUARTER_MAPPING,
    RANDOM_SEED,
    SEASONAL_MULTIPLIERS,
    TOTAL_RECORDS,
    YEARS,
)
from src.department_rules import DepartmentProfile, get_department_profile


def calculate_scaled_headcount_and_projects(
    profile: DepartmentProfile, rng: np.random.Generator
) -> Tuple[int, int]:
    """Generate random headcount and active projects within department bounds.

    Args:
        profile: Department business profile.
        rng: NumPy Random Generator instance for reproducibility.

    Returns:
        Tuple containing (employee_count, project_count).
    """
    employee_count = int(
        rng.integers(profile.employee_count_range[0], profile.employee_count_range[1] + 1)
    )
    project_count = int(
        rng.integers(profile.project_count_range[0], profile.project_count_range[1] + 1)
    )
    return employee_count, project_count


def calculate_planned_budget_and_revenue(
    profile: DepartmentProfile,
    employee_count: int,
    quarter: str,
    year: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Calculate planned budget and revenue based on headcount, seasonal demand, and year.

    Business Rules:
        - Higher revenue departments scale budget proportionally (Revenue -> Budget).
        - Higher headcount moderately elevates base planned budget.

    Args:
        profile: Department profile.
        employee_count: Number of employees.
        quarter: Quarter string (Q1-Q4).
        year: Year integer.
        rng: Random generator instance.

    Returns:
        Tuple of (planned_budget, revenue).
    """
    min_b, max_b = profile.budget_range
    min_r, max_r = profile.revenue_range

    # Generate baseline revenue (Sales generates significant revenue, others 0)
    if max_r > 0.0:
        base_revenue = rng.uniform(min_r, max_r)
        # Seasonal boost for revenue in Q3/Q4
        season_mod = 1.15 if quarter in ("Q3", "Q4") else 1.0
        revenue = round(base_revenue * season_mod, 2)
    else:
        revenue = 0.0

    # Base budget scaled within profile range
    base_budget = rng.uniform(min_b, max_b)

    # Business Dynamics: Headcount & Revenue scale planned budget
    headcount_factor = 1.0 + (
        (employee_count - profile.employee_count_range[0])
        / max(1, (profile.employee_count_range[1] - profile.employee_count_range[0]))
    ) * 0.15
    revenue_factor = 1.0 + (revenue / max_r * 0.10) if max_r > 0.0 else 1.0

    planned_budget = round(base_budget * headcount_factor * revenue_factor, 2)
    return planned_budget, revenue


def calculate_department_expenses(
    profile: DepartmentProfile,
    planned_budget: float,
    employee_count: int,
    project_count: int,
    quarter: str,
    year: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], float]:
    """Calculate individual cost breakdown components applying business relationships.

    Business Logic Relationships:
        - Employee Count -> Increases Operational Cost
        - Project Count -> Increases Vendor Cost
        - Inflation Rate -> Escalates Utility & Vendor Costs
        - Marketing Department in Q4 -> Massive Marketing Cost boost
        - Quarterly Seasonal Multiplier -> Applied dynamically

    Args:
        profile: Department profile.
        planned_budget: Baseline planned budget.
        employee_count: Department headcount.
        project_count: Active project count.
        quarter: Current quarter (Q1-Q4).
        year: Current year.
        rng: Random generator instance.

    Returns:
        Tuple of (cost_components_dict, business_variation).
    """
    inflation = INFLATION_RATES[year]
    base_seasonal = SEASONAL_MULTIPLIERS[quarter]
    dept_seasonal = profile.seasonal_sensitivity.get(quarter, 1.0)
    effective_seasonal = base_seasonal * dept_seasonal

    # Calculate baseline total target spending center
    baseline_spending = planned_budget * effective_seasonal

    dist = profile.expense_distribution

    # 1. Operational Cost (Scaled by Employee Count)
    emp_min, emp_max = profile.employee_count_range
    emp_ratio = (employee_count - emp_min) / max(1, (emp_max - emp_min))
    op_multiplier = 1.0 + (emp_ratio * 0.20)
    op_cost = baseline_spending * dist["Operational_Cost"] * op_multiplier

    # 2. Marketing Cost (Massive boost for Marketing dept in Q4)
    mkt_mod = 1.0
    if profile.name == "Marketing" and quarter == "Q4":
        mkt_mod = 1.35
    mkt_cost = baseline_spending * dist["Marketing_Cost"] * mkt_mod

    # 3. Travel Cost (Sales highest)
    trv_cost = baseline_spending * dist["Travel_Cost"]

    # 4. Utility Cost (Inflated over years)
    utl_cost = baseline_spending * dist["Utility_Cost"] * (1.0 + inflation)

    # 5. Vendor Cost (Scaled by Project Count and Inflation)
    prj_min, prj_max = profile.project_count_range
    prj_ratio = (project_count - prj_min) / max(1, (prj_max - prj_min))
    vnd_multiplier = 1.0 + (prj_ratio * 0.25)
    vnd_cost = baseline_spending * dist["Vendor_Cost"] * vnd_multiplier * (1.0 + inflation)

    # 6. Miscellaneous Cost (R&D highest)
    misc_cost = baseline_spending * dist["Miscellaneous_Cost"]

    # Introduce Gaussian business variation (+/- 3-8% variance)
    business_variation = rng.normal(0.0, 0.04 * baseline_spending)

    expenses = {
        "Operational_Cost": round(max(100.0, op_cost), 2),
        "Marketing_Cost": round(max(0.0, mkt_cost), 2),
        "Travel_Cost": round(max(0.0, trv_cost), 2),
        "Utility_Cost": round(max(50.0, utl_cost), 2),
        "Vendor_Cost": round(max(0.0, vnd_cost), 2),
        "Miscellaneous_Cost": round(max(0.0, misc_cost), 2),
    }

    return expenses, round(business_variation, 2)


def generate_single_record(
    record_id: str,
    year: int,
    month: str,
    department_name: str,
    prev_month_exp: float,
    prev_qtr_exp: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Generate a single raw ERP financial record complying with business rules.

    Args:
        record_id: Unique record string identifier (e.g. REC_00001).
        year: Target year (2021-2025).
        month: Target month name.
        department_name: Name of target department.
        prev_month_exp: Trailing month expense amount.
        prev_qtr_exp: Trailing quarter expense amount.
        rng: Random generator instance.

    Returns:
        Dictionary containing all 21 dataset columns for the generated record.
    """
    profile = get_department_profile(department_name)
    quarter = QUARTER_MAPPING[month]
    inflation_rate = INFLATION_RATES[year]

    # Headcount & Projects
    emp_count, prj_count = calculate_scaled_headcount_and_projects(profile, rng)

    # Budget & Revenue
    planned_budget, revenue = calculate_planned_budget_and_revenue(
        profile, emp_count, quarter, year, rng
    )

    # Expense Components
    expenses, variation = calculate_department_expenses(
        profile, planned_budget, emp_count, prj_count, quarter, year, rng
    )

    # Calculate Target Variables deterministically
    actual_expense = round(
        expenses["Operational_Cost"]
        + expenses["Marketing_Cost"]
        + expenses["Travel_Cost"]
        + expenses["Utility_Cost"]
        + expenses["Vendor_Cost"]
        + expenses["Miscellaneous_Cost"]
        + variation,
        2,
    )

    # Target calculation
    budget_deviation = round(actual_expense - planned_budget, 2)

    return {
        "Record_ID": record_id,
        "Year": year,
        "Quarter": quarter,
        "Month": month,
        "Department": department_name,
        "Cost_Center": profile.cost_center,
        "Planned_Budget": planned_budget,
        "Revenue": revenue,
        "Employee_Count": emp_count,
        "Project_Count": prj_count,
        "Operational_Cost": expenses["Operational_Cost"],
        "Marketing_Cost": expenses["Marketing_Cost"],
        "Travel_Cost": expenses["Travel_Cost"],
        "Utility_Cost": expenses["Utility_Cost"],
        "Vendor_Cost": expenses["Vendor_Cost"],
        "Miscellaneous_Cost": expenses["Miscellaneous_Cost"],
        "Inflation_Rate": inflation_rate,
        "Previous_Month_Expense": round(prev_month_exp, 2),
        "Previous_Quarter_Expense": round(prev_qtr_exp, 2),
        "Actual_Expense": actual_expense,
        "Budget_Deviation": budget_deviation,
    }


def inject_anomalies_and_missing_values(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Inject realistic financial outliers and controlled missing values (2-3%).

    Args:
        df: Processed complete dataset DataFrame.
        rng: Random generator instance.

    Returns:
        DataFrame with realistic noise, outliers, and missing value artifacts.
    """
    df_result = df.copy()

    # 1. Inject Realistic Financial Outliers (~0.8% of rows)
    num_outliers = int(len(df_result) * OUTLIER_RATE)
    outlier_indices = rng.choice(df_result.index, size=num_outliers, replace=False)

    for idx in outlier_indices:
        outlier_type = rng.choice(["vendor_spike", "emergency_maintenance", "campaign_overflow"])
        if outlier_type == "vendor_spike":
            df_result.at[idx, "Vendor_Cost"] = round(df_result.at[idx, "Vendor_Cost"] * 2.8, 2)
        elif outlier_type == "emergency_maintenance":
            df_result.at[idx, "Operational_Cost"] = round(
                df_result.at[idx, "Operational_Cost"] * 3.0, 2
            )
        elif outlier_type == "campaign_overflow":
            df_result.at[idx, "Marketing_Cost"] = round(
                df_result.at[idx, "Marketing_Cost"] * 2.5, 2
            )

        # Recalculate Actual_Expense and Budget_Deviation for consistency
        recalculated_actual = round(
            df_result.at[idx, "Operational_Cost"]
            + df_result.at[idx, "Marketing_Cost"]
            + df_result.at[idx, "Travel_Cost"]
            + df_result.at[idx, "Utility_Cost"]
            + df_result.at[idx, "Vendor_Cost"]
            + df_result.at[idx, "Miscellaneous_Cost"],
            2,
        )
        df_result.at[idx, "Actual_Expense"] = recalculated_actual
        df_result.at[idx, "Budget_Deviation"] = round(
            recalculated_actual - df_result.at[idx, "Planned_Budget"], 2
        )

    # 2. Inject Controlled Missing Values (Targeting 2.5% missingness across total dataset cells)
    total_cells = len(df_result) * len(df_result.columns)
    num_missing = int(total_cells * MISSING_VALUE_RATE)

    for _ in range(num_missing):
        col = rng.choice(INJECTABLE_MISSING_COLUMNS)
        row_idx = rng.choice(df_result.index)
        df_result.at[row_idx, col] = np.nan

    return df_result


def generate_synthetic_dataset(
    num_records: int = TOTAL_RECORDS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Orchestrate synthetic generation of full financial ERP dataset.

    Maintains sequential time tracking per department across 2021-2025 to populate
    Previous_Month_Expense and Previous_Quarter_Expense dynamically.

    Args:
        num_records: Total records to generate (default 15,000).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Pandas DataFrame containing exactly 15,000 validated ERP records.
    """
    rng = np.random.default_rng(seed)

    records: List[Dict[str, Any]] = []

    # Time-series trailing expense trackers per department
    # Format: dept_history[dept] = list of recent monthly actual expenses
    dept_history: Dict[str, List[float]] = {dept: [] for dept in DEPARTMENTS}

    # Generate records stratified across Time (Years, Months) and Departments
    # Total 480 bins = 5 years x 12 months x 8 departments
    total_bins = len(YEARS) * len(MONTHS) * len(DEPARTMENTS)
    records_per_bin = num_records // total_bins
    remainder = num_records % total_bins

    record_counter = 1
    bin_index = 0

    for year in YEARS:
        for month in MONTHS:
            for dept in DEPARTMENTS:
                count = records_per_bin + (1 if bin_index < remainder else 0)
                bin_index += 1

                for _ in range(count):
                    rec_id = f"REC_{record_counter:05d}"

                    # Compute trailing month expense
                    history = dept_history[dept]
                    prev_month_exp = (
                        history[-1]
                        if len(history) >= 1
                        else get_department_profile(dept).budget_range[0] * 0.95
                    )
                    prev_qtr_exp = (
                        sum(history[-3:])
                        if len(history) >= 3
                        else prev_month_exp * 3.0
                    )

                    rec = generate_single_record(
                        record_id=rec_id,
                        year=year,
                        month=month,
                        department_name=dept,
                        prev_month_exp=prev_month_exp,
                        prev_qtr_exp=prev_qtr_exp,
                        rng=rng,
                    )

                    # Update history tracker with generated Actual_Expense
                    dept_history[dept].append(rec["Actual_Expense"])

                    records.append(rec)
                    record_counter += 1

    df = pd.DataFrame(records)

    # Reorder DataFrame strictly according to schema
    df = df[DATASET_COLUMNS]

    # Inject realistic artifacts (Outliers & Controlled 2.5% missingness)
    df = inject_anomalies_and_missing_values(df, rng)

    return df
