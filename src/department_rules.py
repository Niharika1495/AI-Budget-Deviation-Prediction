"""Department rules and profile definitions for financial synthesis.

This module encapsulates business profiles for corporate departments using immutable
dataclasses. Each profile establishes distinct budget boundaries, revenue generation,
headcount, project density, cost center mapping, and specific expense distribution dynamics.
Follows SOLID design principles and eliminates hardcoded logic across the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class DepartmentProfile:
    """Immutable domain representation of a corporate department's financial profile.

    Attributes:
        name: Name of the department.
        cost_center: Enterprise ERP cost center identifier code.
        budget_range: Min and Max monthly planned budget in USD.
        revenue_range: Min and Max monthly revenue generation in USD (0 for cost centers).
        employee_count_range: Min and Max headcount.
        project_count_range: Min and Max active projects.
        expense_distribution: Baseline percentage weights for cost categories
            (Operational, Marketing, Travel, Utility, Vendor, Miscellaneous).
        business_description: Functional domain summary.
        seasonal_sensitivity: Department-specific multipliers for Q1-Q4.
    """

    name: str
    cost_center: str
    budget_range: Tuple[float, float]
    revenue_range: Tuple[float, float]
    employee_count_range: Tuple[int, int]
    project_count_range: Tuple[int, int]
    expense_distribution: Dict[str, float]
    business_description: str
    seasonal_sensitivity: Dict[str, float] = field(
        default_factory=lambda: {"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 1.0}
    )

    def validate(self) -> None:
        """Validate structural integrity of department profile weights and ranges."""
        total_weight = sum(self.expense_distribution.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Expense distribution weights for {self.name} must sum to 1.0, got {total_weight}"
            )
        if self.budget_range[0] > self.budget_range[1]:
            raise ValueError(f"Invalid budget range for {self.name}: {self.budget_range}")
        if self.employee_count_range[0] > self.employee_count_range[1]:
            raise ValueError(
                f"Invalid employee count range for {self.name}: {self.employee_count_range}"
            )


# ==============================================================================
# DEPARTMENT PROFILES CONFIGURATION
# ==============================================================================
DEPARTMENT_PROFILES: Mapping[str, DepartmentProfile] = {
    "Finance": DepartmentProfile(
        name="Finance",
        cost_center="CC-FIN-101",
        budget_range=(400000.0, 1500000.0),
        revenue_range=(0.0, 0.0),
        employee_count_range=(15, 45),
        project_count_range=(3, 10),
        expense_distribution={
            "Operational_Cost": 0.45,
            "Marketing_Cost": 0.02,
            "Travel_Cost": 0.05,
            "Utility_Cost": 0.15,
            "Vendor_Cost": 0.18,
            "Miscellaneous_Cost": 0.15,
        },
        business_description="Corporate financial control, compliance, and stable administrative spending.",
        seasonal_sensitivity={"Q1": 1.00, "Q2": 1.02, "Q3": 1.03, "Q4": 1.10},
    ),
    "HR": DepartmentProfile(
        name="HR",
        cost_center="CC-HR-102",
        budget_range=(350000.0, 1200000.0),
        revenue_range=(0.0, 0.0),
        employee_count_range=(20, 60),
        project_count_range=(4, 12),
        expense_distribution={
            "Operational_Cost": 0.55,
            "Marketing_Cost": 0.03,
            "Travel_Cost": 0.10,
            "Utility_Cost": 0.12,
            "Vendor_Cost": 0.12,
            "Miscellaneous_Cost": 0.08,
        },
        business_description="Human resources management, payroll operations, recruiting, and training.",
        seasonal_sensitivity={"Q1": 1.05, "Q2": 1.00, "Q3": 1.02, "Q4": 1.15},
    ),
    "IT": DepartmentProfile(
        name="IT",
        cost_center="CC-IT-103",
        budget_range=(1200000.0, 5500000.0),
        revenue_range=(0.0, 0.0),
        employee_count_range=(40, 150),
        project_count_range=(10, 40),
        expense_distribution={
            "Operational_Cost": 0.32,
            "Marketing_Cost": 0.01,
            "Travel_Cost": 0.07,
            "Utility_Cost": 0.10,
            "Vendor_Cost": 0.42,  # High vendor cost (cloud, licenses, SaaS)
            "Miscellaneous_Cost": 0.08,
        },
        business_description="IT infrastructure, enterprise cloud services, cybersecurity, and vendor tools.",
        seasonal_sensitivity={"Q1": 0.98, "Q2": 1.04, "Q3": 1.08, "Q4": 1.25},
    ),
    "Marketing": DepartmentProfile(
        name="Marketing",
        cost_center="CC-MKT-104",
        budget_range=(800000.0, 4500000.0),
        revenue_range=(0.0, 0.0),
        employee_count_range=(25, 80),
        project_count_range=(8, 25),
        expense_distribution={
            "Operational_Cost": 0.18,
            "Marketing_Cost": 0.58,  # Highest marketing cost
            "Travel_Cost": 0.08,
            "Utility_Cost": 0.04,
            "Vendor_Cost": 0.08,
            "Miscellaneous_Cost": 0.04,
        },
        business_description="Global marketing campaigns, brand development, and advertising acquisition.",
        seasonal_sensitivity={"Q1": 0.90, "Q2": 1.05, "Q3": 1.10, "Q4": 1.45},  # Significant Q4 campaign spike
    ),
    "Sales": DepartmentProfile(
        name="Sales",
        cost_center="CC-SLS-105",
        budget_range=(1000000.0, 5000000.0),
        revenue_range=(8000000.0, 60000000.0),  # Major revenue generator
        employee_count_range=(50, 200),
        project_count_range=(15, 50),
        expense_distribution={
            "Operational_Cost": 0.30,
            "Marketing_Cost": 0.12,
            "Travel_Cost": 0.38,  # Highest travel cost
            "Utility_Cost": 0.05,
            "Vendor_Cost": 0.10,
            "Miscellaneous_Cost": 0.05,
        },
        business_description="Enterprise sales execution, client acquisition, client relations, and travel.",
        seasonal_sensitivity={"Q1": 0.95, "Q2": 1.05, "Q3": 1.12, "Q4": 1.30},
    ),
    "Operations": DepartmentProfile(
        name="Operations",
        cost_center="CC-OPS-106",
        budget_range=(2500000.0, 12000000.0),
        revenue_range=(0.0, 0.0),
        employee_count_range=(100, 450),
        project_count_range=(12, 35),
        expense_distribution={
            "Operational_Cost": 0.62,  # Highest operational cost
            "Marketing_Cost": 0.01,
            "Travel_Cost": 0.04,
            "Utility_Cost": 0.18,  # High utility cost
            "Vendor_Cost": 0.10,
            "Miscellaneous_Cost": 0.05,
        },
        business_description="Core business infrastructure execution, logistics, facilities, and maintenance.",
        seasonal_sensitivity={"Q1": 1.00, "Q2": 1.03, "Q3": 1.06, "Q4": 1.15},
    ),
    "Procurement": DepartmentProfile(
        name="Procurement",
        cost_center="CC-PRC-107",
        budget_range=(500000.0, 2800000.0),
        revenue_range=(0.0, 0.0),
        employee_count_range=(18, 50),
        project_count_range=(5, 15),
        expense_distribution={
            "Operational_Cost": 0.25,
            "Marketing_Cost": 0.01,
            "Travel_Cost": 0.08,
            "Utility_Cost": 0.06,
            "Vendor_Cost": 0.52,  # Vendor driven department
            "Miscellaneous_Cost": 0.08,
        },
        business_description="Vendor sourcing, contract management, inventory procurement, and supply chain.",
        seasonal_sensitivity={"Q1": 0.95, "Q2": 1.02, "Q3": 1.20, "Q4": 1.25},  # Q3/Q4 procurement growth
    ),
    "Research & Development": DepartmentProfile(
        name="Research & Development",
        cost_center="CC-RND-108",
        budget_range=(1800000.0, 8500000.0),
        revenue_range=(0.0, 0.0),
        employee_count_range=(35, 120),
        project_count_range=(8, 30),
        expense_distribution={
            "Operational_Cost": 0.35,
            "Marketing_Cost": 0.02,
            "Travel_Cost": 0.08,
            "Utility_Cost": 0.10,
            "Vendor_Cost": 0.18,
            "Miscellaneous_Cost": 0.27,  # High miscellaneous cost (prototypes, IP, lab gear)
        },
        business_description="Advanced innovation, prototyping, patent filings, research, and lab operations.",
        seasonal_sensitivity={"Q1": 1.00, "Q2": 1.04, "Q3": 1.05, "Q4": 1.18},
    ),
}


def get_department_profile(department_name: str) -> DepartmentProfile:
    """Retrieve validated department profile by name.

    Args:
        department_name: Name of the target department.

    Returns:
        DepartmentProfile instance corresponding to the department.

    Raises:
        KeyError: If department_name is not registered in DEPARTMENT_PROFILES.
    """
    if department_name not in DEPARTMENT_PROFILES:
        raise KeyError(
            f"Department '{department_name}' not found. Available: {list(DEPARTMENT_PROFILES.keys())}"
        )
    profile = DEPARTMENT_PROFILES[department_name]
    profile.validate()
    return profile
