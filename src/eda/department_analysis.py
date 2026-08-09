"""Department analysis module for cross-functional organizational comparison.

This module evaluates cross-departmental metrics:
- Department Planned Budget vs. Actual Expense
- Department Average Budget Deviation (Overspending / Underspending)
- Department Employee Headcount & Active Project Density
- Department Cost Breakdown Composition Stacked Shares
- Total Corporate Spending Contribution per Department

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.visualization_utils import DEPT_COLORS, PALETTE, VisualizationUtils


class DepartmentAnalyzer:
    """Cross-departmental financial performance analysis engine."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize DepartmentAnalyzer with cleaned Pandas DataFrame.

        Args:
            df: Cleaned financial DataFrame.
        """
        self.df: pd.DataFrame = df
        VisualizationUtils.setup_matplotlib_style()

    def generate_department_charts(self) -> List[str]:
        """Generate high-resolution department comparative charts.

        Returns:
            List of generated image filenames saved in images/eda/.
        """
        generated_files: List[str] = []

        dept_grp = self.df.groupby("Department").agg({
            "Planned_Budget": "mean",
            "Actual_Expense": "mean",
            "Budget_Deviation": "mean",
            "Employee_Count": "mean",
            "Project_Count": "mean",
            "Revenue": "mean",
        }).reset_index()

        # Sort departments by Actual Expense descending
        dept_grp = dept_grp.sort_values("Actual_Expense", ascending=False)

        # 26. Department Planned Budget vs. Actual Expense Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(dept_grp))
        width = 0.35
        ax.bar(x - width / 2, dept_grp["Planned_Budget"], width, label="Mean Planned Budget", color=PALETTE["secondary"])
        ax.bar(x + width / 2, dept_grp["Actual_Expense"], width, label="Mean Actual Expense", color=PALETTE["accent_red"])
        ax.set_title("Department Performance: Planned Budget vs. Actual Expense ($)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(dept_grp["Department"], rotation=30, ha="right")
        ax.set_ylabel("Average Amount ($)")
        ax.legend()

        fn26 = "chart_26_dept_budget_vs_actual.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn26)
        generated_files.append(fn26)

        # 27. Department Mean Budget Deviation (Ranked)
        dept_dev = dept_grp.sort_values("Budget_Deviation", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [PALETTE["accent_red"] if v > 0 else PALETTE["accent_green"] for v in dept_dev["Budget_Deviation"]]
        ax.barh(dept_dev["Department"], dept_dev["Budget_Deviation"], color=colors, alpha=0.85)
        ax.axvline(0.0, color="black", linestyle="-", linewidth=1.5)
        ax.set_title("Department Budget Deviation Ranking ($) [Over/Under Spending]", fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean Budget Deviation ($)")

        for i, (dept, val) in enumerate(zip(dept_dev["Department"], dept_dev["Budget_Deviation"])):
            offset = 1000 if val >= 0 else -15000
            ax.text(val + offset, i, f"${val:+,.0f}", va="center", fontsize=9, fontweight="bold")

        fn27 = "chart_27_dept_budget_deviation.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn27)
        generated_files.append(fn27)

        # 28. Department Employee Headcount & Active Projects
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        ax1.bar(x - width / 2, dept_grp["Employee_Count"], width, label="Mean Headcount", color=PALETTE["primary"])
        ax2.bar(x + width / 2, dept_grp["Project_Count"], width, label="Mean Active Projects", color=PALETTE["accent_amber"])

        ax1.set_title("Department Operating Scale: Headcount vs. Active Projects", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(dept_grp["Department"], rotation=30, ha="right")
        ax1.set_ylabel("Employee Headcount", color=PALETTE["primary"])
        ax2.set_ylabel("Active Projects Count", color=PALETTE["accent_amber"])

        fn28 = "chart_28_dept_headcount_and_projects.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn28)
        generated_files.append(fn28)

        # 29. Department Expense Composition (100% Stacked Bar)
        cost_cols = [
            "Operational_Cost",
            "Marketing_Cost",
            "Travel_Cost",
            "Utility_Cost",
            "Vendor_Cost",
            "Miscellaneous_Cost",
        ]
        comp_df = self.df.groupby("Department")[cost_cols].mean()
        comp_pct = comp_df.div(comp_df.sum(axis=1), axis=0) * 100.0

        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(comp_pct))
        colors_stack = [
            PALETTE["secondary"],
            PALETTE["accent_red"],
            PALETTE["accent_amber"],
            PALETTE["accent_green"],
            PALETTE["accent_purple"],
            PALETTE["primary"],
        ]

        for col, col_color in zip(cost_cols, colors_stack):
            ax.bar(comp_pct.index, comp_pct[col], bottom=bottom, label=col.replace("_", " "), color=col_color)
            bottom += comp_pct[col].values

        ax.set_title("Department Cost Breakdown Composition Share (%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Share of Actual Expense (%)")
        ax.set_xticklabels(comp_pct.index, rotation=30, ha="right")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        fn29 = "chart_29_dept_expense_composition.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn29)
        generated_files.append(fn29)

        # 30. Total Corporate Spending Share by Department (Donut Chart)
        total_dept_spending = self.df.groupby("Department")["Actual_Expense"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            total_dept_spending,
            labels=total_dept_spending.index,
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=140,
            colors=DEPT_COLORS,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        )
        plt.setp(autotexts, size=9, weight="bold", color="white")
        ax.set_title("Total Company Actual Expense Contribution by Department", fontsize=14, fontweight="bold")

        fn30 = "chart_30_dept_total_spending_share.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn30)
        generated_files.append(fn30)

        return generated_files
