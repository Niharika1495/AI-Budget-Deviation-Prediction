"""Bivariate analysis module for financial metrics and relationships.

This module evaluates paired variable relationships, regression trends, scatter patterns,
and financial hypothesis checks:
- Planned Budget vs. Actual Expense
- Revenue vs. Planned Budget
- Employee Count vs. Operational Cost
- Project Count vs. Vendor Cost
- Inflation vs. Actual Expense / Utility Cost
- Budget Deviation vs. Revenue

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.visualization_utils import PALETTE, VisualizationUtils


class BivariateAnalyzer:
    """Bivariate relationship analysis engine using Matplotlib (No Seaborn)."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize BivariateAnalyzer with cleaned Pandas DataFrame.

        Args:
            df: Cleaned financial DataFrame.
        """
        self.df: pd.DataFrame = df
        VisualizationUtils.setup_matplotlib_style()

    def generate_bivariate_charts(self) -> List[str]:
        """Generate high-resolution bivariate analysis charts.

        Returns:
            List of generated image filenames saved in images/eda/.
        """
        generated_files: List[str] = []

        # 9. Budget vs. Actual Expense Scatter
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.df["Planned_Budget"], self.df["Actual_Expense"], alpha=0.35, color=PALETTE["secondary"], edgecolors="none")
        max_val = max(self.df["Planned_Budget"].max(), self.df["Actual_Expense"].max())
        ax.plot([0, max_val], [0, max_val], color=PALETTE["accent_red"], linestyle="--", linewidth=2, label="Parity Line (Zero Deviation: Actual = Budget)")
        ax.set_title("Bivariate Analysis: Planned Budget vs. Actual Expense ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Planned Budget ($)")
        ax.set_ylabel("Actual Expense ($)")
        ax.legend()

        fn9 = "chart_09_bivariate_budget_vs_actual.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn9)
        generated_files.append(fn9)

        # 10. Revenue vs. Planned Budget Scatter (Sales Department focus)
        sales_df = self.df[self.df["Revenue"] > 0]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sales_df["Planned_Budget"], sales_df["Revenue"], alpha=0.4, color=PALETTE["accent_purple"])
        # Fit trend line
        z = np.polyfit(sales_df["Planned_Budget"], sales_df["Revenue"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(sales_df["Planned_Budget"].min(), sales_df["Planned_Budget"].max(), 100)
        ax.plot(x_trend, p(x_trend), color=PALETTE["accent_red"], linestyle="-", linewidth=2, label=f"Trend Line (Slope: {z[0]:.2f})")
        ax.set_title("Bivariate Analysis: Revenue vs. Planned Budget ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Planned Budget ($)")
        ax.set_ylabel("Generated Revenue ($)")
        ax.legend()

        fn10 = "chart_10_bivariate_revenue_vs_planned_budget.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn10)
        generated_files.append(fn10)

        # 11. Employee Count vs. Operational Cost
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.df["Employee_Count"], self.df["Operational_Cost"], alpha=0.35, color=PALETTE["secondary"])
        z = np.polyfit(self.df["Employee_Count"], self.df["Operational_Cost"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.df["Employee_Count"].min(), self.df["Employee_Count"].max(), 100)
        ax.plot(x_trend, p(x_trend), color=PALETTE["accent_amber"], linestyle="--", linewidth=2.5, label=f"Linear Trend (${z[0]:,.0f} / employee)")
        ax.set_title("Business Logic Verification: Headcount vs. Operational Cost ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Employee Headcount")
        ax.set_ylabel("Operational Cost ($)")
        ax.legend()

        fn11 = "chart_11_bivariate_employee_vs_operational_cost.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn11)
        generated_files.append(fn11)

        # 12. Project Count vs. Vendor Cost
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.df["Project_Count"], self.df["Vendor_Cost"], alpha=0.35, color=PALETTE["accent_green"])
        z = np.polyfit(self.df["Project_Count"], self.df["Vendor_Cost"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.df["Project_Count"].min(), self.df["Project_Count"].max(), 100)
        ax.plot(x_trend, p(x_trend), color=PALETTE["accent_red"], linestyle="--", linewidth=2.5, label=f"Linear Trend (${z[0]:,.0f} / project)")
        ax.set_title("Business Logic Verification: Active Projects vs. Vendor Cost ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Active Projects Count")
        ax.set_ylabel("Vendor Cost ($)")
        ax.legend()

        fn12 = "chart_12_bivariate_projects_vs_vendor_cost.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn12)
        generated_files.append(fn12)

        # 13. Inflation vs. Actual Expense
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped_inf = self.df.groupby("Year").agg({"Inflation_Rate": "first", "Actual_Expense": "mean"}).reset_index()
        ax.plot(grouped_inf["Year"], grouped_inf["Actual_Expense"], marker="o", color=PALETTE["accent_red"], linewidth=2.5, label="Mean Actual Expense")
        ax2 = ax.twinx()
        ax2.plot(grouped_inf["Year"], grouped_inf["Inflation_Rate"] * 100.0, marker="s", color=PALETTE["accent_amber"], linestyle="--", linewidth=2, label="Inflation Rate (%)")
        ax.set_title("Macroeconomic Dynamics: Inflation Rate vs. Mean Actual Expense ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Actual Expense ($)", color=PALETTE["accent_red"])
        ax2.set_ylabel("Inflation Rate (%)", color=PALETTE["accent_amber"])
        ax.set_xticks(grouped_inf["Year"])

        fn13 = "chart_13_bivariate_inflation_vs_actual_expense.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn13)
        generated_files.append(fn13)

        # 14. Inflation vs. Utility Cost
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped_utl = self.df.groupby("Year").agg({"Inflation_Rate": "first", "Utility_Cost": "mean"}).reset_index()
        ax.plot(grouped_utl["Year"], grouped_utl["Utility_Cost"], marker="o", color=PALETTE["secondary"], linewidth=2.5, label="Mean Utility Cost")
        ax2 = ax.twinx()
        ax2.plot(grouped_utl["Year"], grouped_utl["Inflation_Rate"] * 100.0, marker="^", color=PALETTE["accent_amber"], linestyle="--", linewidth=2, label="Inflation Rate (%)")
        ax.set_title("Inflation Impact Analysis: Inflation Rate vs. Mean Utility Cost ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Utility Cost ($)", color=PALETTE["secondary"])
        ax2.set_ylabel("Inflation Rate (%)", color=PALETTE["accent_amber"])
        ax.set_xticks(grouped_utl["Year"])

        fn14 = "chart_14_bivariate_inflation_vs_utility_cost.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn14)
        generated_files.append(fn14)

        # 15. Budget Deviation vs. Revenue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sales_df["Revenue"], sales_df["Budget_Deviation"], alpha=0.35, color=PALETTE["accent_purple"])
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
        ax.set_title("Bivariate Analysis: Budget Deviation vs. Generated Revenue ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Generated Revenue ($)")
        ax.set_ylabel("Budget Deviation ($)")

        fn15 = "chart_15_bivariate_deviation_vs_revenue.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn15)
        generated_files.append(fn15)

        return generated_files
