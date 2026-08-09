"""Univariate analysis module for financial metrics.

This module evaluates single-variable distributions, central tendencies, spread, skewness, and outliers
for key financial dimensions (Planned_Budget, Revenue, Actual_Expense, Budget_Deviation, Employee_Count,
Project_Count, Operational_Cost, Vendor_Cost) using Matplotlib and Plotly (No Seaborn).

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.eda.visualization_utils import PALETTE, VisualizationUtils


class UnivariateAnalyzer:
    """Analyzer engine executing single-variable statistical distribution analysis."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize UnivariateAnalyzer with cleaned Pandas DataFrame.

        Args:
            df: Cleaned financial DataFrame.
        """
        self.df: pd.DataFrame = df
        VisualizationUtils.setup_matplotlib_style()

    def generate_univariate_charts(self) -> List[str]:
        """Generate high-resolution univariate distribution charts.

        Returns:
            List of generated image filenames saved in images/eda/.
        """
        generated_files: List[str] = []

        # 1. Planned Budget Distribution (Histogram + Box)
        fig, (ax_box, ax_hist) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}
        )
        ax_box.boxplot(self.df["Planned_Budget"], vert=False, patch_artist=True,
                       boxprops=dict(facecolor=PALETTE["secondary"], color=PALETTE["primary"]),
                       medianprops=dict(color=PALETTE["accent_amber"], linewidth=2))
        ax_box.set_title("Planned Budget Distribution & Outliers ($)", fontsize=14, fontweight="bold")
        ax_box.set_yticks([])

        ax_hist.hist(self.df["Planned_Budget"], bins=40, color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
        ax_hist.axvline(self.df["Planned_Budget"].mean(), color=PALETTE["accent_red"], linestyle="--", linewidth=2, label=f"Mean: ${self.df['Planned_Budget'].mean():,.2f}")
        ax_hist.axvline(self.df["Planned_Budget"].median(), color=PALETTE["accent_amber"], linestyle="-", linewidth=2, label=f"Median: ${self.df['Planned_Budget'].median():,.2f}")
        ax_hist.set_xlabel("Planned Budget ($)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()

        fn1 = "chart_01_univariate_planned_budget_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn1)
        generated_files.append(fn1)

        # 2. Revenue Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        # Filter non-zero revenue (Sales department)
        sales_rev = self.df[self.df["Revenue"] > 0]["Revenue"]
        ax.hist(sales_rev, bins=35, color=PALETTE["accent_purple"], edgecolor="white", alpha=0.85)
        ax.axvline(sales_rev.mean(), color=PALETTE["accent_red"], linestyle="--", linewidth=2, label=f"Mean Sales Revenue: ${sales_rev.mean():,.2f}")
        ax.axvline(sales_rev.median(), color=PALETTE["accent_amber"], linestyle="-", linewidth=2, label=f"Median: ${sales_rev.median():,.2f}")
        ax.set_title("Sales Revenue Distribution ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Revenue ($)")
        ax.set_ylabel("Frequency")
        ax.legend()

        fn2 = "chart_02_univariate_revenue_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn2)
        generated_files.append(fn2)

        # 3. Actual Expense Distribution
        fig, (ax_box, ax_hist) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}
        )
        ax_box.boxplot(self.df["Actual_Expense"], vert=False, patch_artist=True,
                       boxprops=dict(facecolor=PALETTE["accent_green"], color=PALETTE["primary"]),
                       medianprops=dict(color=PALETTE["accent_amber"], linewidth=2))
        ax_box.set_title("Actual Expense Distribution ($)", fontsize=14, fontweight="bold")
        ax_box.set_yticks([])

        ax_hist.hist(self.df["Actual_Expense"], bins=40, color=PALETTE["accent_green"], edgecolor="white", alpha=0.85)
        ax_hist.axvline(self.df["Actual_Expense"].mean(), color=PALETTE["accent_red"], linestyle="--", linewidth=2, label=f"Mean: ${self.df['Actual_Expense'].mean():,.2f}")
        ax_hist.axvline(self.df["Actual_Expense"].median(), color=PALETTE["accent_amber"], linestyle="-", linewidth=2, label=f"Median: ${self.df['Actual_Expense'].median():,.2f}")
        ax_hist.set_xlabel("Actual Expense ($)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()

        fn3 = "chart_03_univariate_actual_expense_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn3)
        generated_files.append(fn3)

        # 4. Budget Deviation Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.df["Budget_Deviation"], bins=50, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
        ax.axvline(0.0, color="black", linestyle="-", linewidth=1.5, label="Zero Deviation Target")
        ax.axvline(self.df["Budget_Deviation"].mean(), color=PALETTE["accent_red"], linestyle="--", linewidth=2, label=f"Mean Dev: ${self.df['Budget_Deviation'].mean():,.2f}")
        ax.set_title("Target Variable: Budget Deviation Distribution ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Budget Deviation ($) [Actual Expense - Planned Budget]")
        ax.set_ylabel("Frequency")
        ax.legend()

        fn4 = "chart_04_univariate_budget_deviation_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn4)
        generated_files.append(fn4)

        # 5. Employee Count Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.df["Employee_Count"], bins=30, color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
        ax.set_title("Department Employee Count Distribution (Headcount)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Headcount (Employees)")
        ax.set_ylabel("Frequency")

        fn5 = "chart_05_univariate_employee_count_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn5)
        generated_files.append(fn5)

        # 6. Project Count Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.df["Project_Count"], bins=25, color=PALETTE["accent_amber"], edgecolor="white", alpha=0.85)
        ax.set_title("Active Project Count Distribution per Department", fontsize=14, fontweight="bold")
        ax.set_xlabel("Active Projects")
        ax.set_ylabel("Frequency")

        fn6 = "chart_06_univariate_project_count_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn6)
        generated_files.append(fn6)

        # 7. Operational Cost Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.df["Operational_Cost"], bins=40, color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
        ax.set_title("Operational Cost Breakdown Distribution ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Operational Cost ($)")
        ax.set_ylabel("Frequency")

        fn7 = "chart_07_univariate_operational_cost_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn7)
        generated_files.append(fn7)

        # 8. Vendor Cost Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.df["Vendor_Cost"], bins=40, color=PALETTE["accent_purple"], edgecolor="white", alpha=0.85)
        ax.set_title("Vendor Cost Breakdown Distribution ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Vendor Cost ($)")
        ax.set_ylabel("Frequency")

        fn8 = "chart_08_univariate_vendor_cost_dist.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn8)
        generated_files.append(fn8)

        return generated_files
