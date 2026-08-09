"""Time series trend analysis module for financial temporal patterns.

This module evaluates monthly, quarterly, and yearly financial trajectories (2021-2025):
- Monthly Budget vs. Actual Expense Trajectory
- Monthly Budget Deviation Cycles
- Quarterly Overspending & Spending Spikes (Q1-Q4)
- Yearly Corporate Spending Evolution
- Yearly Budget Utilization Rate (%)

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from typing import List
import matplotlib.pyplot as plt
import pandas as pd

from src.config import MONTHS
from src.eda.visualization_utils import PALETTE, VisualizationUtils


class TrendAnalyzer:
    """Time-series trend and temporal seasonality analysis engine."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize TrendAnalyzer with cleaned Pandas DataFrame.

        Args:
            df: Cleaned financial DataFrame.
        """
        self.df: pd.DataFrame = df
        VisualizationUtils.setup_matplotlib_style()

    def generate_trend_charts(self) -> List[str]:
        """Generate high-resolution time series analysis charts.

        Returns:
            List of generated image filenames saved in images/eda/.
        """
        generated_files: List[str] = []

        # Ensure Month chronological ordering
        month_order = list(MONTHS)
        df_month = self.df.copy()
        df_month["Month"] = pd.Categorical(df_month["Month"], categories=month_order, ordered=True)

        # 20. Monthly Budget vs. Actual Expense Trend
        monthly_grp = df_month.groupby("Month", observed=True).agg({
            "Planned_Budget": "mean",
            "Actual_Expense": "mean"
        }).reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly_grp["Month"], monthly_grp["Planned_Budget"], marker="o", linewidth=2.5, color=PALETTE["secondary"], label="Mean Planned Budget ($)")
        ax.plot(monthly_grp["Month"], monthly_grp["Actual_Expense"], marker="s", linewidth=2.5, color=PALETTE["accent_red"], label="Mean Actual Expense ($)")
        ax.set_title("Monthly Financial Trajectory: Planned Budget vs. Actual Expense (2021-2025)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Amount ($)")
        ax.set_xticklabels(monthly_grp["Month"], rotation=30)
        ax.legend()

        fn20 = "chart_20_trend_monthly_actual_vs_budget.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn20)
        generated_files.append(fn20)

        # 21. Monthly Budget Deviation Bar Chart
        monthly_dev = df_month.groupby("Month", observed=True)["Budget_Deviation"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = [PALETTE["accent_red"] if v > 0 else PALETTE["accent_green"] for v in monthly_dev["Budget_Deviation"]]
        ax.bar(monthly_dev["Month"], monthly_dev["Budget_Deviation"], color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0.0, color="black", linestyle="-", linewidth=1)
        ax.set_title("Monthly Mean Budget Deviation ($) [Overspending Cycles]", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Deviation ($)")
        ax.set_xticklabels(monthly_dev["Month"], rotation=30)

        fn21 = "chart_21_trend_monthly_budget_deviation.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn21)
        generated_files.append(fn21)

        # 22. Quarterly Budget & Spending Bar Chart
        qtr_order = ["Q1", "Q2", "Q3", "Q4"]
        df_qtr = self.df.copy()
        df_qtr["Quarter"] = pd.Categorical(df_qtr["Quarter"], categories=qtr_order, ordered=True)
        qtr_grp = df_qtr.groupby("Quarter", observed=True).agg({
            "Planned_Budget": "mean",
            "Actual_Expense": "mean"
        }).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(qtr_order))
        width = 0.35
        ax.bar([i - width / 2 for i in x], qtr_grp["Planned_Budget"], width=width, label="Planned Budget", color=PALETTE["secondary"])
        ax.bar([i + width / 2 for i in x], qtr_grp["Actual_Expense"], width=width, label="Actual Expense", color=PALETTE["accent_red"])
        ax.set_title("Quarterly Financial Spending & Budget Comparison (Q1 - Q4)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(qtr_order)
        ax.set_ylabel("Mean Amount ($)")
        ax.legend()

        fn22 = "chart_22_trend_quarterly_budget_and_spending.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn22)
        generated_files.append(fn22)

        # 23. Quarterly Budget Deviation Box Plot
        qtr_dev = df_qtr.groupby("Quarter", observed=True)["Budget_Deviation"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(qtr_dev["Quarter"], qtr_dev["Budget_Deviation"], color=PALETTE["accent_amber"], edgecolor="white", alpha=0.85)
        ax.set_title("Quarterly Average Overspending / Budget Deviation ($)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Average Budget Deviation ($)")

        fn23 = "chart_23_trend_quarterly_budget_deviation.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn23)
        generated_files.append(fn23)

        # 24. Yearly Spending Evolution (2021-2025)
        yearly_grp = self.df.groupby("Year").agg({
            "Planned_Budget": "sum",
            "Actual_Expense": "sum"
        }).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(yearly_grp["Year"], yearly_grp["Planned_Budget"] / 1e6, marker="o", linewidth=2.5, color=PALETTE["secondary"], label="Total Planned Budget ($ Millions)")
        ax.plot(yearly_grp["Year"], yearly_grp["Actual_Expense"] / 1e6, marker="s", linewidth=2.5, color=PALETTE["accent_red"], label="Total Actual Expense ($ Millions)")
        ax.set_title("Yearly Enterprise Financial Spending Evolution (2021 - 2025)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Amount ($ Millions)")
        ax.set_xticks(yearly_grp["Year"])
        ax.legend()

        fn24 = "chart_24_trend_yearly_spending_evolution.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn24)
        generated_files.append(fn24)

        # 25. Yearly Budget Utilization Rate (%)
        yearly_grp["Utilization_Rate"] = (yearly_grp["Actual_Expense"] / yearly_grp["Planned_Budget"]) * 100.0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(yearly_grp["Year"], yearly_grp["Utilization_Rate"], color=PALETTE["accent_purple"], edgecolor="white", alpha=0.85)
        ax.axhline(100.0, color=PALETTE["accent_red"], linestyle="--", linewidth=2, label="100% Target Utilization Baseline")
        ax.set_title("Yearly Budget Utilization Rate (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Budget Utilization Rate (%)")
        ax.set_xticks(yearly_grp["Year"])
        ax.legend()

        for i, row in yearly_grp.iterrows():
            ax.text(row["Year"], row["Utilization_Rate"] + 1.0, f"{row['Utilization_Rate']:.1f}%", ha="center", fontweight="bold")

        fn25 = "chart_25_trend_yearly_budget_utilization_rate.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn25)
        generated_files.append(fn25)

        return generated_files
