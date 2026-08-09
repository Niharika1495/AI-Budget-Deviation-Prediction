"""Financial analysis module for corporate cost center decomposition.

This module executes executive financial analysis:
- Company-wide Cost Category Breakdown (OpEx, MktEx, TrvEx, UtlEx, VndEx, MiscEx)
- Operational Cost vs. Vendor Cost Scaling
- Marketing Campaign & Travel Expenditure Profiles
- Corporate Budget Utilization Efficiency

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from typing import List
import matplotlib.pyplot as plt
import pandas as pd

from src.eda.visualization_utils import PALETTE, VisualizationUtils


class FinancialAnalyzer:
    """Corporate financial decomposition and cost driver analysis engine."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize FinancialAnalyzer with cleaned Pandas DataFrame.

        Args:
            df: Cleaned financial DataFrame.
        """
        self.df: pd.DataFrame = df
        VisualizationUtils.setup_matplotlib_style()

    def generate_financial_charts(self) -> List[str]:
        """Generate high-resolution executive financial analysis charts.

        Returns:
            List of generated image filenames saved in images/eda/.
        """
        generated_files: List[str] = []

        cost_cols = [
            "Operational_Cost",
            "Marketing_Cost",
            "Travel_Cost",
            "Utility_Cost",
            "Vendor_Cost",
            "Miscellaneous_Cost",
        ]
        cost_sums = self.df[cost_cols].sum()

        # 31. Company Total Expense Breakdown Donut Chart
        fig, ax = plt.subplots(figsize=(8, 8))
        colors_pie = [
            PALETTE["secondary"],
            PALETTE["accent_red"],
            PALETTE["accent_amber"],
            PALETTE["accent_green"],
            PALETTE["accent_purple"],
            PALETTE["primary"],
        ]
        wedges, texts, autotexts = ax.pie(
            cost_sums,
            labels=[c.replace("_", " ") for c in cost_cols],
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=140,
            colors=colors_pie,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        )
        plt.setp(autotexts, size=9.5, weight="bold", color="white")
        ax.set_title("Total Company Expense Breakdown Allocation", fontsize=14, fontweight="bold")

        fn31 = "chart_31_financial_company_expense_breakdown.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn31)
        generated_files.append(fn31)

        # 32. Operational Cost vs Vendor Cost Comparison by Department
        op_vs_vnd = self.df.groupby("Department")[["Operational_Cost", "Vendor_Cost"]].mean().reset_index()
        op_vs_vnd = op_vs_vnd.sort_values("Operational_Cost", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(op_vs_vnd))
        width = 0.35
        ax.bar([i - width / 2 for i in x], op_vs_vnd["Operational_Cost"], width, label="Operational Cost", color=PALETTE["secondary"])
        ax.bar([i + width / 2 for i in x], op_vs_vnd["Vendor_Cost"], width, label="Vendor Cost", color=PALETTE["accent_purple"])
        ax.set_title("Department Cost Driver Deep Dive: Operational Cost vs. Vendor Cost ($)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(op_vs_vnd["Department"], rotation=30, ha="right")
        ax.set_ylabel("Average Cost ($)")
        ax.legend()

        fn32 = "chart_32_financial_operational_vs_vendor_cost.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn32)
        generated_files.append(fn32)

        # 33. Marketing Cost vs Travel Cost Comparison by Department
        mkt_vs_trv = self.df.groupby("Department")[["Marketing_Cost", "Travel_Cost"]].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(mkt_vs_trv))
        ax.bar([i - width / 2 for i in x], mkt_vs_trv["Marketing_Cost"], width, label="Marketing Cost", color=PALETTE["accent_red"])
        ax.bar([i + width / 2 for i in x], mkt_vs_trv["Travel_Cost"], width, label="Travel Cost", color=PALETTE["accent_amber"])
        ax.set_title("Specialized Cost Drivers: Marketing Cost vs. Travel Cost ($)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(mkt_vs_trv["Department"], rotation=30, ha="right")
        ax.set_ylabel("Average Cost ($)")
        ax.legend()

        fn33 = "chart_33_financial_marketing_and_travel_cost.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn33)
        generated_files.append(fn33)

        return generated_files
