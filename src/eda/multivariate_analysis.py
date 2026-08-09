"""Multivariate analysis module for multi-dimensional financial feature interactions.

This module evaluates multi-variable correlations, correlation heatmaps, feature target rankings,
variance decomposition across cost categories, and pairwise interaction matrices using Matplotlib.

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.visualization_utils import PALETTE, VisualizationUtils


class MultivariateAnalyzer:
    """Multivariate interaction and correlation analysis engine."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize MultivariateAnalyzer with cleaned Pandas DataFrame.

        Args:
            df: Cleaned financial DataFrame.
        """
        self.df: pd.DataFrame = df
        VisualizationUtils.setup_matplotlib_style()

    def generate_multivariate_charts(self) -> List[str]:
        """Generate high-resolution multivariate analysis charts.

        Returns:
            List of generated image filenames saved in images/eda/.
        """
        generated_files: List[str] = []

        # Filter numeric columns for correlation analysis
        numeric_cols = [
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
        num_df = self.df[numeric_cols].dropna()
        corr_matrix = num_df.corr()

        # 16. Correlation Heatmap (Matplotlib custom rendering - No Seaborn)
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax, shrink=0.8)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(numeric_cols, fontsize=9)
        ax.set_title("Multivariate Financial Correlation Heatmap Matrix", fontsize=14, fontweight="bold", pad=40)

        # Annotate correlation values inside cells
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                val = corr_matrix.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

        fn16 = "chart_16_multivariate_correlation_heatmap.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn16)
        generated_files.append(fn16)

        # 17. Target Correlation Ranking Bar Chart
        target_corr = corr_matrix["Budget_Deviation"].drop("Budget_Deviation").sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [PALETTE["accent_red"] if v > 0 else PALETTE["secondary"] for v in target_corr.values]
        ax.barh(target_corr.index, target_corr.values, color=colors, alpha=0.85)
        ax.axvline(0.0, color="black", linestyle="-", linewidth=1)
        ax.set_title("Feature Correlation Ranking with Target: Budget Deviation", fontsize=14, fontweight="bold")
        ax.set_xlabel("Pearson Correlation Coefficient (r)")

        for i, (col, val) in enumerate(target_corr.items()):
            offset = 0.02 if val >= 0 else -0.07
            ax.text(val + offset, i, f"{val:+.2f}", va="center", fontsize=9, fontweight="semibold")

        fn17 = "chart_17_multivariate_target_correlation_bar.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn17)
        generated_files.append(fn17)

        # 18. Cost Variance Breakdown Analysis
        cost_cols = [
            "Operational_Cost",
            "Marketing_Cost",
            "Travel_Cost",
            "Utility_Cost",
            "Vendor_Cost",
            "Miscellaneous_Cost",
        ]
        cost_vars = self.df[cost_cols].var().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(cost_vars.index, cost_vars.values / 1e9, color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
        ax.set_title("Variance Breakdown Across Expense Breakdown Categories ($² Billions)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Variance ($² Billions)")
        ax.set_xticklabels(cost_vars.index, rotation=25, ha="right")

        fn18 = "chart_18_multivariate_cost_variance_breakdown.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn18)
        generated_files.append(fn18)

        # 19. Pairwise Interaction Scatter Grid (OpEx vs Vendor vs Actual vs Budget)
        pair_cols = ["Operational_Cost", "Vendor_Cost", "Actual_Expense", "Planned_Budget"]
        fig, axes = plt.subplots(len(pair_cols), len(pair_cols), figsize=(12, 12))
        fig.suptitle("Pairwise Cost Feature Interaction Scatter Matrix", fontsize=14, fontweight="bold", y=0.92)

        for i, col1 in enumerate(pair_cols):
            for j, col2 in enumerate(pair_cols):
                ax = axes[i, j]
                if i == j:
                    ax.hist(self.df[col1], color=PALETTE["secondary"], alpha=0.7)
                else:
                    ax.scatter(self.df[col2], self.df[col1], alpha=0.15, s=8, color=PALETTE["primary"])
                if i == len(pair_cols) - 1:
                    ax.set_xlabel(col2.replace("_", " "), fontsize=8)
                if j == 0:
                    ax.set_ylabel(col1.replace("_", " "), fontsize=8)
                ax.tick_params(labelsize=7)

        fn19 = "chart_19_multivariate_pairwise_cost_matrix.png"
        VisualizationUtils.save_matplotlib_figure(fig, fn19)
        generated_files.append(fn19)

        return generated_files
