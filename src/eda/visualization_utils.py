"""Visualization utilities module for corporate financial graphics.

This module enforces executive-level styling using Matplotlib and Plotly (Strictly No Seaborn).
Configures consistent corporate color palettes, typography, layout parameters, and high-DPI (300 DPI)
export capabilities into images/eda/.

Follows PEP8, Clean Architecture, SOLID design, type annotations, and docstrings.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from src.config import PROJECT_ROOT

# Base Directory for Saving EDA Charts
IMAGE_EDA_DIR: Path = PROJECT_ROOT / "images" / "eda"
IMAGE_EDA_DIR.mkdir(parents=True, exist_ok=True)

# Executive Corporate Palette Definition
PALETTE = {
    "primary": "#1E293B",     # Deep Slate
    "secondary": "#0284C7",   # Corporate Blue
    "accent_green": "#059669",# Emerald Green
    "accent_red": "#E11D48",  # Crimson Red
    "accent_amber": "#D97706",# Amber Warning
    "accent_purple": "#7C3AED",# Royal Purple
    "neutral_light": "#F8FAFC",# Card Light
    "grid_color": "#E2E8F0",  # Soft Grid Line
}

# Sequential Palette for Categorical Bar Charts
DEPT_COLORS = [
    "#0284C7", "#059669", "#7C3AED", "#D97706",
    "#E11D48", "#2563EB", "#0D9488", "#475569"
]


class VisualizationUtils:
    """Helper utilities for rendering, styling, and saving high-DPI financial charts."""

    @staticmethod
    def setup_matplotlib_style() -> None:
        """Apply standardized global corporate styling rules to Matplotlib."""
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": PALETTE["grid_color"],
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.color": PALETTE["grid_color"],
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.autolayout": True,
        })

    @staticmethod
    def save_matplotlib_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
        """Save a Matplotlib figure as a high-resolution PNG image.

        Args:
            fig: Target Matplotlib Figure object.
            filename: Output filename (e.g. chart_01_budget_dist.png).
            dpi: Dots per inch resolution (default 300 DPI).

        Returns:
            Path object pointing to saved image artifact.
        """
        target_path = IMAGE_EDA_DIR / filename
        fig.savefig(target_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return target_path

    @staticmethod
    def save_plotly_figure(fig: go.Figure, filename: str, width: int = 1200, height: int = 700) -> Path:
        """Save a Plotly figure as a high-resolution static PNG image.

        Args:
            fig: Target Plotly Figure object.
            filename: Output filename (e.g. chart_plotly_01.png).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Path object pointing to saved image artifact.
        """
        target_path = IMAGE_EDA_DIR / filename
        # Format layout nicely before write_image
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Segoe UI, sans-serif", size=12),
            margin=dict(l=50, r=50, t=60, b=50),
        )
        try:
            fig.write_image(str(target_path), width=width, height=height, scale=2)
        except Exception:
            # Fallback to HTML write if kaleido engine encounters system-level driver issues
            html_path = target_path.with_suffix(".html")
            fig.write_html(str(html_path))
        return target_path
