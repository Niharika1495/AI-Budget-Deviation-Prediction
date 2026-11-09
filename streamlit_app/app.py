"""Interactive Streamlit Web Dashboard for AI Budget Deviation Prediction System.

Provides executive financial overview, interactive visual analytics, real-time AI budget
deviation prediction tool with risk assessment & recommendations, model explainability,
and ML integrity audit reports.

Follows Clean Architecture, PEP8 standards, type annotations, and enterprise UI/UX guidelines.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATASET_PATH,
    MODELS_DIR,
    DEPARTMENTS,
    QUARTER_MAPPING,
    INFLATION_RATES,
)
from src.inference.predictor import BudgetDeviationPredictor
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AI Budget Deviation Prediction System",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# THEME CONFIGURATION & DESIGN SYSTEM TOKENS
# ==============================================================================
THEME = {
        "bg": "#07110C",
        "sidebar_bg": "#091710",
        "card_bg": "#0D1D14",
        "card_border": "#1E3828",
        "card_hover": "#12291B",
        "text_primary": "#F1F7F3",
        "text_secondary": "#91A49A",
        "primary_dark_green": "#092918",
        "primary_green": "#16A34A",
        "bright_green": "#35D06F",
        "soft_mint": "#1A3826",
        "very_light_mint": "#0F2619",
        "warning": "#E4B44C",
        "danger": "#F06464",
        "chart_grid": "#1E3828",
        "nav_active_bg": "#16A34A",
        "nav_active_text": "#FFFFFF",
        "planned_color": "#16A34A",
        "actual_color": "#91A49A",
        "deviation_color": "#F06464",
        "forecast_color": "#35D06F",
}

USD_TO_INR = 83.0


def to_inr(amount: float) -> float:
    """Convert a stored/model amount (USD) to the INR amount shown in the UI."""
    return float(amount) * USD_TO_INR


def to_model_currency(amount_inr: float) -> float:
    """Convert an INR form value to the USD unit expected by the trained model."""
    return float(amount_inr) / USD_TO_INR


def format_inr(amount: float, decimals: int = 0, signed: bool = False) -> str:
    """Format currency using Indian digit grouping and the rupee symbol."""
    sign = "+" if signed and amount > 0 else "-" if amount < 0 else ""
    absolute = abs(float(amount))
    formatted = f"{absolute:.{decimals}f}"
    whole, _, fraction = formatted.partition(".")
    if len(whole) > 3:
        last_three = whole[-3:]
        prefix = whole[:-3]
        groups = []
        while prefix:
            groups.append(prefix[-2:])
            prefix = prefix[:-2]
        whole = ",".join(reversed(groups)) + "," + last_three
    value = whole if decimals == 0 else f"{whole}.{fraction}"
    return f"{sign}₹{value}"


def apply_custom_css() -> None:
    """Inject CSS for the dashboard's fixed dark theme."""
    t = THEME
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Global Viewport Background & Base Typography */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {t['bg']};
            color: {t['text_primary']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }}

        /* Transparent Header */
        [data-testid="stHeader"] {{
            background-color: transparent;
        }}

        /* Sidebar Container Styling */
        [data-testid="stSidebar"] {{
            background-color: {t['sidebar_bg']};
            border-right: 1px solid {t['card_border']};
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.5rem;
        }}

        /* Heading Styles */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif;
            color: {t['text_primary']};
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        /* Sidebar Branding */
        .sidebar-brand {{
            padding: 4px 4px 18px 4px;
            border-bottom: 1px solid {t['card_border']};
            margin-bottom: 20px;
        }}
        .brand-title {{
            font-size: 19px;
            font-weight: 800;
            color: {t['text_primary']};
            letter-spacing: 0.04em;
            line-height: 1.2;
        }}
        .brand-accent {{
            color: {t['primary_green']};
        }}
        .brand-subtitle {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {t['text_secondary']};
            margin-top: 4px;
            font-weight: 600;
        }}

        /* Main Header Card */
        .header-card {{
            background-color: {t['card_bg']};
            border: 1px solid {t['card_border']};
            border-radius: 8px;
            padding: 22px 26px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }}
        .header-title {{
            font-size: 26px;
            font-weight: 800;
            color: {t['text_primary']};
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .header-subtitle {{
            font-size: 13px;
            color: {t['text_secondary']};
            margin-top: 6px;
            font-weight: 500;
        }}

        /* KPI Card Styling */
        .kpi-card {{
            background-color: {t['card_bg']};
            border: 1px solid {t['card_border']};
            border-radius: 8px;
            padding: 18px 20px;
            text-align: left;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
            transition: border-color 0.2s ease, transform 0.15s ease;
            height: 100%;
        }}
        .kpi-card:hover {{
            border-color: {t['primary_green']};
        }}
        .kpi-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {t['text_secondary']};
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px;
            font-weight: 700;
            color: {t['text_primary']};
            line-height: 1.25;
        }}
        .kpi-subtext {{
            font-size: 12px;
            color: {t['text_secondary']};
            margin-top: 6px;
            font-weight: 500;
        }}

        /* Hero Prediction Box */
        .hero-card {{
            background-color: {t['card_bg']};
            border: 1.5px solid {t['primary_green']};
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }}

        /* Risk Badges */
        .risk-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .badge-underbudget, .badge-onbudget {{
            background-color: {t['soft_mint']};
            color: {t['primary_green']};
            border: 1px solid {t['primary_green']};
        }}
        .badge-modover {{
            background-color: rgba(216, 155, 36, 0.12);
            color: {t['warning']};
            border: 1px solid {t['warning']};
        }}
        .badge-severeover {{
            background-color: rgba(217, 74, 74, 0.12);
            color: {t['danger']};
            border: 1px solid {t['danger']};
        }}

        /* Standard Section Card Container */
        .section-card {{
            background-color: {t['card_bg']};
            border: 1px solid {t['card_border']};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        /* Navigation Radio Override */
        div[data-testid="stRadio"] > label {{
            display: none;
        }}
        div[data-testid="stRadio"] > div {{
            gap: 6px;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 10px 14px;
            margin: 0;
            width: 100%;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
            background-color: {t['card_hover']};
            border-color: {t['card_border']};
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {{
            background-color: {t['nav_active_bg']};
            color: {t['nav_active_text']} !important;
            border-color: {t['nav_active_bg']};
            font-weight: 600;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] span {{
            color: inherit !important;
        }}

        /* Streamlit Input Controls Customization */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{
            background-color: {t['card_bg']} !important;
            border-color: {t['card_border']} !important;
            color: {t['text_primary']} !important;
            border-radius: 6px !important;
        }}

        /* Primary Form Button */
        div.stButton > button {{
            background-color: {t['primary_green']} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 12px 24px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            width: 100% !important;
            letter-spacing: 0.02em !important;
            transition: opacity 0.2s ease !important;
        }}
        div.stButton > button:hover {{
            opacity: 0.92 !important;
        }}

        /* Table & Dataframe styling */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {t['card_border']};
            border-radius: 8px;
            overflow: hidden;
        }}

        /* Recommendation Item */
        .rec-item {{
            background-color: {t['very_light_mint']};
            border-left: 3px solid {t['primary_green']};
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-size: 13.5px;
            color: {t['text_primary']};
        }}

        /* Hide top decorative red/blue line from Streamlit */
        div[data-testid="stDecoration"] {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_layout(title: str = "") -> dict:
    """Generate consistent Plotly layout matching the dark theme tokens."""
    t = THEME
    return dict(
        title=dict(
            text=title,
            font=dict(family="Inter", size=15, color=t["text_primary"]),
            x=0.0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=t["text_secondary"], size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            gridcolor=t["chart_grid"],
            zerolinecolor=t["chart_grid"],
            tickfont=dict(color=t["text_secondary"]),
        ),
        yaxis=dict(
            gridcolor=t["chart_grid"],
            zerolinecolor=t["chart_grid"],
            tickfont=dict(color=t["text_secondary"]),
        ),
        legend=dict(
            font=dict(color=t["text_primary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


# ==============================================================================
# DATA & PREDICTOR LOADERS
# ==============================================================================
@st.cache_data
def load_datasets() -> pd.DataFrame:
    """Load clean dataset for analytics and raw dataset fallback."""
    clean_csv = PROCESSED_DATA_DIR / "financial_budget_deviation_clean.csv"
    raw_csv = RAW_DATASET_PATH

    if clean_csv.exists():
        df_clean = pd.read_csv(clean_csv)
    elif raw_csv.exists():
        df_raw = pd.read_csv(raw_csv)
        pipeline = PreprocessingPipeline()
        df_clean, _, _ = pipeline.run(df_raw)
    else:
        df_clean = pd.DataFrame()

    return df_clean


@st.cache_resource
def load_predictor() -> BudgetDeviationPredictor:
    """Load inference engine predictor."""
    return BudgetDeviationPredictor(models_dir=MODELS_DIR)


@st.cache_data
def load_reports() -> tuple[dict, dict]:
    """Load pre-computed model evaluation benchmark and integrity audit reports."""
    eval_json = PROJECT_ROOT / "reports" / "model_evaluation_report.json"
    audit_json = PROJECT_ROOT / "reports" / "ml_integrity_audit.json"

    eval_data = {}
    audit_data = {}

    if eval_json.exists():
        with open(eval_json, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

    if audit_json.exists():
        with open(audit_json, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

    return eval_data, audit_data


# Load cached singletons
df_data = load_datasets()
predictor = load_predictor()
eval_data, audit_data = load_reports()


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-title">AI BUDGET<br><span class="brand-accent">DEVIATION</span></div>
            <div class="brand-subtitle">Predict • Explain • Optimize</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    nav_selection = st.radio(
        "Navigation",
        options=[
            "Executive Overview",
            "Financial Analytics",
            "AI Prediction",
            "Model Intelligence",
            "Reports",
        ],
        key="main_nav",
    )

# Apply the dashboard's fixed dark theme
apply_custom_css()
current_theme = THEME


# ==============================================================================
# HEADER BANNER
# ==============================================================================
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">AI Budget Deviation Prediction</div>
        <div class="header-subtitle">Predict • Explain • Optimize Financial Risk</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION 1: EXECUTIVE OVERVIEW
# ==============================================================================
if nav_selection == "Executive Overview":
    st.markdown("### Executive Overview")

    if not df_data.empty:
        total_planned = df_data["Planned_Budget"].sum()
        total_actual = df_data["Actual_Expense"].sum()
        net_dev = df_data["Budget_Deviation"].sum()

        # Calculate High-Risk Record Rate (percentage of records categorized as Moderate or Severe Overbudget)
        if "Deviation_Risk_Category" in df_data.columns:
            high_risk_records = df_data["Deviation_Risk_Category"].isin(
                ["Moderate Overbudget", "Severe Overbudget"]
            ).sum()
        else:
            high_risk_records = (df_data["Budget_Deviation"] > 0).sum()
        
        high_risk_rate = (high_risk_records / len(df_data)) * 100.0 if len(df_data) > 0 else 0.0

        # KPI Cards Display
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Planned Budget</div>
                    <div class="kpi-value">{format_inr(to_inr(total_planned))}</div>
                    <div class="kpi-subtext">Approved Financial Allocation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Actual Expenses</div>
                    <div class="kpi-value">{format_inr(to_inr(total_actual))}</div>
                    <div class="kpi-subtext">Realized Expenditures</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            dev_color = current_theme["danger"] if net_dev > 0 else current_theme["bright_green"]
            var_pct = (net_dev / total_planned * 100.0) if total_planned > 0 else 0.0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Net Budget Deviation</div>
                    <div class="kpi-value" style="color: {dev_color};">{format_inr(to_inr(net_dev), signed=True)}</div>
                    <div class="kpi-subtext">Variance Rate: {var_pct:+.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            risk_color = current_theme["danger"] if high_risk_rate > 50 else current_theme["warning"]
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">High-Risk Record Rate</div>
                    <div class="kpi-value" style="color: {risk_color};">{high_risk_rate:.1f}%</div>
                    <div class="kpi-subtext">Moderate & Severe Overbudget Shares</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Executive Charts
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Department Budget vs Actual Expenses")
            dept_df = (
                df_data.groupby("Department")[["Planned_Budget", "Actual_Expense"]]
                .sum()
                .reset_index()
            )
            dept_df[["Planned_Budget", "Actual_Expense"]] *= USD_TO_INR
            
            fig_dept = px.bar(
                dept_df,
                x="Department",
                y=["Planned_Budget", "Actual_Expense"],
                barmode="group",
                color_discrete_map={
                    "Planned_Budget": current_theme["planned_color"],
                    "Actual_Expense": current_theme["actual_color"],
                },
                labels={"value": "Amount (₹)", "variable": "Metric"},
            )
            layout = get_plotly_layout()
            layout["legend_title_text"] = ""
            fig_dept.update_layout(**layout)
            st.plotly_chart(fig_dept, use_container_width=True)

        with c2:
            st.markdown("#### Department Deviation Ranking")
            dept_dev = (
                df_data.groupby("Department")["Budget_Deviation"]
                .sum()
                .reset_index()
                .sort_values(by="Budget_Deviation", ascending=True)
            )
            dept_dev["Budget_Deviation"] *= USD_TO_INR

            # Assign colors based on deviation status
            dept_dev["Color"] = dept_dev["Budget_Deviation"].apply(
                lambda x: current_theme["danger"] if x > 0 else current_theme["primary_green"]
            )

            fig_rank = px.bar(
                dept_dev,
                x="Budget_Deviation",
                y="Department",
                orientation="h",
                labels={"Budget_Deviation": "Net Deviation (₹)", "Department": "Department"},
            )
            fig_rank.update_traces(marker_color=dept_dev["Color"])
            layout = get_plotly_layout()
            fig_rank.update_layout(**layout)
            st.plotly_chart(fig_rank, use_container_width=True)

    else:
        st.info("Dataset loading. Run data generator script if necessary.")


# ==============================================================================
# SECTION 2: FINANCIAL ANALYTICS & TRENDS
# ==============================================================================
elif nav_selection == "Financial Analytics":
    st.markdown("### Financial Analytics & Trend Explorer")

    if not df_data.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_depts = st.multiselect(
                "Filter Departments",
                options=list(DEPARTMENTS),
                default=list(DEPARTMENTS),
            )
        with col_f2:
            selected_years = st.multiselect(
                "Filter Years",
                options=sorted(df_data["Year"].unique()),
                default=sorted(df_data["Year"].unique()),
            )

        df_filtered = df_data[
            (df_data["Department"].isin(selected_depts))
            & (df_data["Year"].isin(selected_years))
        ]

        # Quarterly Expense Trajectory
        st.markdown("#### Quarterly Expense & Deviation Trajectory")
        q_df = (
            df_filtered.groupby(["Year", "Quarter"])[
                ["Planned_Budget", "Actual_Expense", "Budget_Deviation"]
            ]
            .sum()
            .reset_index()
        )
        q_df["Period"] = (
            q_df["Year"].astype(str) + " " + q_df["Quarter"].astype(str)
        )
        q_df[["Planned_Budget", "Actual_Expense", "Budget_Deviation"]] *= USD_TO_INR

        fig_trend = px.line(
            q_df,
            x="Period",
            y=["Planned_Budget", "Actual_Expense", "Budget_Deviation"],
            markers=True,
            color_discrete_map={
                "Planned_Budget": current_theme["planned_color"],
                "Actual_Expense": current_theme["actual_color"],
                "Budget_Deviation": current_theme["deviation_color"],
            },
        )
        layout = get_plotly_layout()
        layout["legend_title_text"] = ""
        fig_trend.update_layout(**layout)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Cost Structure Breakdown
        st.markdown("#### Aggregate Cost Structure Breakdown")
        cost_cols = [
            "Operational_Cost",
            "Marketing_Cost",
            "Travel_Cost",
            "Utility_Cost",
            "Vendor_Cost",
            "Miscellaneous_Cost",
        ]
        cost_sums = df_filtered[cost_cols].sum().reset_index()
        cost_sums.columns = ["Cost Category", "Amount (₹)"]
        cost_sums["Amount (₹)"] *= USD_TO_INR
        cost_sums = cost_sums.sort_values(by="Amount (₹)", ascending=False)

        # Green intensity gradient for cost categories
        green_palette = [
            "#16A34A", "#22C55E", "#35D06F", "#4ADE80", "#86EFAC", "#A7F3D0",
        ]

        fig_cost = px.bar(
            cost_sums,
            x="Cost Category",
            y="Amount (₹)",
            color="Cost Category",
            color_discrete_sequence=green_palette,
        )
        layout = get_plotly_layout()
        layout["showlegend"] = False
        fig_cost.update_layout(**layout)
        st.plotly_chart(fig_cost, use_container_width=True)


# ==============================================================================
# SECTION 3: AI REAL-TIME PREDICTOR
# ==============================================================================
elif nav_selection == "AI Prediction":
    st.markdown("### AI Real-Time Budget Deviation Predictor")
    st.markdown(
        "Enter corporate budget parameters below to generate machine learning forecasts of expected expense overruns, risk categories, and AI cost optimization strategies."
    )

    with st.form("prediction_form"):
        col_i1, col_i2, col_i3 = st.columns(3)

        with col_i1:
            dept = st.selectbox("Department", list(DEPARTMENTS))
            planned_budget = st.number_input(
                "Planned Budget (₹)", min_value=to_inr(1000), value=to_inr(250000), step=to_inr(5000)
            )
            revenue = st.number_input(
                "Department Revenue (₹)", min_value=0.0, value=to_inr(350000), step=to_inr(10000)
            )
            emp_count = st.number_input(
                "Employee Count", min_value=1, value=25, step=1
            )
            proj_count = st.number_input(
                "Active Project Count", min_value=1, value=5, step=1
            )

        with col_i2:
            op_cost = st.number_input(
                "Operational Cost (₹)", min_value=0.0, value=to_inr(90000), step=to_inr(2000)
            )
            mktg_cost = st.number_input(
                "Marketing Cost (₹)", min_value=0.0, value=to_inr(40000), step=to_inr(2000)
            )
            vendor_cost = st.number_input(
                "Vendor Cost (₹)", min_value=0.0, value=to_inr(55000), step=to_inr(2000)
            )
            travel_cost = st.number_input(
                "Travel Cost (₹)", min_value=0.0, value=to_inr(20000), step=to_inr(1000)
            )

        with col_i3:
            utility_cost = st.number_input(
                "Utility Cost (₹)", min_value=0.0, value=to_inr(15000), step=to_inr(1000)
            )
            misc_cost = st.number_input(
                "Miscellaneous Cost (₹)", min_value=0.0, value=to_inr(10000), step=to_inr(1000)
            )
            inflation_rate = st.slider(
                "Inflation Rate", min_value=0.01, max_value=0.15, value=0.054, step=0.001
            )
            prev_m_exp = st.number_input(
                "Previous Month Expense (₹)", min_value=0.0, value=to_inr(80000), step=to_inr(2000)
            )
            prev_q_exp = st.number_input(
                "Previous Quarter Expense (₹)", min_value=0.0, value=to_inr(240000), step=to_inr(5000)
            )

        submit_btn = st.form_submit_button("Generate AI Budget Prediction")

    if submit_btn:
        input_record = {
            "Record_ID": "SIM_PRED_001",
            "Year": 2025,
            "Quarter": "Q4",
            "Month": "December",
            "Department": dept,
            "Cost_Center": f"CC-{dept[:3].upper()}-101",
            "Planned_Budget": to_model_currency(planned_budget),
            "Revenue": to_model_currency(revenue),
            "Employee_Count": emp_count,
            "Project_Count": proj_count,
            "Operational_Cost": to_model_currency(op_cost),
            "Marketing_Cost": to_model_currency(mktg_cost),
            "Travel_Cost": to_model_currency(travel_cost),
            "Utility_Cost": to_model_currency(utility_cost),
            "Vendor_Cost": to_model_currency(vendor_cost),
            "Miscellaneous_Cost": to_model_currency(misc_cost),
            "Inflation_Rate": inflation_rate,
            "Previous_Month_Expense": to_model_currency(prev_m_exp),
            "Previous_Quarter_Expense": to_model_currency(prev_q_exp),
        }

        # Predict using inference engine (Backend execution untouched)
        res = predictor.predict_single(input_record)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Prediction Result & Risk Assessment")

        # Hero Prediction Result Display
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Predicted Actual Expense</div>
                    <div class="kpi-value">{format_inr(to_inr(res['predicted_actual_expense']), decimals=2)}</div>
                    <div class="kpi-subtext">Baseline Budget: {format_inr(planned_budget, decimals=2)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p_col2:
            dev_val = res["predicted_budget_deviation"]
            color = current_theme["danger"] if dev_val > 0 else current_theme["bright_green"]
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Predicted Budget Deviation</div>
                    <div class="kpi-value" style="color: {color};">{format_inr(to_inr(dev_val), decimals=2, signed=True)}</div>
                    <div class="kpi-subtext">Variance: {res['deviation_percentage']:+.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p_col3:
            badge_class = "badge-onbudget"
            if "Severe" in res["predicted_risk_level"]:
                badge_class = "badge-severeover"
            elif "Moderate" in res["predicted_risk_level"]:
                badge_class = "badge-modover"
            elif "Underbudget" in res["predicted_risk_level"]:
                badge_class = "badge-underbudget"

            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Risk Assessment Tier</div>
                    <div style="margin-top: 10px;">
                        <span class="risk-badge {badge_class}">{res['predicted_risk_level']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Cost Driver Attribution & AI Recommendations
        d_col, r_col = st.columns(2)

        with d_col:
            st.markdown("#### Cost Driver Attribution")
            drivers_df = pd.DataFrame(res["top_cost_drivers"])
            if not drivers_df.empty:
                drivers_df["amount"] *= USD_TO_INR
                fig_drv = px.bar(
                    drivers_df,
                    x="amount",
                    y="driver_name",
                    orientation="h",
                    color="share_of_total_cost_percent",
                    labels={
                        "amount": "Expense Amount (₹)",
                        "driver_name": "Cost Component",
                        "share_of_total_cost_percent": "Share (%)",
                    },
                    color_continuous_scale=[
                        current_theme["soft_mint"],
                        current_theme["primary_green"],
                        current_theme["danger"],
                    ],
                )
                layout = get_plotly_layout()
                fig_drv.update_layout(**layout)
                st.plotly_chart(fig_drv, use_container_width=True)

        with r_col:
            st.markdown("#### AI Recommendations")
            for rec in res["recommendations"]:
                st.markdown(f'<div class="rec-item">• {rec}</div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 4: MODEL INTELLIGENCE & METRICS
# ==============================================================================
elif nav_selection == "Model Intelligence":
    st.markdown("### Model Intelligence & Architecture")

    # Global Feature Importance
    if predictor.explainer:
        st.markdown("#### Global Feature Importance Ranking")
        importances = predictor.explainer.get_global_feature_importance()

        if importances:
            imp_df = (
                pd.DataFrame(
                    list(importances.items())[:15], columns=["Feature", "Importance Score"]
                )
                .sort_values(by="Importance Score", ascending=True)
            )

            green_scale = [
                current_theme["soft_mint"],
                current_theme["primary_green"],
                current_theme["bright_green"],
            ]

            fig_imp = px.bar(
                imp_df,
                x="Importance Score",
                y="Feature",
                orientation="h",
                color="Importance Score",
                color_continuous_scale=green_scale,
            )
            layout = get_plotly_layout()
            fig_imp.update_layout(**layout)
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Saved Evaluation Benchmark Details
    if eval_data:
        st.markdown("#### Saved Model Performance Benchmarks")

        m_col1, m_col2 = st.columns(2)

        # Regression Metrics (MAPE Headline REMOVED as instructed)
        with m_col1:
            reg = eval_data.get("regression_evaluation", {})
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="kpi-label">Regression Model ({reg.get('model_name', 'HistGradientBoosting')})</div>
                    <table style="width:100%; margin-top:12px; font-size:13.5px; border-collapse:collapse;">
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">Mean Absolute Error (MAE):</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{format_inr(to_inr(reg.get('mae', 0)), decimals=2)}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">Root Mean Squared Error (RMSE):</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{format_inr(to_inr(reg.get('rmse', 0)), decimals=2)}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">R² Score:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'; color:{current_theme['bright_green']}">{reg.get('r2_score', 0):.3f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">5-Fold CV RMSE:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{format_inr(to_inr(reg.get('cv_rmse', 0)), decimals=2)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">Max Absolute Error:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{format_inr(to_inr(reg.get('max_error', 0)), decimals=2)}</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Classification Metrics
        with m_col2:
            clf = eval_data.get("classification_evaluation", {})
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="kpi-label">Classification Model ({clf.get('model_name', 'RandomForestClassifier')})</div>
                    <table style="width:100%; margin-top:12px; font-size:13.5px; border-collapse:collapse;">
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">Accuracy Score:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'; color:{current_theme['bright_green']}">{clf.get('accuracy', 0)*100:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">Weighted F1-Score:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{clf.get('f1_weighted', 0)*100:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">Macro F1-Score:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{clf.get('f1_macro', 0)*100:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid {current_theme['card_border']};">
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">Weighted Precision:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{clf.get('precision_weighted', 0)*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; color:{current_theme['text_secondary']}">5-Fold CV Accuracy:</td>
                            <td style="font-weight:700; text-align:right; font-family:'JetBrains Mono'">{clf.get('cv_accuracy', 0)*100:.2f}%</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Confusion Matrix Heatmap
        cm = clf.get("confusion_matrix", [])
        labels = clf.get("labels", [])
        if cm and labels:
            st.markdown("#### Classification Confusion Matrix")
            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                labels=dict(x="Predicted Risk Tier", y="True Risk Tier", color="Sample Count"),
                text_auto=True,
                color_continuous_scale="Greens",
            )
            layout = get_plotly_layout()
            fig_cm.update_layout(**layout)
            st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# SECTION 5: REPORTS & INTEGRITY AUDIT
# ==============================================================================
elif nav_selection == "Reports":
    st.markdown("### Reports & ML Integrity Audit")

    if audit_data:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h4 style="margin:0;">{audit_data.get('audit_title', 'ML Integrity Audit Report')}</h4>
                        <div style="font-size:12px; color:{current_theme['text_secondary']}; margin-top:4px;">
                            Target Objective: {audit_data.get('target_definition', {}).get('prediction_objective', '')}
                        </div>
                    </div>
                    <div>
                        <span class="risk-badge badge-underbudget">STATUS: {audit_data.get('integrity_status', 'PASS')}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Passed Checks summary
        st.markdown("#### Passed System Integrity Checks")
        for check in audit_data.get("passed_checks", []):
            st.markdown(f'<div class="rec-item">✓ {check}</div>', unsafe_allow_html=True)

        # Feature Lineage Table
        lineage = audit_data.get("feature_lineage_table", [])
        if lineage:
            st.markdown("#### Feature Lineage & Target Leakage Prevention Status")
            lineage_df = pd.DataFrame(lineage)
            st.dataframe(lineage_df, use_container_width=True)

        # Methodological Isolation Summary
        methodology = audit_data.get("methodology", {})
        if methodology:
            st.markdown("#### Methodological Data Isolation")
            t_split = methodology.get("temporal_split", {})
            st.markdown(
                f"""
                - **Train Period**: {t_split.get('train_period', '2021-2024')}
                - **Unseen Test Period**: {t_split.get('test_period', '2025 Unseen')}
                - **Preprocessing Pipeline Isolation**: {methodology.get('preprocessing_isolation', '')}
                """
            )

    else:
        st.info("No saved audit logs located in reports directory.")
