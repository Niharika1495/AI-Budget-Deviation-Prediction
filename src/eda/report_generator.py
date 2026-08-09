"""Executive report generator for financial analytics and business intelligence.

This module compiles multi-format reports from Phase 3 EDA analytics:
- reports/eda/EDA_Report.html (Interactive HTML report with embedded PNG charts)
- reports/eda/EDA_Report.pdf (Printable executive PDF document via ReportLab)
- reports/eda/EDA_Summary.json (Structured JSON summary with all metrics and Q&A answers)

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import REPORTS_DIR

# Target Output Directory
REPORTS_EDA_DIR: Path = REPORTS_DIR / "eda"
REPORTS_EDA_DIR.mkdir(parents=True, exist_ok=True)

HTML_REPORT_PATH: Path = REPORTS_EDA_DIR / "EDA_Report.html"
PDF_REPORT_PATH: Path = REPORTS_EDA_DIR / "EDA_Report.pdf"
JSON_SUMMARY_PATH: Path = REPORTS_EDA_DIR / "EDA_Summary.json"


class ReportGenerator:
    """Multi-format executive report compiler engine (HTML, PDF, JSON)."""

    def __init__(
        self,
        profile_data: Dict[str, Any],
        business_questions: Dict[str, Any],
        correlation_data: Dict[str, Any],
        recommendations: List[Dict[str, str]],
        chart_files: List[str],
    ) -> None:
        """Initialize ReportGenerator with analytical outputs.

        Args:
            profile_data: DataProfiler statistics dictionary.
            business_questions: Answers to 15 business questions.
            correlation_data: Correlation ranking and driver analytics.
            recommendations: Strategic management recommendations.
            chart_files: Filenames of generated PNG charts in images/eda/.
        """
        self.profile: Dict[str, Any] = profile_data
        self.questions: Dict[str, Any] = business_questions
        self.correlations: Dict[str, Any] = correlation_data
        self.recommendations: List[Dict[str, str]] = recommendations
        self.charts: List[str] = chart_files

    def generate_json_summary(self) -> Path:
        """Export structured EDA summary to JSON format.

        Returns:
            Path object pointing to saved JSON file.
        """
        summary_payload = {
            "title": "AI Financial Budget Deviation - Business Intelligence Summary",
            "profiling": self.profile,
            "business_questions_answered": self.questions,
            "correlation_analysis": self.correlations,
            "management_recommendations": self.recommendations,
            "charts_generated_count": len(self.charts),
            "chart_files": self.charts,
        }

        with open(JSON_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4)

        return JSON_SUMMARY_PATH

    def generate_html_report(self) -> Path:
        """Generate styled interactive HTML Business Intelligence report.

        Returns:
            Path object pointing to saved HTML file.
        """
        # Relative image path helper
        def get_img_tag(fn: str) -> str:
            rel_path = f"../../images/eda/{fn}"
            return f'<div class="chart-box"><img src="{rel_path}" alt="{fn}" class="chart-img"/></div>'

        qa_rows = ""
        for q_id, q_data in self.questions.items():
            qa_rows += f"""
            <tr>
                <td><strong>{q_id}</strong></td>
                <td>{q_data['question']}</td>
                <td style="color: #0284C7; font-weight: 600;">{q_data['metric_name']}</td>
                <td style="color: #059669; font-weight: 700;">{q_data['value']}</td>
                <td>{q_data['answer']}</td>
            </tr>
            """

        recs_rows = ""
        for r in self.recommendations:
            recs_rows += f"""
            <tr>
                <td><strong>{r['pillar']}</strong></td>
                <td>{r['recommendation']}</td>
                <td style="color: #059669; font-weight: 600;">{r['expected_impact']}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analytics & Business Intelligence Report - AI Budget Deviation</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-green: #4ade80;
            --accent-purple: #c084fc;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            color: var(--accent);
        }}
        .badge {{
            background-color: rgba(74, 222, 128, 0.15);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 35px;
        }}
        .kpi-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--accent);
            margin-top: 5px;
        }}
        .kpi-label {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        .section h2 {{
            margin-top: 0;
            font-size: 20px;
            color: var(--accent);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            font-size: 13px;
            text-transform: uppercase;
        }}
        tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .chart-box {{
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>AI Financial Budget Deviation - Business Intelligence Report</h1>
                <div style="color: var(--text-muted); font-size: 14px; margin-top: 5px;">
                    Phase 3 Enterprise Financial Analytics & Business Intelligence Module
                </div>
            </div>
            <div class="badge">QUALITY SCORE: {self.profile.get('data_quality_score', 100)}%</div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Analyzed Records</div>
                <div class="kpi-value">{self.profile.get('dataset_shape', {}).get('rows', 0):,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Financial Features</div>
                <div class="kpi-value">{self.profile.get('dataset_shape', {}).get('columns', 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Charts Generated</div>
                <div class="kpi-value" style="color: var(--accent-purple);">{len(self.charts)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Business Q&A Answered</div>
                <div class="kpi-value" style="color: var(--accent-green);">15 / 15</div>
            </div>
        </div>

        <div class="section">
            <h2>15 Executive Business Questions Answered</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Executive Business Question</th>
                        <th>Key Metric</th>
                        <th>Empirical Result</th>
                        <th>Detailed Analytical Finding</th>
                    </tr>
                </thead>
                <tbody>
                    {qa_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Univariate Feature Distribution Analytics</h2>
            <div class="charts-grid">
                {get_img_tag("chart_01_univariate_planned_budget_dist.png")}
                {get_img_tag("chart_03_univariate_actual_expense_dist.png")}
                {get_img_tag("chart_04_univariate_budget_deviation_dist.png")}
                {get_img_tag("chart_07_univariate_operational_cost_dist.png")}
            </div>
        </div>

        <div class="section">
            <h2>Bivariate Financial Relationships</h2>
            <div class="charts-grid">
                {get_img_tag("chart_09_bivariate_budget_vs_actual.png")}
                {get_img_tag("chart_11_bivariate_employee_vs_operational_cost.png")}
                {get_img_tag("chart_12_bivariate_projects_vs_vendor_cost.png")}
                {get_img_tag("chart_13_bivariate_inflation_vs_actual_expense.png")}
            </div>
        </div>

        <div class="section">
            <h2>Time Series & Seasonal Trends (2021 - 2025)</h2>
            <div class="charts-grid">
                {get_img_tag("chart_20_trend_monthly_actual_vs_budget.png")}
                {get_img_tag("chart_21_trend_monthly_budget_deviation.png")}
                {get_img_tag("chart_22_trend_quarterly_budget_and_spending.png")}
                {get_img_tag("chart_25_trend_yearly_budget_utilization_rate.png")}
            </div>
        </div>

        <div class="section">
            <h2>Department & Cost Breakdown Analytics</h2>
            <div class="charts-grid">
                {get_img_tag("chart_26_dept_budget_vs_actual.png")}
                {get_img_tag("chart_27_dept_budget_deviation.png")}
                {get_img_tag("chart_29_dept_expense_composition.png")}
                {get_img_tag("chart_31_financial_company_expense_breakdown.png")}
            </div>
        </div>

        <div class="section">
            <h2>Multivariate Correlation & Target Drivers</h2>
            <div class="charts-grid">
                {get_img_tag("chart_16_multivariate_correlation_heatmap.png")}
                {get_img_tag("chart_17_multivariate_target_correlation_bar.png")}
            </div>
        </div>

        <div class="section">
            <h2>Strategic Management Recommendations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Strategic Pillar</th>
                        <th>Recommendation Action Item</th>
                        <th>Expected Business Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {recs_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(html_content)

        return HTML_REPORT_PATH

    def generate_pdf_report(self) -> Path:
        """Generate multi-page executive PDF report using ReportLab.

        Returns:
            Path object pointing to saved PDF file.
        """
        doc = SimpleDocTemplate(
            str(PDF_REPORT_PATH),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1E293B"),
            alignment=0,
            spaceAfter=10,
        )

        h1_style = ParagraphStyle(
            "Heading1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0284C7"),
            spaceBefore=15,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "BodyText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        )

        story: List[Any] = []

        # Header Block
        story.append(Paragraph("AI Financial Budget Deviation Prediction", title_style))
        story.append(Paragraph("Executive Business Intelligence & Financial Analytics Report (Phase 3)", h1_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=15))

        # Executive Summary Table
        summary_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Records", f"{self.profile.get('dataset_shape', {}).get('rows', 0):,}", "Data Quality Score", f"{self.profile.get('data_quality_score', 100)}%"],
            ["Total Features", str(self.profile.get('dataset_shape', {}).get('columns', 0)), "Business Q&A Answered", "15 / 15"],
            ["Generated Charts", f"{len(self.charts)} Charts", "Memory Footprint", f"{self.profile.get('memory_usage_mb', 0)} MB"],
        ]

        t_summary = Table(summary_data, colWidths=[130, 130, 130, 130])
        t_summary.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t_summary)
        story.append(Spacer(1, 15))

        # 15 Business Questions Table
        story.append(Paragraph("15 Executive Business Questions Answered", h1_style))
        qa_data = [["#", "Business Question", "Metric", "Result"]]
        for q_id, q_info in self.questions.items():
            qa_data.append([
                q_id,
                Paragraph(q_info["question"], body_style),
                Paragraph(q_info["metric_name"], body_style),
                Paragraph(q_info["value"], body_style),
            ])

        t_qa = Table(qa_data, colWidths=[30, 240, 120, 130])
        t_qa.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t_qa)
        story.append(Spacer(1, 15))

        # Key Chart Highlights
        story.append(Paragraph("Key Visual Analytics Artifacts", h1_style))
        chart_samples = [
            "chart_04_univariate_budget_deviation_dist.png",
            "chart_09_bivariate_budget_vs_actual.png",
            "chart_20_trend_monthly_actual_vs_budget.png",
            "chart_26_dept_budget_vs_actual.png",
        ]

        img_dir = REPORTS_EDA_DIR.parent.parent / "images" / "eda"
        for c_fn in chart_samples:
            img_p = img_dir / c_fn
            if img_p.exists():
                story.append(Paragraph(f"<b>Figure:</b> {c_fn}", body_style))
                story.append(Image(str(img_p), width=480, height=240))
                story.append(Spacer(1, 10))

        # Strategic Management Recommendations Table
        story.append(Paragraph("Strategic Management Recommendations", h1_style))
        rec_data = [["Pillar", "Actionable Recommendation", "Expected Impact"]]
        for r in self.recommendations:
            rec_data.append([
                Paragraph(r["pillar"], body_style),
                Paragraph(r["recommendation"], body_style),
                Paragraph(r["expected_impact"], body_style),
            ])

        t_rec = Table(rec_data, colWidths=[120, 260, 140])
        t_rec.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#059669")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t_rec)

        doc.build(story)
        return PDF_REPORT_PATH
