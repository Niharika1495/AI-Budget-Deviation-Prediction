"""Main execution orchestrator script for Phase 3 Financial Analytics & BI Module (EDA).

This script coordinates the entire Phase 3 analytical pipeline:
1. Loads data/processed/financial_budget_deviation_clean.csv
2. Executes Data Profiling & Quality Scoring
3. Generates 33 high-resolution charts in images/eda/ (Univariate, Bivariate, Multivariate, Trends, Dept, Financial)
4. Executes Correlation Driver Ranking
5. Computes empirical answers to all 15 executive business questions
6. Generates multi-format report outputs in reports/eda/ (HTML, PDF, JSON)
7. Logs progress, timing, and metrics

Follows PEP8, Clean Architecture, SOLID principles, type annotations, and docstrings.
"""

import logging
import sys
import time
from pathlib import Path
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import PROJECT_ROOT
from src.eda.bivariate_analysis import BivariateAnalyzer
from src.eda.correlation_analysis import CorrelationAnalyzer
from src.eda.data_profiler import DataProfiler
from src.eda.department_analysis import DepartmentAnalyzer
from src.eda.financial_analysis import FinancialAnalyzer
from src.eda.insight_generator import BusinessInsightGenerator
from src.eda.multivariate_analysis import MultivariateAnalyzer
from src.eda.report_generator import ReportGenerator
from src.eda.trend_analysis import TrendAnalyzer
from src.eda.univariate_analysis import UnivariateAnalyzer

# File Paths
CLEAN_DATASET_PATH: Path = PROJECT_ROOT / "data" / "processed" / "financial_budget_deviation_clean.csv"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_eda")


def main() -> None:
    """Orchestrate Phase 3 Financial Analytics & Business Intelligence pipeline."""
    start_time = time.time()
    logger.info("Starting Phase 3 Financial Analytics & Business Intelligence Pipeline...")

    # 1. Load Processed Clean Dataset
    if not CLEAN_DATASET_PATH.exists():
        logger.error("Clean dataset not found at %s. Please run Phase 2 preprocessing first.", CLEAN_DATASET_PATH)
        sys.exit(1)

    logger.info("Loading cleaned dataset from %s ...", CLEAN_DATASET_PATH)
    df_clean = pd.read_csv(CLEAN_DATASET_PATH)
    logger.info("Clean dataset loaded successfully. Shape: %s", df_clean.shape)

    all_generated_charts = []

    # 2. Data Profiling
    logger.info("Executing Data Profiler...")
    profiler = DataProfiler(df_clean)
    profile_data = profiler.profile()
    logger.info("Data Quality Score: %.2f%%", profile_data["data_quality_score"])

    # 3. Univariate Analysis
    logger.info("Executing Univariate Analysis (Charts 01-08)...")
    uni_analyzer = UnivariateAnalyzer(df_clean)
    uni_charts = uni_analyzer.generate_univariate_charts()
    all_generated_charts.extend(uni_charts)
    logger.info("Generated %d univariate charts.", len(uni_charts))

    # 4. Bivariate Analysis
    logger.info("Executing Bivariate Analysis (Charts 09-15)...")
    biv_analyzer = BivariateAnalyzer(df_clean)
    biv_charts = biv_analyzer.generate_bivariate_charts()
    all_generated_charts.extend(biv_charts)
    logger.info("Generated %d bivariate charts.", len(biv_charts))

    # 5. Multivariate Analysis
    logger.info("Executing Multivariate Analysis (Charts 16-19)...")
    multi_analyzer = MultivariateAnalyzer(df_clean)
    multi_charts = multi_analyzer.generate_multivariate_charts()
    all_generated_charts.extend(multi_charts)
    logger.info("Generated %d multivariate charts.", len(multi_charts))

    # 6. Time Series & Seasonal Trend Analysis
    logger.info("Executing Time Series Trend Analysis (Charts 20-25)...")
    trend_analyzer = TrendAnalyzer(df_clean)
    trend_charts = trend_analyzer.generate_trend_charts()
    all_generated_charts.extend(trend_charts)
    logger.info("Generated %d time series trend charts.", len(trend_charts))

    # 7. Department Deep-Dive Analysis
    logger.info("Executing Department Breakdown Analysis (Charts 26-30)...")
    dept_analyzer = DepartmentAnalyzer(df_clean)
    dept_charts = dept_analyzer.generate_department_charts()
    all_generated_charts.extend(dept_charts)
    logger.info("Generated %d department charts.", len(dept_charts))

    # 8. Financial Category Breakdown Analysis
    logger.info("Executing Financial Cost Category Analysis (Charts 31-33)...")
    fin_analyzer = FinancialAnalyzer(df_clean)
    fin_charts = fin_analyzer.generate_financial_charts()
    all_generated_charts.extend(fin_charts)
    logger.info("Generated %d financial category charts.", len(fin_charts))

    # 9. Correlation Analysis
    logger.info("Executing Target Correlation Driver Analysis...")
    corr_analyzer = CorrelationAnalyzer(df_clean)
    corr_data = corr_analyzer.analyze_correlations()

    # 10. Business Insight Generator (15 Business Questions & Recommendations)
    logger.info("Solving 15 Executive Business Questions & Formulating Recommendations...")
    insight_gen = BusinessInsightGenerator(df_clean)
    bq_answers = insight_gen.answer_business_questions()
    recommendations = insight_gen.generate_management_recommendations()

    # 11. Multi-Format Report Generation (HTML, PDF, JSON)
    logger.info("Compiling multi-format executive reports (HTML, PDF, JSON) in reports/eda/ ...")
    report_gen = ReportGenerator(
        profile_data=profile_data,
        business_questions=bq_answers,
        correlation_data=corr_data,
        recommendations=recommendations,
        chart_files=all_generated_charts,
    )

    json_path = report_gen.generate_json_summary()
    html_path = report_gen.generate_html_report()
    pdf_path = report_gen.generate_pdf_report()

    elapsed = time.time() - start_time
    logger.info("--------------------------------------------------------------------------------")
    logger.info("Phase 3 Financial Analytics & BI Pipeline Completed in %.2f seconds.", elapsed)
    logger.info("Total Charts Generated : %d charts saved in images/eda/", len(all_generated_charts))
    logger.info("Business Questions Answered: 15 / 15")
    logger.info("HTML Report Saved      : %s", html_path)
    logger.info("PDF Report Saved       : %s", pdf_path)
    logger.info("JSON Summary Saved     : %s", json_path)
    logger.info("--------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
