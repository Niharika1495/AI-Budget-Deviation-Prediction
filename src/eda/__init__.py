"""Financial Analytics and Business Intelligence (EDA) Package.

Provides clean architecture modules for data profiling, univariate/bivariate/multivariate
analysis, time series trend evaluation, department deep dives, financial cost breakdowns,
correlation target drivers, business question answering, interactive visual rendering,
and automated multi-format report generation (HTML, PDF, JSON).
"""

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
from src.eda.visualization_utils import VisualizationUtils

__all__ = [
    "VisualizationUtils",
    "DataProfiler",
    "UnivariateAnalyzer",
    "BivariateAnalyzer",
    "MultivariateAnalyzer",
    "TrendAnalyzer",
    "DepartmentAnalyzer",
    "FinancialAnalyzer",
    "CorrelationAnalyzer",
    "BusinessInsightGenerator",
    "ReportGenerator",
]
