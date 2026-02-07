"""
Package initialization for Financial Operations Analytics
"""

__version__ = "1.0.0"
__author__ = "Financial Operations Analytics"

from .data_generator import FinancialDataGenerator
from .data_preprocessing import DataPreprocessor
from .revenue_forecasting import RevenueForecaster
from .churn_prediction import ChurnPredictor
from .profitability_analysis import ProfitabilityAnalyzer
from .cohort_rfm_analysis import CohortRFMAnalyzer
from .executive_reporting import ExecutiveReporter
from .main_pipeline import run_full_pipeline

__all__ = [
    'FinancialDataGenerator',
    'DataPreprocessor', 
    'RevenueForecaster',
    'ChurnPredictor',
    'ProfitabilityAnalyzer',
    'CohortRFMAnalyzer',
    'ExecutiveReporter',
    'run_full_pipeline'
]
