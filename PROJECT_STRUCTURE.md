# Financial Operations Analytics - Project Structure
# =================================================
# This file documents the complete project structure for GitHub deployment
#
# Generated: 2026-02-07
# Version: 1.0.0

```
financial-operations-analytics/
│
├── 📜 README.md                              # Project documentation
├── 📜 requirements.txt                       # Python dependencies
├── 📜 run_pipeline.py                        # Main pipeline runner
├── 📜 LICENSE                                # MIT License
├── 📜 .gitignore                             # Git ignore rules
│
├── 📂 data/                                  # Data directory
│   ├── customers.csv                         # Customer master data (2,500 records)
│   └── transactions.csv                      # Transaction history (~31,000 records)
│
├── 📂 src/                                   # Source code
│   ├── __init__.py                           # Package initialization
│   ├── data_generator.py                     # Synthetic data generation
│   ├── data_preprocessing.py                 # Data cleaning & feature engineering
│   ├── revenue_forecasting.py                # Prophet & ARIMA forecasting
│   ├── churn_prediction.py                   # ML churn models (LR, RF)
│   ├── profitability_analysis.py             # Profitability calculations
│   ├── customer_segmentation.py              # RFM & clustering analysis
│   ├── retention_recommendations.py          # Retention action engine
│   └── unified_analytics.py                  # Dashboard consolidation
│
├── 📂 notebooks/                             # Jupyter notebooks
│   └── financial_analytics.ipynb             # Interactive analysis notebook
│
├── 📂 models/                                # Trained ML models
│   ├── logistic_regression.pkl               # Churn - Logistic Regression
│   ├── random_forest.pkl                     # Churn - Random Forest
│   └── scaler.pkl                            # Feature scaler
│
├── 📂 outputs/                               # Analysis outputs
│   │
│   ├── processed_customers.csv               # Engineered customer features
│   ├── processed_transactions.csv            # Engineered transaction features
│   ├── monthly_revenue.csv                   # Monthly aggregated revenue
│   ├── daily_revenue.csv                     # Daily revenue data
│   ├── monthly_acquisitions.csv              # Customer acquisition by month
│   │
│   ├── 📂 forecasts/                         # Revenue forecasting outputs
│   │   ├── prophet_forecast.csv              # Prophet model predictions
│   │   ├── arima_forecast.csv                # ARIMA model predictions
│   │   ├── combined_forecast.csv             # Ensemble predictions
│   │   ├── forecast_dashboard_ready.csv      # Dashboard-ready format
│   │   ├── time_series_decomposition.csv     # Trend, seasonal, residual
│   │   ├── seasonal_indices.csv              # Monthly seasonality
│   │   ├── model_comparison.csv              # Model performance metrics
│   │   └── forecast_summary_report.txt       # Forecast summary
│   │
│   ├── 📂 churn/                             # Churn prediction outputs
│   │   ├── churn_predictions.csv             # Customer churn scores
│   │   ├── churn_risk_summary.csv            # Risk level summary
│   │   ├── churn_model_comparison.csv        # Model comparison
│   │   ├── lr_feature_importance.csv         # LR coefficients
│   │   ├── rf_feature_importance.csv         # RF feature importance
│   │   └── high_risk_customers_action.csv    # Actionable high-risk list
│   │
│   ├── 📂 profitability/                     # Profitability analysis outputs
│   │   ├── customer_profitability_full.csv   # Complete profitability data
│   │   ├── profitability_dashboard.csv       # Dashboard-ready format
│   │   ├── profitability_kpis.csv            # KPI metrics
│   │   ├── profitability_tier_summary.csv    # Tier breakdown
│   │   ├── contribution_by_subscription.csv  # Revenue/profit by subscription
│   │   ├── contribution_by_country.csv       # Revenue/profit by country
│   │   ├── contribution_by_industry.csv      # Revenue/profit by industry
│   │   └── profitability_report.txt          # Executive report
│   │
│   ├── 📂 segmentation/                      # Customer segmentation outputs
│   │   ├── customer_segmentation_full.csv    # Complete RFM data
│   │   ├── segmentation_dashboard.csv        # Dashboard-ready format
│   │   ├── rfm_segment_summary.csv           # Segment profiles
│   │   ├── segment_kpis.csv                  # Segment KPIs
│   │   ├── behavioral_clusters.csv           # Clustering results
│   │   ├── cohort_retention_dashboard.csv    # Retention heatmap data
│   │   ├── cohort_revenue_dashboard.csv      # Cohort revenue
│   │   ├── retention_curve.csv               # Average retention by month
│   │   └── segmentation_report.txt           # Analysis report
│   │
│   ├── 📂 retention/                         # Retention recommendations
│   │   ├── retention_recommendations_full.csv # All recommendations
│   │   ├── high_risk_action_list.csv         # High-risk customer actions
│   │   ├── medium_risk_engagement_list.csv   # Medium-risk actions
│   │   ├── low_risk_loyalty_list.csv         # Low-risk actions
│   │   ├── retention_action_summary.csv      # Action summary
│   │   └── retention_executive_summary.txt   # Executive summary
│   │
│   └── 📂 unified/                           # Consolidated outputs
│       ├── master_customer_analytics.csv     # Master customer dataset
│       ├── monthly_business_summary.csv      # Monthly business KPIs
│       ├── executive_summary_report.txt      # Executive report
│       │
│       └── 📂 dashboard/                     # Power BI-ready files
│           ├── fact_customer_metrics.csv     # Customer fact table
│           ├── fact_monthly_summary.csv      # Monthly fact table
│           ├── fact_revenue_forecast.csv     # Forecast fact table
│           ├── dim_customer_attributes.csv   # Customer dimension
│           ├── dim_retention_actions.csv     # Actions dimension
│           ├── dim_date.csv                  # Date dimension
│           ├── kpi_summary.csv               # Executive KPIs
│           └── segment_summary.csv           # Segment summary
│
└── 📂 visuals/                               # Visualization outputs
    │
    ├── # Forecasting
    ├── time_series_decomposition.png         # Trend, seasonal, residual
    ├── prophet_forecast.png                  # Prophet predictions
    ├── arima_forecast.png                    # ARIMA predictions
    ├── forecast_comparison.png               # All models comparison
    ├── forecast_monthly_breakdown.png        # Monthly forecast bars
    │
    ├── # Churn
    ├── logistic_regression_features.png      # LR feature importance
    ├── logistic_regression_roc.png           # LR ROC curve
    ├── random_forest_features.png            # RF feature importance
    ├── random_forest_roc.png                 # RF ROC curve
    ├── churn_distribution.png                # Churn probability dist
    │
    ├── # Profitability
    ├── profitability_overview.png            # Margin analysis
    ├── revenue_profit_contribution.png       # Contribution charts
    ├── profitability_by_country.png          # Geographic analysis
    ├── cost_structure_analysis.png           # Cost breakdown
    │
    ├── # Segmentation
    ├── rfm_segmentation.png                  # RFM distribution
    ├── cohort_retention_heatmap.png          # Retention heatmap
    ├── retention_and_revenue.png             # Retention curve
    ├── behavioral_segmentation.png           # Cluster analysis
    │
    └── # Executive
    ├── executive_dashboard_final.png         # Main dashboard
    ├── business_performance_summary.png      # Performance charts
    └── risk_retention_analysis.png           # Risk analysis
```

## File Statistics

| Category | Files | Total Size |
|----------|-------|------------|
| Source Code | 10 | ~150 KB |
| Data Files | 2 | ~2 MB |
| Output CSVs | 40+ | ~10 MB |
| Visualizations | 20+ | ~5 MB |
| Models | 3 | ~2 MB |
| Documentation | 5 | ~50 KB |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/financial-operations-analytics.git
cd financial-operations-analytics
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# View results
ls outputs/unified/
cat outputs/unified/executive_summary_report.txt
```
