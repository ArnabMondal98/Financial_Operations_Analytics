# Financial Operations Analytics

<p align="center">
  <strong>End-to-End Financial Analytics Platform</strong><br>
  Revenue Forecasting | Churn Prediction | Profitability Analysis | Customer Segmentation
</p>

---

## 📊 Business Objectives

This project delivers a comprehensive financial operations analytics solution designed to:

1. **Forecast Revenue** - Predict future revenue using Prophet and ARIMA models for strategic planning
2. **Predict Customer Churn** - Identify at-risk customers before they leave using ML models
3. **Analyze Profitability** - Calculate customer-level profitability and identify margin optimization opportunities
4. **Segment Customers** - Group customers using RFM analysis and behavioral clustering for targeted marketing
5. **Enable Data-Driven Decisions** - Provide executive dashboards and Power BI-ready datasets

### Target Outcomes
- 📈 **12-month revenue forecast** with confidence intervals
- ⚠️ **Early churn warning** with actionable retention recommendations
- 💰 **Customer profitability tiers** for resource allocation
- 🎯 **Behavioral segments** for personalized engagement
- 📋 **Executive KPI dashboard** for leadership visibility

---

## 🔬 Analytics Methodology

### 1. Revenue Forecasting

| Model | Approach | Use Case |
|-------|----------|----------|
| **Prophet** | Additive model with seasonality | Captures holiday effects, yearly patterns |
| **ARIMA** | Statistical time series | Short-term predictions, trend analysis |
| **Ensemble** | Weighted average | Combines strengths of both models |

**Process:**
- Aggregate transaction data to monthly revenue
- Decompose time series (trend, seasonality, residual)
- Train and validate models with holdout data
- Generate 12-month forecast with confidence intervals

### 2. Churn Prediction

| Model | Accuracy | ROC-AUC | Key Features |
|-------|----------|---------|--------------|
| **Logistic Regression** | ~85% | ~0.82 | Interpretable coefficients |
| **Random Forest** | ~88% | ~0.85 | Feature importance ranking |

**Features Used:**
- Transaction frequency and recency
- Payment failure rate
- Customer tenure
- Subscription tier
- Revenue trends

### 3. Profitability Analysis

**Cost Model:**
```
Total Cost = Acquisition + Servicing + Support + Infrastructure + Payment Processing
Profit = Revenue - Total Cost
Margin = Profit / Revenue × 100
```

**Profitability Tiers:**
- 🟢 **High Profit** (Margin > 50%) - Priority retention
- 🔵 **Medium Profit** (Margin 20-50%) - Optimization focus
- 🟡 **Low Profit** (Margin 0-20%) - Cost reduction
- 🔴 **Loss-Making** (Margin < 0%) - Review or exit

### 4. Customer Segmentation

**RFM Analysis:**
- **R**ecency - Days since last transaction
- **F**requency - Number of transactions
- **M**onetary - Total spend

**Segments:**
| Segment | Criteria | Action |
|---------|----------|--------|
| Champions | High R, F, M | Reward & referral programs |
| Loyal | High F, M | Upsell opportunities |
| At Risk | Low R, High F | Win-back campaigns |
| New | High R, Low F | Onboarding support |
| Hibernating | Low R, F, M | Reactivation offers |

---

## 📈 Model Performance Summary

### Revenue Forecasting
| Metric | Prophet | ARIMA | Ensemble |
|--------|---------|-------|----------|
| MAE | $45,000 | $52,000 | $48,000 |
| RMSE | $62,000 | $71,000 | $65,000 |
| MAPE | 8.2% | 9.5% | 8.8% |

### Churn Prediction
| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | 84.5% | 87.2% |
| Precision | 78.3% | 82.1% |
| Recall | 71.2% | 74.8% |
| F1 Score | 74.6% | 78.3% |
| ROC-AUC | 0.823 | 0.856 |

---

## 📊 KPI Results Summary

### Financial Performance
| KPI | Value |
|-----|-------|
| Total Revenue | $10.89M |
| Total Profit | $7.02M |
| Average Profit Margin | 52.0% |
| 12-Month Forecast | $7.92M |

### Customer Metrics
| KPI | Value |
|-----|-------|
| Total Customers | 2,500 |
| Active Customers | 2,264 |
| Churn Rate | 9.4% |
| High-Risk Customers | 230 |
| Revenue at Risk | $993K |

### Segmentation
| KPI | Value |
|-----|-------|
| Customer Segments | 11 |
| Top Segment | Potential Loyalists |
| Avg Customer LTV | $4,357 |

---

## 📁 Project Structure

```
financial-operations-analytics/
├── 📂 data/                          # Raw and generated datasets
│   ├── customers.csv                 # Customer master data
│   └── transactions.csv              # Transaction records
│
├── 📂 src/                           # Source code modules
│   ├── data_generator.py             # Synthetic data generation
│   ├── data_preprocessing.py         # Feature engineering
│   ├── revenue_forecasting.py        # Prophet & ARIMA models
│   ├── churn_prediction.py           # ML churn models
│   ├── profitability_analysis.py     # Profitability calculations
│   ├── customer_segmentation.py      # RFM & clustering
│   ├── retention_recommendations.py  # Retention engine
│   ├── unified_analytics.py          # Dashboard consolidation
│   └── __init__.py                   # Package initialization
│
├── 📂 outputs/                       # Analysis outputs
│   ├── processed_customers.csv       # Engineered features
│   ├── forecasts/                    # Revenue forecasts
│   ├── churn/                        # Churn predictions
│   ├── profitability/                # Profitability analysis
│   ├── segmentation/                 # Customer segments
│   ├── retention/                    # Retention recommendations
│   └── unified/                      # Consolidated outputs
│       └── dashboard/                # Power BI-ready files
│
├── 📂 models/                        # Trained ML models
│   ├── logistic_regression.pkl       # Churn - LR model
│   ├── random_forest.pkl             # Churn - RF model
│   └── scaler.pkl                    # Feature scaler
│
├── 📂 visuals/                       # Visualization outputs
│   ├── *_forecast.png                # Forecast charts
│   ├── *_segmentation.png            # Segment charts
│   ├── *_profitability.png           # Profit charts
│   └── executive_dashboard_final.png # Executive summary
│
├── 📂 notebooks/                     # Jupyter notebooks
│   └── financial_analytics.ipynb     # Interactive analysis
│
├── 📜 run_pipeline.py                # Main pipeline runner
├── 📜 requirements.txt               # Python dependencies
└── 📜 README.md                      # This documentation
```

---

## 🖥️ Dashboard Explanation

### Power BI-Ready Datasets

The pipeline exports 8 dashboard-ready files in `outputs/unified/dashboard/`:

| File | Description | Usage |
|------|-------------|-------|
| `fact_customer_metrics.csv` | Customer-level KPIs | Main fact table |
| `fact_monthly_summary.csv` | Monthly aggregations | Time intelligence |
| `fact_revenue_forecast.csv` | 12-month predictions | Forecast visuals |
| `dim_customer_attributes.csv` | Customer dimensions | Slicers/filters |
| `dim_retention_actions.csv` | Recommended actions | Action tracking |
| `dim_date.csv` | Date dimension | Calendar hierarchy |
| `kpi_summary.csv` | Executive KPIs | KPI cards |
| `segment_summary.csv` | Segment breakdown | Segment analysis |

### Executive Dashboard Views

1. **Overview** - Revenue, customers, churn rate, forecast
2. **Profitability** - Margin analysis, tier breakdown, trends
3. **Segmentation** - RFM distribution, segment performance
4. **Risk Analysis** - Churn risk, revenue at risk, actions

---

## 🚀 Instructions to Run Project

### Prerequisites

- Python 3.9+
- pip package manager
- 8GB+ RAM (for forecasting models)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/financial-operations-analytics.git
cd financial-operations-analytics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Full Pipeline (Recommended)
```bash
python run_pipeline.py
```
This will:
- Generate synthetic data (2,500 customers, ~30K transactions)
- Run all analytics modules
- Export dashboards and visualizations
- Generate executive reports

#### Option 2: Use Existing Data
```bash
python run_pipeline.py --no-generate
```

#### Option 3: Quick Run (Skip Forecasting)
```bash
python run_pipeline.py --skip-forecasting
```

#### Option 4: Custom Output Directory
```bash
python run_pipeline.py --output-dir ./my_output
```

### Running Individual Modules

```bash
cd financial-operations-analytics

# Generate data
python src/data_generator.py

# Preprocess
python src/data_preprocessing.py

# Forecast
python src/revenue_forecasting.py

# Churn prediction
python src/churn_prediction.py

# Profitability
python src/profitability_analysis.py

# Segmentation
python src/customer_segmentation.py

# Retention
python src/retention_recommendations.py

# Unified dashboard
python src/unified_analytics.py
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/financial_analytics.ipynb
```

---

## 📋 Output Files Reference

### Key Outputs

| Category | File | Description |
|----------|------|-------------|
| **Master Data** | `outputs/unified/master_customer_analytics.csv` | Complete customer view |
| **Forecast** | `outputs/forecasts/combined_forecast.csv` | 12-month predictions |
| **Churn** | `outputs/churn/churn_predictions.csv` | Risk scores |
| **Actions** | `outputs/retention/high_risk_action_list.csv` | Retention tasks |
| **Executive** | `outputs/unified/executive_summary_report.txt` | KPI summary |

### Visualization Outputs

| File | Description |
|------|-------------|
| `executive_dashboard_final.png` | 4-panel executive summary |
| `time_series_decomposition.png` | Trend/seasonal analysis |
| `forecast_comparison.png` | Prophet vs ARIMA vs Ensemble |
| `rfm_segmentation.png` | Customer segments |
| `cohort_retention_heatmap.png` | Retention by cohort |
| `profitability_overview.png` | Margin analysis |

---

## 🔧 Configuration

### Data Schema

**customers.csv**
| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique identifier |
| signup_date | date | Registration date |
| country | string | Customer location |
| industry | string | Business sector |
| subscription_type | string | Basic/Professional/Enterprise/Premium |
| monthly_fee | float | Monthly subscription amount |

**transactions.csv**
| Column | Type | Description |
|--------|------|-------------|
| transaction_id | string | Unique identifier |
| customer_id | string | Customer reference |
| transaction_date | date | Transaction timestamp |
| amount | float | Transaction value |
| payment_method | string | Payment type |
| transaction_status | string | Completed/Failed/Pending/Refunded |

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -am 'Add new analysis'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Create Pull Request

---

👨‍💻 Author

Arnab Mondal
Data Analyst | SQL | Python | Power BI | Databricks |

LinkedIn: https://www.linkedin.com/in/arnabmondal98/
GitHub: https://github.com/ArnabMondal98

---

<p align="center">
  <strong>Built with ❤️ for Data-Driven Decision Making</strong>
</p>
